#include "nanodet_openvino.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"
#include "ros/ros.h"
#include <iostream>
#include <sstream>

auto detector = NanoDet("/home/px4vision/catkin_ws/src/object_tracking/src/nanodet.xml", "MYRIAD", 32);


struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    return 0;
}

std::vector<float> get_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi)
{
    cv::Mat image = bgr.clone();

    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;
    std::vector<float> boundingboxes;


    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];

        int point = (bbox.x1 - effect_roi.x) * width_ratio;
        boundingboxes.push_back(point);
        point = (bbox.y1 - effect_roi.y) * height_ratio;
        boundingboxes.push_back(point);
        point = (bbox.x2 - effect_roi.x) * width_ratio;
        boundingboxes.push_back(point);
        point = (bbox.y2 - effect_roi.y) * height_ratio;
        boundingboxes.push_back(point);
        boundingboxes.push_back(bbox.score * 100);
    }

    return boundingboxes;
}

int intelrealsense_inference(ros::Publisher pub)
{
    using namespace cv;
    using namespace rs2;

    const size_t inWidth      = 512;
    const size_t inHeight     = 288;
    const float WHRatio       = inWidth / (float)inHeight;
    const float inScaleFactor = 0.007843f;
    const float meanVal       = 127.5;
    int height = detector.input_size[0];
    int width = detector.input_size[1];

    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR)
                         .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)
    {
        cropSize = Size(static_cast<int>(profile.height() * WHRatio),
                        profile.height());
    }
    else
    {
        cropSize = Size(profile.width(),
                        static_cast<int>(profile.width() / WHRatio));
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
                    (profile.height() - cropSize.height) / 2),
              cropSize);
    ros::Rate loop_rate(10);
    std::vector<float> old_bboxes;
    old_bboxes.push_back(0.4);
    old_bboxes.push_back(0.4);
    old_bboxes.push_back(0.6);
    old_bboxes.push_back(0.6);

    while (1)
    {
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();

        auto color_frame = data.get_color_frame();

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);

        cv::Mat resized_img;
        object_rect effect_roi;
        resize_uniform(color_mat, resized_img, cv::Size(width, height), effect_roi);
        auto results = detector.detect(resized_img, 0.4, 0.5);
        std::vector<float> bboxes = get_bboxes(color_mat, results, effect_roi);
        if(bboxes.size() == 0) {
            bboxes = old_bboxes;
        }
        else {
            bboxes[0] /= color_mat.cols;
            bboxes[1] /= color_mat.rows;
            bboxes[2] /= color_mat.cols;
            bboxes[3] /= color_mat.rows;
            old_bboxes = bboxes;
        }
        geometry_msgs::Quaternion msg;
        msg.x = bboxes[0];
        msg.y = bboxes[1];
        msg.z = bboxes[2];
        msg.w = bboxes[3];
        geometry_msgs::QuaternionStamped stamped_msg;
        stamped_msg.header = std_msgs::Header();
        stamped_msg.quaternion = msg;
        pub.publish(msg);
        ros::spinOnce();
    }
    return 0;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "object_tracking");
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<geometry_msgs::QuaternionStamped>("rover/bounding_box",500);
    intelrealsense_inference(pub);
    return 0;
}
