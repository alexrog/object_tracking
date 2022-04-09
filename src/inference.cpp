#include "inference.h"
#include <ros/package.h>

Inference::Inference() : it(n)
{
	std::string path = ros::package::getPath("object_tracking");
    detector = NanoDet((path + "/src/nanodet.xml").c_str(), "MYRIAD", 32);
    height = detector.input_size[0];
    width = detector.input_size[1];

    sub_rgb = it.subscribe("/drone/camera/color/image_raw", 1, &Inference::imageCallback, this);
    //sub_depth = it.subscribe("camera/depth/image_raw", 1, &Inference::imageCallback, this);
    pub_bbox = n.advertise<geometry_msgs::QuaternionStamped>("/drone/rover/bounding_box", 5);
    //pub_rel_pos = n.advertise<geometry_msgs::PointStamped>("rover/rel_pos", 5);
    old_bboxes.push_back(-1);
    old_bboxes.push_back(-1);
    old_bboxes.push_back(-1);
    old_bboxes.push_back(-1);

    old_point[0] = -1;
	old_point[1] = -1;
	old_point[2] = -1;
}

void Inference::imageCallback(const sensor_msgs::ImageConstPtr &img_msg)
{
    using namespace cv;
    using namespace rs2;

    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    Mat color_mat = cv_ptr->image;

    Mat resized_img;
    object_rect effect_roi;
    resize_uniform(color_mat, resized_img, cv::Size(width, height), effect_roi);
    auto results = detector.detect(resized_img, 0.4, 0.5);
    std::vector<float> bboxes = get_bboxes(color_mat, results, effect_roi);

    float point[3];
    float mean_depth = 0;

    if (bboxes.size() == 0)
    {
        bboxes = old_bboxes;
        point[0] = old_point[0];
        point[1] = old_point[1];
        point[2] = old_point[2];

        count++;
        if (count > 30)
        {
            bboxes[0] = -1;
            bboxes[1] = -1;
            bboxes[2] = -1;
            bboxes[3] = -1;

            point[0] = -1;
            point[1] = -1;
            point[2] = -1;
        }
    }
    else
    {
        old_bboxes = bboxes;
        count = 0;
    }

    ROS_INFO("%f, %f, %f, %f, %d, %d\n", bboxes[0], bboxes[1], bboxes[2], bboxes[3], color_mat.cols, color_mat.rows);
    geometry_msgs::Quaternion msg;
    msg.x = bboxes[0]; // x of top left
    msg.y = bboxes[1]; // y of top left
    msg.z = bboxes[2]-bboxes[0]; // width
    msg.w = bboxes[3]-bboxes[1]; // height
    geometry_msgs::QuaternionStamped stamped_msg;
    stamped_msg.header = std_msgs::Header();
    stamped_msg.quaternion = msg;
    pub_bbox.publish(stamped_msg);
}

int Inference::resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size, object_rect &effect_area)
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
    if (ratio_src > ratio_dst)
    {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst)
    {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else
    {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w)
    {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        for (int i = 0; i < dst_h; i++)
        {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h)
    {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else
    {
        printf("error\n");
    }
    return 0;
}

std::vector<float> Inference::get_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes, object_rect effect_roi)
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
        const BoxInfo &bbox = bboxes[i];

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
