#include "inference.h"

Inference::Inference() : it(n)
{
    /*pipeline pipe;
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

    intrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    pipe.stop();*/

    detector = NanoDet("/home/px4vision/catkin/src/auav_2022_sample/object_tracking/src/nanodet.xml", "MYRIAD", 32);
    height = detector.input_size[0];
    width = detector.input_size[1];

    sub_rgb = it.subscribe("camera/color/image_raw", 1, &Inference::imageCallback, this);
    //sub_depth = it.subscribe("camera/depth/image_raw", 1, &Inference::imageCallback, this);
    pub_bbox = n.advertise<geometry_msgs::QuaternionStamped>("rover/bounding_box", 5);
    //pub_rel_pos = n.advertise<geometry_msgs::PointStamped>("rover/rel_pos", 5);
}

Inference::void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    using namespace cv;
    using namespace rs2;

    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
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
        /*Rect object((int)(bboxes[0]), (int)(bboxes[3]),
                    (int)(bboxes[2] - bboxes[0]),
                    (int)(bboxes[3] - bboxes[1]));
        object = object & Rect(0, 0, depth_mat.cols, depth_mat.rows);
        float pixel[2];
        pixel[0] = bboxes[0] + (bboxes[2] - bboxes[0]) / 2;
        pixel[1] = bboxes[1] + (bboxes[3] - bboxes[1]) / 2;
        Scalar m = mean(depth_mat(object));
        mean_depth = (float)m[0];
        rs2_deproject_pixel_to_point(point, &intrinsics, pixel, (float)m[0]);
        old_point[0] = point[0];
        old_point[1] = point[1];
        old_point[2] = point[2];*/

        bboxes[0] /= color_mat.cols;
        bboxes[1] /= color_mat.rows;
        bboxes[2] /= color_mat.cols;
        bboxes[3] /= color_mat.rows;
        old_bboxes = bboxes;
        count = 0;
    }

    ROS_INFO("%f, %f, %f, %f, %d, %d\n", bboxes[0], bboxes[1], bboxes[2], bboxes[3], color_mat.cols, color_mat.rows);
    //ROS_INFO("%f, %f, %f, depth: %f", point[0], point[1], point[2], mean_depth);
    geometry_msgs::Quaternion msg;
    msg.x = bboxes[0];
    msg.y = bboxes[1];
    msg.z = bboxes[2];
    msg.w = bboxes[3];
    geometry_msgs::QuaternionStamped stamped_msg;
    stamped_msg.header = std_msgs::Header();
    stamped_msg.quaternion = msg;
    pub_bbox.publish(stamped_msg);

    /*geometry_msgs::Point msg_pos;
    msg_pos.x = point[0];
    msg_pos.y = point[1];
    msg_pos.z = point[2];
    geometry_msgs::PointStamped stamped_msg_pos;
    stamped_msg_pos.header = std_msgs::Header();
    stamped_msg_pos.point = msg_pos;
    pub_rel_pos.publish(stamped_msg_pos);*/

    ros::spinOnce();   
}

Inference::int resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size, object_rect &effect_area)
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

Inference::std::vector<float> get_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes, object_rect effect_roi)
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