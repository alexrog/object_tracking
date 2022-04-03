
#ifndef _INFERENCE_H_
#define _INFERENCE_H_

#include "nanodet_openvino.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include "std_msgs/String.h"
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include "cv-helpers.hpp"
#include "ros/ros.h"
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <iostream>
#include <sstream>

struct object_rect
{
    int x;
    int y;
    int width;
    int height;
};

class Inference
{
    public:
        Inference();
        NanoDet detector; 
        ros::NodeHandle n;
        image_transport::ImageTransport it(n);
        image_transport::Subscriber sub_rgb;
        //image_transport::Subscriber sub_depth;
        ros::Publisher pub_bbox;
        ros::Publisher pub_rel_pos;
        rs2_intrinsics intrinsics;
        const size_t inWidth = 512;
        const size_t inHeight = 288;
        const float WHRatio = inWidth / (float)inHeight;
        const float inScaleFactor = 0.007843f;
        const float meanVal = 127.5;
        int height;
        int width;
        int count = 0;
        std::vector<float> old_bboxes;
        float old_point[3];

        void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    private:
        int resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size, object_rect &effect_area);
        std::vector<float> get_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes, object_rect effect_roi);
}

#endif