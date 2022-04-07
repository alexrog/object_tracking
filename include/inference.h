
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
#include "BYTETracker.h"

struct object_rect
{
    int x;
    int y;
    int width;
    int height;
};

struct bbox_tlbr {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct xyz{
    float x;
    float y;
    float z;
};

class Inference
{
    public:
        Inference();
        ~Inference();
        NanoDet detector; 
        std::shared_ptr<BYTETracker> tracker;
        ros::NodeHandle n;
        image_transport::ImageTransport it;
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
        const float conf_threshold = 0.4;
        const float nms_threshold = 0.5;
        const bbox_tlbr null_bbox = {.x1=-1.0, .y1=-1.0, .x2=-1.0, .y2=-1.0};
        const xyz null_xyz = {.x = -1.0, .y=-1.0, .z=-1.0};
        int elapsed_frames = 0; //Modified by update_bbox
        cv::Size model_in_size;
        bbox_tlbr old_bbox;
        bbox_tlbr new_bbox;
        xyz old_cords;
        xyz new_cords;
        std::vector<float> old_bboxes;
        float old_point[3];

        void imageCallback(const sensor_msgs::ImageConstPtr& img_msg);
    private:
        int resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size, object_rect &effect_area);
        std::vector<float> get_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes, object_rect effect_roi);
        void update_bbox_cord(vector<STrack> stracks, bbox_tlbr& old_bbox, bbox_tlbr& new_bbox, xyz& old_cords, xyz& new_cords);
        vector<Object> convert_bytetrack(const std::vector<BoxInfo>& results, const cv::Mat& image, const object_rect effect_roi);
};

#endif