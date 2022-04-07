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
#include <iostream>
#include <sstream>
#include "inference.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "object_tracking");
	ROS_INFO("init the ros node");
    Inference inference;
    ros::spin();
	return 0;
}
