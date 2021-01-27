#pragma once

// C
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// C++
#include <string>
#include <thread>
#include <mutex>
#include <vector>

// ROS
#include <actionlib/server/simple_action_server.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>

// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

// darknet_ros_msgs
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/DetectObjects.h>

// Darknet
extern "C" {
	#include "darknet.h"
}

namespace darknet_ros {

class YoloObjectDetectorLite {

	ros::NodeHandle m_nh;

	std::vector<std::string> m_classLabels;
	network* m_net;
	float m_threshold;

	ros::ServiceServer m_service;
	ros::Publisher m_publisher;

	image_transport::ImageTransport m_imgTransport;
	image_transport::Subscriber m_imgSubscriber;

	std::mutex m_mutex;
	cv_bridge::CvImagePtr m_imagePtr;

	char** m_classLabelsForDebug;
	image** m_alphabetForDebug;

	void cameraCallback(const sensor_msgs::ImageConstPtr& msg);
	bool serviceCallback(darknet_ros_msgs::DetectObjects::Request& req, darknet_ros_msgs::DetectObjects::Response& res);

public:
	explicit YoloObjectDetectorLite(ros::NodeHandle nh);
	~YoloObjectDetectorLite();
};

}
