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
#include <algorithm>

// ROS
#include <actionlib/server/simple_action_server.h>
#include <geometry_msgs/Point.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>
#include <image_geometry/pinhole_camera_model.h>

// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

// darknet_ros_msgs
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/DetectObjects.h>
#include <darknet_ros_msgs/DetectObjects3D.h>

// Darknet
extern "C" {
	#include "darknet.h"
}

namespace darknet_ros {

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicy;

class YoloObjectDetectorLite {

	ros::NodeHandle m_nh;

	std::vector<std::string> m_classLabels;
	network* m_net;
	float m_threshold;

	ros::ServiceServer m_service, m_service3d;
	ros::Publisher m_publisher;

	image_transport::ImageTransport m_imgTransport;
	image_transport::SubscriberFilter m_imgSubscriber, m_imgSubscriberDepth;
	message_filters::Subscriber<sensor_msgs::CameraInfo> m_infoSubscriber;
	message_filters::Synchronizer<SyncPolicy> m_imgSync;

	std::mutex m_mutex;
	cv_bridge::CvImagePtr m_imagePtr;
	cv_bridge::CvImagePtr m_imagePtrDepth;
	image_geometry::PinholeCameraModel m_camModel;

	char** m_classLabelsForDebug;
	image** m_alphabetForDebug;

	void cameraCallback(const sensor_msgs::ImageConstPtr& msg);
	void cameraCallback3D(const sensor_msgs::ImageConstPtr& msgColor, const sensor_msgs::ImageConstPtr& msgDepth, const sensor_msgs::CameraInfoConstPtr& msgInfo);

	void calcZData(cv::Mat&& dimg, int64_t& zmin, int64_t& zmax);

	template <typename Lambda>
	bool serviceCallbackCommon(IplImage& ipl_image, Lambda lambda);
	bool serviceCallback(darknet_ros_msgs::DetectObjects::Request& req, darknet_ros_msgs::DetectObjects::Response& res);
	bool serviceCallback3D(darknet_ros_msgs::DetectObjects3D::Request& req, darknet_ros_msgs::DetectObjects3D::Response& res);

public:
	explicit YoloObjectDetectorLite(ros::NodeHandle nh);
	~YoloObjectDetectorLite();
};

}
