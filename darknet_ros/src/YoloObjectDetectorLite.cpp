#include "darknet_ros/YoloObjectDetectorLite.hpp"

extern "C" image ipl_to_image(IplImage* src);
extern "C" image** load_alphabet_with_file(const char* datafile);
extern "C" void generate_image(image p, IplImage* disp);

namespace darknet_ros {

YoloObjectDetectorLite::YoloObjectDetectorLite(ros::NodeHandle nh) :
	m_nh(nh), m_imgTransport(nh), m_classLabelsForDebug(), m_alphabetForDebug()
{
	ROS_INFO("[YoloObjectDetectorLite] Node started.");

	m_nh.param("yolo_model/detection_classes/names", m_classLabels, std::vector<std::string>{});
	for (unsigned i = 0; i < m_classLabels.size(); i ++) {
		ROS_INFO("Loaded category '%s'", m_classLabels[i].c_str());
	}

	m_nh.param("yolo_model/threshold/value", m_threshold, 0.3f);

	std::string cameraTopicName;
	m_nh.param("subscribers/camera_reading/topic", cameraTopicName, std::string("/camera/image_raw"));

	int cameraQueueSize;
	m_nh.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);

	std::string serviceName;
	m_nh.param("services/camera_reading/name", serviceName, std::string("detect_objects"));

	std::string debugOutputTopicName;
	m_nh.param("publishers/debug_image_output/topic", debugOutputTopicName, std::string(""));

	if (!debugOutputTopicName.empty()) {
		m_classLabelsForDebug = new char*[m_classLabels.size()];
		for (unsigned i = 0; i < m_classLabels.size(); i ++) {
			m_classLabelsForDebug[i] = const_cast<char*>(m_classLabels[i].c_str());
		}

		m_alphabetForDebug = load_alphabet_with_file(DARKNET_FILE_PATH "/data");
	}

	// Path to weights file
	std::string weightsModel, weightsPath;
	m_nh.param("yolo_model/weight_file/name", weightsModel, std::string("yolov2-tiny.weights"));
	m_nh.param("weights_path", weightsPath, std::string("/default"));
	weightsPath += "/" + weightsModel;

	// Path to config file
	std::string configModel, configPath;
	m_nh.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
	m_nh.param("config_path", configPath, std::string("/default"));
	configPath += "/" + configModel;

	// Create the network!
	m_net = load_network(const_cast<char*>(configPath.c_str()), const_cast<char*>(weightsPath.c_str()), 0);
	set_batch_network(m_net, 1);

	// ROS
	m_imgSubscriber = m_imgTransport.subscribe(cameraTopicName, cameraQueueSize, &YoloObjectDetectorLite::cameraCallback, this);
	if (!debugOutputTopicName.empty()) {
		m_publisher = m_nh.advertise<sensor_msgs::Image>(debugOutputTopicName, 1, true);
	}
	m_service = m_nh.advertiseService(serviceName, &YoloObjectDetectorLite::serviceCallback, this);
}

YoloObjectDetectorLite::~YoloObjectDetectorLite()
{
	ROS_INFO("[YoloObjectDetectorLite] Node exiting.");
	free_network(m_net);
	if (m_alphabetForDebug)
		free(m_alphabetForDebug);
	if (m_classLabelsForDebug)
		delete[] m_classLabelsForDebug;
}

void YoloObjectDetectorLite::cameraCallback(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr image;

	try {
		image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	{
		boost::unique_lock<std::mutex> l(m_mutex);
		m_imagePtr = std::move(image);
	}
}

bool YoloObjectDetectorLite::serviceCallback(darknet_ros_msgs::DetectObjects::Request& req, darknet_ros_msgs::DetectObjects::Response& res)
{
	cv_bridge::CvImagePtr cv_image;
	{
		boost::unique_lock<std::mutex> l(m_mutex);
		cv_image = std::move(m_imagePtr);
	}

	if (!cv_image) {
		ROS_ERROR("No image received");
		return false;
	}

	IplImage ipl_image(cv_image->image);
	image dn_image = ipl_to_image(&ipl_image);
	image dn_image_letterboxed = letterbox_image(dn_image, m_net->w, m_net->h);

	network_predict(m_net, dn_image_letterboxed.data);
	int nboxes;
	detection* dets = get_network_boxes(m_net, dn_image.w, dn_image.h, m_threshold, 0.5f, 0, 1, &nboxes);
	do_nms_obj(dets, nboxes, m_net->layers[m_net->n - 1].classes, 0.4f);

	if (m_publisher) {
		draw_detections(dn_image, dets, nboxes, m_threshold, m_classLabelsForDebug, m_alphabetForDebug, m_classLabels.size());

		IplImage* ipl = cvCreateImage(cvSize(dn_image.w, dn_image.h), IPL_DEPTH_8U, dn_image.c);
		generate_image(dn_image, ipl);
		cv::Mat cvMat = cv::cvarrToMat(ipl);

		cv_bridge::CvImage cvImage;
		cvImage.header.stamp = ros::Time::now();
		cvImage.header.frame_id = "detection_image";
		cvImage.encoding = sensor_msgs::image_encodings::BGR8;
		cvImage.image = cvMat;
		m_publisher.publish(*cvImage.toImageMsg());

		cvReleaseImage(&ipl);
	}

	free_image(dn_image);

	for (int i = 0; i < nboxes; i ++) {
		int cat_id = -1;
		float cat_prob = 0.0f;
		for (int j = 0; j < m_classLabels.size(); j ++) {
			if (dets[i].prob[j] >= m_threshold) {
				ROS_DEBUG("Box %d Probability of being '%s': %.2f", i, m_classLabels[j].c_str(), dets[i].prob[j]);
				if (dets[i].prob[j] > cat_prob) {
					cat_id = j;
					cat_prob = dets[i].prob[j];
				}
			}
		}

		if (cat_id >= 0) {
			darknet_ros_msgs::BoundingBox bb;
			bb.Class = m_classLabels[cat_id];
			bb.id = cat_id;
			bb.probability = cat_prob;
			bb.xmin = dn_image.w*(dets[i].bbox.x);
			bb.xmax = dn_image.w*(dets[i].bbox.x + 0.5f*dets[i].bbox.w);
			bb.ymin = dn_image.h*(dets[i].bbox.y);
			bb.ymax = dn_image.h*(dets[i].bbox.y + 0.5f*dets[i].bbox.h);
			res.bounding_boxes.push_back(bb);
		}
	}

	free_detections(dets, nboxes);

	return true;
}

}
