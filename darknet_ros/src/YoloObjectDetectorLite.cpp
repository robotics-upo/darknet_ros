#include "darknet_ros/YoloObjectDetectorLite.hpp"

extern "C" image ipl_to_image(IplImage* src);
extern "C" image** load_alphabet_with_file(const char* datafile);
extern "C" void generate_image(image p, IplImage* disp);

namespace darknet_ros {

YoloObjectDetectorLite::YoloObjectDetectorLite(ros::NodeHandle nh) :
	m_nh(nh), m_imgTransport(nh), m_imgSync(SyncPolicy(10)), m_classLabelsForDebug(), m_alphabetForDebug()
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

	std::string cameraDepthTopicName;
	m_nh.param("subscribers/camera_depth_reading/topic", cameraDepthTopicName, std::string("/camera/image_raw"));

	int cameraDepthQueueSize;
	m_nh.param("subscribers/camera_depth_reading/queue_size", cameraDepthQueueSize, 1);

	std::string serviceName;
	m_nh.param("services/camera_reading/name", serviceName, std::string("detect_objects"));

	std::string serviceName3D;
	m_nh.param("services/camera_reading_3d/name", serviceName3D, std::string("detect_objects_3d"));

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
	m_imgSubscriber.subscribe(m_imgTransport, cameraTopicName, cameraQueueSize,
		image_transport::TransportHints("compressed", ros::TransportHints(), nh, "RGB_image_transport"));
	m_imgSubscriberDepth.subscribe(m_imgTransport, cameraDepthTopicName, cameraDepthQueueSize,
		image_transport::TransportHints("raw", ros::TransportHints(), nh, "D_image_transport"));
	m_imgSync.connectInput(m_imgSubscriber, m_imgSubscriberDepth);
	m_imgSync.registerCallback(boost::bind(&YoloObjectDetectorLite::cameraCallback3D, this, _1, _2));
	//m_imgSubscriber.registerCallback(&YoloObjectDetectorLite::cameraCallback, this);
	if (!debugOutputTopicName.empty()) {
		m_publisher = m_nh.advertise<sensor_msgs::Image>(debugOutputTopicName, 1, true);
	}
	m_service = m_nh.advertiseService(serviceName, &YoloObjectDetectorLite::serviceCallback, this);
	m_service3d = m_nh.advertiseService(serviceName3D, &YoloObjectDetectorLite::serviceCallback3D, this);
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
		image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	{
		boost::unique_lock<std::mutex> l(m_mutex);
		m_imagePtr = std::move(image);
	}
}

void YoloObjectDetectorLite::cameraCallback3D(const sensor_msgs::ImageConstPtr& msgColor, const sensor_msgs::ImageConstPtr& msgDepth)
{
	cv_bridge::CvImagePtr imageColor, imageDepth;

	try {
		imageColor = cv_bridge::toCvCopy(msgColor, sensor_msgs::image_encodings::RGB8);
		imageDepth = cv_bridge::toCvCopy(msgDepth, sensor_msgs::image_encodings::MONO16);
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	{
		boost::unique_lock<std::mutex> l(m_mutex);
		m_imagePtr = std::move(imageColor);
		m_imagePtrDepth = std::move(imageDepth);
	}
}

void YoloObjectDetectorLite::calcZData(cv::Mat&& dimg, int64_t& zmin, int64_t& zmax)
{
	// Retrieve and sort the raw array of depth pixels
	uint16_t* pixels = dimg.ptr<uint16_t>();
	size_t num_pixels = dimg.total();
	std::sort(&pixels[0], &pixels[num_pixels]);

	// Find the first valid (non-0) depth value
	size_t first_valid = 0;
	while (first_valid < num_pixels && pixels[first_valid] == 0) first_valid ++;
	if (first_valid == num_pixels) {
		// No depth information available
		zmin = zmax = 0;
		return;
	}

	// Remove all invalid values from the array
	pixels += first_valid;
	num_pixels -= first_valid;

	// Reduce the array to its 80th percentile in order to remove background pixels
	num_pixels = ceilf(num_pixels * 0.8f);

	// Calculate average
	uint64_t average = 0;
	for (size_t i = 0; i < num_pixels; i ++)
		average += pixels[i];
	average /= num_pixels;

	// Calculate standard deviation
	uint64_t std_dev = 0;
	for (size_t i = 0; i < num_pixels; i ++) {
		int64_t err = pixels[i]-average;
		std_dev += err*err;
	}
	std_dev = sqrt(std_dev/num_pixels);

	// Return the range corresponding to a normal [-1,1]
	zmin = average - std_dev;
	zmax = average + std_dev;
}

template <typename Lambda>
bool YoloObjectDetectorLite::serviceCallbackCommon(IplImage& ipl_image, Lambda lambda)
{
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
			int64_t xmin = floorf(dn_image.w*(dets[i].bbox.x - 0.5f*dets[i].bbox.w));
			int64_t xmax =  ceilf(dn_image.w*(dets[i].bbox.x + 0.5f*dets[i].bbox.w));
			int64_t ymin = floorf(dn_image.h*(dets[i].bbox.y - 0.5f*dets[i].bbox.h));
			int64_t ymax =  ceilf(dn_image.h*(dets[i].bbox.y + 0.5f*dets[i].bbox.h));
			if (xmin < 0) xmin = 0;
			if (xmax > dn_image.w) xmax = dn_image.w;
			if (ymin < 0) ymin = 0;
			if (ymax > dn_image.h) ymax = dn_image.h;

			ROS_DEBUG("bbox[%d] x=%f y=%f w=%f h=%f\n", i, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
			ROS_DEBUG("  -->xmin=%ld xmax=%ld ymin=%ld ymax=%ld\n", xmin, xmax, ymin, ymax);
			lambda(
				m_classLabels[cat_id],
				cat_id,
				cat_prob,
				xmin, xmax, ymin, ymax
			);
		}
	}

	free_detections(dets, nboxes);
	return true;
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
	return serviceCallbackCommon(ipl_image,
	[&](const std::string& label, int16_t id, double prob, int64_t xmin, int64_t xmax, int64_t ymin, int64_t ymax) {
		darknet_ros_msgs::BoundingBox bb;
		bb.Class = label;
		bb.id = id;
		bb.probability = prob;
		bb.xmin = xmin;
		bb.xmax = xmax;
		bb.ymin = ymin;
		bb.ymax = ymax;
		res.bounding_boxes.push_back(bb);
	});
}

bool YoloObjectDetectorLite::serviceCallback3D(darknet_ros_msgs::DetectObjects3D::Request& req, darknet_ros_msgs::DetectObjects3D::Response& res)
{
	cv_bridge::CvImagePtr cv_color, cv_depth;
	{
		boost::unique_lock<std::mutex> l(m_mutex);
		cv_color = std::move(m_imagePtr);
		cv_depth = std::move(m_imagePtrDepth);
	}

	if (!cv_color || !cv_depth) {
		ROS_ERROR("No image received");
		return false;
	}

	IplImage ipl_image(cv_color->image);
	return serviceCallbackCommon(ipl_image,
	[&](const std::string& label, int16_t id, double prob, int64_t xmin, int64_t xmax, int64_t ymin, int64_t ymax) {
		darknet_ros_msgs::BoundingBox3D bb;
		bb.Class = label;
		bb.id = id;
		bb.probability = prob;
		bb.xmin = xmin;
		bb.xmax = xmax;
		bb.ymin = ymin;
		bb.ymax = ymax;
		calcZData(cv_depth->image(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin)).clone(), bb.zmin, bb.zmax);
		res.bounding_boxes.push_back(bb);
	});
}

}
