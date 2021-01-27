/*
 * yolo_object_detector_node.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#include <ros/ros.h>
#include <darknet_ros/YoloObjectDetectorLite.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "darknet_ros");
  ros::NodeHandle nodeHandle("~");
  darknet_ros::YoloObjectDetectorLite yoloObjectDetector(nodeHandle);

  ros::spin();
  return 0;
}
