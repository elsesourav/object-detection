#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
   std::cout << "OpenCV version: " << CV_VERSION << std::endl;

   // Create a simple black image
   cv::Mat image = cv::Mat::zeros(300, 300, CV_8UC3);

   // Draw a circle
   cv::circle(image, cv::Point(150, 150), 100, cv::Scalar(0, 255, 0), -1);

   // Display the image (if a windowing system is available)
   cv::imshow("OpenCV Test", image);
   cv::waitKey(0);
   std::cout << "Hello Sourav: " << CV_VERSION << std::endl;
   cv::waitKey(0);
   std::cout << "Image created successfully!" << std::endl;

   return 0;
}
