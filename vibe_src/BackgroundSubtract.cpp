#include "BackgroundSubtract.h"

BackgroundSubtract::BackgroundSubtract()
{
	model = libvibeModelNew();
}

BackgroundSubtract::~BackgroundSubtract()
{
	libvibeModelFree(model);
}

void BackgroundSubtract::init(cv::Mat &image)
{
	int32_t width = image.size().width;
	int32_t height = image.size().height;
	int32_t stride = image.channels()*image.size().width;
	uint8_t *image_data = (uint8_t*)image.data;

	libvibeModelInit(model, image_data, width, height, stride);
}

void BackgroundSubtract::subtract(const cv::Mat &image, cv::Mat &foreground)
{
	uint8_t *image_data = (uint8_t*)image.data;
	uint8_t *segmentation_map = (uint8_t*)foreground.data;
	cv::Mat erodeElement = cv::getStructuringElement( 0, cv::Size( 5, 5 ), cv::Point( -1, -1 ) ); 
	cv::Mat dilateElement = cv::getStructuringElement( 0, cv::Size( 3, 3 ), cv::Point( -1, -1 ) ); 

	libvibeModelUpdate(model, image_data, segmentation_map);

	//cv::dilate(foreground, foreground, dilateElement, cv::Point(-1,-1), 1);
	cv::erode(foreground, foreground, erodeElement, cv::Point(-1, -1), 1);
	cv::dilate(foreground, foreground, dilateElement, cv::Point(-1,-1), 2);
}