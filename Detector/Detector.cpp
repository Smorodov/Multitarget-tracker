#include "Detector.h"

CDetector::CDetector(cv::Mat& gray)
{
	fg=gray.clone();
	bs=new BackgroundSubtract;
	bs->init(gray);
	std::vector<cv::Rect> rects;
	std::vector<cv::Point2d> centers;
	
}

//----------------------------------------------------------------------
// Detector
//----------------------------------------------------------------------
void CDetector::DetectContour(cv::Mat& img, std::vector<cv::Rect>& Rects,std::vector<cv::Point2d>& centers)
{
	Rects.clear();
	centers.clear();
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Mat edges=img.clone();
	Canny(img, edges, 50, 190, 3);
	findContours(edges,contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
	if(contours.size()>0)
	{
        for (size_t i = 0; i < contours.size(); i++ )
		{
			cv::Rect r=cv::boundingRect(contours[i]);
			Rects.push_back(r);
			centers.push_back((r.br()+r.tl())*0.5);
		}
	}
}

std::vector<cv::Point2d> CDetector::Detect(cv::Mat& gray)
{
		bs->subtract(gray,fg);
		// rects - bounding rectangles
		// centers - centers of bounding rectangles
		/*
		cv::Mat fg2;
		fg.convertTo(fg2,CV_32FC1);
		cv::GaussianBlur(fg2,fg2,Size(5,5),1.0);
		cv::Laplacian(fg2,fg2,CV_32FC1);

		normalize(fg2,fg2,0,255,cv::NORM_MINMAX);
		fg2.convertTo(fg2,CV_8UC1);
		cv::applyColorMap(fg2,fg2,COLORMAP_JET);
		imshow("Foreground",fg2);
		*/
		DetectContour(fg,rects,centers);
		return centers;
}

CDetector::~CDetector(void)
{
	delete bs;
}
