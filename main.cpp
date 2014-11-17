#include "opencv2/opencv.hpp"
#include "BackgroundSubtract.h"
#include "Detector.h"

#include <opencv2/highgui/highgui_c.h>
#include "CTracker.h"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

float X=0,Y=0;
float Xmeasured=0,Ymeasured=0;
RNG rng;
//------------------------------------------------------------------------
// Mouse callbacks
//------------------------------------------------------------------------
void mv_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if(event == cv::EVENT_MOUSEMOVE)
	{
		X=(float)x;
		Y=(float)y;
	}
}

// ----------------------------------------------------------------------
// set to 0 for Bugs tracking example
// set to 1 for mouse tracking example
// ----------------------------------------------------------------------
#define ExampleNum 1

int main(int ac, char** av)
{
#if ExampleNum==0
	Scalar Colors[]={Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(0,255,255),Scalar(255,0,255),Scalar(255,127,255),Scalar(127,0,255),Scalar(127,0,127)};
	VideoCapture capture("..\\..\\data\\TrackingBugs.mp4");
	if(!capture.isOpened())
	{
	return 0;
	}
	namedWindow("Video");
	Mat frame;
	Mat gray;

	CTracker tracker(0.2,0.5,60.0,10,10);
	
	capture >> frame;
	cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
	CDetector* detector=new CDetector(gray);
	int k=0;
	vector<Point2d> centers;
	while(k!=27)
	{
	capture >> frame;
	if(frame.empty())
	{
	capture.set(CAP_PROP_POS_FRAMES,0);
	continue;
	}
	cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

	centers=detector->Detect(gray);

	for(int i=0; i<centers.size(); i++)
	{
	circle(frame,centers[i],3,Scalar(0,255,0),1,CV_AA);
	}


	if(centers.size()>0)
	{
		tracker.Update(centers);
		
		cout << tracker.tracks.size()  << endl;

		for(int i=0;i<tracker.tracks.size();i++)
		{
			if(tracker.tracks[i]->trace.size()>1)
			{
				for(int j=0;j<tracker.tracks[i]->trace.size()-1;j++)
				{
					line(frame,tracker.tracks[i]->trace[j],tracker.tracks[i]->trace[j+1],Colors[tracker.tracks[i]->track_id%9],2,CV_AA);
				}
			}
		}
	}

	imshow("Video",frame);

	k=waitKey(20);
	}
	delete detector;
	destroyAllWindows();
	return 0;
#else

	int k=0;
	Scalar Colors[]={Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(0,255,255),Scalar(255,255,255)};
	cv::namedWindow("Video");
	Mat frame=Mat(800,800,CV_8UC3);

	VideoWriter vw=VideoWriter::VideoWriter("output.mpeg", CV_FOURCC('P','I','M','1'), 20, frame.size());

	// Set mouse callback
	cv::setMouseCallback("Video",mv_MouseCallback,0);

	CTracker tracker(0.2,0.5,60.0,25,25);
	float alpha=0;
	while(k!=27)
	{
		frame=Scalar::all(0);
		
		// Noise addition (measurements/detections simulation )
		Xmeasured=X+rng.gaussian(2.0);
		Ymeasured=Y+rng.gaussian(2.0);

		// Append circulating around mouse points (frequently intersecting)
		vector<Point2d> pts;
		pts.push_back(Point2d(Xmeasured+100.0*sin(-alpha),Ymeasured+100.0*cos(-alpha)));
		pts.push_back(Point2d(Xmeasured+100.0*sin(alpha),Ymeasured+100.0*cos(alpha)));
		pts.push_back(Point2d(Xmeasured+100.0*sin(alpha/2.0),Ymeasured+100.0*cos(alpha/2.0)));
		pts.push_back(Point2d(Xmeasured+100.0*sin(alpha/3.0),Ymeasured+100.0*cos(alpha/1.0)));
		alpha+=0.05;


	for(int i=0; i<pts.size(); i++)
	{
	circle(frame,pts[i],3,Scalar(0,255,0),1,CV_AA);
	}

		tracker.Update(pts);
		
		std::cout << tracker.tracks.size()  << endl;

		for(int i=0;i<tracker.tracks.size();i++)
		{
			if(tracker.tracks[i]->trace.size()>1)
			{
				for(int j=0;j<tracker.tracks[i]->trace.size()-1;j++)
				{
					line(frame,tracker.tracks[i]->trace[j],tracker.tracks[i]->trace[j+1],Colors[i%6],2,CV_AA);
				}
			}
		}

		imshow("Video",frame);
		vw << frame;

		k=waitKey(10);
	}

	vw.release();
	cv::destroyAllWindows();
	return 0;

#endif
}
