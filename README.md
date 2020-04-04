![travis ci:](https://travis-ci.org/Smorodov/Multitarget-tracker.svg?branch=master)

# New videos!

* Traffic counting

[![Traffic counting:](https://img.youtube.com/vi/LzCv6Dr46kw/0.jpg)](https://youtu.be/LzCv6Dr46kw)

* ADAS

Coming soon...

# Multitarget (multiple objects) tracker

#### 1. Objects detector can be created with function [CreateDetector](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Detector/BaseDetector.cpp) with different values of the detectorType:

1.1. Based on background substraction: built-in Vibe (tracking::Motion_VIBE), SuBSENSE (tracking::Motion_SuBSENSE) and LOBSTER (tracking::Motion_LOBSTER); MOG2 (tracking::Motion_MOG2) from [opencv](https://github.com/opencv/opencv/blob/master/modules/video/include/opencv2/video/background_segm.hpp); MOG (tracking::Motion_MOG), GMG (tracking::Motion_GMG) and CNT (tracking::Motion_CNT) from [opencv_contrib](https://github.com/opencv/opencv_contrib/tree/master/modules/bgsegm). For foreground segmentation used contours from OpenCV with result as cv::RotatedRect

1.2. Haar face detector from OpenCV (tracking::Face_HAAR)

1.3. HOG pedestrian detector from OpenCV (tracking::Pedestrian_HOG) and C4 pedestrian detector from [sturkmen72](https://github.com/sturkmen72/C4-Real-time-pedestrian-detection)  (tracking::Pedestrian_C4)

1.4. MobileNet SSD detector (tracking::SSD_MobileNet) with opencv_dnn inference and pretrained models from [chuanqi305](https://github.com/chuanqi305/MobileNet-SSD)

1.5. YOLO detector (tracking::Yolo_OCV) with opencv_dnn inference and pretrained models from [pjreddie](https://pjreddie.com/darknet/yolo/)

1.6. YOLO detector (tracking::Yolo_Darknet) with darknet inference from [AlexeyAB](https://github.com/AlexeyAB/darknet) and pretrained models from [pjreddie](https://pjreddie.com/darknet/yolo/)

1.7. YOLO detector (tracking::Yolo_TensorRT) with NVidia TensorRT inference from [enazoe](https://github.com/enazoe/yolo-tensorrt) and pretrained models from [pjreddie](https://pjreddie.com/darknet/yolo/)

1.8. You can to use custom detector with bounding or rotated rectangle as output.

#### 2. Matching or solve an [assignment problem](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h):

2.1. Hungrian algorithm (tracking::MatchHungrian) with cubic time O(N^3) where N is objects count

2.2. Algorithm based on weighted bipartite graphs (tracking::MatchBipart) from [rdmpage](https://github.com/rdmpage/maximum-weighted-bipartite-matching) with time O(M * N^2) where N is objects count and M is connections count between detections on frame and tracking objects. It can be faster than Hungrian algorithm

2.3. [Distance](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h) from detections and objects: euclidean distance in pixels between centers (tracking::DistCenters), euclidean distance in pixels between rectangles (tracking::DistRects), Jaccard or IoU distance from 0 to 1 (tracking::DistJaccard)

#### 3. [Smoothing trajectories and predict missed objects](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h):

3.1. Linear Kalman filter from OpenCV (tracking::KalmanLinear)

3.2. Unscented Kalman filter from OpenCV (tracking::KalmanUnscented)

3.3. [Kalman goal](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h) is only coordinates (tracking::FilterCenter) or coordinates and size (tracking::FilterRect)

3.4. Simple [Abandoned detector](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h)

3.5. [Line intersection](https://github.com/Smorodov/Multitarget-tracker/blob/master/cars_counting/CarsCounting.cpp) counting

#### 4. [Advanced visual search](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h) for objects if they have not been detected:

4.1. No search (tracking::TrackNone)

4.2. Built-in DAT (tracking::TrackDAT) from [foolwood](https://github.com/foolwood/DAT), STAPLE (tracking::TrackSTAPLE) from [xuduo35](https://github.com/xuduo35/STAPLE) or LDES (tracking::TrackLDES) from [yfji](https://github.com/yfji/LDESCpp); KCF (tracking::TrackSTAPLE), MIL (tracking::TrackSTAPLE), MedianFlow (tracking::TrackSTAPLE), GOTURN (tracking::TrackSTAPLE), MOSSE (tracking::TrackSTAPLE) or CSRT (tracking::TrackSTAPLE) from [opencv_contrib](https://github.com/opencv/opencv_contrib/tree/master/modules/tracking)

With this option the tracking can work match slower but more accuracy.

#### 5. Pipeline

5.1. Syncronous [pipeline - SyncProcess](https://github.com/Smorodov/Multitarget-tracker/blob/master/example/VideoExample.h):
- get frame from capture device;
- decoding;
- objects detection (1);
- tracking (2-4);
- show result.

This pipeline is good if all algorithms are fast and works faster than time between two frames (40 ms for device with 25 fps). Or it can be used if we have only 1 core for all (no parallelization).

5.2. Pipeline with [2 threads - AsyncProcess](https://github.com/Smorodov/Multitarget-tracker/blob/master/example/VideoExample.h):
- 1th thread takes frame t and makes capture, decoding and objects detection;
- 2th thread takes frame t-1, results from first thread and makes tracking and results presentation (this is the Main read).

So we have a latency on 1 frame but on two free CPU cores we can increase performance on 2 times.

5.3. Fully [acynchronous pipeline](https://github.com/Smorodov/Multitarget-tracker/tree/master/async_detector) can be used if the objects detector works with low fps and we have a free 2 CPU cores. In this case we use 4 threads:
- 1th main thread is not busy and used for GUI and result presentation;
- 2th thread makes capture and decoding, puts frames in threadsafe queue;
- 3th thread is used for objects detection on the newest frame from the queue;
- 4th thread is used for objects tracking: waits the frame with detection from 3th tread and used advanced visual search (4) in intermediate frames from queue until it ges a frame with detections.

This pipeline can used with slow but accuracy DNN and track objects in intermediate frame in realtime without latency.


#### Demo Videos

* MobileNet SSD and tracking for low resolution and low quality videos from car DVR:

[![Tracking:](https://img.youtube.com/vi/Qssz6tVGoOc/0.jpg)](https://youtu.be/Qssz6tVGoOc)

* Mouse tracking:

[![Tracking:](https://img.youtube.com/vi/2fW5TmAtAXM/0.jpg)](https://www.youtube.com/watch?v=2fW5TmAtAXM)

* Motion Detection and tracking:

[![Motion Detection and tracking:](https://img.youtube.com/vi/GjN8jOy4kVw/0.jpg)](https://www.youtube.com/watch?v=GjN8jOy4kVw)

* Multiple Faces tracking:

[![Multiple Faces tracking:](https://img.youtube.com/vi/j67CFwFtciU/0.jpg)](https://www.youtube.com/watch?v=j67CFwFtciU)

* Simple Abandoned detector:

[![Simple Abandoned detector:](https://img.youtube.com/vi/fpkHRsFzspA/0.jpg)](https://www.youtube.com/watch?v=fpkHRsFzspA)

#### Tested Platforms
1. Ubuntu Linux 18.04 with x86 processors
2. Ubuntu Linux 18.04 with Nvidia Jetson Nano (YOLO + darknet on GPU works!)
3. Windows 10 (x64 and x32 builds)

#### Build
1. Download project sources
2. Install CMake
3. Install OpenCV (https://github.com/opencv/opencv) and OpenCV contrib (https://github.com/opencv/opencv_contrib) repositories
4. Configure project CmakeLists.txt, set OpenCV_DIR.
5. If opencv_contrib don't installed then disable options USE_OCV_BGFG=OFF, USE_OCV_KCF=OFF and USE_OCV_UKF=OFF
6. If you want to use native darknet YOLO detector with CUDA + cuDNN then set BUILD_YOLO_LIB=ON
7. For building example with low fps detector (now native darknet YOLO detector) and Tracker worked on each frame: BUILD_ASYNC_DETECTOR=ON
8. For building example with line crossing detection (cars counting): BUILD_CARS_COUNTING=ON
9. Go to the build directory and run make

**Usage:**

           Usage:
             ./MultitargetTracker <path to movie file> [--example]=<number of example 0..6> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs> [--gpu]=<use OpenCL> [--async]=<async pipeline>
             ./MultitargetTracker ../data/atrium.avi -e=1 -o=../data/atrium_motion.avi
           Press:
           * 'm' key for change mode: play|pause. When video is paused you can press any key for get next frame.
           * Press Esc to exit from video

           Params: 
           1. Movie file, for example ../data/atrium.avi
           2. [Optional] Number of example: 0 - MouseTracking, 1 - MotionDetector, 2 - FaceDetector, 3 - PedestrianDetector, 4 - MobileNet SSD detector, 5 - YOLO OpenCV detector, 6 - Yolo Darknet detector, 7 - YOLO TensorRT Detector
              -e=0 or --example=1
           3. [Optional] Frame number to start a video from this position
              -sf=0 or --start_frame==1500
           4. [Optional] Play a video to this position (if 0 then played to the end of file)
              -ef=0 or --end_frame==200
           5. [Optional] Delay in milliseconds after video ending
              -ed=0 or --end_delay=1000
           6. [Optional] Name of result video file
              -o=out.avi or --out=result.mp4
           7. [Optional] Show Trackers logs in terminal
              -sl=1 or --show_logs=0
           8. [Optional] Use built-in OpenCL
              -g=1 or --gpu=0
           9. [Optional] Use 2 threads for processing pipeline
              -a=1 or --async=0


#### Thirdparty libraries
* OpenCV (and contrib): https://github.com/opencv/opencv and https://github.com/opencv/opencv_contrib
* Vibe: https://github.com/BelBES/VIBE
* SuBSENSE and LOBSTER: https://github.com/ethereon/subsense
* GTL: https://github.com/rdmpage/graph-template-library
* MWBM: https://github.com/rdmpage/maximum-weighted-bipartite-matching
* Pedestrians detector: https://github.com/sturkmen72/C4-Real-time-pedestrian-detection
* Non Maximum Suppression: https://github.com/Nuzhny007/Non-Maximum-Suppression
* MobileNet SSD models: https://github.com/chuanqi305/MobileNet-SSD
* YOLO models: https://pjreddie.com/darknet/yolo/
* Darknet inference: https://github.com/AlexeyAB/darknet
* NVidia TensorRT inference: https://github.com/enazoe/yolo-tensorrt
* GOTURN models: https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking
* DAT tracker: https://github.com/foolwood/DAT
* STAPLE tracker: https://github.com/xuduo35/STAPLE
* LDES tracker: https://github.com/yfji/LDESCpp

#### License
GNU GPLv3: http://www.gnu.org/licenses/gpl-3.0.txt 

#### Project cititations
1. Jeroen PROVOOST "Camera gebaseerde analysevan de verkeersstromen aaneen kruispunt", 2014 ( https://iiw.kuleuven.be/onderzoek/eavise/mastertheses/provoost.pdf )
2. Roberto Ciano, Dimitrij Klesev "Autonome Roboterschwarme in geschlossenen Raumen", 2015 ( https://www.hs-furtwangen.de/fileadmin/user_upload/fak_IN/Dokumente/Forschung_InformatikJournal/informatikJournal_2016.pdf#page=18 )
3. Wenda Qin, Tian Zhang, Junhe Chen "Traffic Monitoring By Video: Vehicles Tracking and Vehicle Data Analysing", 2016 ( http://cs-people.bu.edu/wdqin/FinalProject/CS585%20FinalProjectReport.html )
4. Ipek BARIS "CLASSIFICATION AND TRACKING OF VEHICLES WITH HYBRID CAMERA SYSTEMS", 2016 ( http://cvrg.iyte.edu.tr/publications/IpekBaris_MScThesis.pdf )
5. Cheng-Ta Lee, Albert Y. Chen, Cheng-Yi Chang "In-building Coverage of Automated External Defibrillators Considering Pedestrian Flow", 2016 ( http://www.see.eng.osaka-u.ac.jp/seeit/icccbe2016/Proceedings/Full_Papers/092-132.pdf )
6. Omid Noorshams "Automated systems to assess weights and activity in grouphoused mice", 2017 ( https://pdfs.semanticscholar.org/e5ff/f04b4200c149fb39d56f171ba7056ab798d3.pdf ) 
7. RADEK VOPÁLENSKÝ "DETECTION,TRACKING AND CLASSIFICATION OF VEHICLES", 2018 ( https://www.vutbr.cz/www_base/zav_prace_soubor_verejne.php?file_id=181063 )
8. Márk Rátosi, Gyula Simon "Real-Time Localization and Tracking  using Visible Light Communication", 2018, ( https://ieeexplore.ieee.org/abstract/document/8533800 )
9. Thi Nha Ngo, Kung-Chin Wu, En-Cheng Yang, Ta-Te Lin "Areal-time imaging system for multiple honey bee tracking and activity monitoring", 2019 ( https://www.sciencedirect.com/science/article/pii/S0168169919301498 )
