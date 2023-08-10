[![Status](https://github.com/Nuzhny007/Multitarget-tracker/actions/workflows/cmake.yml/badge.svg?branch=master)](https://github.com/Nuzhny007/Multitarget-tracker/actions?query=workflow%3Abuild-Ubuntu)
[![CodeQL](https://github.com/Smorodov/Multitarget-tracker/workflows/CodeQL/badge.svg?branch=master)](https://github.com/Smorodov/Multitarget-tracker/actions?query=workflow%3ACodeQL)

# Last changes

* YOLOv8 detector worked with TensorRT! Export pretrained Pytorch models [here (ultralytics/ultralytics)](https://github.com/ultralytics/ultralytics) to onnx format and run Multitarget-tracker with -e=6 example

* Some experiments with YOLOv7_mask and results with rotated rectangles: detector works tracker in progress

* YOLOv7 worked with TensorRT! Export pretrained Pytorch models [here (WongKinYiu/yolov7)](https://github.com/WongKinYiu/yolov7) to onnx format and run Multitarget-tracker with -e=6 example

* YOLOv6 worked with TensorRT! Download pretrained onnx models [here (meituan/YOLOv6)](https://github.com/meituan/YOLOv6/releases/tag/0.1.0) and run Multitarget-tracker with -e=6 example

# New videos!

* YOLOv7 instance segmentation

[![YOLOv7 instance segmentation:](https://img.youtube.com/vi/gZxuYyFz1dU/0.jpg)](https://youtu.be/gZxuYyFz1dU)


* Very fast and small objects tracking [Thnx Scianand](https://github.com/Smorodov/Multitarget-tracker/issues/367)

[![Fast and small motion:](https://img.youtube.com/vi/PalIIAfgX88/0.jpg)](https://youtu.be/PalIIAfgX88)

* Vehicles speed calculation with YOLO v4

[![Vehicles speed:](https://img.youtube.com/vi/qOHYvDwpsO0/0.jpg)](https://youtu.be/qOHYvDwpsO0)


* First step to ADAS with YOLO v4

[![Simple ADAS:](https://img.youtube.com/vi/5cgg5fy90Xg/0.jpg)](https://youtu.be/5cgg5fy90Xg)

# Multitarget (multiple objects) tracker

#### 1. Objects detector can be created with function [CreateDetector](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Detector/BaseDetector.cpp) with different values of the detectorType:

1.1. Based on background substraction: built-in Vibe (tracking::Motion_VIBE), SuBSENSE (tracking::Motion_SuBSENSE) and LOBSTER (tracking::Motion_LOBSTER); MOG2 (tracking::Motion_MOG2) from [opencv](https://github.com/opencv/opencv/blob/master/modules/video/include/opencv2/video/background_segm.hpp); MOG (tracking::Motion_MOG), GMG (tracking::Motion_GMG) and CNT (tracking::Motion_CNT) from [opencv_contrib](https://github.com/opencv/opencv_contrib/tree/master/modules/bgsegm). For foreground segmentation used contours from OpenCV with result as cv::RotatedRect

1.2. Haar face detector from OpenCV (tracking::Face_HAAR)

1.3. HOG pedestrian detector from OpenCV (tracking::Pedestrian_HOG) and C4 pedestrian detector from [sturkmen72](https://github.com/sturkmen72/C4-Real-time-pedestrian-detection)  (tracking::Pedestrian_C4)

1.4. Detector based on opencv_dnn (tracking::DNN_OCV) and pretrained models from [chuanqi305](https://github.com/chuanqi305/MobileNet-SSD) and [pjreddie](https://pjreddie.com/darknet/yolo/)

1.5. YOLO detector (tracking::Yolo_Darknet) with darknet inference from [AlexeyAB](https://github.com/AlexeyAB/darknet) and pretrained models from [pjreddie](https://pjreddie.com/darknet/yolo/)

1.6. YOLO detector (tracking::Yolo_TensorRT) with NVidia TensorRT inference from [enazoe](https://github.com/enazoe/yolo-tensorrt) and pretrained models from [pjreddie](https://pjreddie.com/darknet/yolo/)

1.7. You can to use custom detector with bounding or rotated rectangle as output.

#### 2. Matching or solve an [assignment problem](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h):

2.1. Hungrian algorithm (tracking::MatchHungrian) with cubic time O(N^3) where N is objects count

2.2. Algorithm based on weighted bipartite graphs (tracking::MatchBipart) from [rdmpage](https://github.com/rdmpage/maximum-weighted-bipartite-matching) with time O(M * N^2) where N is objects count and M is connections count between detections on frame and tracking objects. It can be faster than Hungrian algorithm

2.3. [Distance](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h) from detections and objects: euclidean distance in pixels between centers (tracking::DistCenters), euclidean distance in pixels between rectangles (tracking::DistRects), Jaccard or IoU distance from 0 to 1 (tracking::DistJaccard)

#### 3. [Smoothing trajectories and predict missed objects](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h):

3.1. Linear Kalman filter from OpenCV (tracking::KalmanLinear)

3.2. Unscented Kalman filter from OpenCV (tracking::KalmanUnscented) with constant velocity or constant acceleration models

3.3. [Kalman goal](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h) is only coordinates (tracking::FilterCenter) or coordinates and size (tracking::FilterRect)

3.4. Simple [Abandoned detector](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h)

3.5. [Line intersection](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/CarsCounting.cpp) counting

#### 4. [Advanced visual search](https://github.com/Smorodov/Multitarget-tracker/blob/master/src/Tracker/Ctracker.h) for objects if they have not been detected:

4.1. No search (tracking::TrackNone)

4.2. Built-in DAT (tracking::TrackDAT) from [foolwood](https://github.com/foolwood/DAT), STAPLE (tracking::TrackSTAPLE) from [xuduo35](https://github.com/xuduo35/STAPLE) or LDES (tracking::TrackLDES) from [yfji](https://github.com/yfji/LDESCpp); KCF (tracking::TrackKCF), MIL (tracking::TrackMIL), MedianFlow (tracking::TrackMedianFlow), GOTURN (tracking::TrackGOTURN), MOSSE (tracking::TrackMOSSE) or CSRT (tracking::TrackCSRT) from [opencv_contrib](https://github.com/opencv/opencv_contrib/tree/master/modules/tracking)

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

Also you can read [Wiki in Russian](https://github.com/Smorodov/Multitarget-tracker/wiki).

#### Demo Videos

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
4. Configure project CmakeLists.txt, set OpenCV_DIR (-DOpenCV_DIR=/path/to/opencv/build).
5. If opencv_contrib don't installed then disable options USE_OCV_BGFG=OFF, USE_OCV_KCF=OFF and USE_OCV_UKF=OFF
6. If you want to use native darknet YOLO detector with CUDA + cuDNN then set BUILD_YOLO_LIB=ON  (Install first CUDA and cuDNN libraries from Nvidia)
7. If you want to use YOLO detector with TensorRT then set BUILD_YOLO_TENSORRT=ON (Install first TensorRT library from Nvidia)
8. For building example with low fps detector (now native darknet YOLO detector) and Tracker worked on each frame: BUILD_ASYNC_DETECTOR=ON
9. For building example with line crossing detection (cars counting): BUILD_CARS_COUNTING=ON
10. Go to the build directory and run make

**Full build:**

           git clone https://github.com/Smorodov/Multitarget-tracker.git
           cd Multitarget-tracker
           mkdir build
           cd build
           cmake . .. -DUSE_OCV_BGFG=ON -DUSE_OCV_KCF=ON -DUSE_OCV_UKF=ON -DBUILD_YOLO_LIB=ON -DBUILD_YOLO_TENSORRT=ON -DBUILD_ASYNC_DETECTOR=ON -DBUILD_CARS_COUNTING=ON
           make -j

How to run cmake on Windows for Visual Studio 15 2017 Win64: [example](https://github.com/Smorodov/Multitarget-tracker/blob/master/data/cmake_vs2017.bat). You need to add directory with cmake.exe to PATH and change build params in cmake.bat


**Usage:**

           Usage:
             ./MultitargetTracker <path to movie file> [--example]=<number of example 0..7> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs> [--gpu]=<use OpenCL> [--async]=<async pipeline> [--res]=<csv log file> [--settings]=<ini file> [--batch_size=<number of frames>]
             ./MultitargetTracker ../data/atrium.avi -e=1 -o=../data/atrium_motion.avi
           Press:
           * 'm' key for change mode: play|pause. When video is paused you can press any key for get next frame.
           * Press Esc to exit from video

           Params:
           1. Movie file, for example ../data/atrium.avi
           2. [Optional] Number of example: 0 - MouseTracking, 1 - MotionDetector, 2 - FaceDetector, 3 - PedestrianDetector, 4 - OpenCV dnn objects detector, 5 - Yolo Darknet detector, 6 - YOLO TensorRT Detector, Cars counting
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
           10. [Optional] Path to the csv file with tracking result
              -r=res.csv or --res=res.csv
           11. [Optional] Path to the ini file with tracker settings
              -s=settings.ini or --settings=settings.ini
           12. [Optional] Batch size - simultaneous detection on several consecutive frames
              -bs=2 or --batch_size=1

More details here: [How to run examples](https://github.com/Smorodov/Multitarget-tracker/wiki/Run-examples).

#### Using MT Tracking as a library in your CMake project

Build MTTracking in the usual way, and choose an installation prefix where the library will be installed
(see [CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html) for the defaults).

In the `build` directory run
```
$ cmake --install .
```
This will generate the CMake files needed to find the MTTracking package with libraries and include files for
your project. E.g.
```
MTTrackingConfig.cmake
MTTrackingConfigVersion.cmake
MTTrackingTargets.cmake
```

In your CMake project, do the following:
```
    find_package(MTTracking REQUIRED)
    target_include_directories(MyProjectTarget PUBLIC ${MTTracking_INCLUDE_DIR})
    target_link_libraries(MyProjectTarget PUBLIC MTTracking::mtracking MTTracking::mdetection)
```

You may need to provide CMake with the location to find the above `.cmake` files, e.g.
```
$ cmake -DMTTracking_DIR=<location_of_cmake_files> ..
```

If CMake succeeds at finding the package, you can use MTTracking in your project e.g.
```
#include <mtracking/Ctracker.h>
//...
    std::unique_ptr<BaseTracker> m_tracker;

	TrackerSettings settings;
	settings.SetDistance(tracking::DistJaccard);
    m_tracker = BaseTracker::CreateTracker(settings);
//...
```
And so on.

#### Thirdparty libraries
* OpenCV (and contrib): https://github.com/opencv/opencv and https://github.com/opencv/opencv_contrib
* Vibe: https://github.com/BelBES/VIBE
* SuBSENSE and LOBSTER: https://github.com/ethereon/subsense
* GTL: https://github.com/rdmpage/graph-template-library
* MWBM: https://github.com/rdmpage/maximum-weighted-bipartite-matching
* Pedestrians detector: https://github.com/sturkmen72/C4-Real-time-pedestrian-detection
* Non Maximum Suppression: https://github.com/Nuzhny007/Non-Maximum-Suppression
* MobileNet SSD models: https://github.com/chuanqi305/MobileNet-SSD
* YOLO v3 models: https://pjreddie.com/darknet/yolo/
* Darknet inference and YOLO v4 models: https://github.com/AlexeyAB/darknet
* NVidia TensorRT inference and YOLO v5 models: https://github.com/enazoe/yolo-tensorrt
* YOLOv6 models: https://github.com/meituan/YOLOv6/releases
* YOLOv7 models: https://github.com/WongKinYiu/yolov7
* GOTURN models: https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking
* DAT tracker: https://github.com/foolwood/DAT
* STAPLE tracker: https://github.com/xuduo35/STAPLE
* LDES tracker: https://github.com/yfji/LDESCpp
* Ini file parser: https://github.com/benhoyt/inih
* Circular Code from Lior Kogan

#### License
Apache 2.0: [LICENSE text](https://github.com/Smorodov/Multitarget-tracker/blob/master/LICENSE)

#### Project cititations
1. Jeroen PROVOOST "Camera gebaseerde analysevan de verkeersstromen aaneen kruispunt", 2014 ( https://iiw.kuleuven.be/onderzoek/eavise/mastertheses/provoost.pdf )
2. Roberto Ciano, Dimitrij Klesev "Autonome Roboterschwarme in geschlossenen Raumen", 2015 ( https://www.hs-furtwangen.de/fileadmin/user_upload/fak_IN/Dokumente/Forschung_InformatikJournal/informatikJournal_2016.pdf#page=18 )
3. Wenda Qin, Tian Zhang, Junhe Chen "Traffic Monitoring By Video: Vehicles Tracking and Vehicle Data Analysing", 2016 ( http://cs-people.bu.edu/wdqin/FinalProject/CS585%20FinalProjectReport.html )
4. Ipek BARIS "CLASSIFICATION AND TRACKING OF VEHICLES WITH HYBRID CAMERA SYSTEMS", 2016 ( http://cvrg.iyte.edu.tr/publications/IpekBaris_MScThesis.pdf )
5. Cheng-Ta Lee, Albert Y. Chen, Cheng-Yi Chang "In-building Coverage of Automated External Defibrillators Considering Pedestrian Flow", 2016 ( http://www.see.eng.osaka-u.ac.jp/seeit/icccbe2016/Proceedings/Full_Papers/092-132.pdf )
6. Roberto Ciano, Dimitrij Klesev "Autonome Roboterschwarme in geschlossenen Raumen" in "informatikJournal 2016/17", 2017 ( https://docplayer.org/124538994-2016-17-informatikjournal-2016-17-aktuelle-berichte-aus-forschung-und-lehre-der-fakultaet-informatik.html )
7. Omid Noorshams "Automated systems to assess weights and activity in grouphoused mice", 2017 ( https://pdfs.semanticscholar.org/e5ff/f04b4200c149fb39d56f171ba7056ab798d3.pdf )
8. RADEK VOPÁLENSKÝ "DETECTION,TRACKING AND CLASSIFICATION OF VEHICLES", 2018 ( https://www.vutbr.cz/www_base/zav_prace_soubor_verejne.php?file_id=181063 )
9. Márk Rátosi, Gyula Simon "Real-Time Localization and Tracking  using Visible Light Communication", 2018 ( https://ieeexplore.ieee.org/abstract/document/8533800 )
10. Thi Nha Ngo, Kung-Chin Wu, En-Cheng Yang, Ta-Te Lin "A real-time imaging system for multiple honey bee tracking and activity monitoring", 2019 ( https://www.sciencedirect.com/science/article/pii/S0168169919301498 )
11. Tiago Miguel, Rodrigues de Almeida "Multi-Camera and Multi-Algorithm Architecture for VisualPerception onboard the ATLASCAR2", 2019 ( http://lars.mec.ua.pt/public/LAR%20Projects/Vision/2019_TiagoAlmeida/Thesis_Tiago_AlmeidaVF_26Jul2019.pdf )
12. ROS, http://docs.ros.org/lunar/api/costmap_converter/html/Ctracker_8cpp_source.html
13. Sangeeth Kochanthara, Yanja Dajsuren, Loek Cleophas, Mark van den Brand "Painting the Landscape of Automotive Software in GitHub", 2022 ( https://arxiv.org/abs/2203.08936 )
