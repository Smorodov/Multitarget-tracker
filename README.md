# Multitarget Tracker

[![Build Ubuntu](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/ubuntu.yml)
[![Build MacOS](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/macos.yml/badge.svg)](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/macos.yml)
[![CodeQL](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/codeql-analysis.yml)

## Latest Features
- D-FINE detector works with TensorRT! Export pre-trained PyTorch models [here (Peterande/D-FINE)](https://github.com/Peterande/D-FINE) to ONNX format and run Multitarget-tracker with `-e=6` example
- RF-DETR detector works with TensorRT! Export pre-trained PyTorch models [here (roboflow/rf-detr)](https://github.com/roboflow/rf-detr) to ONNX format and run Multitarget-tracker with `-e=6` example
- YOLOv12 detector works with TensorRT! Export pre-trained PyTorch models [here (sunsmarterjie/yolov12)](https://github.com/sunsmarterjie/yolov12) to ONNX format and run Multitarget-tracker with `-e=6` example
- TensorRT 10 supported
- YOLOv11, YOLOv11-obb and YOLOv11-seg detectors work with TensorRT! Export pre-trained PyTorch models [here (ultralytics/ultralytics)](https://github.com/ultralytics/ultralytics) to ONNX format and run Multitarget-tracker with `-e=6` example
- YOLOv8-obb detector works with TensorRT! Export pre-trained PyTorch models [here (ultralytics/ultralytics)](https://github.com/ultralytics/ultralytics) to ONNX format and run Multitarget-tracker with `-e=6` example
- YOLOv10 detector works with TensorRT! Export pre-trained PyTorch models [here (THU-MIG/yolov10)](https://github.com/THU-MIG/yolov10) to ONNX format and run Multitarget-tracker with `-e=6` example
- YOLOv9 detector works with TensorRT! Export pre-trained PyTorch models [here (WongKinYiu/yolov9)](https://github.com/WongKinYiu/yolov9) to ONNX format and run Multitarget-tracker with `-e=6` example
- YOLOv8 instance segmentation models work with TensorRT! Export pre-trained PyTorch models [here (ultralytics/ultralytics)](https://github.com/ultralytics/ultralytics) to ONNX format and run Multitarget-tracker with `-e=6` example
- Re-identification model `osnet_x0_25_msmt17` from [mikel-brostrom/yolo_tracking](https://github.com/mikel-brostrom/yolo_tracking)

## Demo Videos

### Detection & Tracking
[![Satellite planes detection and tracking with YOLOv11-obb](https://img.youtube.com/vi/gTpWnejRpY4/0.jpg)](https://youtu.be/gTpWnejRpY4)
[![4-in-1 latest SOTA detectors](https://img.youtube.com/vi/Pb_HnejRpY4/0.jpg)](https://youtu.be/Pb_HnejRpY4)
[![YOLOv8-obb detection with rotated boxes](https://img.youtube.com/vi/1e6ur57Fhzs/0.jpg)](https://youtu.be/1e6ur57Fhzs)
[![Very fast and small objects tracking](https://img.youtube.com/vi/PalIIAfgX88/0.jpg)](https://youtu.be/PalIIAfgX88)

### Other Demos
[![Mouse tracking](https://img.youtube.com/vi/2fW5TmAtAXM/0.jpg)](https://youtu.be/2fW5TmAtAXM)
[![Motion Detection and tracking](https://img.youtube.com/vi/GjN8jOy4kVw/0.jpg)](https://youtu.be/GjN8jOy4kVw)
[![Simple Abandoned detector](https://img.youtube.com/vi/fpkHRsFzspA/0.jpg)](https://youtu.be/fpkHRsFzspA)

## Documentation

### Core Components

#### 1. Object Detectors
Available through `CreateDetector` function with different `detectorType`:
1. **Background Subtraction**:
   - Built-in: VIBE (`tracking::Motion_VIBE`), SuBSENSE (`tracking::Motion_SuBSENSE`), LOBSTER (`tracking::Motion_LOBSTER`)
   - OpenCV: MOG2 (`tracking::Motion_MOG2`)
   - OpenCV Contrib: MOG (`tracking::Motion_MOG`), GMG (`tracking::Motion_GMG`), CNT (`tracking::Motion_CNT`)
   - Foreground segmentation uses OpenCV contours producing `cv::RotatedRect`
2. **Face Detection**: Haar cascade from OpenCV (`tracking::Face_HAAR`)
3. **Pedestrian Detection**:
   - HOG descriptor (`tracking::Pedestrian_HOG`)
   - C4 algorithm from sturkmen72 ([C4-Real-time-pedestrian-detection](https://github.com/sturkmen72/C4-Real-time-pedestrian-detection)) (`tracking::Pedestrian_C4`)
4. **Deep Learning Models**:
   - OpenCV DNN module (`tracking::DNN_OCV`) with models from [chuanqi305](https://github.com/chuanqi305/MobileNet-SSD) and [pjreddie](https://pjreddie.com/darknet/yolo/)
   - Darknet/YOLO (`tracking::Yolo_Darknet`) with AlexeyAB's implementation
   - TensorRT-accelerated YOLO (`tracking::Yolo_TensorRT`)

#### 2. Matching Algorithms
For solving assignment problems:
- **Hungarian Algorithm** (`tracking::MatchHungrian`) - O(N³) complexity
- **Weighted Bipartite Graph Matching** (`tracking::MatchBipart`) - O(M*N²) complexity
- **Distance Metrics**:
  - Center distance (`tracking::DistCenters`)
  - Bounding box distance (`tracking::DistRects`)
  - Jaccard/IoU similarity (`tracking::DistJaccard`)

#### 3. Trajectory Smoothing
- Kalman filters: Linear (`tracking::KalmanLinear`) and Unscented (`tracking::KalmanUnscented`)
- State models: Constant velocity and constant acceleration
- Tracking modes: Position-only (`tracking::FilterCenter`) and position+size (`tracking::FilterRect`)
- Specialized features: Abandoned object detection, line intersection counting

#### 4. Visual Search
When targets disappear:
- DAT (`tracking::TrackDAT`), STAPLE (`tracking::TrackSTAPLE`), LDES (`tracking::TrackLDES`)
- KCF (`tracking::TrackKCF`), MIL (`tracking::TrackMIL`), MedianFlow (`tracking::TrackMedianFlow`)
- GOTURN (`tracking::TrackGOTURN`), MOSSE (`tracking::TrackMOSSE`), CSRT (`tracking::TrackCSRT`) etc

### Processing Pipelines
1. **Synchronous** (`SyncProcess`): Single-threaded processing
2. **Asynchronous (2 threads)** (`AsyncProcess`): Decouples detection and tracking
3. **Fully Asynchronous (4 threads)**: For low-FPS deep learning detectors

### Installation & Building
```bash
git clone https://github.com/Smorodov/Multitarget-tracker.git
cd Multitarget-tracker
mkdir build && cd build
cmake . .. \
  -DUSE_OCV_BGFG=ON \
  -DUSE_OCV_KCF=ON \
  -DUSE_OCV_UKF=ON \
  -DBUILD_YOLO_LIB=ON \
  -DBUILD_YOLO_TENSORRT=ON \
  -DBUILD_ASYNC_DETECTOR=ON \
  -DBUILD_CARS_COUNTING=ON
make -j
```

### Usage Guide
Basic command syntax:
```bash
./MultitargetTracker <video_path> [--example=<num>] [--start_frame=<num>] 
                     [--end_frame=<num>] [--end_delay=<ms>] [--out=<filename>]
                     [--show_logs] [--gpu] [--async] [--res=<filename>]
                     [--settings=<filename>] [--batch_size=<num>]
```

Example:
```bash
./MultitargetTracker ../data/atrium.avi -e=1 -o=../data/atrium_motion.avi
```

Keyboard Controls:
- `m`: Toggle play/pause
- Any key: Step forward when paused
- `Esc`: Exit

### Integration as Library
```cpp
#include <mtracking/Ctracker.h>

std::unique_ptr<BaseTracker> m_tracker;
TrackerSettings settings;
settings.SetDistance(tracking::DistJaccard);
m_tracker = BaseTracker::CreateTracker(settings);
```

### Third-party Dependencies

- [OpenCV (and contrib)](https://github.com/opencv/opencv)
- [Vibe](https://github.com/BelBES/VIBE)
- [SuBSENSE and LOBSTER](https://github.com/ethereon/subsense)
- [GTL](https://github.com/rdmpage/graph-template-library)
- [MWBM](https://github.com/rdmpage/maximum-weighted-bipartite-matching)
- [Pedestrians detector](https://github.com/sturkmen72/C4-Real-time-pedestrian-detection)
- [Non Maximum Suppression](https://github.com/Nuzhny007/Non-Maximum-Suppression)
- [MobileNet SSD models](https://github.com/chuanqi305/MobileNet-SSD)
- [YOLO v3 models](https://pjreddie.com/darknet/yolo/)
- [Darknet inference and YOLO v4 models](https://github.com/AlexeyAB/darknet)
- [NVidia TensorRT inference and YOLO v5 models](https://github.com/enazoe/yolo-tensorrt)
- [YOLOv6 models](https://github.com/meituan/YOLOv6/releases)
- [YOLOv7 models](https://github.com/WongKinYiu/yolov7)
- [GOTURN models](https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking)
- [DAT tracker](https://github.com/foolwood/DAT)
- [STAPLE tracker](https://github.com/xuduo35/STAPLE)
- [LDES tracker](https://github.com/yfji/LDESCpp)
- [Ini file parser](https://github.com/benhoyt/inih)
- [Circular Code](https://github.com/LiorKogan/Circular)

### License
[Apache 2.0 License](https://github.com/Smorodov/Multitarget-tracker/blob/master/LICENSE)

#### Project citations
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
14. Fesus, A., Kovari, B., Becsi, T., Leginusz, L. "Dynamic Prompt-Based Approach for Open Vocabulary Multi-Object Tracking", 2025 ( https://link.springer.com/chapter/10.1007/978-3-031-81799-1_25 )
