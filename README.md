# Multitarget Tracker

[![Build Ubuntu](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/ubuntu.yml)
[![Build MacOS](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/macos.yml/badge.svg)](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/macos.yml)
[![CodeQL](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/Smorodov/Multitarget-tracker/actions/workflows/codeql-analysis.yml)

## Latest Features
- Add D-FINE seg detection model [ArgoHA/D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg)
- Add ByteTrack MOT algorithm based on [Vertical-Beach/ByteTrack-cpp](https://github.com/Vertical-Beach/ByteTrack-cpp)
- Big code cleanup from old style algorithms and detectors: some bgfg detectors, some VOT trackes, Face and Pedestrin detectors, Darknet based backend for old YOLO etc
- YOLOv13 detector works with TensorRT! Export pre-trained PyTorch models [here (iMoonLab/yolov13)](https://github.com/iMoonLab/yolov13) to ONNX format and run Multitarget-tracker with `-e=3` example
- Instance segmentation model from RF-DETR detector works with TensorRT! Export pre-trained PyTorch models [here (roboflow/rf-detr)](https://github.com/roboflow/rf-detr) to ONNX format and run Multitarget-tracker with `-e=3` example
- New linear assignment algorithm - [Jonker-Volgenant / LAPJV algorithm](https://github.com/yongyanghz/LAPJV-algorithm-c) used in [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) as alternative for Hungarian allgorithm
- D-FINE detector works with TensorRT! Export pre-trained PyTorch models [here (Peterande/D-FINE)](https://github.com/Peterande/D-FINE) to ONNX format and run Multitarget-tracker with `-e=3` example
- RF-DETR detector works with TensorRT! Export pre-trained PyTorch models [here (roboflow/rf-detr)](https://github.com/roboflow/rf-detr) to ONNX format and run Multitarget-tracker with `-e=3` example
- YOLOv12 detector works with TensorRT! Export pre-trained PyTorch models [here (sunsmarterjie/yolov12)](https://github.com/sunsmarterjie/yolov12) to ONNX format and run Multitarget-tracker with `-e=3` example

## Demo Videos

### Detection & Tracking

[![RF-DETR: detection vs instance segmentation](https://img.youtube.com/vi/oKy7jEKT83c/0.jpg)](https://youtu.be/oKy7jEKT83c)
[![Satellite planes detection and tracking with YOLOv11-obb](https://img.youtube.com/vi/gTpWnkMF7Lg/0.jpg)](https://youtu.be/gTpWnkMF7Lg)
[![4-in-1 latest SOTA detectors](https://img.youtube.com/vi/Pb_HnejRpY4/0.jpg)](https://youtu.be/Pb_HnejRpY4)
[![YOLOv8-obb detection with rotated boxes](https://img.youtube.com/vi/1e6ur57Fhzs/0.jpg)](https://youtu.be/1e6ur57Fhzs)
[![Very fast and small objects tracking](https://img.youtube.com/vi/PalIIAfgX88/0.jpg)](https://youtu.be/PalIIAfgX88)

## Documentation

### Core Components

#### 1. Object Detectors
Available through `CreateDetector` function with different `detectorType`:
1. **Background Subtraction**:
   - Built-in: VIBE (`tracking::Motion_VIBE`), SuBSENSE (`tracking::Motion_SuBSENSE`), LOBSTER (`tracking::Motion_LOBSTER`)
   - OpenCV: MOG2 (`tracking::Motion_MOG2`)
   - OpenCV Contrib: MOG (`tracking::Motion_MOG`), GMG (`tracking::Motion_GMG`), CNT (`tracking::Motion_CNT`)
   - Foreground segmentation uses OpenCV contours producing `cv::RotatedRect`
2. **Deep Learning Models**:
   - OpenCV DNN module (`tracking::DNN_OCV`)
   - TensorRT-accelerated YOLO (`tracking::Yolo_TensorRT`)

#### 2. Matching Algorithms
For solving assignment problems:
- **Hungarian Algorithm** (`tracking::MatchHungrian`) - O(N³) complexity
- **LAPJV** (`tracking::MatchBipart`) - O(M*N²) complexity
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
- KCF (`tracking::TrackKCF`)
- CSRT (`tracking::TrackCSRT`)
- DaSiamRPN (`tracking::TrackDaSiamRPN`)
- Vit (`tracking::TrackVit`)
- Nano (`tracking::TrackNano`)

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
  -DBUILD_ONNX_TENSORRT=ON \
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
- [Non Maximum Suppression](https://github.com/Nuzhny007/Non-Maximum-Suppression)
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
