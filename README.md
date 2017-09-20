# Multitarget-tracker

Hungarian algorithm + Kalman filter multitarget tracker implementation.

#### Demo Videos

* Mouse tracking:

[![Tracking:](https://img.youtube.com/vi/2fW5TmAtAXM/0.jpg)](https://www.youtube.com/watch?v=2fW5TmAtAXM)

* Motion Detection and tracking:

[![Motion Detection and tracking:](https://img.youtube.com/vi/GjN8jOy4kVw/0.jpg)](https://www.youtube.com/watch?v=GjN8jOy4kVw)

* Multiple Faces tracking:

[![Multiple Faces tracking:](https://img.youtube.com/vi/j67CFwFtciU/0.jpg)](https://www.youtube.com/watch?v=j67CFwFtciU)

#### Parameters
1. Background substraction: built-in Vibe, SuBSENSE and LOBSTER; MOG2 from opencv; MOG, GMG and CNT from opencv_contrib
2. Foreground segmentation: contours
3. Matching: Hungrian algorithm or algorithm based on weighted bipartite graphs
4. Tracking: Linear or Unscented Kalman filter for objects center or for object coordinates and size
5. Use or not local tracker (LK optical flow) for smooth trajectories
6. KCF, MIL or MedianFlow tracking for lost objects and collision resolving

#### Build
1. Download project sources
2. Install CMake
3. Install OpenCV (https://github.com/opencv/opencv) and OpenCV contrib (https://github.com/opencv/opencv_contrib) repositories
4. Configure project CmakeLists.txt, set OpenCV_DIR. If opencv_contrib don't installed then set disable options USE_OCV_BGFG, USE_OCV_KCF and USE_OCV_UKF
5. Go to the build directory and run make

**Usage:**

           Usage:
             ./MultitargetTracker <path to movie file> [--example]=<number of example 0..3> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs>
             ./MultitargetTracker ../data/atrium.avi -e=1 -o=../data/atrium_motion.avi
           Press:
           * 'm' key for change mode: play|pause. When video is paused you can press any key for get next frame.
           * Press Esc to exit from video

           Params: 
           1. Movie file, for example ../data/atrium.avi
           2. [Optional] Number of example: 0 - MouseTracking, 1 - MotionDetector, 2 - FaceDetector, 3 - PedestrianDetector
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

#### Thirdparty libraries
* OpenCV (and contrib): https://github.com/opencv/opencv and https://github.com/opencv/opencv_contrib
* Vibe: https://github.com/BelBES/VIBE
* SuBSENSE and LOBSTER: https://github.com/ethereon/subsense
* GTL: https://github.com/rdmpage/graph-template-library
* MWBM: https://github.com/rdmpage/maximum-weighted-bipartite-matching
* Pedestrians detector: https://github.com/sturkmen72/C4-Real-time-pedestrian-detection
* Non Maximum Suppression: https://github.com/Nuzhny007/Non-Maximum-Suppression

#### License
GNU GPLv3: http://www.gnu.org/licenses/gpl-3.0.txt 
