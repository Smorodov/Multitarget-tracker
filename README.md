# Multitarget-tracker

Demo Videos:

Tracking: https://www.youtube.com/watch?v=2fW5TmAtAXM

Detection and tracking: https://www.youtube.com/watch?v=GjN8jOy4kVw

Multiple Faces tracking: https://www.youtube.com/watch?v=j67CFwFtciU

Hungarian algorithm + Kalman filter multitarget tracker implementation.

And you can select:
1. Background substraction: Vibe, MOG or GMG
2. Segmentation: contours.
3. Matching: Hungrian algorithm or algorithm based on weighted bipartite graphs.
4. Tracking: Linear or Unscented Kalman filter for objects center or for object coordinates and size.
5. Use or not local tracker (LK optical flow) for smooth trajectories.
6. KCF tracking for lost objects and collision resolving

License: GNU GPLv3 http://www.gnu.org/licenses/gpl-3.0.txt 


In project uses libraries:
- OpenCV (and contrib): https://github.com/opencv/opencv and https://github.com/opencv/opencv_contrib
- Vibe: https://github.com/BelBES/VIBE
- GTL: https://github.com/rdmpage/graph-template-library
- MWBM: https://github.com/rdmpage/maximum-weighted-bipartite-matching
- Pedestrians detector: https://github.com/sturkmen72/C4-Real-time-pedestrian-detection
- Non Maximum Suppression: https://github.com/Nuzhny007/Non-Maximum-Suppression
