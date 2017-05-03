# Multitarget-tracker

Demo Videos:

Tracking: https://www.youtube.com/watch?v=2fW5TmAtAXM

Detection and tracking: https://www.youtube.com/watch?v=GjN8jOy4kVw

Hungarian algorithm + Kalman filter multitarget tracker implementation.

And you can select:
1. Background substraction: Vibe, MOG or GMG
2. Segmentation: contours.
3. Matching: Hungrian algorithm or algorithm based on weighted bipartite graphs.
4. Tracking: Kalman filter for objects center or for object coordinates and size.
5. Use or not local tracker (LK optical flow) for smooth trajectories.

License: GNU GPLv3 http://www.gnu.org/licenses/gpl-3.0.txt 


In project uses libraries:
- OpenCV (and contrib): https://github.com/opencv/opencv and https://github.com/opencv/opencv_contrib
- Vibe: https://github.com/BelBES/VIBE
- GTL: https://github.com/rdmpage/graph-template-library
- MWBM: https://github.com/rdmpage/maximum-weighted-bipartite-matching
