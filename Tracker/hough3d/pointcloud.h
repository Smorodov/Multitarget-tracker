//
// pointcloud.h
//     Class for holding a set of 3D points
//
// Author:  Tilman Schramke, Christoph Dalitz
// Date:    2017-03-16
// License: see LICENSE-BSD2
//

#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include "vector3d.h"
#include <vector>
#include <string>

class PointCloud {
public:
  // translation of pointCloud as done by shiftToOrigin()
  Vector3d shift;
  // points of the point cloud
  std::vector<Vector3d> points;

  // translate point cloud so that center = origin
  void shiftToOrigin();
  // mean value of all points (center of gravity)
  Vector3d meanValue() const;
  // bounding box corners
  void getMinMax3D(Vector3d* min_pt, Vector3d* max_pt);
  // reads point cloud data from the given file
  int readFromFile(const char* path);
  // store points closer than dx to line (a, b) in Y
  void pointsCloseToLine(const Vector3d &a, const Vector3d &b,
                         double dx, PointCloud* Y);
  // removes the points in Y from PointCloud
  // WARNING: only works when points in same order as in pointCloud!
  void removePoints(const PointCloud &Y);
};



#endif /* POINTCLOUD_H_ */
