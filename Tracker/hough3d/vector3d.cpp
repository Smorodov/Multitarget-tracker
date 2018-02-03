//
// vector3d.cpp
//     Class providing common math operations for 3D points
//
// Author:  Tilman Schramke, Christoph Dalitz
// Date:    2017-03-16
// License: see LICENSE-BSD2
//

#include "vector3d.h"
#include <math.h>

Vector3d::Vector3d() {
  x = 0; y = 0; z = 0;
}

Vector3d::Vector3d(double a, double b, double c) {
  x = a; y = b; z = c;
}

bool Vector3d::operator==(const Vector3d &rhs) const {
  if((x == rhs.x) && (y == rhs.y) && (z == rhs.z))
    return true;
  return false;
}

Vector3d& Vector3d::operator=(const Vector3d& other) {
  x = other.x; y = other.y; z = other.z;
  return *this;
}

// nicely formatted output
std::ostream& operator<<(std::ostream& strm, const Vector3d& vec) {
  return strm << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
}

// Euclidean norm
double Vector3d::norm() const {
  return sqrt((x * x) + (y * y) + (z * z));
}

// mathematical vector operations

// vector addition
Vector3d operator+(Vector3d x, Vector3d y) {
  Vector3d v(x.x + y.x, x.y + y.y, x.z + y.z);
  return v;
}

// vector subtraction
Vector3d operator-(Vector3d x, Vector3d y) {
  Vector3d v(x.x - y.x, x.y - y.y, x.z - y.z);
  return v;
}

// scalar product
double operator*(Vector3d x, Vector3d y) {
  return (x.x*y.x + x.y*y.y +  x.z*y.z);
}

// scalar multiplication
Vector3d operator*(Vector3d x, double c) {
  Vector3d v(c*x.x, c*x.y, c*x.z);
  return v;
}
Vector3d operator*(double c, Vector3d x) {
  Vector3d v(c*x.x, c*x.y, c*x.z);
  return v;
}
Vector3d operator/(Vector3d x, double c) {
  Vector3d v(x.x/c, x.y/c, x.z/c);
  return v;
}
