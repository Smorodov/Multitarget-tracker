//
// vector3d.h
//     Class providing common math operations for 3D points
//
// Author:  Tilman Schramke, Christoph Dalitz
// Date:    2017-03-16
// License: see LICENSE-BSD2
//

#ifndef VECTOR3D_H_
#define VECTOR3D_H_

#include <iostream>
#include "defines.h"

class Vector3d
{
public:
  track_t x;
  track_t y;
  track_t z;

  Vector3d();
  Vector3d(track_t a, track_t b, track_t c);
  bool operator==(const Vector3d &rhs) const;
  Vector3d& operator=(const Vector3d& other);
  // nicely formatted output
  friend std::ostream& operator<<(std::ostream& os, const Vector3d& vec);
  // Euclidean norm
  track_t norm() const;

};

// mathematical vector operations

// vector addition
Vector3d operator+(Vector3d x, Vector3d y);
// vector subtraction
Vector3d operator-(Vector3d x, Vector3d y);
// scalar product
track_t operator*(Vector3d x, Vector3d y);
// scalar multiplication
Vector3d operator*(Vector3d x, track_t c);
Vector3d operator*(track_t c, Vector3d x);
Vector3d operator/(Vector3d x, track_t c);


#endif /* VECTOR3D_H_ */
