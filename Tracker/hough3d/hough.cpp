//
// hough.h
//     Implementation of Algorithm 2 (Hough transform) from IPOL paper
//
// Author:  Tilman Schramke, Manuel Jeltsch, Christoph Dalitz
// Date:    2017-03-16
// License: see LICENSE-BSD2
//

#include "hough.h"
#include <math.h>

#include <cstdlib>


static double roundToNearest(double num) {
  return (num > 0.0) ? floor(num + 0.5) : ceil(num - 0.5);
}

Hough::Hough(const Vector3d& minP, const Vector3d& maxP, double var_dx,
             unsigned int sphereGranularity) {

  // compute directional vectors
  sphere = new Sphere();
  sphere->fromIcosahedron(sphereGranularity);
  num_b = sphere->vertices.size();

  // compute x'y' discretization
  max_x = std::max(maxP.norm(), minP.norm());
  double range_x = 2 * max_x;
  dx = var_dx;
  if (dx == 0.0) {
    dx = range_x / 64.0;
  }
  num_x = roundToNearest(range_x / dx);

  // allocate voting space
  VotingSpace.resize(num_x * num_x * num_b);
}

Hough::~Hough() {
  delete sphere;
}

// add all points from point cloud to voting space
void Hough::add(const PointCloud &pc) {
  for (std::vector<Vector3d>::const_iterator it = pc.points.begin();
       it != pc.points.end(); it++) {
    pointVote((*it), true);
  }
}

// subtract all points from point cloud to voting space
void Hough::subtract(const PointCloud &pc) {
  for (std::vector<Vector3d>::const_iterator it = pc.points.begin();
       it != pc.points.end(); it++) {
    pointVote((*it), false);
  }
}

// add or subtract (add==false) one point from voting space
// (corresponds to inner loop of Algorithm 2 in IPOL paper)
void Hough::pointVote(const Vector3d& point, bool add){

  // loop over directions B
  for(size_t j = 0; j < sphere->vertices.size(); j++) {

    Vector3d b = sphere->vertices[j];
    double beta = 1 / (1 + b.z);	// denominator in Eq. (2)

    // compute x' according to left hand side of Eq. (2)
    double x_new = ((1 - (beta * (b.x * b.x))) * point.x)
      - ((beta * (b.x * b.y)) * point.y)
      - (b.x * point.z);

    // compute y' according to right hand side Eq. (2)
    double y_new = ((-beta * (b.x * b.y)) * point.x)
      + ((1 - (beta * (b.y * b.y))) * point.y)
      - (b.y * point.z);

    size_t x_i = roundToNearest((x_new + max_x) / dx);
    size_t y_i = roundToNearest((y_new + max_x) / dx);

    // compute one-dimensional index from three indices
	// x_i * #planes * #direction_Vec + y_i * #direction_Vec + #loop
    size_t index = (x_i * num_x * num_b) + (y_i * num_b) + j;

    if (index < VotingSpace.size()) {
      if(add){
        VotingSpace[index]++;
      } else {
        VotingSpace[index]--;
      }
    }
  }
}

// returns the line with most votes (rc = number of votes)
unsigned int Hough::getLine(Vector3d* a, Vector3d* b){
  unsigned int votes = 0;
  unsigned int index = 0;

  for(unsigned int i = 0; i < VotingSpace.size(); i++){
    if (VotingSpace[i] > votes) {
      votes = VotingSpace[i];
      index = i;
    }
  }

  // retrieve x' coordinate from VotingSpace[num_x * num_x * num_b]
  double x = (int) (index / (num_x * num_b));
  index -= (int) x * num_x * num_b;
  x = x * dx - max_x;

  // retrieve y' coordinate from VotingSpace[num_x * num_x * num_b]
  double y = (int) index / num_b;
  index -= (int) y * num_b;
  y = y * dx - max_x;

  // retrieve directional vector
  *b = sphere->vertices[index];

  // compute anchor point according to Eq. (3)
  a->x = x * (1 - ((b->x * b->x) / (1 + b->z)))
    - y * ((b->x * b->y) / (1 + b->z));
  a->y = x * (-((b->x * b->y) / (1 + b->z)))
     + y * (1 - ((b->y * b->y) / (1 + b->z)));
  a->z = - x * b->x - y * b->y;

  return votes;
}
