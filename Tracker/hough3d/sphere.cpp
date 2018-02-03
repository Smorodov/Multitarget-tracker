//
// sphere.cpp
//     Class that implements direction quantization as described in
//     Jeltsch, Dalitz, Pohle-Froehlich: "Hough Parameter Space
//     Regularisation for Line Detection in 3D." VISAPP, pp. 345-352, 2016 
//
// Author:  Manuel Jeltsch, Tilman Schramke, Christoph Dalitz
// Date:    2017-03-16
// License: see LICENSE-BSD2
//

#include "sphere.h"

#include <math.h>


// creates the directions by subdivisions of icosahedron
void Sphere::fromIcosahedron(int subDivisions){
  this->getIcosahedron();
  for(int i = 0; i < subDivisions; i++) {
    subDivide();
  }
  this->makeUnique();
}

  // creates nodes and edges of icosahedron
void Sphere::getIcosahedron(){
  vertices.clear();
  triangles.clear();
  float tau = 1.61803399; // golden_ratio
  float norm = sqrt(1 + tau * tau);
  float v = 1 / norm;
  tau = tau / norm;

  Vector3d vec;
  vec.x = -v;
  vec.y = tau;
  vec.z = 0;
  vertices.push_back(vec); // 1
  vec.x = v;
  vec.y = tau;
  vec.z = 0;
  vertices.push_back(vec); // 2
  vec.x = 0;
  vec.y = v;
  vec.z = -tau;
  vertices.push_back(vec); // 3
  vec.x = 0;
  vec.y = v;
  vec.z = tau;
  vertices.push_back(vec); // 4
  vec.x = -tau;
  vec.y = 0;
  vec.z = -v;
  vertices.push_back(vec); // 5
  vec.x = tau;
  vec.y = 0;
  vec.z = -v;
  vertices.push_back(vec); // 6
  vec.x = -tau;
  vec.y = 0;
  vec.z = v;
  vertices.push_back(vec); // 7
  vec.x = tau;
  vec.y = 0;
  vec.z = v;
  vertices.push_back(vec); // 8
  vec.x = 0;
  vec.y = -v;
  vec.z = -tau;
  vertices.push_back(vec); // 9
  vec.x = 0;
  vec.y = -v;
  vec.z = tau;
  vertices.push_back(vec); // 10
  vec.x = -v;
  vec.y = -tau;
  vec.z = 0;
  vertices.push_back(vec); // 11
  vec.x = v;
  vec.y = -tau;
  vec.z = 0;
  vertices.push_back(vec); // 12
  // add all edges of all triangles
  triangles.push_back(0);
  triangles.push_back(1);
  triangles.push_back(2); // 1
  triangles.push_back(0);
  triangles.push_back(1);
  triangles.push_back(3); // 2
  triangles.push_back(0);
  triangles.push_back(2);
  triangles.push_back(4); // 3
  triangles.push_back(0);
  triangles.push_back(4);
  triangles.push_back(6); // 4
  triangles.push_back(0);
  triangles.push_back(3);
  triangles.push_back(6); // 5
  triangles.push_back(1);
  triangles.push_back(2);
  triangles.push_back(5); // 6
  triangles.push_back(1);
  triangles.push_back(3);
  triangles.push_back(7); // 7
  triangles.push_back(1);
  triangles.push_back(5);
  triangles.push_back(7); // 8
  triangles.push_back(2);
  triangles.push_back(4);
  triangles.push_back(8); // 9
  triangles.push_back(2);
  triangles.push_back(5);
  triangles.push_back(8); // 10
  triangles.push_back(3);
  triangles.push_back(6);
  triangles.push_back(9); // 1
  triangles.push_back(3);
  triangles.push_back(7);
  triangles.push_back(9); // 12
  triangles.push_back(4);
  triangles.push_back(8);
  triangles.push_back(10); // 13
  triangles.push_back(8);
  triangles.push_back(10);
  triangles.push_back(11); // 14
  triangles.push_back(5);
  triangles.push_back(8);
  triangles.push_back(11); // 15
  triangles.push_back(5);
  triangles.push_back(7);
  triangles.push_back(11); // 16
  triangles.push_back(7);
  triangles.push_back(9);
  triangles.push_back(11); // 17
  triangles.push_back(9);
  triangles.push_back(10);
  triangles.push_back(11); // 18
  triangles.push_back(6);
  triangles.push_back(9);
  triangles.push_back(10); // 19
  triangles.push_back(4);
  triangles.push_back(6);
  triangles.push_back(10); // 20
}

// one subdivision step
void Sphere::subDivide(){
  unsigned int vert_num = vertices.size();
  double norm;
  // subdivide each triangle
  int num = triangles.size() / 3;
  for(int i = 0; i < num; i++) {
    Vector3d a, b, c, d, e, f;

    unsigned int ai, bi, ci, di, ei, fi;
    ai = triangles.front();
    triangles.pop_front();
    bi = triangles.front();
    triangles.pop_front();
    ci = triangles.front();
    triangles.pop_front();

    a = vertices[ai];
    b = vertices[bi];
    c = vertices[ci];

    //  d = a+b
    d.x = (a.x + b.x);
    d.y = (a.y + b.y);
    d.z = (a.z + b.z);
    norm = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    d.x /= norm;
    d.y /= norm;
    d.z /= norm;
    //  e = b+c
    e.x = (c.x + b.x);
    e.y = (c.y + b.y);
    e.z = (c.z + b.z);
    norm = sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
    e.x /= norm;
    e.y /= norm;
    e.z /= norm;
    //  f = c+a
    f.x = (a.x + c.x);
    f.y = (a.y + c.y);
    f.z = (a.z + c.z);
    norm = sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
    f.x /= norm;
    f.y /= norm;
    f.z /= norm;

    // add all new edge indices of new triangles to the triangles deque
    bool d_found = false;
    bool e_found = false;
    bool f_found = false;
    for(unsigned int j = vert_num; j < vertices.size(); j++) {
      if(vertices[j] == d) {
        d_found = true;
        di = j;
        continue;
      }
      if(vertices[j] == e) {
        e_found = true;
        ei = j;
        continue;
      }
      if(vertices[j] == f) {
        f_found = true;
        fi = j;
        continue;
      }

    }
    if(!d_found) {
      di = vertices.size();
      vertices.push_back(d);
    }
    if(!e_found) {
      ei = vertices.size();
      vertices.push_back(e);
    }
    if(!f_found) {
      fi = vertices.size();
      vertices.push_back(f);
    }

    triangles.push_back(ai);
    triangles.push_back(di);
    triangles.push_back(fi);

    triangles.push_back(di);
    triangles.push_back(bi);
    triangles.push_back(ei);

    triangles.push_back(fi);
    triangles.push_back(ei);
    triangles.push_back(ci);

    triangles.push_back(fi);
    triangles.push_back(di);
    triangles.push_back(ei);
  }
}

// make vectors nondirectional and unique
void Sphere::makeUnique(){
  for(unsigned int i = 0; i < vertices.size(); i++) {
    if(vertices[i].z < 0) { // make hemisphere
      vertices.erase(vertices.begin() + i);

      // delete all triangles with vertex_i
      int t = 0;
      for(std::deque<unsigned int>::iterator it = triangles.begin();
          it != triangles.end();) {
        if(triangles[t] == i || triangles[t + 1] == i ||
           triangles[t + 2] == i) {
          it = triangles.erase(it);
          it = triangles.erase(it);
          it = triangles.erase(it);
        } else {
          ++it;
          ++it;
          ++it;
          t += 3;
        }
      }
      // update indices
      for(unsigned int j = 0; j < triangles.size(); j ++) {
        if(triangles[j] > i) {
          triangles[j]--;
        }
      }

      i--;
    } else if(vertices[i].z == 0) { // make equator vectors unique
      if(vertices[i].x < 0) {
        vertices.erase(vertices.begin() + i);
        // delete all triangles with vertex_i
        int t = 0;
        for(std::deque<unsigned int>::iterator it = triangles.begin();
            it != triangles.end();) {
          if(triangles[t] == i || triangles[t + 1] == i ||
             triangles[t + 2] == i) {
            it = triangles.erase(it);
            it = triangles.erase(it);
            it = triangles.erase(it);
          } else {
            ++it;
            ++it;
            ++it;
            t += 3;
          }
        }
        // update indices
        for(unsigned int j = 0; j < triangles.size(); j ++) {
          if(triangles[j] > i) {
            triangles[j]--;
          }
        }
        i--;
      } else if(vertices[i].x == 0 && vertices[i].y == -1) {
        vertices.erase(vertices.begin() + i);
        // delete all triangles with vertex_i
        int t = 0;
        for(std::deque<unsigned int>::iterator it = triangles.begin();
            it != triangles.end();) {
          if(triangles[t] == i || triangles[t + 1] == i ||
             triangles[t + 2] == i) {
            it = triangles.erase(it);
            it = triangles.erase(it);
            it = triangles.erase(it);
          } else {
            ++it;
            ++it;
            ++it;
            t += 3;
          }
        }
        // update indices
        for(unsigned int j = 0; j < triangles.size(); j ++) {
          if(triangles[j] > i) {
            triangles[j]--;
          }
        }
        i--;
      }
    }
  }
}



