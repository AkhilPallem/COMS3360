#ifndef RAY_H
#define RAY_H

#include "vec3.h"

struct Ray {
    Vec3 origin, direction;
    double time;
    
    Ray();
    Ray(const Vec3& o, const Vec3& d);
    Ray(const Vec3& o, const Vec3& d, double t);
    
    Vec3 at(double t) const;
};

#endif