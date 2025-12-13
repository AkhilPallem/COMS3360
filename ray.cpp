#include "ray.h"

Ray::Ray() : origin(Vec3(0, 0, 0)), direction(Vec3(0, 0, 1)), time(0.0) {}
Ray::Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d), time(0.0) {}
Ray::Ray(const Vec3& o, const Vec3& d, double t) : origin(o), direction(d), time(t) {}

Vec3 Ray::at(double t) const { return origin + direction * t; }