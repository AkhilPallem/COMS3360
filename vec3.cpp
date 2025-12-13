#include "vec3.h"
#include <cmath>

extern double randomDouble();
extern double randomDouble(double min, double max);

Vec3::Vec3() : x(0), y(0), z(0) {}
Vec3::Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

Vec3 Vec3::operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
Vec3 Vec3::operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
Vec3 Vec3::operator-() const { return Vec3(-x, -y, -z); }
Vec3 Vec3::operator*(double t) const { return Vec3(x * t, y * t, z * t); }
Vec3 Vec3::operator/(double t) const { return Vec3(x / t, y / t, z / t); }
Vec3 Vec3::operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }

double Vec3::dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
Vec3 Vec3::cross(const Vec3& v) const { 
    return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); 
}

double Vec3::length() const { return sqrt(x*x + y*y + z*z); }
double Vec3::lengthSquared() const { return x*x + y*y + z*z; }
Vec3 Vec3::normalize() const { double l = length(); return Vec3(x/l, y/l, z/l); }

bool Vec3::nearZero() const {
    const double s = 1e-8;
    return (fabs(x) < s) && (fabs(y) < s) && (fabs(z) < s);
}

Vec3 Vec3::random() {
    return Vec3(randomDouble(), randomDouble(), randomDouble());
}

Vec3 Vec3::random(double min, double max) {
    return Vec3(randomDouble(min, max), randomDouble(min, max), randomDouble(min, max));
}