#ifndef VEC3_H
#define VEC3_H

struct Vec3 {
    double x, y, z;
    
    Vec3();
    Vec3(double x, double y, double z);

    Vec3 operator+(const Vec3& v) const;
    Vec3 operator-(const Vec3& v) const;
    Vec3 operator-() const;
    Vec3 operator*(double t) const;
    Vec3 operator/(double t) const;
    Vec3 operator*(const Vec3& v) const;
    
    double dot(const Vec3& v) const;
    Vec3 cross(const Vec3& v) const;
    double length() const;
    double lengthSquared() const;
    Vec3 normalize() const;
    
    bool nearZero() const;
    static Vec3 random();
    static Vec3 random(double min, double max);
};

#endif