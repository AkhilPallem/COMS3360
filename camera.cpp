#include "camera.h"
#include <cmath>

extern Vec3 randomInUnitDisk();
extern double randomDouble();

Camera::Camera(Vec3 pos, Vec3 tar, Vec3 vup, double fovDegrees, double aspect, 
               double aperture, double focusDist) 
    : position(pos), target(tar), up(vup), fov(fovDegrees), 
      aspectRatio(aspect), aperture(aperture), focusDist(focusDist) {
    
    lensRadius = aperture / 2.0;
    
    double theta = fov * M_PI / 180.0;
    double halfHeight = tan(theta / 2.0);
    double halfWidth = aspectRatio * halfHeight;
    
    w = (position - target).normalize();
    u = up.cross(w).normalize();
    v = w.cross(u);
    
    horizontal = u * (2.0 * halfWidth * focusDist);
    vertical = v * (2.0 * halfHeight * focusDist);
    lowerLeftCorner = position - horizontal * 0.5 - vertical * 0.5 - w * focusDist;
}

Ray Camera::getRay(double s, double t) const {
    Vec3 rd = randomInUnitDisk() * lensRadius;
    Vec3 offset = u * rd.x + v * rd.y;

    Vec3 direction = lowerLeftCorner + horizontal * s + vertical * t - position - offset;
    double time = randomDouble();
    
    return Ray(position + offset, direction.normalize(), time);
}