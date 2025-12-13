#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include "ray.h"

class Camera {
public:
    Vec3 position;
    Vec3 target;
    Vec3 up;
    double fov;
    double aspectRatio;
    double aperture;
    double focusDist;
    
    Vec3 u, v, w;
    Vec3 horizontal, vertical, lowerLeftCorner;
    double lensRadius;
    
    Camera(Vec3 pos, Vec3 tar, Vec3 vup, double fovDegrees, double aspect, 
           double aperture = 0.0, double focusDist = 1.0);
    
    Ray getRay(double s, double t) const;
};

#endif