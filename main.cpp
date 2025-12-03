#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <random>
#include <algorithm>
#include <sstream>
#include <string>

// Random number gen
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

double randomDouble() {
    return dis(gen);
}

double randomDouble(double min, double max) {
    return min + (max - min) * randomDouble();
}

// Basic 3D vector calcs 
struct Vec3 {
    double x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3 operator*(double t) const { return Vec3(x * t, y * t, z * t); }
    Vec3 operator/(double t) const { return Vec3(x / t, y / t, z / t); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    
    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const { 
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); 
    }
    
    double length() const { return sqrt(x*x + y*y + z*z); }
    double lengthSquared() const { return x*x + y*y + z*z; }
    Vec3 normalize() const { double l = length(); return Vec3(x/l, y/l, z/l); }
    
    bool nearZero() const {
        const double s = 1e-8;
        return (fabs(x) < s) && (fabs(y) < s) && (fabs(z) < s);
    }
    
    static Vec3 random() {
        return Vec3(randomDouble(), randomDouble(), randomDouble());
    }
    
    static Vec3 random(double min, double max) {
        return Vec3(randomDouble(min, max), randomDouble(min, max), randomDouble(min, max));
    }
};

// Random point in unit sphere for diffuse materials
Vec3 randomInUnitSphere() {
    while (true) {
        Vec3 p = Vec3::random(-1, 1);
        if (p.lengthSquared() < 1)
            return p;
    }
}

Vec3 randomUnitVector() {
    return randomInUnitSphere().normalize();
}

// Random point in disk for depth of field
Vec3 randomInUnitDisk() {
    while (true) {
        Vec3 p = Vec3(randomDouble(-1, 1), randomDouble(-1, 1), 0);
        if (p.lengthSquared() < 1)
            return p;
    }
}

// Reflection for mirrors
Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - n * (2.0 * v.dot(n));
}

// Refraction for glass
Vec3 refract(const Vec3& uv, const Vec3& n, double etaiOverEtat) {
    double cosTheta = fmin((-uv).dot(n), 1.0);
    Vec3 rOutPerp = (uv + n * cosTheta) * etaiOverEtat;
    Vec3 rOutParallel = n * (-sqrt(fabs(1.0 - rOutPerp.lengthSquared())));
    return rOutPerp + rOutParallel;
}

// Ray for tracing through scene
struct Ray {
    Vec3 origin, direction;
    Ray() : origin(Vec3(0, 0, 0)), direction(Vec3(0, 0, 1)) {}
    Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}
    Vec3 at(double t) const { return origin + direction * t; }
};

// Different material types
enum MaterialType {
    DIFFUSE,
    METAL,
    DIELECTRIC,
    EMISSIVE
};

struct Material {
    MaterialType type;
    Vec3 albedo;
    double fuzz;
    double ior;
    Vec3 emission;
    
    Material() : type(DIFFUSE), albedo(0.5, 0.5, 0.5), fuzz(0), ior(1.5), emission(0, 0, 0) {}
    
    static Material makeDiffuse(const Vec3& color) {
        Material m;
        m.type = DIFFUSE;
        m.albedo = color;
        return m;
    }
    
    static Material makeMetal(const Vec3& color, double fuzz) {
        Material m;
        m.type = METAL;
        m.albedo = color;
        m.fuzz = fuzz < 1 ? fuzz : 1;
        return m;
    }
    
    static Material makeDielectric(double ior) {
        Material m;
        m.type = DIELECTRIC;
        m.ior = ior;
        m.albedo = Vec3(1, 1, 1);
        return m;
    }
    
    static Material makeEmissive(const Vec3& color, double intensity) {
        Material m;
        m.type = EMISSIVE;
        m.emission = color * intensity;
        m.albedo = Vec3(0, 0, 0);
        return m;
    }
};

// Stores ray hit information
struct HitRecord {
    Vec3 point;
    Vec3 normal;
    double t;
    bool hit;
    bool frontFace;
    Material material;
    double u, v;
    
    HitRecord() : hit(false), frontFace(true) {}
    
    void setFaceNormal(const Ray& ray, const Vec3& outwardNormal) {
        frontFace = ray.direction.dot(outwardNormal) < 0;
        normal = frontFace ? outwardNormal : outwardNormal * -1;
    }
};

// Texture loading and sampling
class Texture {
public:
    std::vector<std::vector<Vec3>> data;
    int width, height;
    
    Texture() : width(0), height(0) {}
    
    Vec3 sample(double u, double v) const {
        if (width == 0 || height == 0) return Vec3(1, 1, 1);
        
        u = fmax(0.0, fmin(1.0, u));
        v = 1.0 - fmax(0.0, fmin(1.0, v));
        
        int i = int(u * (width - 1));
        int j = int(v * (height - 1));
        
        return data[j][i];
    }
    
    // Create checkerboard pattern on image final
    static Texture makeCheckerboard(int size = 8) {
        Texture tex;
        tex.width = size;
        tex.height = size;
        tex.data.resize(size, std::vector<Vec3>(size));
        
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                bool isEven = ((i / 2) + (j / 2)) % 2 == 0;
                tex.data[j][i] = isEven ? Vec3(0.9, 0.9, 0.9) : Vec3(0.1, 0.1, 0.1);
            }
        }
        return tex;
    }
    
    // Load PPM image file
    static Texture loadFromPPM(const std::string& filename) {
        Texture tex;
        std::ifstream file(filename, std::ios::binary);
        
        if (!file.is_open()) {
            std::cout << "Failed to open texture: " << filename << std::endl;
            return makeCheckerboard();
        }
        
        std::string magic;
        file >> magic;
        
        if (magic != "P6" && magic != "P3") {
            std::cout << "Unsupported PPM format" << std::endl;
            return makeCheckerboard();
        }
        
        file >> tex.width >> tex.height;
        int maxVal;
        file >> maxVal;
        file.ignore();
        
        tex.data.resize(tex.height, std::vector<Vec3>(tex.width));
        
        if (magic == "P6") {
            // Binary format
            for (int j = 0; j < tex.height; j++) {
                for (int i = 0; i < tex.width; i++) {
                    unsigned char rgb[3];
                    file.read(reinterpret_cast<char*>(rgb), 3);
                    tex.data[j][i] = Vec3(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0);
                }
            }
        } else {
            // ASCII format
            for (int j = 0; j < tex.height; j++) {
                for (int i = 0; i < tex.width; i++) {
                    int r, g, b;
                    file >> r >> g >> b;
                    tex.data[j][i] = Vec3(r / 255.0, g / 255.0, b / 255.0);
                }
            }
        }
        
        std::cout << "Loaded texture: " << filename << " (" << tex.width << "x" << tex.height << ")" << std::endl;
        return tex;
    }
};

// Bounding box for acceleration
struct AABB {
    Vec3 min, max;
    
    AABB() : min(Vec3(0, 0, 0)), max(Vec3(0, 0, 0)) {}
    AABB(const Vec3& a, const Vec3& b) : min(a), max(b) {}
    
    bool hit(const Ray& ray, double tMin, double tMax) const {
        for (int a = 0; a < 3; a++) {
            double coord = a == 0 ? ray.origin.x : (a == 1 ? ray.origin.y : ray.origin.z);
            double dir = a == 0 ? ray.direction.x : (a == 1 ? ray.direction.y : ray.direction.z);
            double minCoord = a == 0 ? min.x : (a == 1 ? min.y : min.z);
            double maxCoord = a == 0 ? max.x : (a == 1 ? max.y : max.z);
            
            double invD = 1.0 / dir;
            double t0 = (minCoord - coord) * invD;
            double t1 = (maxCoord - coord) * invD;
            
            if (invD < 0.0) std::swap(t0, t1);
            
            tMin = t0 > tMin ? t0 : tMin;
            tMax = t1 < tMax ? t1 : tMax;
            
            if (tMax <= tMin) return false;
        }
        return true;
    }
    
    static AABB surroundingBox(const AABB& box0, const AABB& box1) {
        Vec3 small(fmin(box0.min.x, box1.min.x),
                   fmin(box0.min.y, box1.min.y),
                   fmin(box0.min.z, box1.min.z));
        Vec3 big(fmax(box0.max.x, box1.max.x),
                 fmax(box0.max.y, box1.max.y),
                 fmax(box0.max.z, box1.max.z));
        return AABB(small, big);
    }
};

// Sphere primitive
class Sphere {
public:
    Vec3 center;
    double radius;
    Material material;
    Texture texture;
    bool hasTexture;
    
    Sphere(Vec3 c, double r, Material mat) 
        : center(c), radius(r), material(mat), hasTexture(false) {}
    
    Sphere(Vec3 c, double r, Material mat, Texture tex) 
        : center(c), radius(r), material(mat), texture(tex), hasTexture(true) {}
    
    HitRecord intersect(const Ray& ray) const {
        HitRecord rec;
        Vec3 oc = ray.origin - center;
        double a = ray.direction.dot(ray.direction);
        double b = 2.0 * oc.dot(ray.direction);
        double c = oc.dot(oc) - radius * radius;
        double discriminant = b * b - 4 * a * c;
        
        if (discriminant >= 0) {
            double t = (-b - sqrt(discriminant)) / (2.0 * a);
            if (t > 0.001) { 
                rec.t = t;
                rec.point = ray.at(t);
                Vec3 outwardNormal = (rec.point - center) / radius;
                rec.setFaceNormal(ray, outwardNormal);
                rec.hit = true;
                rec.material = material;
                
                // Calculate UV coordinates
                Vec3 p = outwardNormal;
                double phi = atan2(p.z, p.x);
                double theta = asin(p.y);
                rec.u = 1.0 - (phi + M_PI) / (2.0 * M_PI);
                rec.v = (theta + M_PI / 2.0) / M_PI;
                
                return rec;
            }
        }
        return rec;
    }
    
    AABB boundingBox() const {
        return AABB(center - Vec3(radius, radius, radius),
                    center + Vec3(radius, radius, radius));
    }
};

// Triangle primitive
class Triangle {
public:
    Vec3 v0, v1, v2;
    Vec3 normal;
    Material material;
    Texture texture;
    bool hasTexture;
    
    Triangle(Vec3 a, Vec3 b, Vec3 c, Material mat) 
        : v0(a), v1(b), v2(c), material(mat), hasTexture(false) {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        normal = edge1.cross(edge2).normalize();
    }
    
    HitRecord intersect(const Ray& ray) const {
        HitRecord rec;
        
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = ray.direction.cross(edge2);
        double a = edge1.dot(h);
        
        if (a > -0.0001 && a < 0.0001)
            return rec;
        
        double f = 1.0 / a;
        Vec3 s = ray.origin - v0;
        double u = f * s.dot(h);
        
        if (u < 0.0 || u > 1.0)
            return rec;
        
        Vec3 q = s.cross(edge1);
        double v = f * ray.direction.dot(q);
        
        if (v < 0.0 || u + v > 1.0)
            return rec;
        
        double t = f * edge2.dot(q);
        
        if (t > 0.001) {
            rec.t = t;
            rec.point = ray.at(t);
            rec.setFaceNormal(ray, normal);
            rec.hit = true;
            rec.material = material;
            rec.u = u;
            rec.v = v;
        }
        
        return rec;
    }
    
    AABB boundingBox() const {
        double minX = fmin(fmin(v0.x, v1.x), v2.x) - 0.0001;
        double minY = fmin(fmin(v0.y, v1.y), v2.y) - 0.0001;
        double minZ = fmin(fmin(v0.z, v1.z), v2.z) - 0.0001;
        double maxX = fmax(fmax(v0.x, v1.x), v2.x) + 0.0001;
        double maxY = fmax(fmax(v0.y, v1.y), v2.y) + 0.0001;
        double maxZ = fmax(fmax(v0.z, v1.z), v2.z) + 0.0001;
        return AABB(Vec3(minX, minY, minZ), Vec3(maxX, maxY, maxZ));
    }
};


// Smooth triangle for normal interpolation feature addition
class SmoothTriangle {
public:
    Vec3 v0, v1, v2;
    Vec3 n0, n1, n2;
    Material material;
    
    SmoothTriangle(Vec3 a, Vec3 b, Vec3 c, Vec3 na, Vec3 nb, Vec3 nc, Material mat) 
        : v0(a), v1(b), v2(c), n0(na.normalize()), n1(nb.normalize()), n2(nc.normalize()), material(mat) {}
    
    HitRecord intersect(const Ray& ray) const {
        HitRecord rec;
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = ray.direction.cross(edge2);
        double a = edge1.dot(h);
        
        if (a > -0.0001 && a < 0.0001)
            return rec;
        
        double f = 1.0 / a;
        Vec3 s = ray.origin - v0;
        double u = f * s.dot(h);
        
        if (u < 0.0 || u > 1.0)
            return rec;
        
        Vec3 q = s.cross(edge1);
        double v = f * ray.direction.dot(q);
        
        if (v < 0.0 || u + v > 1.0)
            return rec;
        
        double t = f * edge2.dot(q);
        
        if (t > 0.001) {
            rec.t = t;
            rec.point = ray.at(t);
            double w = 1.0 - u - v;
            Vec3 interpolatedNormal = (n0 * w + n1 * u + n2 * v).normalize();
            rec.setFaceNormal(ray, interpolatedNormal);
            rec.hit = true;
            rec.material = material;
            rec.u = u;
            rec.v = v;
        }
        return rec;
    }
    
    AABB boundingBox() const {
        double minX = fmin(fmin(v0.x, v1.x), v2.x) - 0.0001;
        double minY = fmin(fmin(v0.y, v1.y), v2.y) - 0.0001;
        double minZ = fmin(fmin(v0.z, v1.z), v2.z) - 0.0001;
        double maxX = fmax(fmax(v0.x, v1.x), v2.x) + 0.0001;
        double maxY = fmax(fmax(v0.y, v1.y), v2.y) + 0.0001;
        double maxZ = fmax(fmax(v0.z, v1.z), v2.z) + 0.0001;
        return AABB(Vec3(minX, minY, minZ), Vec3(maxX, maxY, maxZ));
    }
};

// BVH node for acceleration
struct BVHNode {
    AABB box;
    std::shared_ptr<BVHNode> left;
    std::shared_ptr<BVHNode> right;
    int sphereIdx;
    int triangleIdx;
    bool isLeaf;
    
    BVHNode() : sphereIdx(-1), triangleIdx(-1), isLeaf(false) {}
};

// Camera with configurable settings
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
           double aperture = 0.0, double focusDist = 1.0) 
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
    
    Ray getRay(double s, double t) const {
        Vec3 rd = randomInUnitDisk() * lensRadius;
        Vec3 offset = u * rd.x + v * rd.y;
        
        Vec3 direction = lowerLeftCorner + horizontal * s + vertical * t - position - offset;
        return Ray(position + offset, direction.normalize());
    }
};

// Material scattering behavior
bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered) {
    if (rec.material.type == DIFFUSE) {
        Vec3 scatterDirection = rec.normal + randomUnitVector();
        if (scatterDirection.nearZero())
            scatterDirection = rec.normal;
        scattered = Ray(rec.point, scatterDirection.normalize());
        attenuation = rec.material.albedo;
        return true;
    }
    else if (rec.material.type == METAL) {
        Vec3 reflected = reflect(rayIn.direction, rec.normal);
        scattered = Ray(rec.point, (reflected + randomInUnitSphere() * rec.material.fuzz).normalize());
        attenuation = rec.material.albedo;
        return scattered.direction.dot(rec.normal) > 0;
    }
    else if (rec.material.type == DIELECTRIC) {
        attenuation = Vec3(1.0, 1.0, 1.0);
        double refractionRatio = rec.frontFace ? (1.0 / rec.material.ior) : rec.material.ior;
        
        Vec3 unitDirection = rayIn.direction;
        double cosTheta = fmin((-unitDirection).dot(rec.normal), 1.0);
        double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
        
        bool cannotRefract = refractionRatio * sinTheta > 1.0;
        Vec3 direction;
        
        // Schlick's approximation
        auto reflectance = [](double cosine, double refIdx) {
            double r0 = (1 - refIdx) / (1 + refIdx);
            r0 = r0 * r0;
            return r0 + (1 - r0) * pow((1 - cosine), 5);
        };
        
        if (cannotRefract || reflectance(cosTheta, refractionRatio) > randomDouble())
            direction = reflect(unitDirection, rec.normal);
        else
            direction = refract(unitDirection, rec.normal, refractionRatio);
        
        scattered = Ray(rec.point, direction);
        return true;
    }
    
    return false;
}

// Scene with BVH acceleration
class Scene {
public:
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    std::vector<SmoothTriangle> smoothTriangles;
    Vec3 backgroundColor;
    std::shared_ptr<BVHNode> bvhRoot;
    
    Scene() : backgroundColor(0.7, 0.8, 1.0) {}
    
    void addSphere(const Sphere& sphere) {
        spheres.push_back(sphere);
    }
    
    void addTriangle(const Triangle& tri) {
        triangles.push_back(tri);
    }

    void addSmoothTriangle(const SmoothTriangle& tri){
        smoothTriangles.push_back(tri);
    }
    
    // Build BVH tree for faster rendering
    void buildBVH() {
        if (spheres.empty() && triangles.empty()) return;
        
        std::vector<int> sphereIndices;
        std::vector<int> triangleIndices;
        
        for (size_t i = 0; i < spheres.size(); i++)
            sphereIndices.push_back(i);
        for (size_t i = 0; i < triangles.size(); i++)
            triangleIndices.push_back(i);
        
        bvhRoot = buildBVHRecursive(sphereIndices, triangleIndices, 0);
    }
    
    std::shared_ptr<BVHNode> buildBVHRecursive(std::vector<int>& sphereIndices, 
                                                std::vector<int>& triangleIndices, 
                                                int depth) {
        auto node = std::make_shared<BVHNode>();
        
        bool first = true;
        for (int idx : sphereIndices) {
            AABB box = spheres[idx].boundingBox();
            if (first) {
                node->box = box;
                first = false;
            } else {
                node->box = AABB::surroundingBox(node->box, box);
            }
        }
        for (int idx : triangleIndices) {
            AABB box = triangles[idx].boundingBox();
            if (first) {
                node->box = box;
                first = false;
            } else {
                node->box = AABB::surroundingBox(node->box, box);
            }
        }
        
        // Leaf node if small enough
        if (sphereIndices.size() + triangleIndices.size() <= 4 || depth > 20) {
            node->isLeaf = true;
            if (!sphereIndices.empty()) node->sphereIdx = sphereIndices[0];
            if (!triangleIndices.empty()) node->triangleIdx = triangleIndices[0];
            return node;
        }
        
        // Split along longest axis
        Vec3 extent = node->box.max - node->box.min;
        int axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2);
        
        double midpoint = axis == 0 ? (node->box.min.x + node->box.max.x) / 2 :
                         axis == 1 ? (node->box.min.y + node->box.max.y) / 2 :
                                    (node->box.min.z + node->box.max.z) / 2;
        
        std::vector<int> leftSpheres, rightSpheres;
        std::vector<int> leftTriangles, rightTriangles;
        
        for (int idx : sphereIndices) {
            double center = axis == 0 ? spheres[idx].center.x :
                           axis == 1 ? spheres[idx].center.y : spheres[idx].center.z;
            if (center < midpoint)
                leftSpheres.push_back(idx);
            else
                rightSpheres.push_back(idx);
        }
        
        for (int idx : triangleIndices) {
            Vec3 center = (triangles[idx].v0 + triangles[idx].v1 + triangles[idx].v2) / 3.0;
            double centerCoord = axis == 0 ? center.x : axis == 1 ? center.y : center.z;
            if (centerCoord < midpoint)
                leftTriangles.push_back(idx);
            else
                rightTriangles.push_back(idx);
        }

        if (leftSpheres.empty() && leftTriangles.empty()) {
            if (!rightSpheres.empty()) leftSpheres.push_back(rightSpheres.back()), rightSpheres.pop_back();
            if (!rightTriangles.empty()) leftTriangles.push_back(rightTriangles.back()), rightTriangles.pop_back();
        }
        if (rightSpheres.empty() && rightTriangles.empty()) {
            if (!leftSpheres.empty()) rightSpheres.push_back(leftSpheres.back()), leftSpheres.pop_back();
            if (!leftTriangles.empty()) rightTriangles.push_back(leftTriangles.back()), leftTriangles.pop_back();
        }
        
        if (!leftSpheres.empty() || !leftTriangles.empty())
            node->left = buildBVHRecursive(leftSpheres, leftTriangles, depth + 1);
        if (!rightSpheres.empty() || !rightTriangles.empty())
            node->right = buildBVHRecursive(rightSpheres, rightTriangles, depth + 1);
        
        return node;
    }
    
    HitRecord intersectBVH(const Ray& ray, const std::shared_ptr<BVHNode>& node) const {
        HitRecord rec;
        if (!node || !node->box.hit(ray, 0.001, std::numeric_limits<double>::max()))
            return rec;
        
        if (node->isLeaf) {
            HitRecord closest;
            closest.t = std::numeric_limits<double>::max();
            
            if (node->sphereIdx >= 0) {
                HitRecord hit = spheres[node->sphereIdx].intersect(ray);
                if (hit.hit && hit.t < closest.t) {
                    closest = hit;
                    if (spheres[node->sphereIdx].hasTexture) {
                        Vec3 texColor = spheres[node->sphereIdx].texture.sample(hit.u, hit.v);
                        closest.material.albedo = closest.material.albedo * texColor;
                    }
                }
            }
            
            if (node->triangleIdx >= 0) {
                HitRecord hit = triangles[node->triangleIdx].intersect(ray);
                if (hit.hit && hit.t < closest.t)
                    closest = hit;
            }
            
            return closest;
        }
        
        HitRecord leftHit = node->left ? intersectBVH(ray, node->left) : HitRecord();
        HitRecord rightHit = node->right ? intersectBVH(ray, node->right) : HitRecord();
        
        if (leftHit.hit && rightHit.hit)
            return leftHit.t < rightHit.t ? leftHit : rightHit;
        if (leftHit.hit) return leftHit;
        if (rightHit.hit) return rightHit;
        
        return rec;
    }
    
    HitRecord intersect(const Ray& ray) const {
        if (bvhRoot)
            return intersectBVH(ray, bvhRoot);
        
        HitRecord closestHit;
        closestHit.t = std::numeric_limits<double>::max();
        
        for (const auto& sphere : spheres) {
            HitRecord hit = sphere.intersect(ray);
            if (hit.hit && hit.t < closestHit.t) {
                closestHit = hit;
                if (sphere.hasTexture) {
                    Vec3 texColor = sphere.texture.sample(hit.u, hit.v);
                    closestHit.material.albedo = closestHit.material.albedo * texColor;
                }
            }
        }
        
        for (const auto& tri : triangles) {
            HitRecord hit = tri.intersect(ray);
            if (hit.hit && hit.t < closestHit.t)
                closestHit = hit;
        }

        for (const auto& smoothTri : smoothTriangles) {
            HitRecord hit = smoothTri.intersect(ray);
            if (hit.hit && hit.t < closestHit.t)
                closestHit = hit;
        }
        
        return closestHit;
    }
    
    Vec3 traceRay(const Ray& ray, int depth) const {
        if (depth <= 0)
            return Vec3(0, 0, 0);
        
        HitRecord hit = intersect(ray);
        
        if (hit.hit) {
            if (hit.material.type == EMISSIVE) {
                return hit.material.emission;
            }
            
            Ray scattered;
            Vec3 attenuation;
            
            if (scatter(ray, hit, attenuation, scattered)) {
                return attenuation * traceRay(scattered, depth - 1);
            }
            return Vec3(0, 0, 0);
        }
        //Creating backgound gradient
        Vec3 unitDirection = ray.direction;
        double t = 0.5 * (unitDirection.y + 1.0);
        return Vec3(1.0, 1.0, 1.0) * (1.0 - t) + backgroundColor * t;
    }
    
    // Load OBJ mesh file
    bool loadOBJ(const std::string& filename, Material mat) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "Failed to open OBJ file: " << filename << std::endl;
            return false;
        }
        
        std::vector<Vec3> vertices;
        std::string line;
        int triCount = 0;
        
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;
            
            if (type == "v") {
                // Vertex position
                double x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(Vec3(x, y, z));
            }
            else if (type == "f") {
                // Face (triangle)
                std::string v1Str, v2Str, v3Str;
                iss >> v1Str >> v2Str >> v3Str;
                
                // Parse vertex indices
                int v1 = std::stoi(v1Str.substr(0, v1Str.find('/'))) - 1;
                int v2 = std::stoi(v2Str.substr(0, v2Str.find('/'))) - 1;
                int v3 = std::stoi(v3Str.substr(0, v3Str.find('/'))) - 1;
                
                if (v1 >= 0 && v1 < vertices.size() &&
                    v2 >= 0 && v2 < vertices.size() &&
                    v3 >= 0 && v3 < vertices.size()) {
                    addTriangle(Triangle(vertices[v1], vertices[v2], vertices[v3], mat));
                    triCount++;
                }
            }
        }
        
        std::cout << "Loaded OBJ: " << filename << " (" << triCount << " triangles)" << std::endl;
        return true;
    }
};

// Write image to PPM file
void writePPM(const std::string& filename, const std::vector<std::vector<Vec3>>& image) {
    int width = image[0].size();
    int height = image.size();
    
    FILE* f = fopen(filename.c_str(), "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    
    std::vector<uint8_t> buffer(width * height * 3);
    int idx = 0;
    
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            Vec3 color = image[j][i];
            color.x = sqrt(color.x);
            color.y = sqrt(color.y);
            color.z = sqrt(color.z);
            
            buffer[idx++] = uint8_t(255.99 * std::min(1.0, std::max(0.0, color.x)));
            buffer[idx++] = uint8_t(255.99 * std::min(1.0, std::max(0.0, color.y)));
            buffer[idx++] = uint8_t(255.99 * std::min(1.0, std::max(0.0, color.z)));
        }
    }
    
    fwrite(buffer.data(), sizeof(uint8_t), buffer.size(), f);
    fclose(f);
}

// Main rendering loop
int main() {
    const int imageWidth = 1200;
    const int imageHeight = 900;
    const double aspectRatio = double(imageWidth) / imageHeight;
    const int samplesPerPixel = 150;  
    const int maxDepth = 50;
    
    Scene scene;
    
    // Ground plane with checkerboard
    Texture checker = Texture::makeCheckerboard(16);
    scene.addSphere(Sphere(Vec3(0, -100.5, -1), 100, 
                            Material::makeDiffuse(Vec3(1, 1, 1)), checker));
    
    // Diffuse sphere
    scene.addSphere(Sphere(Vec3(0, 0, -1), 0.5, 
                            Material::makeDiffuse(Vec3(0.7, 0.3, 0.3))));
    
    // Glass sphere
    scene.addSphere(Sphere(Vec3(-1, 0, -1), 0.5, 
                            Material::makeDielectric(1.5)));
    scene.addSphere(Sphere(Vec3(-1, 0, -1), -0.45, 
                            Material::makeDielectric(1.5)));
    
    // Metal sphere
    scene.addSphere(Sphere(Vec3(1, 0, -1), 0.5, 
                            Material::makeMetal(Vec3(0.8, 0.8, 0.8), 0.3)));
    
    // Light sources
    scene.addSphere(Sphere(Vec3(-2, 3, 0), 0.5, 
                            Material::makeEmissive(Vec3(1, 0.9, 0.8), 5.0)));
    scene.addSphere(Sphere(Vec3(2, 2.5, -2), 0.4, 
                            Material::makeEmissive(Vec3(0.8, 0.9, 1), 4.0)));
    
    // Triangle 
    scene.addTriangle(Triangle(Vec3(-1.5, -0.5, -2.5), 
                                Vec3(-0.5, -0.5, -2.5), 
                                Vec3(-1, 1, -2.5),
                                Material::makeDiffuse(Vec3(0.2, 0.8, 0.3))));
    
    // Add a second triangle for fun
    scene.addTriangle(Triangle(Vec3(1.5, 0.5, -2), 
                                Vec3(2.5, 0.5, -2), 
                                Vec3(2, -0.5, -2),
                                Material::makeMetal(Vec3(0.9, 0.5, 0.1), 0.2)));

    
    // Add more random spheres
    for (int i = 0; i < 10; i++) {
        Vec3 center(randomDouble(-3, 3), randomDouble(-0.3, 0.3), randomDouble(-3, -1));
        double radius = randomDouble(0.1, 0.3);
        
        int matType = int(randomDouble(0, 3));
        Material mat;
        if (matType == 0)
            mat = Material::makeDiffuse(Vec3::random());
        else if (matType == 1)
            mat = Material::makeMetal(Vec3::random(), randomDouble(0, 0.5));
        else
            mat = Material::makeDielectric(1.5);
        
        scene.addSphere(Sphere(center, radius, mat));
    }
    
    Vec3 sphereCenter(0.5, 0.3, -2);
    double r = 0.5;
    
    Vec3 top = sphereCenter + Vec3(0, r, 0);
    Vec3 bottom = sphereCenter + Vec3(0, -r, 0);
    Vec3 front = sphereCenter + Vec3(0, 0, r);
    Vec3 back = sphereCenter + Vec3(0, 0, -r);
    Vec3 left = sphereCenter + Vec3(-r, 0, 0);
    Vec3 right = sphereCenter + Vec3(r, 0, 0);
    
    Vec3 nTop = Vec3(0, 1, 0);
    Vec3 nBottom = Vec3(0, -1, 0);
    Vec3 nFront = Vec3(0, 0, 1);
    Vec3 nBack = Vec3(0, 0, -1);
    Vec3 nLeft = Vec3(-1, 0, 0);
    Vec3 nRight = Vec3(1, 0, 0);
    
    Material smoothMat = Material::makeDiffuse(Vec3(0.9, 0.5, 0.7));
    
    // Top pyramid
    scene.addSmoothTriangle(SmoothTriangle(top, front, right, nTop, nFront, nRight, smoothMat));
    scene.addSmoothTriangle(SmoothTriangle(top, right, back, nTop, nRight, nBack, smoothMat));
    scene.addSmoothTriangle(SmoothTriangle(top, back, left, nTop, nBack, nLeft, smoothMat));
    scene.addSmoothTriangle(SmoothTriangle(top, left, front, nTop, nLeft, nFront, smoothMat));
    
    // Bottom pyramid
    scene.addSmoothTriangle(SmoothTriangle(bottom, right, front, nBottom, nRight, nFront, smoothMat));
    scene.addSmoothTriangle(SmoothTriangle(bottom, back, right, nBottom, nBack, nRight, smoothMat));
    scene.addSmoothTriangle(SmoothTriangle(bottom, left, back, nBottom, nLeft, nBack, smoothMat));
    scene.addSmoothTriangle(SmoothTriangle(bottom, front, left, nBottom, nFront, nLeft, smoothMat));

    
    std::cout << "Building BVH acceleration structure..." << std::endl;
    scene.buildBVH();
    
    // Setup camera
    Camera camera(Vec3(-2.5, 2, 2.5), Vec3(0, 0, -1), Vec3(0, 1, 0), 
                  60.0, aspectRatio, 0.05, 5.0);
    
    // Render
    std::vector<std::vector<Vec3>> image(imageHeight, std::vector<Vec3>(imageWidth));
    
    std::cout << "Rendering " << imageWidth << "x" << imageHeight 
              << " with " << samplesPerPixel << " samples per pixel..." << std::endl;
    
    for (int j = 0; j < imageHeight; ++j) {
        if (j % 50 == 0) {
            std::cout << "Progress: " << (100 * j / imageHeight) << "%" << std::endl;
        }
        
        for (int i = 0; i < imageWidth; ++i) {
            Vec3 color(0, 0, 0);
            for (int s = 0; s < samplesPerPixel; ++s) {
                double u = (i + randomDouble()) / (imageWidth - 1);
                double v = (j + randomDouble()) / (imageHeight - 1);
                
                Ray ray = camera.getRay(u, v);
                color = color + scene.traceRay(ray, maxDepth);
            }
            
            color = color / samplesPerPixel;
            image[j][i] = color;
        }
    }
    
    writePPM("output.ppm", image);
    std::cout << "Done! Image saved as output.ppm" << std::endl;
    return 0;
}