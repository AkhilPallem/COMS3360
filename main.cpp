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
#include <thread>
#include <mutex>

#include "vec3.h"
#include "ray.h"
#include "camera.h"

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

Vec3 getCheckerboardColor(const Vec3& p, double tileSize = 1.0);

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

// Different material types
enum MaterialType {
    DIFFUSE,
    METAL,
    DIELECTRIC,
    EMISSIVE,
    PERLIN_NOISE,
    PERLIN_MARBLE 
};

struct Material {
    MaterialType type;
    Vec3 albedo;
    double fuzz;
    double ior;
    Vec3 emission;
    double noiseScale;
    
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

    static Material makePerlinNoise(const Vec3& color1, const Vec3& color2, double scale) {
        Material m;
        m.type = PERLIN_NOISE;
        m.albedo = color1;     
        m.emission = color2;  
        m.noiseScale = scale;
        return m;
    }
    
    static Material makePerlinMarble(const Vec3& color1, const Vec3& color2, double scale) {
        Material m;
        m.type = PERLIN_MARBLE;
        m.albedo = color1;
        m.emission = color2;
        m.noiseScale = scale;
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

class PerlinNoise {
private:
    static const int pointCount = 256;
    Vec3* randVec;
    int* permX;
    int* permY;
    int* permZ;
    
    static int* perlinGeneratePerm() {
        int* p = new int[pointCount];
        for (int i = 0; i < pointCount; i++)
            p[i] = i;
        
        for (int i = pointCount - 1; i > 0; i--) {
            int target = int(randomDouble(0, i + 1));
            std::swap(p[i], p[target]);
        }
        return p;
    }
    
    static double perlinInterp(Vec3 c[2][2][2], double u, double v, double w) {
        double uu = u * u * (3 - 2 * u);
        double vv = v * v * (3 - 2 * v);
        double ww = w * w * (3 - 2 * w);
        double accum = 0.0;
        
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    Vec3 weightV(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu)) *
                            (j * vv + (1 - j) * (1 - vv)) *
                            (k * ww + (1 - k) * (1 - ww)) *
                            c[i][j][k].dot(weightV);
                }
        return accum;
    }
    
public:
    PerlinNoise() {
        randVec = new Vec3[pointCount];
        for (int i = 0; i < pointCount; i++) {
            randVec[i] = Vec3::random(-1, 1).normalize();
        }
        
        permX = perlinGeneratePerm();
        permY = perlinGeneratePerm();
        permZ = perlinGeneratePerm();
    }
    
    ~PerlinNoise() {
        delete[] randVec;
        delete[] permX;
        delete[] permY;
        delete[] permZ;
    }
    
    double noise(const Vec3& p) const {
        double u = p.x - floor(p.x);
        double v = p.y - floor(p.y);
        double w = p.z - floor(p.z);
        
        int i = int(floor(p.x));
        int j = int(floor(p.y));
        int k = int(floor(p.z));
        
        Vec3 c[2][2][2];
        
        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++)
                    c[di][dj][dk] = randVec[
                        permX[(i + di) & 255] ^
                        permY[(j + dj) & 255] ^
                        permZ[(k + dk) & 255]
                    ];
        
        return perlinInterp(c, u, v, w);
    }
    
    double turbulence(const Vec3& p, int depth = 7) const {
        double accum = 0.0;
        Vec3 tempP = p;
        double weight = 1.0;
        
        for (int i = 0; i < depth; i++) {
            accum += weight * noise(tempP);
            weight *= 0.5;
            tempP = tempP * 2.0;
        }
        
        return fabs(accum);
    }
};

PerlinNoise globalPerlin;

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
    bool isMoving;
    Vec3 centerEnd;
    
    Sphere(Vec3 c, double r, Material mat, Texture tex) 
    : center(c), radius(r), material(mat), texture(tex), hasTexture(true),
      isMoving(false), centerEnd(c) {}

    Sphere(Vec3 c, double r, Material mat) 
    : center(c), radius(r), material(mat), hasTexture(false), 
      isMoving(false), centerEnd(c) {}

    Sphere(Vec3 centerStart, Vec3 centerEnd, double r, Material mat) : center(centerStart), radius(r), material(mat), hasTexture(false), isMoving(true), centerEnd(centerEnd) {}
    
    Vec3 centerAt(double time) const {
        if (!isMoving) return center;
        return center + (centerEnd - center) * time;
    }
    
    HitRecord intersect(const Ray& ray) const {
        HitRecord rec;

        Vec3 currentCenter = centerAt(ray.time);
        
        Vec3 oc = ray.origin - currentCenter;
        double a = ray.direction.dot(ray.direction);
        double half_b = oc.dot(ray.direction);
        double c = oc.dot(oc) - radius * radius;

        double discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return rec;

        double sqrtd = sqrt(discriminant);

        double t = (-half_b - sqrtd) / a;
        if (t <= 0.001) {
            t = (-half_b + sqrtd) / a;
            if (t <= 0.001) return rec;
        }

        rec.t = t;
        rec.point = ray.at(t);
        Vec3 outwardNormal = (rec.point - currentCenter) / radius;  
        rec.setFaceNormal(ray, outwardNormal);
        rec.hit = true;
        rec.material = material;

        Vec3 p = outwardNormal;
        double phi = atan2(p.z, p.x);
        double theta = asin(p.y);
        rec.u = 1.0 - (phi + M_PI) / (2.0 * M_PI);
        rec.v = (theta + M_PI / 2.0) / M_PI;

        return rec;
    }
    
    AABB boundingBox() const {
        if (!isMoving) {
            return AABB(center - Vec3(radius, radius, radius),
                       center + Vec3(radius, radius, radius));
        }

        AABB box0(center - Vec3(radius, radius, radius),
                  center + Vec3(radius, radius, radius));
        AABB box1(centerEnd - Vec3(radius, radius, radius),
                  centerEnd + Vec3(radius, radius, radius));
        return AABB::surroundingBox(box0, box1);
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
    
    Triangle(Vec3 a, Vec3 b, Vec3 c, Material mat) : v0(a), v1(b), v2(c), material(mat), hasTexture(false) {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        normal = edge1.cross(edge2).normalize();
    }

    Triangle(Vec3 a, Vec3 b, Vec3 c, Material mat, Texture tex) : v0(a), v1(b), v2(c), material(mat), texture(tex), hasTexture(true) {
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

// Quad feature class 
class Quad {
public:
    Vec3 corner;
    Vec3 u, v;  
    Material material;
    
    Quad(Vec3 c, Vec3 edge1, Vec3 edge2, Material mat): corner(c), u(edge1), v(edge2), material(mat) {}
    
    HitRecord intersect(const Ray& ray) const {
        Vec3 p0 = corner;
        Vec3 p1 = corner + u;
        Vec3 p2 = corner + u + v;
        Vec3 p3 = corner + v;
        
        Triangle tri1(p0, p1, p2, material);
        HitRecord hit1 = tri1.intersect(ray);
        
        Triangle tri2(p0, p2, p3, material);
        HitRecord hit2 = tri2.intersect(ray);
        
        if (hit1.hit && hit2.hit)
            return hit1.t < hit2.t ? hit1 : hit2;
        if (hit1.hit) return hit1;
        if (hit2.hit) return hit2;
        
        return HitRecord();
    }
    
    AABB boundingBox() const {
        Vec3 p0 = corner;
        Vec3 p1 = corner + u;
        Vec3 p2 = corner + u + v;
        Vec3 p3 = corner + v;
        
        double minX = fmin(fmin(fmin(p0.x, p1.x), p2.x), p3.x) - 0.0001;
        double minY = fmin(fmin(fmin(p0.y, p1.y), p2.y), p3.y) - 0.0001;
        double minZ = fmin(fmin(fmin(p0.z, p1.z), p2.z), p3.z) - 0.0001;
        double maxX = fmax(fmax(fmax(p0.x, p1.x), p2.x), p3.x) + 0.0001;
        double maxY = fmax(fmax(fmax(p0.y, p1.y), p2.y), p3.y) + 0.0001;
        double maxZ = fmax(fmax(fmax(p0.z, p1.z), p2.z), p3.z) + 0.0001;
        
        return AABB(Vec3(minX, minY, minZ), Vec3(maxX, maxY, maxZ));
    }
};

// BVH node for acceleration
struct BVHNode {
    AABB box;
    std::shared_ptr<BVHNode> left;
    std::shared_ptr<BVHNode> right;

    std::vector<int> sphereIndices;
    std::vector<int> triangleIndices;
    std::vector<int> smoothTriangleIndices;
    std::vector<int> quadIndices;

    bool isLeaf = false;
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
    std::vector<Quad> quads;
    
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
    
    void addQuad(const Quad& quad) { 
        quads.push_back(quad);
    }

    // Build BVH tree for faster rendering
   void buildBVH() {
    std::vector<int> s, t, st, q;
    for (size_t i = 0; i < spheres.size(); ++i) s.push_back(i);
    for (size_t i = 0; i < triangles.size(); ++i) t.push_back(i);
    for (size_t i = 0; i < smoothTriangles.size(); ++i) st.push_back(i);
    for (size_t i = 0; i < quads.size(); ++i) q.push_back(i);
    bvhRoot = buildBVHRecursive(s, t, st, q, 0);
}
    
    std::shared_ptr<BVHNode> buildBVHRecursive(
    std::vector<int> sphereIdx,
    std::vector<int> triIdx,
    std::vector<int> smoothTriIdx,
    std::vector<int> quadIdx,
    int depth)
{
    auto node = std::make_shared<BVHNode>();

    bool first = true;
    for (int i : sphereIdx) {
        AABB b = spheres[i].boundingBox();
        node->box = first ? b : AABB::surroundingBox(node->box, b);
        first = false;
    }
    for (int i : triIdx) {
        AABB b = triangles[i].boundingBox();
        node->box = first ? b : AABB::surroundingBox(node->box, b);
        first = false;
    }
    for (int i : smoothTriIdx) {
        AABB b = smoothTriangles[i].boundingBox();
        node->box = first ? b : AABB::surroundingBox(node->box, b);
        first = false;
    }
    for (int i : quadIdx) {
        AABB b = quads[i].boundingBox();
        node->box = first ? b : AABB::surroundingBox(node->box, b);
        first = false;
    }

    int total = sphereIdx.size() + triIdx.size() + smoothTriIdx.size();
    if (total <= 4 || depth > 20) {
        node->isLeaf = true;
        node->sphereIndices = std::move(sphereIdx);
        node->triangleIndices = std::move(triIdx);
        node->smoothTriangleIndices = std::move(smoothTriIdx);
        node->quadIndices = std::move(quadIdx);
        return node;
    }

    Vec3 diag = node->box.max - node->box.min;
    int axis = (diag.x > diag.y) ? (diag.x > diag.z ? 0 : 2) : (diag.y > diag.z ? 1 : 2);

    auto getCoord = [&](const Vec3& p) { return axis == 0 ? p.x : (axis == 1 ? p.y : p.z); };

    std::vector<int> leftS, rightS, leftT, rightT, leftST, rightST, leftQ, rightQ;

    for (int i : sphereIdx)   (getCoord(spheres[i].center) < getCoord(node->box.min + node->box.max) * 0.5 ? leftS : rightS).push_back(i);
    for (int i : triIdx)      (getCoord((triangles[i].v0 + triangles[i].v1 + triangles[i].v2) / 3.0) < getCoord(node->box.min + node->box.max) * 0.5 ? leftT : rightT).push_back(i);
    for (int i : smoothTriIdx)(getCoord((smoothTriangles[i].v0 + smoothTriangles[i].v1 + smoothTriangles[i].v2) / 3.0) < getCoord(node->box.min + node->box.max) * 0.5 ? leftST : rightST).push_back(i);
    for (int i : quadIdx)(getCoord(quads[i].corner + (quads[i].u + quads[i].v) * 0.5) < getCoord(node->box.min + node->box.max) * 0.5 ? leftQ : rightQ).push_back(i);

    if (leftS.empty() && leftT.empty() && leftST.empty() && leftQ.empty()) { leftS = std::move(rightS); rightS.clear(); leftT = std::move(rightT); rightT.clear(); leftST = std::move(rightST); rightST.clear(); leftQ = std::move(rightQ); rightQ.clear(); } 
    if (rightS.empty() && rightT.empty() && rightST.empty() && rightQ.empty()) { rightS = std::move(leftS); leftS.clear(); rightT = std::move(leftT); leftT.clear(); rightST = std::move(leftST); leftST.clear(); rightQ = std::move(leftQ); leftQ.clear(); }

    node->left  = buildBVHRecursive(leftS,  leftT,  leftST,  leftQ, depth + 1);
    node->right = buildBVHRecursive(rightS, rightT, rightST, rightQ, depth + 1);
    return node;
}
    
    HitRecord intersectBVH(const Ray& ray, const std::shared_ptr<BVHNode>& node) const {
    HitRecord best; best.t = 1e30;
    if (!node || !node->box.hit(ray, 0.001, best.t)) return best;

    if (node->isLeaf) {
        for (int i : node->sphereIndices) {
            HitRecord h = spheres[i].intersect(ray);
            if (h.hit && h.t < best.t) { best = h;
                if (spheres[i].hasTexture) best.material.albedo = best.material.albedo * spheres[i].texture.sample(h.u, h.v);
            }
        }
        for (int i : node->triangleIndices) { 
            HitRecord h = triangles[i].intersect(ray); 
            if (h.hit && h.t < best.t) {
                best = h;
                if (triangles[i].hasTexture) 
                    best.material.albedo = best.material.albedo * triangles[i].texture.sample(h.u, h.v);
            }
        }
        for (int i : node->smoothTriangleIndices) { 
            HitRecord h = smoothTriangles[i].intersect(ray); 
            if (h.hit && h.t < best.t) best = h; 
        }
        for (int i : node->quadIndices) { 
            HitRecord h = quads[i].intersect(ray); 
            if (h.hit && h.t < best.t) best = h; 
        }
        return best;
    }

    HitRecord l = intersectBVH(ray, node->left);
    HitRecord r = intersectBVH(ray, node->right);
    return (l.hit && (!r.hit || l.t < r.t)) ? l : r;
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

        for (const auto& quad : quads) { 
            HitRecord hit = quad.intersect(ray);
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
            if (fabs(hit.point.y + 0.5) < 0.01) { 
            hit.material.albedo = getCheckerboardColor(hit.point, 1.0);
        }

        if (hit.material.type == PERLIN_NOISE) {
            double noiseValue = globalPerlin.turbulence(hit.point * hit.material.noiseScale);
            noiseValue = (noiseValue + 1.0) * 0.5;
            
            Vec3 color = hit.material.albedo * (1.0 - noiseValue) + 
                        hit.material.emission * noiseValue;
            
            hit.material.type = DIFFUSE;
            hit.material.albedo = color;
        }
        else if (hit.material.type == PERLIN_MARBLE) {
            double noiseValue = globalPerlin.turbulence(hit.point * hit.material.noiseScale);
            double pattern = sin(hit.point.z * hit.material.noiseScale + 10.0 * noiseValue);
            pattern = (pattern + 1.0) * 0.5;
            
            Vec3 color = hit.material.albedo * (1.0 - pattern) + 
                        hit.material.emission * pattern;
            
            hit.material.type = DIFFUSE;
            hit.material.albedo = color;
        }

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
            
            color.x = color.x / (1.0 + color.x);
            color.y = color.y / (1.0 + color.y);
            color.z = color.z / (1.0 + color.z);
            
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

Vec3 getCheckerboardColor(const Vec3& p, double tileSize) {
    double x = floor(p.x / tileSize);
    double z = floor(p.z / tileSize);
    bool isBlack = ((int)(x + z) & 1);        
    return isBlack ? Vec3(0.15, 0.15, 0.15) : Vec3(0.93, 0.93, 0.93);
}

int main() {
    const int imageWidth = 1200;
    const int imageHeight = 900;
    const double aspectRatio = double(imageWidth) / imageHeight;
    const int samplesPerPixel = 600;  
    const int maxDepth = 50;
    
    Scene scene;
    
    // Ground checkerboard pattern
    auto checkerMaterial = Material::makeDiffuse(Vec3(1,1,1));
    Vec3 floorCorner(-100, -0.5, -100);
    Vec3 floorEdge1(200, 0, 0);
    Vec3 floorEdge2(0, 0, 200);
    scene.addQuad(Quad(floorCorner, floorEdge1, floorEdge2, checkerMaterial));
    
    //all spheres in the scene
    scene.addSphere(Sphere(Vec3(-1.1, 0.0, -0.8), 0.5,  Material::makeDielectric(1.5)));
    scene.addSphere(Sphere(Vec3(-1.1, 0.0, -0.8), -0.49, Material::makeDielectric(1.5)));

    // Red diffuse (center)
    scene.addSphere(Sphere(Vec3( 0.0, 0.0, -1.0), 0.5,  Material::makeDiffuse(Vec3(0.8, 0.2, 0.2))));

    // Shiny metal (right)
    scene.addSphere(Sphere(Vec3( 1.1, 0.0, -0.9), 0.5,  Material::makeMetal(Vec3(0.9, 0.9, 0.95), 0.05)));
    
    // Light sources
    scene.addSphere(Sphere(Vec3(-2.0, 3.0,  0.0), 0.6, Material::makeEmissive(Vec3(1.0, 0.95, 0.9), 70)));
    //scene.addSphere(Sphere(Vec3( 2.0, 2.8, -1.0), 0.5, Material::makeEmissive(Vec3(0.9, 0.95, 1.0), 80)));
    //scene.addSphere(Sphere(Vec3(-0.3, 2.2,  0.5), 0.4, Material::makeEmissive(Vec3(1.0, 1.0, 1.0), 90)));

    // Triangles 
    scene.addTriangle(Triangle(Vec3(-2, -0.3, -2), 
                                Vec3(-1, -0.3, -2), 
                                Vec3(-1.5, 1.2, -2),
                                Material::makeDiffuse(Vec3(0.2, 0.9, 0.3))));
    
    scene.addTriangle(Triangle(Vec3(1.5, 0.2, -1.5), 
                                Vec3(2.8, 0.2, -1.5), 
                                Vec3(2.15, 1.5, -1.5),
                                Material::makeMetal(Vec3(0.95, 0.6, 0.1), 0.15)));

    //Make a texture triangle more visible with a different pattern from my floor                      
    Texture stripedTexture;
    stripedTexture.width = 16;
    stripedTexture.height = 16;
    stripedTexture.data.resize(16, std::vector<Vec3>(16));

    for (int j = 0; j < 16; j++) {
        for (int i = 0; i < 16; i++) {
            if ((j / 2) % 2 == 0) {
                stripedTexture.data[j][i] = Vec3(0.2, 0.4, 0.9);
            } else {
                stripedTexture.data[j][i] = Vec3(0.95, 0.85, 0.3); 
            }
        }
    }
    scene.addTriangle(Triangle(
        Vec3(1.5, -0.5, -3.5), 
        Vec3(3.5, -0.5, -3.5),  
        Vec3(2.5, 2.0, -3.5),  
        Material::makeDiffuse(Vec3(1.0, 1.0, 1.0)), 
        stripedTexture
    ));


    scene.addSphere(Sphere(
        Vec3(-1.9, 0.85, -5.2),
        Vec3( 2.3, 0.85, -5.2),
        0.22,
        Material::makeEmissive(Vec3(1.0, 0.9, 0.7), 8)  
    ));


    
    Vec3 sphereCenter(0.5, 0.4, -2);
    double r = 1.0;
    
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
    
    Material smoothMat = Material::makeDiffuse(Vec3(0.95, 0.7, 0.85));
    
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

    scene.addQuad(Quad(Vec3(-2.2, -0.4, -2.8), Vec3(0.9, 0, 0), Vec3(0, 1.3, 0), Material::makeDiffuse(Vec3(0.2, 0.7, 0.9))));
    
    scene.addQuad(Quad(Vec3(1.8, -0.3, -2.2), Vec3(0.7, 0, -0.2), Vec3(0, 1.1, 0), Material::makeMetal(Vec3(0.9, 0.4, 0.6), 0.15)));

    //Noise sphere
    scene.addSphere(Sphere(
        Vec3(-2.5, 0.3, -1.2),
        0.5,
        Material::makePerlinNoise(
            Vec3(0.2, 0.3, 0.8), 
            Vec3(0.9, 0.9, 0.95),
            4.0          
        )
    ));

    //Marble sphere
    scene.addSphere(Sphere(
    Vec3(2.8, 0.3, -1.2), 
    0.5,
    Material::makePerlinMarble(
        Vec3(0.1, 0.1, 0.1),    
        Vec3(0.9, 0.85, 0.8),   
        3.0                      
    )
));

    
    scene.buildBVH();
    
    // Setup camera (a lot of diff angles to choose from but thinking this will show everything needed. might adjust as time goes on and more gets added)
    Camera camera(Vec3(-1.8, 1.8, 2.8), Vec3(0.0, 0.1, -1.0), Vec3(0, 1, 0), 65.0, aspectRatio, 0.08, 6.0);
    
    // Render and show progress in terminal 
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
    std::cout << "Command to convert to PNG: convert output.ppm output.png" << std::endl;
    return 0;
}