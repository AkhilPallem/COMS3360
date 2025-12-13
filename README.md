# Ray Tracing Engine - COMS 3360
NetID: apallem  
Email: apallem@iastate.edu

## Overview
A C++ ray tracer implementation featuring physically based materials, spatial acceleration, and procedural textures. Renders to PPM format at 1200x900 resolution with 600 samples per pixel.

## Build and Run
make clean
g++ -O3 -std=c++17 main.cpp -o raytracer (make file)
./raytracer
convert output.ppm output.png (see the image in png format)

## Core Features Implemented
Camera System
  - Configurable position, orientation, and field of view
  - Thin lens model for depth of field effects

Anti-aliasing
  - 600 samples per pixel with jittered sampling
  - Reduces aliasing artifacts on edges

Geometric Primitives
  - Ray/sphere intersections with UV mapping
  - Ray/triangle intersections with UV mapping
  - Quad primitives for planar surfaces
  - Smooth triangles with vertex normal interpolation

Texture System
  - PPM format texture loading
  - Texture mapping on spheres and triangles
  - Procedural checkerboard pattern
  - Custom stripe pattern generation

3D Model Loading
  - OBJ file format support
  - Triangle mesh rendering

Acceleration Structure
  - BVH (Bounding Volume Hierarchy) with recursive subdivision

Material Types
  - Diffuse (Lambertian) with cosine-weighted scattering
  - Specular (Metal) with configurable roughness
  - Dielectric (Glass) with refraction and Schlick approximation
  - Emissive materials for area light sources
  - Perlin noise materials for procedural textures
  - Marble materials with turbulence patterns

## Additional Features (45 Points)
Motion Blur (10 points)
  - Time based sphere animation
  - Temporal sampling across shutter interval
  - Moving sphere support in BVH

Depth of Field (10 points)
  - Thin lens camera model
  - Configurable aperture size
  - Adjustable focus distance

Perlin Noise (10 points)
  - 3D gradient noise implementation
  - Turbulence function with octaves
  - Marble texture with sinusoidal patterns

Normal Interpolation (5 points)
  - Smooth triangle shading
  - Vertex normal interpolation
  - Improved appearance of low-poly geometry

Quads (10 points)
  - Planar quadrilateral primitives
  - Decomposition into two triangles

Custom things done 
  - Gamma correction for color accuracy (was too bright so i adjusted this to make it look cleaner and more visible)
  - Procedural ground plane with checkerboard pattern