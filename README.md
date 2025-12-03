# Ray Tracing Engine - COMS 3360
# netID: apallem
# email: apallem@iastate.edu

# Small note
I forgot to make a git repo when I started this project, so the first commit is basically everything I've done so far with plenty left to do and clean up. 

## What's Working So Far

### Stuff so far complete but still needs tweaking (Required Features)
- Camera that you can move around and change the FOV
- Anti aliasing so edges don't look terrible
- Spheres (3-4 so far)
- Triangles(put 2 just so it was more visible)
- Loading textures from PPM files
- Putting textures on spheres
- Diffuse materials (matte stuff like chalk)
- Metal materials (shiny reflective surfaces)
- Glass materials (see through)
- Lights (emissive materials)
- Loading 3D models from OBJ files
- BVH acceleration so it doesn't take forever to render and does so in a timley manner

### Some extra effects so far that will be built on
- Depth of field blur 
- Multiple light sources
- Gamma correction so colors look right

## building and running:
g++ -O3 -std=c++17 main.cpp -o raytracer
./raytracer 
