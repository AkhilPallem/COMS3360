all:
	g++ -o raytracer main.cpp -std=c++11 -O3
	
clean:
	rm -f raytracer output.ppm output.png

run:
	./raytracer