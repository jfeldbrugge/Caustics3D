compile:
	g++-10 -fopenmp -std=c++11  -O3 -march=native  -ffast-math -o Caustics3D main.cpp
	./Caustics3D
