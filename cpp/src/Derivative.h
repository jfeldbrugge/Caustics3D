#pragma once
#include "Types.h"

// Periodic finite difference method
extern void gradient(vector2d &gradient, vector2d &data, int depth, int Ni);
extern void gradient(vector2d &gradient, vector1d &data, int depth, int Ni);
// Finite difference order dx^2
extern double derivative(double dm1, double d1);
// Finite difference order dx^4
extern double derivative(double dm2, double dm1, double d1, double d2);
// Finite difference order dx^6
extern double derivative(double dm3, double dm2, double dm1, double d1, double d2, double d3);

