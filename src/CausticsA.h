#pragma once
#include "Types.h"

extern void computeA(
     vectorXYZ &A2, vectorXYZ &A3Bound, vectorXYZ &A3, vectorXYZ &A4, vectorXYZ &A5,
     const vector2d &eigen, const vector2d &grad,
     const double Dp, const int Ni);

extern int PolygoniseA2(GRIDCELL g, double iso,
                 std::vector<XYZ> &A2,
                 std::vector<XYZ> &A3Bound,
                 int v0, int v1, int v2, int v3, int Ni);

extern int PolygoniseA(GRIDCELL g, double Dp,
                vectorXYZ &A3, vectorXYZ &A4, vectorXYZ &A5,
                int v0, int v1, int v2, int v3,
                const int index, const vector2d &eigen, const vector2d &grad, int Ni);

