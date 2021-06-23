#pragma once
#include "Types.h"

extern void offsets(const int index, vector1i &indices, const int Ni);
        
extern double inner (int index, int indexRef, const vector2d &eigen, const vector2d &grad, int Ni);

extern double innerD (int index, int indexRef, const vector2d &eigen, const vector2d &grad, int Ni);

extern double innerDD (int index, int indexRef, const vector2d &eigen, const vector2d &grad, int Ni);
        
extern void Eulerian(const vectorXYZ &Lpoints, vectorXYZ &Epoints,
              const double Dp, const vector2d &s, const int Ni);

extern void Scotch(const vectorXYZ &Lpoints, vectorXYZ &Spoints,
            const std::vector<std::vector<double>> &eigen, const std::vector<std::vector<double>> &s, const int Ni);

extern void Nbody(const vectorXYZ &Lpoints, vectorXYZ &Epoints,
           const vector2d &sNbody, const int Ni);

extern void setupPosition(GRIDCELL &g, int i, int j, int k);

XYZ VertexInterp(double isolevel, XYZ p1, XYZ p2, double valp1, double valp2);

extern double interpolationX(const XYZ point, XYZ const &p, double const eigen[8]);

extern double interpolation(const XYZ point, const vector2d &data, const int el, const int Ni);

