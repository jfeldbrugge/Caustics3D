#pragma once
#include <vector>

typedef std::vector<int> vector1i;
typedef std::vector<double> vector1d;
typedef std::vector<std::vector<int> > vector2i;
typedef std::vector<std::vector<double> > vector2d;

typedef struct
{
    double x,y,z;
} XYZ;
typedef struct
{
    XYZ p[8];
    double eigen1[8];
    double eigen2[8];
    double eigen3[8];
    double inner[8];
    double innerD[8];
    double innerDD[8];
    double val1[8];
    double val2[8];
    double check1[8];
    double check2[8];
    double check3[8];
} GRIDCELL;

typedef std::vector<XYZ> vectorXYZ;

