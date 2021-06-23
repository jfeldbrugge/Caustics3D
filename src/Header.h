#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <cstdlib>

#include <Eigen/Eigenvalues>

#if defined(_OPENMP)
#include <omp.h>
extern const bool parallelism_enabled = true;
#else
extern const bool parallelism_enabled = false;
#endif
//#pragma omp parallel for

#include "Types.h"

void loadDisplacementFields(vector2d &s, int Ni, std::string directory);
void computeHessian (vector2d &H, const vector2d &s, int Ni);

void computeA(vectorXYZ &A2, vectorXYZ &A3Bound, vectorXYZ &A3, vectorXYZ &A4, vectorXYZ &A5,
              const vector2d &eigen, const vector2d &grad,
              const double Dp, const int Ni);
void computeD(vectorXYZ &D4ab, vectorXYZ &D4bc, const vector2d &H, const vector2d &a, const vector2d &b, const vector2d &c, const double Dp, const int Ni);

int PolygoniseA2(GRIDCELL g, double iso,
                 vectorXYZ &A2,
                 vectorXYZ &A3Bound,
                 int v0, int v1, int v2, int v3, int Ni);
int PolygoniseA(GRIDCELL g, double Dp,
                vectorXYZ &A3, vectorXYZ &A4, vectorXYZ &A5,
                int v0, int v1, int v2, int v3,
                const int index, const vector2d &eigen, const vector2d &grad, int Ni);
int PolygoniseD(GRIDCELL g, std::vector<XYZ> &D4ab, std::vector<XYZ> &D4bc,
                int v0, int v1, int v2, int v3, const double Dp, const int Ni);

void gradient(vector2d &gradient, vector1d &data, int depth, int Ni);
void gradient(vector2d &gradient, vector2d &data, int depth, int Ni);
double derivative(double dm1, double d1);
double derivative(double dm2, double dm1, double d1, double d2);
double derivative(double dm3, double dm2, double dm1, double d1, double d2, double d3);

void loadEigenvalueFields(vector2d &a, vector2d &b, vector2d &c, vector2d &H, int Ni, std::string directory);
void eigenFields(vector2d &a, vector2d &b, vector2d &c, vector2d &H, int Ni);
void eigen(Eigen::Matrix3d D, double &aa, double &bb, double &cc, Eigen::Vector3d &aav, Eigen::Vector3d &bbv, Eigen::Vector3d &ccv);

int mod(int a, int Ni);
double mod(double a, int Ni);
int indexOf (int i, int j, int k, int Ni);
void pos (int index, int Ni);
int shift (int index, int steps, int direction, int Ni);

double inner   (int index, int indexRef, const vector2d &eigen, const vector2d &grad, int Ni);
double innerD  (int index, int indexRef, const vector2d &eigen, const vector2d &grad, int Ni);
double innerDD (int index, int indexRef, const vector2d &eigen, const vector2d &grad, int Ni);

void offsets(const int index, vector1i &indices, const int Ni);

XYZ VertexInterp(double isolevel, XYZ p1, XYZ p2, double valp1, double valp2);
double interpolationEigen1(const XYZ point, const GRIDCELL g);
double interpolationEigen2(const XYZ point, const GRIDCELL g);
double interpolationEigen3(const XYZ point, const GRIDCELL g);
double interpolationInner(const XYZ point, const GRIDCELL g);
double interpolationInnerD(const XYZ point, const GRIDCELL g);
double interpolationInnerDD(const XYZ point, const GRIDCELL g);
double interpolationVal1(const XYZ point, const GRIDCELL g);
double interpolationVal2(const XYZ point, const GRIDCELL g);
double interpolationCheck1(const XYZ point, const GRIDCELL g);
double interpolationCheck2(const XYZ point, const GRIDCELL g);
double interpolationCheck3(const XYZ point, const GRIDCELL g);
double interpolation(const XYZ point, const vector2d &s, const int dir, const int Ni);

void import(vector2d &data, int depth, std::string fileName, int Ni);
void importB(vector2d &data, int depth, std::string fileName, int Ni);
void outputB(vector2d &data, int depth, std::string fileName, int Ni);
void writePoints(vectorXYZ &points, std::string name);
void writePointsB(std::vector<XYZ> &points, std::string name);
void writeHessian(vector2d &H, std::string directory);
void Eulerian(const vectorXYZ &Lpoints, vectorXYZ &Epoints, const double Dp, const vector2d &s, const int Ni);
void Scotch(const vectorXYZ &Lpoints, vectorXYZ &Spoints,
            const std::vector<std::vector<double>> &eigen, const std::vector<std::vector<double>> &s, const int Ni);
void Nbody(const vectorXYZ &Lpoints, vectorXYZ &Epoints, const vector2d &sNbody, const int Ni);

double length (vectorXYZ &lines);
double area (vectorXYZ &triangles);
double norm (XYZ p1, XYZ p2);
double min(double a, double b);
void setupPosition(GRIDCELL &g, int i, int j, int k);


int PolygoniseD2(GRIDCELL g, std::vector<XYZ> &D4ab, std::vector<XYZ> &D4bc,
                 int v0, int v1, int v2, int v3, const double Dp, const int Ni);

double cond1(const vector1d &H);
double cond2(const vector1d &H);
double cond3(const vector1d &H);

#include "Hessian.h"
#include "Eigenvalues.h"
#include "Derivative.h"
#include "Caustics.h"
#include "CausticsA.h"
#include "CausticsD.h"
#include "Utilities.h"

