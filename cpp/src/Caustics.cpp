#include "Caustics.h"
#include "Utilities.h"
#include "Derivative.h"

/*
 Polygonise a tetrahedron given its vertices within a cube.
 This is an alternative algorithm to polygonisegrid.
 It results in a smoother surface but more triangular facets.
 
 |                 + 0
 |                /|\
 |               / | \
 |              /  |  \
 |             /   |   \
 |            /    |    \
 |           /     |     \
 |        3 +-------------+ 1
 |           \     |     /
 |            \    |    /
 |             \   |   /
 |              \  |  /
 |               \ | /
 |                \|/
 |                 + 2
 
 It's main purpose is still to polygonise a gridded dataset and
 would normally be called 6 times, one for each tetrahedron making
 up the grid cell.
 Given the grid labelling as in PolygniseGrid one would call
 PolygoniseTri(grid,iso,triangles,0,2,3,7);
 PolygoniseTri(grid,iso,triangles,0,2,6,7);
 PolygoniseTri(grid,iso,triangles,0,4,6,7);
 PolygoniseTri(grid,iso,triangles,0,6,1,2);
 PolygoniseTri(grid,iso,triangles,0,6,1,4);
 PolygoniseTri(grid,iso,triangles,5,6,1,4);
 */

/*! \brief Create a grid stencil, given an index.
 *
 * Writes output to indices.
 */
void offsets(const int index, vector1i &indices, const int Ni)
{
    indices =
    {   index,
        shift(index, 1, 0, Ni),
        shift(shift(index, 1, 0, Ni), 1, 1, Ni),
        shift(index, 1, 1, Ni),
        shift(index, 1, 2, Ni),
        shift(shift(index, 1, 0, Ni), 1, 2, Ni),
        shift(shift(shift(index, 1, 0, Ni), 1, 1, Ni), 1, 2, Ni),
        shift(shift(index, 1, 1, Ni), 1, 2, Ni)
    };
}

/*! \brief Compute oriented innerproduct of eigen vector against gradient of
 *         eigenvalue.
 *
 *  Compensates for artificial numeric flips in eigenvector fields. (i.e.
 *  line segments versus vectors).
 */
double inner (int index, int indexRef, const vector2d &eigen, const vector2d &grad, int Ni)
{
    bool flip =
    eigen[index][1] * eigen[indexRef][1] +
    eigen[index][2] * eigen[indexRef][2] +
    eigen[index][3] * eigen[indexRef][3] > 0;
    
    double result = grad[index][0] * eigen[index][1] + grad[index][1] * eigen[index][2] + grad[index][2] * eigen[index][3];
    
    if (flip) {
        return +result;
    } else {
        return -result;
    }
}

/*! \brief Compute inner product between eigenvector and double gradient.
 */
double innerD (int index, int indexRef, const vector2d &eigen, const vector2d &grad, int Ni)
{
    bool flip =
    eigen[index][1] * eigen[indexRef][1] +
    eigen[index][2] * eigen[indexRef][2] +
    eigen[index][3] * eigen[indexRef][3] > 0;
    
    double result =
    eigen[index][1] *
    derivative(inner(shift(index, -2, 0, Ni), indexRef, eigen, grad, Ni),
               inner(shift(index, -1, 0, Ni), indexRef, eigen, grad, Ni),
               inner(shift(index, +1, 0, Ni), indexRef, eigen, grad, Ni),
               inner(shift(index, +2, 0, Ni), indexRef, eigen, grad, Ni)) +
    eigen[index][2] *
    derivative(inner(shift(index, -2, 1, Ni), indexRef, eigen, grad, Ni),
               inner(shift(index, -1, 1, Ni), indexRef, eigen, grad, Ni),
               inner(shift(index, +1, 1, Ni), indexRef, eigen, grad, Ni),
               inner(shift(index, +2, 1, Ni), indexRef, eigen, grad, Ni)) +
    eigen[index][3] *
    derivative(inner(shift(index, -2, 2, Ni), indexRef, eigen, grad, Ni),
               inner(shift(index, -1, 2, Ni), indexRef, eigen, grad, Ni),
               inner(shift(index, +1, 2, Ni), indexRef, eigen, grad, Ni),
               inner(shift(index, +2, 2, Ni), indexRef, eigen, grad, Ni));
    
    if (flip) {
        return +result;
    } else {
        return -result;
    }
}

/*! \brief Compute inner product between eigenvector and third gradient of eigenvalue.
 */
double innerDD (int index, int indexRef, const vector2d &eigen, const vector2d &grad, int Ni)
{
    bool flip =
    eigen[index][1] * eigen[indexRef][1] +
    eigen[index][2] * eigen[indexRef][2] +
    eigen[index][3] * eigen[indexRef][3] > 0;
    
    double result =
    eigen[index][1] *
    derivative(innerD(shift(index, -2, 0, Ni), indexRef, eigen, grad, Ni),
               innerD(shift(index, -1, 0, Ni), indexRef, eigen, grad, Ni),
               innerD(shift(index, +1, 0, Ni), indexRef, eigen, grad, Ni),
               innerD(shift(index, +2, 0, Ni), indexRef, eigen, grad, Ni)) +
    eigen[index][2] *
    derivative(innerD(shift(index, -2, 1, Ni), indexRef, eigen, grad, Ni),
               innerD(shift(index, -1, 1, Ni), indexRef, eigen, grad, Ni),
               innerD(shift(index, +1, 1, Ni), indexRef, eigen, grad, Ni),
               innerD(shift(index, +2, 1, Ni), indexRef, eigen, grad, Ni)) +
    eigen[index][3] *
    derivative(innerD(shift(index, -2, 2, Ni), indexRef, eigen, grad, Ni),
               innerD(shift(index, -1, 2, Ni), indexRef, eigen, grad, Ni),
               innerD(shift(index, +1, 2, Ni), indexRef, eigen, grad, Ni),
               innerD(shift(index, +2, 2, Ni), indexRef, eigen, grad, Ni));
    
    if (flip) {
        return +result;
    } else {
        return -result;
    }
}

// Map from Lagrangian to Eulerian
/*! \brief Linear interpolation of Lagrangian map for
 *         Zeldovich Approximation.
 */
void Eulerian(const vectorXYZ &Lpoints, vectorXYZ &Epoints,
              const double Dp, const vector2d &s, const int Ni)
{
    std::vector<XYZ> points(Lpoints.size());
    for (int index = 0; index < Lpoints.size(); index++)
    {
        XYZ p = Lpoints[index];
        
        points[index].x = p.x + Dp * interpolation(p, s, 0, Ni);
        points[index].y = p.y + Dp * interpolation(p, s, 1, Ni);
        points[index].z = p.z + Dp * interpolation(p, s, 2, Ni);
    }
    Epoints = points;
}

/*! \brief Scotch flow approximation.
 *
 * Output to Spoints.
 */
void Scotch(const vectorXYZ &Lpoints, vectorXYZ &Spoints,
            const std::vector<std::vector<double>> &eigen, const std::vector<std::vector<double>> &s, const int Ni)
{
    vectorXYZ points(Lpoints.size());
    for (int index = 0; index < Lpoints.size(); index++)
    {
        XYZ p = Lpoints[index];
        
        const double Dp = - 1 / interpolation(p, eigen, 0, Ni);
        points[index].x = p.x + Dp * interpolation(p, s, 0, Ni);
        points[index].y = p.y + Dp * interpolation(p, s, 1, Ni);
        points[index].z = p.z + Dp * interpolation(p, s, 2, Ni);
    }
    Spoints = points;
}

/*! \brief Interpolation on any given displacement.
 */
void Nbody(const vectorXYZ &Lpoints, vectorXYZ &Epoints,
           const vector2d &sNbody, const int Ni)
{
    std::vector<XYZ> points(Lpoints.size());
    for (int index = 0; index < Lpoints.size(); index++)
    {
        XYZ p = Lpoints[index];
        
        points[index].x = p.x + interpolation(p, sNbody, 0, Ni);
        points[index].y = p.y + interpolation(p, sNbody, 1, Ni);
        points[index].z = p.z + interpolation(p, sNbody, 2, Ni);
    }
    Epoints = points;
}

/*! \brief Create initial Lagrangian grid.
 *
 * Think np.indices.
 */
void setupPosition(GRIDCELL &g, int i, int j, int k)
{
    g.p[0].x = i;       g.p[0].y = j;       g.p[0].z = k;
    g.p[1].x = i + 1;   g.p[1].y = j;       g.p[1].z = k;
    g.p[2].x = i + 1;   g.p[2].y = j + 1;   g.p[2].z = k;
    g.p[3].x = i;       g.p[3].y = j + 1;   g.p[3].z = k;
    g.p[4].x = i;       g.p[4].y = j;       g.p[4].z = k + 1;
    g.p[5].x = i + 1;   g.p[5].y = j;       g.p[5].z = k + 1;
    g.p[6].x = i + 1;   g.p[6].y = j + 1;   g.p[6].z = k + 1;
    g.p[7].x = i;       g.p[7].y = j + 1;   g.p[7].z = k + 1;
}

/*! \brief Linearly interpolate the position where an isosurface cuts
 *         an edge between two vertices, each with their own scalar value
 */
XYZ VertexInterp(double isolevel, XYZ p1, XYZ p2, double valp1, double valp2)
{
    double mu;
    XYZ p;
    
    if (std::abs(isolevel-valp1) < 0.00001)
        return(p1);
    if (std::abs(isolevel-valp2) < 0.00001)
        return(p2);
    if (std::abs(valp1-valp2) < 0.00001)
        return(p1);
    mu = (isolevel - valp1) / (valp2 - valp1);
    p.x = p1.x + mu * (p2.x - p1.x);
    p.y = p1.y + mu * (p2.y - p1.y);
    p.z = p1.z + mu * (p2.z - p1.z);
    
    return(p);
}

/*! \brief Tri-linear interpolation given a cube with values eigen,
 *         grid position p, and floating position point.
 */
double interpolationX(const XYZ point, XYZ const &p, double const eigen[8])
{
    double xx = point.x - p.x;
    double yy = point.y - p.y;
    double zz = point.z - p.z;
    
    double dataxyz =
    eigen[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    eigen[1] * xx * (1 - yy) * (1 - zz) +
    eigen[2] * xx * yy * (1 - zz) +
    eigen[3] * (1 - xx) * yy * (1 - zz) +
    eigen[4] * (1 - xx) * (1 - yy) * zz +
    eigen[5] * xx * (1 - yy) * zz +
    eigen[6] * xx * yy * zz +
    eigen[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolation(const XYZ point, const vector2d &data, const int el, const int Ni)
{
    int i = (int) point.x; double xx = point.x - i;
    int j = (int) point.y; double yy = point.y - j;
    int k = (int) point.z; double zz = point.z - k;
    
    int indexOfCell = indexOf(i, j, k, Ni);
    
    double dataxyz =
    data[indexOfCell][el] * (1 - xx) * (1 - yy) * (1 - zz) +
    data[shift(indexOfCell, 1, 0, Ni)][el] * xx * (1 - yy) * (1 - zz) +
    data[shift(indexOfCell, 1, 1, Ni)][el] * (1 - xx) * yy * (1 - zz) +
    data[shift(indexOfCell, 1, 2, Ni)][el] * (1 - xx) * (1 - yy) * zz +
    data[shift(shift(indexOfCell, 1, 0, Ni), 1, 2, Ni)][el] * xx * (1 - yy) * zz +
    data[shift(shift(indexOfCell, 1, 1, Ni), 1, 2, Ni)][el] * (1 - xx) * yy * zz +
    data[shift(shift(indexOfCell, 1, 0, Ni), 1, 1, Ni)][el] * xx * yy * (1 - zz) +
    data[shift(shift(shift(indexOfCell, 1, 0, Ni), 1, 1, Ni), 1, 2, Ni)][el] * xx * yy * zz;
    
    return dataxyz;
}

