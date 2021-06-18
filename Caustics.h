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



double min(double a, double b)
{
    if (a > b) {
        return b;
    } else {
        return a;
    }
}

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

/*
 Linearly interpolate the position where an isosurface cuts
 an edge between two vertices, each with their own scalar value
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

// Linear interpolation
double interpolationEigen1(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.eigen1[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.eigen1[1] * xx * (1 - yy) * (1 - zz) +
    g.eigen1[2] * xx * yy * (1 - zz) +
    g.eigen1[3] * (1 - xx) * yy * (1 - zz) +
    g.eigen1[4] * (1 - xx) * (1 - yy) * zz +
    g.eigen1[5] * xx * (1 - yy) * zz +
    g.eigen1[6] * xx * yy * zz +
    g.eigen1[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationEigen2(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.eigen2[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.eigen2[1] * xx * (1 - yy) * (1 - zz) +
    g.eigen2[2] * xx * yy * (1 - zz) +
    g.eigen2[3] * (1 - xx) * yy * (1 - zz) +
    g.eigen2[4] * (1 - xx) * (1 - yy) * zz +
    g.eigen2[5] * xx * (1 - yy) * zz +
    g.eigen2[6] * xx * yy * zz +
    g.eigen2[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationEigen3(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.eigen3[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.eigen3[1] * xx * (1 - yy) * (1 - zz) +
    g.eigen3[2] * xx * yy * (1 - zz) +
    g.eigen3[3] * (1 - xx) * yy * (1 - zz) +
    g.eigen3[4] * (1 - xx) * (1 - yy) * zz +
    g.eigen3[5] * xx * (1 - yy) * zz +
    g.eigen3[6] * xx * yy * zz +
    g.eigen3[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationInner(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.inner[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.inner[1] * xx * (1 - yy) * (1 - zz) +
    g.inner[2] * xx * yy * (1 - zz) +
    g.inner[3] * (1 - xx) * yy * (1 - zz) +
    g.inner[4] * (1 - xx) * (1 - yy) * zz +
    g.inner[5] * xx * (1 - yy) * zz +
    g.inner[6] * xx * yy * zz +
    g.inner[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationInnerD(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.innerD[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.innerD[1] * xx * (1 - yy) * (1 - zz) +
    g.innerD[2] * xx * yy * (1 - zz) +
    g.innerD[3] * (1 - xx) * yy * (1 - zz) +
    g.innerD[4] * (1 - xx) * (1 - yy) * zz +
    g.innerD[5] * xx * (1 - yy) * zz +
    g.innerD[6] * xx * yy * zz +
    g.innerD[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationInnerDD(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.innerDD[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.innerDD[1] * xx * (1 - yy) * (1 - zz) +
    g.innerDD[2] * xx * yy * (1 - zz) +
    g.innerDD[3] * (1 - xx) * yy * (1 - zz) +
    g.innerDD[4] * (1 - xx) * (1 - yy) * zz +
    g.innerDD[5] * xx * (1 - yy) * zz +
    g.innerDD[6] * xx * yy * zz +
    g.innerDD[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationVal1(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.val1[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.val1[1] * xx * (1 - yy) * (1 - zz) +
    g.val1[2] * xx * yy * (1 - zz) +
    g.val1[3] * (1 - xx) * yy * (1 - zz) +
    g.val1[4] * (1 - xx) * (1 - yy) * zz +
    g.val1[5] * xx * (1 - yy) * zz +
    g.val1[6] * xx * yy * zz +
    g.val1[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationVal2(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.val2[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.val2[1] * xx * (1 - yy) * (1 - zz) +
    g.val2[2] * xx * yy * (1 - zz) +
    g.val2[3] * (1 - xx) * yy * (1 - zz) +
    g.val2[4] * (1 - xx) * (1 - yy) * zz +
    g.val2[5] * xx * (1 - yy) * zz +
    g.val2[6] * xx * yy * zz +
    g.val2[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationCheck1(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.check1[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.check1[1] * xx * (1 - yy) * (1 - zz) +
    g.check1[2] * xx * yy * (1 - zz) +
    g.check1[3] * (1 - xx) * yy * (1 - zz) +
    g.check1[4] * (1 - xx) * (1 - yy) * zz +
    g.check1[5] * xx * (1 - yy) * zz +
    g.check1[6] * xx * yy * zz +
    g.check1[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationCheck2(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.check2[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.check2[1] * xx * (1 - yy) * (1 - zz) +
    g.check2[2] * xx * yy * (1 - zz) +
    g.check2[3] * (1 - xx) * yy * (1 - zz) +
    g.check2[4] * (1 - xx) * (1 - yy) * zz +
    g.check2[5] * xx * (1 - yy) * zz +
    g.check2[6] * xx * yy * zz +
    g.check2[7] * (1 - xx) * yy * zz;
    
    return dataxyz;
}

double interpolationCheck3(const XYZ point, const GRIDCELL g)
{
    double xx = point.x - g.p[0].x;
    double yy = point.y - g.p[0].y;
    double zz = point.z - g.p[0].z;
    
    double dataxyz =
    g.check3[0] * (1 - xx) * (1 - yy) * (1 - zz) +
    g.check3[1] * xx * (1 - yy) * (1 - zz) +
    g.check3[2] * xx * yy * (1 - zz) +
    g.check3[3] * (1 - xx) * yy * (1 - zz) +
    g.check3[4] * (1 - xx) * (1 - yy) * zz +
    g.check3[5] * xx * (1 - yy) * zz +
    g.check3[6] * xx * yy * zz +
    g.check3[7] * (1 - xx) * yy * zz;
    
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

