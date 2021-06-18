void computeA(vectorXYZ &A2, vectorXYZ &A3Bound, vectorXYZ &A3, vectorXYZ &A4, vectorXYZ &A5,
              const vector2d &eigen, const vector2d &grad,
              const double Dp, const int Ni)
{
    //    Setup grid
    //    #pragma omp parallel for
    GRIDCELL g;
    for (int index = 0; index < Ni * Ni * Ni; index++)
    {
        int i = index % Ni;
        int j = ((index % (Ni * Ni)) - i) / Ni;
        int k = (index - i - j * Ni) / (Ni * Ni);
        
        setupPosition(g, i, j, k);
        
        vector1i indices;
        offsets(index, indices, Ni);
        for (int i = 0; i < 8; i++)
        {
            g.eigen1[i] = eigen[indices[i]][0];
            g.inner[i]   = inner  (indices[i], index, eigen, grad, Ni);
            //            grid[index].innerD[i]  = innerD (indices[i], index, eigen, grad, Ni);
            //            grid[index].innerDD[i] = innerDD(indices[i], index, eigen, grad, Ni);
        }
        
        PolygoniseA2(g, -1. / Dp, A2, A3Bound, 0, 2, 3, 7, Ni);
        PolygoniseA2(g, -1. / Dp, A2, A3Bound, 0, 2, 6, 7, Ni);
        PolygoniseA2(g, -1. / Dp, A2, A3Bound, 0, 4, 6, 7, Ni);
        PolygoniseA2(g, -1. / Dp, A2, A3Bound, 0, 6, 1, 2, Ni);
        PolygoniseA2(g, -1. / Dp, A2, A3Bound, 0, 6, 1, 4, Ni);
        PolygoniseA2(g, -1. / Dp, A2, A3Bound, 5, 6, 1, 4, Ni);
        
        PolygoniseA(g, Dp, A3, A4, A5, 0, 2, 3, 7, index, eigen, grad, Ni);
        PolygoniseA(g, Dp, A3, A4, A5, 0, 2, 6, 7, index, eigen, grad, Ni);
        PolygoniseA(g, Dp, A3, A4, A5, 0, 4, 6, 7, index, eigen, grad, Ni);
        PolygoniseA(g, Dp, A3, A4, A5, 0, 6, 1, 2, index, eigen, grad, Ni);
        PolygoniseA(g, Dp, A3, A4, A5, 0, 6, 1, 4, index, eigen, grad, Ni);
        PolygoniseA(g, Dp, A3, A4, A5, 5, 6, 1, 4, index, eigen, grad, Ni);
    }
}

int PolygoniseA2(GRIDCELL g, double iso,
                 std::vector<XYZ> &A2,
                 std::vector<XYZ> &A3Bound,
                 int v0, int v1, int v2, int v3, int Ni)
{
    /*
     Determine which of the 16 cases we have given which vertices
     are above or below the isosurface
     */
    
    int ntri = 0;
    int triindex = 0;
    if (g.eigen1[v0] < iso) triindex |= 1;
    if (g.eigen1[v1] < iso) triindex |= 2;
    if (g.eigen1[v2] < iso) triindex |= 4;
    if (g.eigen1[v3] < iso) triindex |= 8;
    
    std::function<void (int, int, int, int, int, int)>
    push = [&] (int v1, int w1, int v2, int w2, int v3, int w3)
    {
        XYZ p1 = VertexInterp(iso, g.p[v1], g.p[w1], g.eigen1[v1], g.eigen1[w1]);
        XYZ p2 = VertexInterp(iso, g.p[v2], g.p[w2], g.eigen1[v2], g.eigen1[w2]);
        XYZ p3 = VertexInterp(iso, g.p[v3], g.p[w3], g.eigen1[v3], g.eigen1[w3]);
        
        A2.push_back(p1);
        A2.push_back(p2);
        A2.push_back(p3);
        
        //        Compute A3Bound
        double inner1 = interpolationInner(p1, g);
        double inner2 = interpolationInner(p2, g);
        double inner3 = interpolationInner(p3, g);
        
        if (inner1 * inner2 < 0)
        {
            A3Bound.push_back(VertexInterp(0., p1, p2, inner1, inner2));
        }
        if (inner1 * inner3 < 0)
        {
            A3Bound.push_back(VertexInterp(0., p1, p3, inner1, inner3));
        }
        if (inner2 * inner3 < 0)
        {
            A3Bound.push_back(VertexInterp(0., p2, p3, inner2, inner3));
        }
    };
    
    /* Form the vertices of the triangles for each case */
    switch (triindex) {
        case 0x00:
        case 0x0F:
            break;
        case 0x0E:
        case 0x01:
            push(v0, v1, v0, v2, v0, v3); ntri++;
            break;
        case 0x0D:
        case 0x02:
            push(v1, v0, v1, v3, v1, v2); ntri++;
            break;
        case 0x0C:
        case 0x03:
            push(v0, v3, v0, v2, v1, v3); ntri++;
            push(v1, v3, v1, v2, v0, v2); ntri++;
            break;
        case 0x0B:
        case 0x04:
            push(v2, v0, v2, v1, v2, v3); ntri++;
            break;
        case 0x0A:
        case 0x05:
            push(v0, v1, v2, v3, v0, v3); ntri++;
            push(v0, v1, v1, v2, v2, v3); ntri++;
            break;
        case 0x09:
        case 0x06:
            push(v0, v1, v1, v3, v2, v3); ntri++;
            push(v0, v1, v0, v2, v2, v3); ntri++;
            break;
        case 0x07:
        case 0x08:
            push(v3, v0, v3, v2, v3, v1); ntri++;
            break;
    }
    return(ntri);
}

int PolygoniseA(GRIDCELL g, double Dp,
                vectorXYZ &A3, vectorXYZ &A4, vectorXYZ &A5,
                int v0, int v1, int v2, int v3,
                const int index, const vector2d &eigen, const vector2d &grad, int Ni)
{
    /*
     Determine which of the 16 cases we have given which vertices
     are above or below the isosurface
     */
    
    int ntri = 0;
    int triindex = 0;
    if (g.inner[v0] < 0) triindex |= 1;
    if (g.inner[v1] < 0) triindex |= 2;
    if (g.inner[v2] < 0) triindex |= 4;
    if (g.inner[v3] < 0) triindex |= 8;
    
    vector1i indices;
    offsets(index, indices, Ni);
    
    std::function<void (XYZ, XYZ)>
    pushA5 = [&] (const XYZ &p1, const XYZ &p2)
    {
        for (int i = 0; i < 8; i++)
        {
            g.innerDD[i] = innerDD(indices[i], index, eigen, grad, Ni);
        }
        
        double innerDD1 = interpolationInnerDD(p1, g);
        double innerDD2 = interpolationInnerDD(p2, g);
        
        if (innerDD1 * innerDD2 < 0)
        {
            A5.push_back(VertexInterp(0., p1, p2, innerDD1, innerDD2));
        }
    };
    
    std::function<void (XYZ, XYZ, XYZ)>
    pushA4A5 = [&] (const XYZ &p1, const XYZ &p2, const XYZ &p3)
    {
        
        for (int i = 0; i < 8; i++)
        {
            g.innerD[i] = innerD (indices[i], index, eigen, grad, Ni);
        }
        
        double innerD1 = interpolationInnerD(p1, g);
        double innerD2 = interpolationInnerD(p2, g);
        double innerD3 = interpolationInnerD(p3, g);
        
        XYZ pp1, pp2;
        
        int triindex = 0;
        if (innerD1 < 0) triindex |= 1;
        if (innerD2 < 0) triindex |= 2;
        if (innerD3 < 0) triindex |= 4;
        
        switch (triindex) {
            case 0x00:
            case 0x07:
                break;
            case 0x01:
            case 0x06:
                pp1 = VertexInterp(0., p1, p2, innerD1, innerD2);
                pp2 = VertexInterp(0., p1, p3, innerD1, innerD3);
                A4.push_back(pp1);
                A4.push_back(pp2);
                pushA5(pp1, pp2);
                break;
            case 0x02:
            case 0x05:
                pp1 = VertexInterp(0., p1, p2, innerD1, innerD2);
                pp2 = VertexInterp(0., p2, p3, innerD2, innerD3);
                A4.push_back(pp1);
                A4.push_back(pp2);
                pushA5(pp1, pp2);
                break;
            case 0x03:
            case 0x04:
                pp1 = VertexInterp(0., p1, p3, innerD1, innerD3);
                pp2 = VertexInterp(0., p2, p3, innerD2, innerD3);
                A4.push_back(pp1);
                A4.push_back(pp2);
                pushA5(pp1, pp2);
                break;
        }
    };
    
    std::function<void (int, int, int, int, int, int)>
    pushA3A4A5 = [&] (int v1, int w1, int v2, int w2, int v3, int w3)
    {
        XYZ p1 = VertexInterp(0, g.p[v1], g.p[w1], g.inner[v1], g.inner[w1]);
        XYZ p2 = VertexInterp(0, g.p[v2], g.p[w2], g.inner[v2], g.inner[w2]);
        XYZ p3 = VertexInterp(0, g.p[v3], g.p[w3], g.inner[v3], g.inner[w3]);
        
        double eigen1 = interpolationEigen1(p1, g);
        double eigen2 = interpolationEigen1(p2, g);
        double eigen3 = interpolationEigen1(p3, g);
        
        int triindex = 0;
        if (eigen1 < -1. / Dp) triindex |= 1;
        if (eigen2 < -1. / Dp) triindex |= 2;
        if (eigen3 < -1. / Dp) triindex |= 4;
        
        XYZ pp1, pp2, pp3;
        
        switch (triindex) {
            case 0x00:
                break;
            case 0x01:
                pp2 = VertexInterp(-1. / Dp, p1, p2, eigen1, eigen2);
                pp3 = VertexInterp(-1. / Dp, p1, p3, eigen1, eigen3);
                A3.push_back(p1);
                A3.push_back(pp2);
                A3.push_back(pp3);
                pushA4A5(p1, pp2, pp3);
                break;
            case 0x02:
                pp1 = VertexInterp(-1. / Dp, p1, p2, eigen1, eigen2);
                pp3 = VertexInterp(-1. / Dp, p2, p3, eigen2, eigen3);
                A3.push_back(pp1);
                A3.push_back(p2);
                A3.push_back(pp3);
                pushA4A5(pp1, p2, pp3);
                break;
            case 0x03:
                pp1 = VertexInterp(-1. / Dp, p1, p3, eigen1, eigen3);
                pp2 = VertexInterp(-1. / Dp, p2, p3, eigen2, eigen3);
                A3.push_back(p1);
                A3.push_back(pp1);
                A3.push_back(p2);
                pushA4A5(p1, pp1, p2);
                A3.push_back(p2);
                A3.push_back(pp1);
                A3.push_back(pp2);
                pushA4A5(p2, pp1, pp2);
                break;
            case 0x04:
                pp1 = VertexInterp(-1. / Dp, p1, p3, eigen1, eigen3);
                pp2 = VertexInterp(-1. / Dp, p2, p3, eigen2, eigen3);
                A3.push_back(pp1);
                A3.push_back(pp2);
                A3.push_back(p3);
                pushA4A5(pp1, pp2, p3);
                break;
            case 0x05:
                pp1 = VertexInterp(-1. / Dp, p1, p2, eigen1, eigen2);
                pp3 = VertexInterp(-1. / Dp, p2, p3, eigen2, eigen3);
                A3.push_back(p1);
                A3.push_back(p3);
                A3.push_back(pp1);
                pushA4A5(p1, p3, pp1);
                A3.push_back(p3);
                A3.push_back(pp3);
                A3.push_back(pp1);
                pushA4A5(p3, pp3, pp1);
                break;
            case 0x06:
                pp2 = VertexInterp(-1. / Dp, p1, p2, eigen1, eigen2);
                pp3 = VertexInterp(-1. / Dp, p1, p3, eigen1, eigen3);
                A3.push_back(p2);
                A3.push_back(pp2);
                A3.push_back(p3);
                pushA4A5(p2, pp2, p3);
                A3.push_back(p3);
                A3.push_back(pp2);
                A3.push_back(pp3);
                pushA4A5(p3, pp2, pp3);
                break;
            case 0x07:
                A3.push_back(p1);
                A3.push_back(p2);
                A3.push_back(p3);
                pushA4A5(p1, p2, p3);
        }
    };
    
    /* Form the vertices of the triangles for each case */
    switch (triindex) {
        case 0x00:
        case 0x0F:
            break;
        case 0x0E:
        case 0x01:
            pushA3A4A5(v0, v1, v0, v2, v0, v3); ntri++;
            break;
        case 0x0D:
        case 0x02:
            pushA3A4A5(v1, v0, v1, v3, v1, v2); ntri++;
            break;
        case 0x0C:
        case 0x03:
            pushA3A4A5(v0, v3, v0, v2, v1, v3); ntri++;
            pushA3A4A5(v1, v3, v1, v2, v0, v2); ntri++;
            break;
        case 0x0B:
        case 0x04:
            pushA3A4A5(v2, v0, v2, v1, v2, v3); ntri++;
            break;
        case 0x0A:
        case 0x05:
            pushA3A4A5(v0, v1, v2, v3, v0, v3); ntri++;
            pushA3A4A5(v0, v1, v1, v2, v2, v3); ntri++;
            break;
        case 0x09:
        case 0x06:
            pushA3A4A5(v0, v1, v1, v3, v2, v3); ntri++;
            pushA3A4A5(v0, v1, v0, v2, v2, v3); ntri++;
            break;
        case 0x07:
        case 0x08:
            pushA3A4A5(v3, v0, v3, v2, v3, v1); ntri++;
            break;
    }
    return(ntri);
}

