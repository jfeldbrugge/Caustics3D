void computeD(vectorXYZ &D4ab, vectorXYZ &D4bc, const vector2d &H, const vector2d &a, const vector2d &b, const vector2d &c, const double Dp, const int Ni)
{
    GRIDCELL g;
    for (int index = 0; index < Ni * Ni * Ni; index++)
    {
        int i = index % Ni;
        int j = ((index % (Ni * Ni)) - i) / Ni;
        int k = (index - i - j * Ni) / (Ni * Ni);
        
        setupPosition(g, i, j, k);
     
        vector1i indices;
        offsets(index, indices, Ni);
        
        double m12 = H[indices[0]][1];
        double m13 = H[indices[0]][3];
        double m23 = H[indices[0]][5];
        
        for (int i = 0; i < 8; i++)
        {
            if (abs(m12) > fmax(abs(m13), abs(m23)))
            {
                g.val1[i] = cond1(H[indices[i]]);
                g.val2[i] = cond2(H[indices[i]]);
            } else if (abs(m13) > fmax(abs(m12), abs(m23))) {
                g.val1[i] = cond2(H[indices[i]]);
                g.val2[i] = cond3(H[indices[i]]);
            } else if (abs(m23) > fmax(abs(m12), abs(m13))) {
                g.val1[i] = cond1(H[indices[i]]);
                g.val2[i] = cond3(H[indices[i]]);
            }
            
            g.check1[i] = H[indices[i]][1];
            g.check2[i] = H[indices[i]][2];
            g.check3[i] = H[indices[i]][4];
            
            g.eigen1[i] = a[indices[i]][0];
            g.eigen2[i] = b[indices[i]][0];
            g.eigen3[i] = c[indices[i]][0];
        }
    
        PolygoniseD(g, D4ab, D4bc, 0, 2, 3, 7, Dp, Ni);
        PolygoniseD(g, D4ab, D4bc, 0, 2, 6, 7, Dp, Ni);
        PolygoniseD(g, D4ab, D4bc, 0, 4, 6, 7, Dp, Ni);
        PolygoniseD(g, D4ab, D4bc, 0, 6, 1, 2, Dp, Ni);
        PolygoniseD(g, D4ab, D4bc, 0, 6, 1, 4, Dp, Ni);
        PolygoniseD(g, D4ab, D4bc, 5, 6, 1, 4, Dp, Ni);
    }
}

int PolygoniseD(GRIDCELL g, std::vector<XYZ> &D4ab, std::vector<XYZ> &D4bc,
                int v0, int v1, int v2, int v3, const double Dp, const int Ni)
{
    std::function<void (XYZ, XYZ)>
    pushD4 = [&] (const XYZ &pp1, const XYZ &pp2)
    {
        double check1_1 = interpolationX(pp1, g.p[0], g.check1);
        double check1_2 = interpolationX(pp2, g.p[0], g.check1);
        double check2_1 = interpolationX(pp1, g.p[0], g.check2);
        double check2_2 = interpolationX(pp2, g.p[0], g.check2);
        double check3_1 = interpolationX(pp1, g.p[0], g.check3);
        double check3_2 = interpolationX(pp2, g.p[0], g.check3);
        
        double eigen1_1 = interpolationX(pp1, g.p[0], g.eigen1);
        double eigen1_2 = interpolationX(pp2, g.p[0], g.eigen1);
        double eigen2_1 = interpolationX(pp1, g.p[0], g.eigen2);
        double eigen2_2 = interpolationX(pp2, g.p[0], g.eigen2);
        double eigen3_1 = interpolationX(pp1, g.p[0], g.eigen3);
        double eigen3_2 = interpolationX(pp2, g.p[0], g.eigen3);

        int triindex1 = 0;
        if (check1_1 < 0) triindex1 |= 1;
        if (check1_2 < 0) triindex1 |= 2;

        int triindex2 = 0;
        if (check2_1 < 0) triindex2 |= 1;
        if (check2_2 < 0) triindex2 |= 2;

        int triindex3 = 0;
        if (check3_1 < 0) triindex3 |= 1;
        if (check3_2 < 0) triindex3 |= 2;

        if(
           (triindex1 == 0 || triindex1 == 3) &&
           (triindex2 == 0 || triindex2 == 3) &&
           (triindex3 == 0 || triindex3 == 3))
        {
            const double tres = 0.5;
//            const double tres = 1.;
            if(abs(eigen1_1 - eigen2_1) < tres &&  abs(eigen1_2 - eigen2_2) < tres && eigen2_1 < -1. / Dp && eigen2_2 < -1. / Dp)
            {
                D4ab.push_back(pp1);
                D4ab.push_back(pp2);
            } else if (abs(eigen2_1 - eigen3_1) < tres &&  abs(eigen2_2 - eigen3_2) < tres && eigen3_1 < -1. / Dp && eigen3_2 < -1. / Dp)
            {
                D4bc.push_back(pp1);
                D4bc.push_back(pp2);
            }
        }
    };
    
    std::function<void (int, int, int, int, int, int)>
    push = [&] (int v1, int w1, int v2, int w2, int v3, int w3)
    {
        XYZ p1 = VertexInterp(0., g.p[v1], g.p[w1], g.val1[v1], g.val1[w1]);
        XYZ p2 = VertexInterp(0., g.p[v2], g.p[w2], g.val1[v2], g.val1[w2]);
        XYZ p3 = VertexInterp(0., g.p[v3], g.p[w3], g.val1[v3], g.val1[w3]);
        
        double val2_1 = interpolationX(p1, g.p[0], g.val2);
        double val2_2 = interpolationX(p2, g.p[0], g.val2);
        double val2_3 = interpolationX(p3, g.p[0], g.val2);
        
        XYZ pp1, pp2;
        
        int triindex = 0;
        if (val2_1 < 0.) triindex |= 1;
        if (val2_2 < 0.) triindex |= 2;
        if (val2_3 < 0.) triindex |= 4;
        
        switch(triindex) {
            case 0x00:
            case 0x07:
                break;
            case 0x01:
            case 0x06:
                pp1 = VertexInterp(0., p1, p2, val2_1, val2_2);
                pp2 = VertexInterp(0., p1, p3, val2_1, val2_3);
                
                pushD4(pp1, pp2);
                break;
            case 0x02:
            case 0x05:
                pp1 = VertexInterp(0., p1, p2, val2_1, val2_2);
                pp2 = VertexInterp(0., p2, p3, val2_2, val2_3);
                
                pushD4(pp1, pp2);
                break;
            case 0x03:
            case 0x04:
                pp1 = VertexInterp(0., p1, p3, val2_1, val2_3);
                pp2 = VertexInterp(0., p2, p3, val2_2, val2_3);
                
                pushD4(pp1, pp2);
                break;
        }
    };
    
    /*
     Determine which of the 16 cases we have given which vertices
     are above or below the isosurface
     */
    
    int ntri = 0;
    int triindex = 0;
    
    if (g.val1[v0] < 0) triindex |= 1;
    if (g.val1[v1] < 0) triindex |= 2;
    if (g.val1[v2] < 0) triindex |= 4;
    if (g.val1[v3] < 0) triindex |= 8;
    
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

double cond1(const vector1d &H)
{
    return H[0] * H[1] * H[4] + H[2] * H[4] * H[4] - H[1] * H[1] * H[2] - H[1] * H[4] * H[5];
}

double cond2(const vector1d &H)
{
    return H[1] * H[2] * H[3] + H[2] * H[2] * H[4] - H[1] * H[1] * H[4] - H[1] * H[2] * H[5];
}

double cond3(const vector1d &H)
{
    return H[3] * H[4] * H[2] + H[1] * H[2] * H[2] - H[1] * H[4] * H[4] - H[0] * H[2] * H[4];
}
