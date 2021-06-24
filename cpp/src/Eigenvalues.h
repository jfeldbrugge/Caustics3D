void eigenFields(vector2d &a, vector2d &b, vector2d &c, vector2d &H, int Ni)
{
    for (int index = 0; index < Ni * Ni * Ni; index++)
    {
        Eigen::Matrix3d D;
        D << H[index][0], H[index][1], H[index][2], H[index][1], H[index][3], H[index][4], H[index][2], H[index][4], H[index][5];
       
        double aa, bb, cc;
        Eigen::Vector3d aav, bbv, ccv;

        eigen(D, aa, bb, cc, aav, bbv, ccv);

        double aNorm = sqrt(aav(0) * aav(0) + aav(1) * aav(1) + aav(2) * aav(2));
        double bNorm = sqrt(bbv(0) * bbv(0) + bbv(1) * bbv(1) + bbv(2) * bbv(2));
        double cNorm = sqrt(ccv(0) * ccv(0) + ccv(1) * ccv(1) + ccv(2) * ccv(2));
        
        a[index][0] = aa;
        a[index][1] = aav(0) / aNorm;
        a[index][2] = aav(1) / aNorm;
        a[index][3] = aav(2) / aNorm;
        b[index][0] = bb;
        b[index][1] = bbv(0) / bNorm;
        b[index][2] = bbv(1) / bNorm;
        b[index][3] = bbv(2) / bNorm;
        c[index][0] = cc;
        c[index][1] = ccv(0) / cNorm;
        c[index][2] = ccv(1) / cNorm;
        c[index][3] = ccv(2) / cNorm;
    }
}


void eigen(Eigen::Matrix3d D, double &aa, double &bb, double &cc, Eigen::Vector3d &aav, Eigen::Vector3d &bbv, Eigen::Vector3d &ccv)
{
    Eigen::EigenSolver<Eigen::MatrixXd> es(D);

    double l1 = es.eigenvalues()[0].real();
    double l2 = es.eigenvalues()[1].real();
    double l3 = es.eigenvalues()[2].real();

    Eigen::Vector3d lv1 = es.eigenvectors().col(0).real();
    Eigen::Vector3d lv2 = es.eigenvectors().col(1).real();
    Eigen::Vector3d lv3 = es.eigenvectors().col(2).real();

    if(l1 < l2 && l2 < l3)
    {
        aa = l1;   bb = l2;   cc = l3;
        aav = lv1; bbv = lv2; ccv = lv3;
    } else if(l1 < l3 && l3 < l2)
    {
        aa = l1;   bb = l3;   cc = l2;
        aav = lv1; bbv = lv3; ccv = lv2;
    } else if(l2 < l1 && l1 < l3)
    {
        aa = l2;   bb = l1;   cc = l3;
        aav = lv2; bbv = lv1; ccv = lv3;
    } else if(l2 < l3 && l3 < l1)
    {
        aa = l2;   bb = l3;   cc = l1;
        aav = lv2; bbv = lv3; ccv = lv1;
    } else if(l3 < l1 && l1 < l2)
    {
        aa = l3;   bb = l1;   cc = l2;
        aav = lv3; bbv = lv1; ccv = lv2;
    } else if(l3 < l2 && l2 < l1)
    {
        aa = l3;   bb = l2;   cc = l1;
        aav = lv3; bbv = lv2; ccv = lv1;
    }
}


