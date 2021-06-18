// Periodic finite difference method
void gradient(vector2d &gradient, vector2d &data, int depth, int Ni)
{
    for (int i = 0; i < Ni * Ni * Ni; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            gradient[i][j] = derivative(data[shift(i, -3, j, Ni)][depth],
                                        data[shift(i, -2, j, Ni)][depth],
                                        data[shift(i, -1, j, Ni)][depth],
                                        data[shift(i, +1, j, Ni)][depth],
                                        data[shift(i, +2, j, Ni)][depth],
                                        data[shift(i, +3, j, Ni)][depth]);
        }
    }
}

void gradient(vector2d &gradient, vector1d &data, int depth, int Ni)
{
    for (int i = 0; i < Ni * Ni * Ni; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            gradient[i][j] = derivative(data[shift(i, -3, j, Ni)],
                                        data[shift(i, -2, j, Ni)],
                                        data[shift(i, -1, j, Ni)],
                                        data[shift(i, +1, j, Ni)],
                                        data[shift(i, +2, j, Ni)],
                                        data[shift(i, +3, j, Ni)]);
        }
    }
}


// Finite difference order dx^2
double derivative(double dm1, double d1)
{
    double m1 = ( d1 - dm1 ) / 2.;
    
    return m1;
}

// Finite difference order dx^4
double derivative(double dm2, double dm1, double d1, double d2)
{
    double m1 = ( d1 - dm1 ) / 2.;
    double m2 = ( d2 - dm2 ) / 4.;
    
    return 4./3. * m1 -1./3. * m2;
}

// Finite difference order dx^6
double derivative(double dm3, double dm2, double dm1, double d1, double d2, double d3)
{
    double m1 = ( d1 - dm1 ) / 2.;
    double m2 = ( d2 - dm2 ) / 4.;
    double m3 = ( d3 - dm3 ) / 6.;
    
    return 3./2. * m1 -3./5. * m2 + 1./10. * m3;
}

