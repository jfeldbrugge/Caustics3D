void computeHessian (vector2d &H, const vector2d &s, const int Ni)
{
    for (int i = 0; i < Ni * Ni * Ni; i++)
    {
        H[i][0] = derivative(s[shift(i, -3, 0, Ni)][0],
                             s[shift(i, -2, 0, Ni)][0],
                             s[shift(i, -1, 0, Ni)][0],
                             s[shift(i, +1, 0, Ni)][0],
                             s[shift(i, +2, 0, Ni)][0],
                             s[shift(i, +3, 0, Ni)][0]);
        H[i][1] = derivative(s[shift(i, -3, 1, Ni)][0],
                             s[shift(i, -2, 1, Ni)][0],
                             s[shift(i, -1, 1, Ni)][0],
                             s[shift(i, +1, 1, Ni)][0],
                             s[shift(i, +2, 1, Ni)][0],
                             s[shift(i, +3, 1, Ni)][0]);
        H[i][2] = derivative(s[shift(i, -3, 2, Ni)][0],
                             s[shift(i, -2, 2, Ni)][0],
                             s[shift(i, -1, 2, Ni)][0],
                             s[shift(i, +1, 2, Ni)][0],
                             s[shift(i, +2, 2, Ni)][0],
                             s[shift(i, +3, 2, Ni)][0]);
        H[i][3] = derivative(s[shift(i, -3, 1, Ni)][1],
                             s[shift(i, -2, 1, Ni)][1],
                             s[shift(i, -1, 1, Ni)][1],
                             s[shift(i, +1, 1, Ni)][1],
                             s[shift(i, +2, 1, Ni)][1],
                             s[shift(i, +3, 1, Ni)][1]);
        H[i][4] = derivative(s[shift(i, -3, 2, Ni)][1],
                             s[shift(i, -2, 2, Ni)][1],
                             s[shift(i, -1, 2, Ni)][1],
                             s[shift(i, +1, 2, Ni)][1],
                             s[shift(i, +2, 2, Ni)][1],
                             s[shift(i, +3, 2, Ni)][1]);
        H[i][5] = derivative(s[shift(i, -3, 2, Ni)][2],
                             s[shift(i, -2, 2, Ni)][2],
                             s[shift(i, -1, 2, Ni)][2],
                             s[shift(i, +1, 2, Ni)][2],
                             s[shift(i, +2, 2, Ni)][2],
                             s[shift(i, +3, 2, Ni)][2]);
    }
}

