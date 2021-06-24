#pragma once
#include "Types.h"
#include <string>

// Modulo functions
/*! \brief Computes modulo correctly for negative numbers
 *  downto -2N
 */
inline int mod(const int a, const int N)
{
    return (a + 2 * N) % N;
}

/*! \brief Computes modulo correctly for negative numbers
 *  between -N  and  +2N (floating point version)
 */
inline double mod(const double a, const int Ni)
{
    if (a < 0.) {
        return a + Ni;
    } else if (a > Ni) {
        return a - Ni;
    } else {
        return a;
    }
}

/*! \brief compute linear index into 3d array, first argument is fastest
 *  moving coordinate.
 */
inline int indexOf (const int i, const int j, const int k, const int Ni)
{
    return mod(i, Ni) + mod(j, Ni) * Ni + mod(k, Ni) * Ni * Ni;
}

extern int shift (const int index, const int steps, const int direction, const int Ni);
extern void pos (const int index, const int Ni);
extern void importB(vector2d &data, int depth, std::string fileName, int Ni);
extern void outputB(vector2d &data, int depth, std::string fileName, int Ni);
extern void writePointsB(std::vector<XYZ> &points, std::string name);

