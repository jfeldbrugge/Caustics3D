#include "Utilities.h"
#include <iostream>
#include <fstream>

/*! \brief translate index a number of steps
 */
int shift (const int index, const int steps, const int direction, const int Ni)
{
    int i = index % Ni;
    int j = ((index % (Ni * Ni)) - i) / Ni;
    int k = (index - i - j * Ni) / (Ni * Ni);
    
    if(direction == 0) {
        return mod(i + steps, Ni) + j * Ni + k * Ni * Ni;
    } else if(direction == 1) {
        return i + mod(j + steps, Ni) * Ni + k * Ni * Ni;
    } else {
        return i + j * Ni + mod(k + steps, Ni) * Ni * Ni;
    }
}

void pos (const int index, const int Ni)
{
    int i = index % Ni;
    int j = ((index % (Ni * Ni)) - i) / Ni;
    int k = (index - i - j * Ni) / (Ni * Ni);
    
    std::cout << "(" << i << "," << j << "," << k << ")" << std::endl;
}


// Input Output Binary Files
void importB(vector2d &data, int depth, std::string fileName, int Ni)
{
    std::ifstream fin;
    fin.open (fileName, std::ios::binary);
    if(fin.is_open()) {
        for (int i = 0; i < Ni; i++) {
            for (int j = 0; j < Ni; j++) {
                for (int k = 0; k < Ni; k++) {
                    for (int d = 0; d < depth; d++) {
                        double f;
                        fin.read((char*) &f, sizeof(double));
                        data[i + j * Ni + k * Ni * Ni][d] = f;
                    }
                }
            }
        }
        fin.close();
    } else
    {
        std::cout << "Error: could not open " << fileName << std::endl;
    }
}

void outputB(vector2d &data, int depth, std::string fileName, int Ni)
{
    std::ofstream fout;
    fout.open (fileName, std::ios::binary);
    if(fout.is_open()) {
        for (int i = 0; i < Ni; i++) {
            for (int j = 0; j < Ni; j++) {
                for (int k = 0; k < Ni; k++) {
                    for (int d = 0; d < depth; d++) {
                        double f = data[i + j * Ni + k * Ni * Ni][d];
                        fout.write((char*) &f, sizeof(double));
                    }
                }
            }
        }
        fout.close();
    } else {
        std::cout << "Error: could not open " << fileName << std::endl;
    }
}

void writePointsB(std::vector<XYZ> &points, std::string name)
{
    std::ofstream pointsFile;
    pointsFile.open(name, std::ios::binary);
    for (int index = 0; index < points.size(); index++)
    {
        double f;
        f = points[index].x;
        pointsFile.write((char*) &f, sizeof(double));
        f = points[index].y;
        pointsFile.write((char*) &f, sizeof(double));
        f = points[index].z;
        pointsFile.write((char*) &f, sizeof(double));
    }
    pointsFile.close();
}
