#include "Header.h"

int main(int argc, const char * argv[]) {
    std::cout << "Start!" << std::endl; std::cout.flush();
    
    // Quijote simulations
//    const int Ni = 512;
//    const double L = 1000.;
//    const std::string directory = "../fiducial_ZA/0/";
//    const std::vector<std::string> simulations = {"000", "001", "002", "003", "004"};
//    const std::vector<double> growingModes = {0.0078125, 0.248497, 0.328681, 0.478234, 0.606323, 0.789246};
//    const std::vector<std::string> scales = {"2", "3", "4", "6", "8", "10"};
//
//    std::string scale;
//    if (argc == 2) {
//        scale = scales[atoi(argv[1])];
//    } else {
//        scale = scales[1];
//    }
    
    // N-body Rien
//    const int Ni = 128;
//    const double L = 100.;
//    const std::string directory = "../N-body_Rien/";
//    const std::vector<std::string> simulations = {"001", "002", "003", "004", "005", "006", "007", "008", "009", "010"};
//    const std::string sim = simulations[9];
//    const double Dp = double(std::stoi(sim)) / 10.;
    
    // N-body Johan
    const int Ni = 128;
    const double L = 50.;
//    const std::string directory = "../N-body_Johan/1/";
    //    const std::string directory = "../N-body_Johan/2/";
    const std::string directory = "../N-body_Johan/3/";
    const std::vector<std::string> simulations = {"000", "001", "002", "003"};
    const std::vector<double> growingModes = {0.015625, 0.0499976, 0.248497, 0.478234, 0.789246};
    const std::vector<std::string> scales = {"4", "6", "8", "10"};
    
    std::string scale;
    if (argc == 2) {
        scale = scales[atoi(argv[1])];
    } else {
        scale = scales[1];
    }
    
    std::cout << "Directory: " << directory << " scale:" << scale << " Ni:" << Ni << std::endl << std::endl;
    
    //    Data containers
    vector2d s(Ni * Ni * Ni, std::vector<double>(3));
    vector2d a(Ni * Ni * Ni, std::vector<double>(4));
    vector2d b(Ni * Ni * Ni, std::vector<double>(4));
    vector2d c(Ni * Ni * Ni, std::vector<double>(4));
    vector2d aGrad(Ni * Ni * Ni, std::vector<double>(3));
    vector2d H(Ni * Ni * Ni, std::vector<double>(6));

    if (1) {
        //    Load displacement, eigenvalue and eigenvector fields
        std::cout << "The displacement fields and Hessian." << std::endl;
        importB(s, 3, directory + "displacement/s_IC_" + scale + ".bin", Ni);
        computeHessian(H, s, Ni);

        std::cout << "The eigenvalue and eigenvector fields." << std::endl; std::cout.flush();
        eigenFields(a, b, c, H, Ni);
        //        b.clear(); c.clear();

        outputB(a, 4, directory + "Caustics_" + scale + "/a.bin", Ni);
        outputB(b, 4, directory + "Caustics_" + scale + "/b.bin", Ni);
        outputB(c, 4, directory + "Caustics_" + scale + "/c.bin", Ni);
    }

//    importB(a, 4, directory + "Caustics_" + scale + "/a.bin", Ni);
//    importB(b, 4, directory + "Caustics_" + scale + "/b.bin", Ni);
//    importB(c, 4, directory + "Caustics_" + scale + "/c.bin", Ni);
    gradient(aGrad, a, 0, Ni);

    for (int simInt = 0; simInt < simulations.size(); simInt++) {
        const std::string sim = simulations[simInt];
        const double Dp = growingModes[simInt + 1] / growingModes[0];

        std::cout << std::endl << " sim:" << sim << " Dp:" << Dp << std::endl << std::endl;

        //     Compute caustics
        vectorXYZ A2a, A3a, A4a, A5a, A3aBound, D4ab, D4bc;
        
        std::cout << "Compute A caustics:" << std::endl; std::cout.flush();
        computeA(A2a, A3aBound, A3a, A4a, A5a, a, aGrad, Dp, Ni);
        std::cout << "Compute D caustics:" << std::endl; std::cout.flush();
        computeD(D4ab, D4bc, H, a, b, c, Dp, Ni);

        //    Give summary
        std::cout << std::endl << "The analysis resulted into: " << std::endl; std::cout.flush();
        std::cout << "A2a.size() = " << A2a.size() << std::endl;
        std::cout << "A3a.size() = " << A3a.size() << std::endl;
        std::cout << "A4a.size() = " << A4a.size() << std::endl;
        std::cout << "A5a.size() = " << A5a.size() << std::endl;
        std::cout << "D4ab.size()  = " << D4ab.size() << std::endl;
        std::cout << "D4bc.size()  = " << D4bc.size() << std::endl;
        std::cout << "A3aBound.size()  = " << A3aBound.size() << std::endl << std::endl;

        // Write Lagrangian skelet
        writePointsB(A2a, directory + "Caustics_" + scale + "/" + sim + "/A2a.bin");
        writePointsB(A3a, directory + "Caustics_" + scale + "/" + sim + "/A3a.bin");
        writePointsB(A4a, directory + "Caustics_" + scale + "/" + sim + "/A4a.bin");
        writePointsB(A5a, directory + "Caustics_" + scale + "/" + sim + "/A5a.bin");
        writePointsB(A3aBound,  directory + "Caustics_" + scale + "/" + sim + "/A3aBound.bin");
        writePointsB(D4ab, directory + "Caustics_" + scale + "/" + sim + "/D4ab.bin");

        //    Map caustics in Lagrangian space to Eulerian space with the Zel'dovich approximation
        {
            vectorXYZ A2aE, A3aE, A4aE, A5aE, A3aBoundE, D4abE;

            std::cout << "Zeldovich flow:" << std::endl; std::cout.flush();
            Eulerian(A2a, A2aE, Dp, s, Ni);
            Eulerian(A3a, A3aE, Dp, s, Ni);
            Eulerian(A3aBound,  A3aBoundE,  Dp, s, Ni);
            Eulerian(A4a, A4aE, Dp, s, Ni);
            Eulerian(A5a, A5aE, Dp, s, Ni);
            Eulerian(D4ab,  D4abE,  Dp, s, Ni);

            writePointsB(A2aE, directory + "Caustics_" + scale + "/" + sim + "/A2aE.bin");
            writePointsB(A3aE, directory + "Caustics_" + scale + "/" + sim + "/A3aE.bin");
            writePointsB(A3aBoundE, directory + "Caustics_" + scale + "/" + sim + "/A3aBoundE.bin");
            writePointsB(A4aE, directory + "Caustics_" + scale + "/" + sim + "/A4aE.bin");
            writePointsB(A5aE, directory + "Caustics_" + scale + "/" + sim + "/A5aE.bin");
            writePointsB(D4abE,  directory + "Caustics_" + scale + "/" + sim + "/D4abE.bin");
        }

        //    Map caustics in Lagrangian space to Eulerian space with the N-body simulation
        {
            vector2d sNbody(Ni * Ni * Ni, std::vector<double>(3));
            importB(sNbody, 3, directory + "displacement/s_" + sim + ".bin", Ni);

            vectorXYZ A2aNb, A3aNb, A4aNb, A5aNb, A3aBoundNb, D4abNb;

            std::cout << "Nbody flow:" << std::endl; std::cout.flush();
            Nbody(A2a, A2aNb, sNbody, Ni);
            Nbody(A3a, A3aNb, sNbody, Ni);
            Nbody(A3aBound, A3aBoundNb, sNbody, Ni);
            Nbody(A4a, A4aNb, sNbody, Ni);
            Nbody(A5a, A5aNb, sNbody, Ni);
            Nbody(D4ab,  D4abNb, sNbody, Ni);

            writePointsB(A2aNb, directory + "Caustics_" + scale + "/" + sim + "/A2aNb.bin");
            writePointsB(A3aNb, directory + "Caustics_" + scale + "/" + sim + "/A3aNb.bin");
            writePointsB(A3aBoundNb, directory + "Caustics_" + scale + "/" + sim + "/A3aBoundNb.bin");
            writePointsB(A4aNb, directory + "Caustics_" + scale + "/" + sim + "/A4aNb.bin");
            writePointsB(A5aNb, directory + "Caustics_" + scale + "/" + sim + "/A5aNb.bin");
            writePointsB(D4abNb,  directory + "Caustics_" + scale + "/" + sim + "/D4abNb.bin");
        }
    }
    
    std::cout << "Done!" << std::endl;
    return 0;
}


