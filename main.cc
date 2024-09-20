/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2024
 * This code is for phase field fracture.
 **/
/**
 * This work is modified from https://github.com/tjhei/cracks and its
 * corresponding papers.
 */
/**
 * @todo
 *  Done: Interface of ABAQUS .inp file
 *  Boundary setting (a script to generate loading sequence)
 *  Parallel solver of PFM
 **/

#include "dealii_includes.h"
#include "parameters.h"
#include "phase_field_fracture.h"

int main(int argc, char *argv[]) {
  try {
    using namespace dealii;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    Parameters::AllParameters params;
    if (argc == 2)
      params.set_parameters(argv[1]);
    else
      params.set_parameters("../parameters/test.prm");

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      // prepare directories
      std::string command;
      command = "mkdir " + params.output_dir_top;
      system(command.c_str());
      command = "mkdir " + params.output_dir;
      system(command.c_str());
      command =
          "cp " + params.param_dir + " " + params.output_dir + "params.prm";
      system(command.c_str());
      command = "cp " + params.boundary_from + " " + params.output_dir +
                "boundary.txt";
      system(command.c_str());
    }

    if (params.dim == 2) {
      PhaseFieldFracture<2> pfm(params);
      pfm.run();
    } else if (params.dim == 3) {
      PhaseFieldFracture<3> pfm(params);
      pfm.run();
    }
  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
}