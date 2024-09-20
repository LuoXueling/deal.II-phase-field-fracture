//
// Created by xlluo on 24-7-31.
//

#ifndef CRACKS_ABSTRACT_MULTIPHYSICS_H
#define CRACKS_ABSTRACT_MULTIPHYSICS_H

#include "abstract_field.h"
#include "adaptive_timestep.h"
#include "controller.h"
#include "dealii_includes.h"
#include "elasticity.h"
#include "parameters.h"
#include "phase_field.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class AbstractMultiphysics {
public:
  explicit AbstractMultiphysics(Parameters::AllParameters &prms);

  void run();
  Controller<dim> ctl;

private:
  virtual void setup_system() { AssertThrow(false, ExcNotImplemented()); };
  virtual bool refine_grid() { AssertThrow(false, ExcNotImplemented()); };
  virtual void record_old_solution() {
    AssertThrow(false, ExcNotImplemented());
  };
  virtual void return_old_solution() {
    AssertThrow(false, ExcNotImplemented());
  };
  virtual void record_checkpoint() { AssertThrow(false, ExcNotImplemented()); };
  virtual void return_checkpoint() { AssertThrow(false, ExcNotImplemented()); };
  virtual double staggered_scheme() {
    AssertThrow(false, ExcNotImplemented());
  };
  virtual void respective_output_results(DataOut<dim> &data_out) {
    AssertThrow(false, ExcNotImplemented());
  };

  void setup_mesh();
  void output_results();
};

template <int dim>
AbstractMultiphysics<dim>::AbstractMultiphysics(Parameters::AllParameters &prms)
    : ctl(prms) {}

template <int dim> void AbstractMultiphysics<dim>::run() {
  ctl.dcout << "Project: " << ctl.params.project_name << std::endl;
  ctl.dcout << "Mesh from: " << ctl.params.mesh_from << std::endl;
  ctl.dcout << "Load sequence from: " << ctl.params.load_sequence_from
            << std::endl;
  ctl.dcout << "Output directory: " << ctl.params.output_dir << std::endl;
  ctl.dcout << "Solving " << ctl.params.dim << " dimensional PFM problem"
            << std::endl;
  ctl.dcout << "Running on " << Utilities::MPI::n_mpi_processes(ctl.mpi_com)
            << " MPI rank(s)" << std::endl;
  ctl.dcout << "Number of threads " << MultithreadInfo::n_threads()
            << std::endl;
  ctl.dcout << "Number of cores " << MultithreadInfo::n_cores() << std::endl;

  ctl.dcout << "Set mesh" << std::endl;
  ctl.timer.enter_subsection("Set mesh");
  setup_mesh();
  ctl.timer.leave_subsection("Set mesh");

  ctl.debug_dcout << "Initialize system" << std::endl;
  ctl.timer.enter_subsection("Initialize system");
  setup_system();
  ctl.initialize_point_history();
  ctl.timer.leave_subsection("Initialize system");

  //  if (ctl.params.enable_phase_field) {
  //    enforce_phase_field_limitation();
  //  }

  ctl.dcout << "Solve Newton system" << std::endl;
  ctl.timer.enter_subsection("Solve Newton system");
  unsigned int refinement_cycle = 0;
  double finishing_timestep_loop = 0;
  double tmp_timestep = 0.0;

  std::unique_ptr<AdaptiveTimeStep<dim>> time_stepping =
      select_adaptive_timestep<dim>(ctl.params.adaptive_timestep, ctl);

  time_stepping->initialize_timestep(ctl);

  ctl.current_timestep = ctl.params.timestep;
  // Initialize old and old_old timestep sizes
  ctl.old_timestep = ctl.current_timestep;

  do {
    double newton_reduction = 1.0;
    if (ctl.timestep_number > ctl.params.switch_timestep)
      ctl.current_timestep = ctl.params.timestep_size_2;

    double tmp_current_timestep = ctl.current_timestep;
    ctl.old_timestep = ctl.current_timestep;

    ctl.dcout << "\n=============================="
              << "===========================================" << std::endl;

    double current_timestep = time_stepping->get_timestep(ctl);
    ctl.time += current_timestep;

    ctl.dcout << "Time (No." << ctl.timestep_number << "): " << ctl.time
              << " (Step: " << current_timestep << ") "
              << "Cells: " << ctl.triangulation.n_global_active_cells();
    ctl.dcout << "\n--------------------------------"
              << "-----------------------------------------" << std::endl;
    ctl.dcout << std::endl;

    try {
      do {
        // The Newton method can either stagnate or the linear solver
        // might not converge. To not abort the program we catch the
        // exception and retry with a smaller step.
        ctl.debug_dcout << "Solve Newton system - enter loop" << std::endl;
        record_old_solution();
        try {
          ctl.debug_dcout << "Solve Newton system - staggered scheme"
                          << std::endl;
          newton_reduction = staggered_scheme();
          while (time_stepping->fail(newton_reduction, ctl)) {
            time_stepping->execute_when_fail(ctl);
            std::string solution_or_checkpoint =
                time_stepping->return_solution_or_checkpoint(ctl);
            if (solution_or_checkpoint == "solution") {
              ctl.dcout << "Returning previous solution" << std::endl;
              return_old_solution();
            } else if (solution_or_checkpoint == "checkpoint") {
              ctl.dcout << "Returning the last checkpoint" << std::endl;
              return_checkpoint();
            }
            newton_reduction = staggered_scheme();
          }

          break;

        } catch (SolverControl::NoConvergence &e) {
          ctl.dcout << "Solver did not converge! Adjusting time step."
                    << std::endl;
          time_stepping->fail(1e8, ctl);
          time_stepping->execute_when_fail(ctl);
          std::string solution_or_checkpoint =
              time_stepping->return_solution_or_checkpoint(ctl);
          if (solution_or_checkpoint == "solution") {
            ctl.dcout << "Returning previous solution" << std::endl;
            return_old_solution();
          } else if (solution_or_checkpoint == "checkpoint") {
            ctl.dcout << "Returning the last checkpoint" << std::endl;
            return_checkpoint();
          }
        }
      } while (true);
    } catch (const std::runtime_error &e) {
      ctl.dcout << "Failed to solve: " << e.what() << std::endl;
      break;
    }
    time_stepping->after_step(ctl);
    ctl.finalize_point_history();
    // Recover time step
    ctl.current_timestep = tmp_current_timestep;
    ctl.timer.leave_subsection("Solve Newton system");
    // Refine mesh.
    if (ctl.params.refine) {
      ctl.dcout << "Refining mesh" << std::endl;
      ctl.timer.enter_subsection("Refine grid");
      ctl.computing_timer.enter_subsection("Refine grid");
      bool refined = refine_grid();
      if (refined)
        ctl.last_refinement_timestep_number = ctl.timestep_number;
      ctl.timer.leave_subsection();
      ctl.computing_timer.leave_subsection("Refine grid");
    }
    // Whether to save checkpoints
    if (time_stepping->save_checkpoint(ctl)) {
      ctl.dcout << "Saving checkpoint" << std::endl;
      record_checkpoint();
    }
    ++ctl.output_timestep_number;
    if (ctl.timestep_number == 0 ||
        (ctl.timestep_number + 1) % ctl.params.save_vtk_per_step == 0 ||
        time_stepping->save_results) {
      time_stepping->save_results = false;
      ctl.timer.enter_subsection("Calculate outputs");
      ctl.dcout << "Computing output (will be saved to No."
                << ctl.output_timestep_number << ")" << std::endl;
      ctl.computing_timer.enter_subsection("Calculate outputs");
      output_results();
      ctl.computing_timer.leave_subsection("Calculate outputs");
      ctl.timer.leave_subsection("Calculate outputs");
    }
    ctl.timer.enter_subsection("Solve Newton system");
    ++ctl.timestep_number;

    ctl.computing_timer.print_summary();
    ctl.computing_timer.reset();
    ctl.dcout << std::endl;
  } while (ctl.timestep_number <= ctl.params.max_no_timesteps &&
           !time_stepping->terminate(ctl));
  ctl.timer.leave_subsection("Solve Newton system");
  ctl.timer.print_summary();
}

template <int dim> void AbstractMultiphysics<dim>::setup_mesh() {
  GridIn<dim> grid_in;
  /**
   * similar to normal use of GridIn.
   */
  grid_in.attach_triangulation(ctl.triangulation);
  if (!checkFileExsit(ctl.params.mesh_from)) {
    throw std::runtime_error("Mesh file does not exist");
  }
  std::filebuf fb;
  if (fb.open(ctl.params.mesh_from, std::ios::in)) {
    std::istream is(&fb);
    grid_in.read_abaqus(is);
    fb.close();
  }
  //  GridGenerator::hyper_cube(ctl.triangulation);
  //  ctl.triangulation.refine_global(5);

  if (dim == 2) {
    std::ofstream out(ctl.params.output_dir + "initial_grid.svg");
    GridOut grid_out;
    grid_out.write_svg(ctl.triangulation, out);
  }

  std::vector<int> boundary_ids;
  std::tuple<std::vector<Point<dim>>, std::vector<CellData<dim>>, SubCellData>
      info;
  info = GridTools::get_coarse_mesh_description(ctl.triangulation);
  ctl.debug_dcout << "Searching boundaries" << std::endl;
  if (dim == 2) {
    for (const CellData<1> i : std::get<2>(info).boundary_lines) {
      int id = i.boundary_id;
      if (id == 0 || id == -1)
        continue;
      if (std::find(boundary_ids.begin(), boundary_ids.end(), id) ==
              boundary_ids.end() &&
          id != -1 && id != 0) {
        ctl.debug_dcout << "Find id" + std::to_string(id) << std::endl;
        boundary_ids.push_back(id);
      }
    }
  } else {
    for (const CellData<2> i : std::get<2>(info).boundary_quads) {
      int id = i.boundary_id;
      if (id == 0 || id == -1)
        continue;
      if (std::find(boundary_ids.begin(), boundary_ids.end(), id) ==
          boundary_ids.end()) {
        ctl.debug_dcout << "Find id" + std::to_string(id) << std::endl;
        boundary_ids.push_back(id);
      }
    }
  }
  ctl.boundary_ids = boundary_ids;
  ctl.dcout << "Find " << ctl.triangulation.n_global_active_cells()
            << " elements" << std::endl;
}

template <int dim> void AbstractMultiphysics<dim>::output_results() {
  DataOut<dim> data_out;
  data_out.attach_triangulation(ctl.triangulation);

  Vector<float> subdomain(ctl.triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = ctl.triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  // Record statistics
  ctl.statistics.add_value("Step", ctl.timestep_number);
  ctl.statistics.set_precision("Step", 1);
  ctl.statistics.set_scientific("Step", false);
  ctl.statistics.add_value("Step-Out", ctl.output_timestep_number);
  ctl.statistics.set_precision("Step-Out", 1);
  ctl.statistics.set_scientific("Step-Out", false);
  ctl.statistics.add_value("Time", ctl.time);
  ctl.statistics.set_precision("Time", 8);
  ctl.statistics.set_scientific("Time", true);

  respective_output_results(data_out);

  ctl.debug_dcout << "Computing output - build patches" << std::endl;
  data_out.build_patches();
  ctl.debug_dcout << "Computing output - writing" << std::endl;
  data_out.write_vtu_with_pvtu_record(ctl.params.output_dir, "solution",
                                      ctl.output_timestep_number, ctl.mpi_com,
                                      2, 8);

  ctl.debug_dcout << "Computing output - report statistics" << std::endl;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::ofstream stat_file(
        (ctl.params.output_dir + "/log-results.txt").c_str());
    ctl.statistics.write_text(stat_file);
    stat_file.close();
  }
  ctl.debug_dcout << "Computing output - done" << std::endl;
}

#endif // CRACKS_ABSTRACT_MULTIPHYSICS_H
