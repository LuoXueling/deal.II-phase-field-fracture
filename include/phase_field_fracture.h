/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef PHASE_FIELD_FRACTURE_H
#define PHASE_FIELD_FRACTURE_H

#include "abstract_field.h"
#include "abstract_multiphysics.h"
#include "controller.h"
#include "dealii_includes.h"
#include "elasticity.h"
#include "parameters.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class PhaseFieldFracture : public AbstractMultiphysics<dim> {
public:
  explicit PhaseFieldFracture(Parameters::AllParameters &prms);

private:
  void setup_system() override;
  bool refine_grid() override;
  void record_old_solution() override;
  void return_old_solution() override;
  void record_checkpoint() override;
  void return_checkpoint() override;
  double staggered_scheme() override;
  double solve_phase_field_subproblem();
  double solve_elasticity_subproblem();
  void respective_output_results(DataOut<dim> &data_out) override;

  Elasticity<dim> elasticity;
  PhaseField<dim> phasefield;
};

template <int dim>
PhaseFieldFracture<dim>::PhaseFieldFracture(Parameters::AllParameters &prms)
    : AbstractMultiphysics<dim>(prms),
      elasticity((this->ctl).params.boundary_from, "newton", this->ctl),
      phasefield(prms.phase_field_scheme, this->ctl) {}

template <int dim> void PhaseFieldFracture<dim>::setup_system() {
  this->ctl.debug_dcout << "Initialize system - elasticity" << std::endl;
  elasticity.setup_system(this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    this->ctl.debug_dcout << "Initialize system - phase field" << std::endl;
    phasefield.setup_system(this->ctl);
  }
}

template <int dim> void PhaseFieldFracture<dim>::record_old_solution() {
  (this->ctl).record_point_history((this->ctl).quadrature_point_history,
                                   (this->ctl).old_quadrature_point_history);
  elasticity.record_old_solution(this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    phasefield.record_old_solution(this->ctl);
  }
}

template <int dim> void PhaseFieldFracture<dim>::return_old_solution() {
  (this->ctl).record_point_history((this->ctl).old_quadrature_point_history,
                                   (this->ctl).quadrature_point_history);
  elasticity.return_old_solution(this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    phasefield.return_old_solution(this->ctl);
  }
}

template <int dim> void PhaseFieldFracture<dim>::record_checkpoint() {
  (this->ctl).record_point_history(
      (this->ctl).quadrature_point_history,
      (this->ctl).quadrature_point_history_checkpoint);
  elasticity.record_checkpoint(this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    phasefield.record_checkpoint(this->ctl);
  }
}

template <int dim> void PhaseFieldFracture<dim>::return_checkpoint() {
  (this->ctl).record_point_history(
      (this->ctl).quadrature_point_history_checkpoint,
      (this->ctl).quadrature_point_history);
  elasticity.return_checkpoint(this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    phasefield.return_checkpoint(this->ctl);
  }
}

template <int dim>
double PhaseFieldFracture<dim>::solve_phase_field_subproblem() {
  (this->ctl).dcout << "Staggered scheme - Solving phase field" << std::endl;
  (this->ctl).computing_timer.enter_subsection("Solve phase field");
  double newton_reduction_phasefield = phasefield.update(this->ctl);
  //  (this->ctl).debug_dcout
  //      << "Staggered scheme - Solving phase field - phase field limitation"
  //      << std::endl;
  //  phasefield.enforce_phase_field_limitation(this->ctl);
  (this->ctl).computing_timer.leave_subsection("Solve phase field");
  return newton_reduction_phasefield;
}

template <int dim>
double PhaseFieldFracture<dim>::solve_elasticity_subproblem() {
  (this->ctl).dcout
      << "Solve Newton system - staggered scheme - Solving elasticity"
      << std::endl;
  (this->ctl).computing_timer.enter_subsection("Solve elasticity");
  double newton_reduction_elasticity = elasticity.update(this->ctl);
  (this->ctl).computing_timer.leave_subsection("Solve elasticity");
  return newton_reduction_elasticity;
}

template <int dim> double PhaseFieldFracture<dim>::staggered_scheme() {
  if ((this->ctl).params.enable_phase_field) {
    double newton_reduction_elasticity = 0, newton_reduction_phasefield = 0;
    double phasefield_residual;
    double last_residual = 1e8, last_last_residual = 1e9;
    bool residual_decreased = false;
    for (unsigned int cnt = 0; cnt < (this->ctl).params.max_multipass; ++cnt) {
      newton_reduction_phasefield = solve_phase_field_subproblem();
      newton_reduction_elasticity = solve_elasticity_subproblem();
      phasefield.update_newton_residual(this->ctl);
      if ((this->ctl).params.multipass_staggered) {
        phasefield_residual =
            get_norm(phasefield.system_rhs, (this->ctl).params.norm_type);
        (this->ctl).dcout << "Phase field residual: " << phasefield_residual
                          << std::endl;
        if (cnt >= 2 && phasefield_residual < last_residual &&
            last_residual < last_last_residual) {
          residual_decreased = true;
        }
        if (phasefield_residual < (this->ctl.params).multipass_residual_tol)
          break;
        if (phasefield.newton_info.i_step == 1 &&
            elasticity.newton_info.i_step == 1) {
          break;
        }
        if (last_residual > last_last_residual &&
            phasefield_residual > last_residual) {
          if ((this->ctl).params.throw_if_multipass_increase &&
              !residual_decreased) {
            throw SolverControl::NoConvergence(0, 0);
          } else if ((this->ctl).params.quit_multipass_if_increase) {
            break;
          }
        }
        last_last_residual = last_residual;
        last_residual = phasefield_residual;
      } else {
        break;
      }
    }
    return std::max(newton_reduction_elasticity, newton_reduction_phasefield);
  } else {
    return solve_elasticity_subproblem();
  }
}

template <int dim>
void PhaseFieldFracture<dim>::respective_output_results(
    DataOut<dim> &data_out) {
  (this->ctl).dcout << "Computing output - elasticity" << std::endl;
  elasticity.output_results(data_out, this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    (this->ctl).dcout << "Computing output - phase field" << std::endl;
    phasefield.output_results(data_out, this->ctl);
  }
}

template <int dim> bool PhaseFieldFracture<dim>::refine_grid() {
  typename DoFHandler<dim>::active_cell_iterator
      cell = phasefield.dof_handler.begin_active(),
      endc = phasefield.dof_handler.end();

  FEValues<dim> fe_values(phasefield.fe, (this->ctl).quadrature_formula,
                          update_gradients);

  unsigned int n_q_points = (this->ctl).quadrature_formula.size();
  std::vector<Tensor<1, dim>> phasefield_grads(n_q_points);

  // Define refinement criterion and mark cells to refine
  unsigned int will_refine = 0;
  double a1 = (this->ctl).params.refine_influence_initial;
  double a2 = (this->ctl).params.refine_influence_final;
  double phi_ref = std::exp(-a2) / std::exp(-a1);
  for (; cell != endc; ++cell) {
    if (cell->is_locally_owned()) {
      if (cell->diameter() < (this->ctl).params.l_phi *
                                 (this->ctl).params.refine_minimum_size_ratio) {
        cell->clear_refine_flag();
        continue;
      }
      fe_values.reinit(cell);
      fe_values[phasefield.fields.extractors_scalar["phasefield"]]
          .get_function_gradients((phasefield.solution), phasefield_grads);
      double max_grad = 0;
      for (unsigned int q = 0; q < n_q_points; ++q) {
        double prod = std::sqrt(phasefield_grads[q] * phasefield_grads[q]);
        max_grad = std::max(max_grad, prod);
      }
      if (max_grad > 1 / (this->ctl).params.l_phi * phi_ref * exp(-a1)) {
        cell->set_refine_flag();
        will_refine = 1;
      }
    }
  }
  (this->ctl).debug_dcout << "Refine - finish marking" << std::endl;
  double will_refine_global =
      Utilities::MPI::sum(will_refine, (this->ctl).mpi_com);
  if (!static_cast<bool>(will_refine_global)) {
    (this->ctl).dcout << "No cell to refine" << std::endl;
    return false;
  } else {
    (this->ctl).debug_dcout << "Refine - prepare" << std::endl;
    // Prepare transferring of point history
    parallel::distributed::ContinuousQuadratureDataTransfer<dim, PointHistory>
        point_history_transfer(
            FE_Q<dim>(QGaussLobatto<1>((this->ctl).params.poly_degree+1)),
            QGauss<dim>((this->ctl).params.poly_degree + 1),
            QGauss<dim>((this->ctl).params.poly_degree + 1));
    point_history_transfer.prepare_for_coarsening_and_refinement(
        (this->ctl).triangulation, (this->ctl).quadrature_point_history);

    // Prepare transferring of fields
    parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector>
        soltrans_elasticity = elasticity.prepare_refine();
    parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector>
        soltrans_phasefield = phasefield.prepare_refine();

    (this->ctl).debug_dcout << "Refine - start refinement" << std::endl;
    // Execute refinement
    (this->ctl).triangulation.execute_coarsening_and_refinement();
    setup_system();

    (this->ctl).debug_dcout << "Refine - after refinement - point history"
                            << std::endl;
    // Finalize transferring of point history
    (this->ctl).initialize_point_history();
    point_history_transfer.interpolate();
    (this->ctl).debug_dcout << "Refine - after refinement - transfer fields"
                            << std::endl;
    // Finalize transferring of fields
    elasticity.post_refine(soltrans_elasticity, this->ctl);
    phasefield.post_refine(soltrans_phasefield, this->ctl);
    (this->ctl).debug_dcout << "Refine - done" << std::endl;
    return true;
  }
}

#endif