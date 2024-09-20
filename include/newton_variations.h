//
// Created by xlluo on 24-8-5.
//

#ifndef CRACKS_NEWTON_VARIATIONS_H
#define CRACKS_NEWTON_VARIATIONS_H

#include "controller.h"
// #include "abstract_field.h"
#include "dealii_includes.h"
// #include "elasticity.h"
#include "utils.h"
using namespace dealii;

template <int dim> class NewtonInformation {
public:
  double residual;
  double old_residual;
  double new_residual;
  double i_step;
  unsigned int adjustment_step;
  unsigned int iterative_solver_nonlinear_step;
  bool system_matrix_rebuilt;
};

template <int dim> class NewtonVariation {
public:
  NewtonVariation(Controller<dim> &ctl){};

  virtual bool allow_skip_first_iteration(NewtonInformation<dim> &info,
                                          Controller<dim> &ctl) {
    return ctl.params.skip_first_iter;
  };
  virtual bool quit_newton(NewtonInformation<dim> &info, Controller<dim> &ctl) {
    return info.residual <= ctl.params.lower_bound_newton_residual;
  };
  virtual bool quit_adjustment(NewtonInformation<dim> &info,
                               Controller<dim> &ctl) {
    // Actually no adjustment is done.
    return true;
  }
  virtual void apply_increment(LA::MPI::BlockVector &negative_increment,
                               LA::MPI::BlockVector &solution,
                               LA::MPI::BlockSparseMatrix &system_matrix,
                               LA::MPI::BlockVector &system_rhs,
                               LA::MPI::BlockVector &neumann_rhs,
                               NewtonInformation<dim> &info,
                               Controller<dim> &ctl) {
    solution -= negative_increment;
  };
  virtual bool re_solve(NewtonInformation<dim> &info, Controller<dim> &ctl) {
    return false;
  };
  virtual bool rebuild_jacobian(NewtonInformation<dim> &info,
                                Controller<dim> &ctl) {
    if (ctl.params.direct_solver) {
      return true;
    } else {
      if (info.i_step == 1 || (info.residual / info.old_residual) > 0.1) {
        return true;
      } else {
        return false;
      }
    }
  };
  virtual void prepare_next_adjustment(
      LA::MPI::BlockVector &negative_increment, LA::MPI::BlockVector &solution,
      LA::MPI::BlockSparseMatrix &system_matrix,
      LA::MPI::BlockVector &system_rhs, LA::MPI::BlockVector &neumann_rhs,
      NewtonInformation<dim> &info, Controller<dim> &ctl) {
    throw SolverControl::NoConvergence(0, 0);
  };
  virtual bool give_up(NewtonInformation<dim> &info, Controller<dim> &ctl) {
    return info.i_step == ctl.params.max_no_newton_steps - 1;
  };
};

/*
 * https://www.sciencedirect.com/science/article/pii/S2590037420300054#sec1
 * Newton-Anderson(1) from https://doi.org/10.1145/321296.321305
 */
template <int dim> class AndersonNewton : public NewtonVariation<dim> {
public:
  AndersonNewton(Controller<dim> &ctl) : NewtonVariation<dim>(ctl){};
  void apply_increment(LA::MPI::BlockVector &negative_increment,
                       LA::MPI::BlockVector &solution,
                       LA::MPI::BlockSparseMatrix &system_matrix,
                       LA::MPI::BlockVector &system_rhs,
                       LA::MPI::BlockVector &neumann_rhs,
                       NewtonInformation<dim> &info,
                       Controller<dim> &ctl) override {
    if (info.i_step == 1) {
      last_solution = solution;
      solution -= negative_increment;
    } else {
      LA::MPI::BlockVector increment_diff = last_negative_increment;
      increment_diff -= negative_increment;
      double gamma =
          negative_increment * increment_diff / increment_diff.l2_norm() * (-1);

      LA::MPI::BlockVector solution_diff = solution;
      solution_diff -= last_solution;
      last_solution = solution;

      solution_diff += increment_diff;
      solution_diff *= gamma;
      LA::MPI::BlockVector anderson_increment = negative_increment;
      anderson_increment += solution_diff;

      solution -= anderson_increment;
    }

    last_negative_increment = negative_increment;
  };

  LA::MPI::BlockVector last_solution;
  LA::MPI::BlockVector last_negative_increment;
};

template <int dim>
class KristensenModifiedNewton : public NewtonVariation<dim> {
public:
  KristensenModifiedNewton(Controller<dim> &ctl)
      : NewtonVariation<dim>(ctl), record_c(0), record_i(0), ever_built(false) {
    AssertThrow(ctl.params.linesearch_parameters != "",
                ExcInternalError("No parameters assigned to modified newton."));
    std::istringstream iss(ctl.params.modified_newton_parameters);
    iss >> n_i >> n_c;
    if (ctl.params.max_no_newton_steps < 8 * n_i) {
      ctl.params.max_no_newton_steps = 8 * n_i;
      ctl.dcout
          << "The maximum allowed newton step is much lower than settings of "
             "KristensenModifiedNewton, making it to "
          << 8 * n_i << std::endl;
    }
  }

  bool rebuild_jacobian(NewtonInformation<dim> &info,
                        Controller<dim> &ctl) override {
    if (info.i_step == 1) {
      record_i = 0;
    }
    if (!ever_built ||
        (record_c <= ctl.last_refinement_timestep_number && info.i_step) ||
        ctl.timestep_number == 0 || record_i > n_i ||
        (ctl.timestep_number - record_c) > n_c ||
        info.new_residual > info.old_residual) {
      record_i = 0;
      record_c = ctl.timestep_number;
      ever_built = true;
      ctl.debug_dcout << "Allow rebuilding system matrix" << std::endl;
      return true;
    } else {
      record_i++;
      ctl.debug_dcout << "Forbid rebuilding system matrix" << std::endl;
      return false;
    };
  };

  double n_i; // One of the subproblems fails to converge in n_i inner Newton
              // iterations.
  double n_c; // A number of load increments n_c have passed without updating
              // the stiffness matrices.
  unsigned int record_i;
  int record_c;
  bool ever_built;
};

template <int dim> class LineSearch : public NewtonVariation<dim> {
public:
  LineSearch(Controller<dim> &ctl) : NewtonVariation<dim>(ctl) {
    AssertThrow(ctl.params.linesearch_parameters != "",
                ExcInternalError("No damping factor is assigned."));
    std::istringstream iss(ctl.params.linesearch_parameters);
    iss >> damping;
  }

  bool quit_adjustment(NewtonInformation<dim> &info, Controller<dim> &ctl) {
    return info.new_residual < info.residual;
  }

  void prepare_next_adjustment(LA::MPI::BlockVector &negative_increment,
                               LA::MPI::BlockVector &solution,
                               LA::MPI::BlockSparseMatrix &system_matrix,
                               LA::MPI::BlockVector &system_rhs,
                               LA::MPI::BlockVector &neumann_rhs,
                               NewtonInformation<dim> &info,
                               Controller<dim> &ctl) override {
    if (damping >= 1) {
      throw SolverControl::NoConvergence(0, 0);
    }
    solution += negative_increment;
    negative_increment *= damping;
  };
  double damping;
};

template <int dim>
std::unique_ptr<NewtonVariation<dim>>
select_newton_variation(std::string method, Controller<dim> &ctl) {
  if (method == "none")
    return std::make_unique<NewtonVariation<dim>>(ctl);
  else if (method == "linesearch")
    return std::make_unique<LineSearch<dim>>(ctl);
  else if (method == "AndersonNewton")
    return std::make_unique<AndersonNewton<dim>>(ctl);
  else if (method == "KristensenModifiedNewton")
    return std::make_unique<KristensenModifiedNewton<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_NEWTON_VARIATIONS_H
