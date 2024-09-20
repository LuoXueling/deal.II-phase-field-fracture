//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_ABSTRACT_FIELD_H
#define CRACKS_ABSTRACT_FIELD_H

#include "boundary.h"
#include "controller.h"
#include "dealii_includes.h"
#include "multi_field.h"
#include "newton_variations.h"
#include <typeinfo>

template <int dim> class AbstractField {
public:
  AbstractField(std::vector<unsigned int> n_components,
                std::vector<std::string> names,
                std::vector<std::string> boundary_from,
                std::string update_scheme, Controller<dim> &ctl);

  virtual void assemble_newton_system(bool residual_only,
                                      LA::MPI::BlockVector &neumann_rhs,
                                      Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void assemble_linear_system(Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual unsigned int solve(NewtonInformation<dim> &info,
                             Controller<dim> &ctl);
  virtual unsigned int solve_linear_system(
      NewtonInformation<dim> &info, Controller<dim> &ctl,
      SolverControl &solver_control,
      BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG> &preconditioner);
  unsigned int solve(Controller<dim> &ctl) {
    NewtonInformation<dim> dummy_info;
    dummy_info.system_matrix_rebuilt =
        true; // Refactorization at every timestep for now.
    return solve(dummy_info, ctl);
  };
  virtual void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented());
  };
  virtual void setup_dirichlet_boundary_condition(Controller<dim> &ctl);
  virtual void
  setup_neumann_boundary_condition(LA::MPI::BlockVector &neumann_rhs,
                                   Controller<dim> &ctl);
  virtual void setup_system(Controller<dim> &ctl);
  virtual void record_old_solution(Controller<dim> &ctl);
  virtual void return_old_solution(Controller<dim> &ctl);
  virtual void record_checkpoint(Controller<dim> &ctl);
  virtual void return_checkpoint(Controller<dim> &ctl);
  virtual void distribute_hanging_node_constraints(LA::MPI::BlockVector &vector,
                                                   Controller<dim> &ctl);
  virtual void distribute_all_constraints(LA::MPI::BlockVector &vector,
                                          Controller<dim> &ctl);

  virtual double update(Controller<dim> &ctl);
  virtual double update_linear_system(Controller<dim> &ctl);
  virtual double update_newton_system(Controller<dim> &ctl);
  virtual void update_newton_residual(Controller<dim> &ctl);
  parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector>
  prepare_refine();
  void
  post_refine(parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector>
                  &soltrans,
              Controller<dim> &ctl);
  bool dof_is_this_field(unsigned int i_dof, std::string name);
  unsigned int block_id(std::string name) {
    return fields.components_to_blocks[fields.component_start_indices[name]];
  }
  /*
   * Solver
   */
  std::string update_scheme_timestep;
  std::vector<std::vector<std::vector<bool>>> fields_constant_modes;

  /*
   * FE system, constraints, and dof handler
   */
  MultiFieldCfg<dim> fields;
  FESystem<dim> fe;
  DoFHandler<dim> dof_handler;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  std::vector<IndexSet> fields_locally_owned_dofs;
  std::vector<IndexSet> fields_locally_relevant_dofs;
  AffineConstraints<double> constraints_hanging_nodes;
  AffineConstraints<double> constraints_all;

  /*
   * Solutions
   */
  LA::MPI::BlockSparseMatrix system_matrix;
  LA::MPI::BlockVector solution, newton_update, old_solution,
      solution_checkpoint;
  //  LA::MPI::BlockVector system_total_residual;
  LA::MPI::BlockVector system_rhs;
  LA::MPI::BlockVector neumann_rhs;
  //  LA::MPI::BlockVector diag_mass, diag_mass_relevant;
  LA::MPI::BlockVector system_solution;

  SolverControl direct_solver_control;
  TrilinosWrappers::SolverDirect direct_solver;

  std::unique_ptr<NewtonVariation<dim>> newton_ctl;
  NewtonInformation<dim> newton_info;
};

template <int dim>
AbstractField<dim>::AbstractField(std::vector<unsigned int> n_components,
                                  std::vector<std::string> names,
                                  std::vector<std::string> boundary_from,
                                  std::string update_scheme,
                                  Controller<dim> &ctl)
    : fields(n_components, names, boundary_from, ctl),
      fe(fields.FE_Q_sequence, fields.FE_Q_dim_sequence),
      dof_handler(ctl.triangulation), update_scheme_timestep(update_scheme),
      direct_solver(direct_solver_control) {
  newton_ctl = select_newton_variation<dim>(ctl.params.adjustment_method, ctl);
}

template <int dim> void AbstractField<dim>::setup_system(Controller<dim> &ctl) {
  system_matrix.clear();
  /**
   * DOF
   **/
  {
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler, fields.components_to_blocks);
#if DEAL_II_VERSION_GTE(9, 2, 0)
    std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler,
                                          fields.components_to_blocks);
#else
    std::vector<types::global_dof_index> dofs_per_block(introspection.n_blocks);
    DoFTools::count_dofs_per_block(dof_handler, dofs_per_block,
                                   fields.components_to_blocks);
#endif
    fields_locally_owned_dofs.clear();
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    compatibility::split_by_block(dofs_per_block, locally_owned_dofs,
                                  fields_locally_owned_dofs);
    fields_locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    compatibility::split_by_block(dofs_per_block, locally_relevant_dofs,
                                  fields_locally_relevant_dofs);

    fields_constant_modes.clear();
    for (unsigned int i = 0; i < fields.n_fields; ++i) {
      std::vector<std::vector<bool>> constant_modes;
      constant_modes.clear();
      DoFTools::extract_constant_modes(
          dof_handler, fields.component_masks[fields.names[i]], constant_modes);
      fields_constant_modes.push_back(constant_modes);
    }
  }
  /**
   * Hanging node and boundary value constraints
   */
  {
    constraints_hanging_nodes.clear();
    constraints_hanging_nodes.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints_hanging_nodes);
    constraints_hanging_nodes.close();

    constraints_all.clear();
    constraints_all.reinit(locally_relevant_dofs);
    setup_dirichlet_boundary_condition(ctl);
    constraints_all.close();
  }

  /**
   * Sparsity pattern
   */
  {
    TrilinosWrappers::BlockSparsityPattern sparsity_pattern(
        fields_locally_owned_dofs, ctl.mpi_com);
    DoFTools::make_sparsity_pattern(
        dof_handler, sparsity_pattern, constraints_all,
        /*keep constrained dofs*/ false,
        Utilities::MPI::this_mpi_process(ctl.mpi_com));
    sparsity_pattern.compress();
    system_matrix.clear();
    system_matrix.reinit(sparsity_pattern);
  }

  /**
   * Initialize solution
   */
  {
    // solution has ghost elements.
    solution.reinit(fields_locally_relevant_dofs);
    solution_checkpoint.reinit(fields_locally_relevant_dofs);
    old_solution.reinit(fields_locally_relevant_dofs);
    solution = 0;
    old_solution = solution;
    solution_checkpoint = solution;
    // system_rhs, system_matrix, and the solution vector system_solution do not
    // have ghost elements
    system_solution.reinit(fields_locally_owned_dofs);
    system_rhs.reinit(fields_locally_owned_dofs);
    neumann_rhs.reinit(fields_locally_owned_dofs);
    // Initialize fields. Trilino does not allow writing into its parallel
    // vector.
    //    VectorTools::interpolate(dof_handler, ZeroFunction<dim>(dim),
    //                             solution);
  }
}

template <int dim> double AbstractField<dim>::update(Controller<dim> &ctl) {
  if (update_scheme_timestep == "linear") {
    update_linear_system(ctl);
    return 0.0;
  } else if (update_scheme_timestep == "newton") {
    return update_newton_system(ctl);
  } else {
    AssertThrow(false, ExcNotImplemented());
  }
}

template <int dim>
bool AbstractField<dim>::dof_is_this_field(unsigned int i_dof,
                                           std::string name) {
  const unsigned int comp_i = fe.system_to_component_index(i_dof).first;
  if (comp_i < fields.component_start_indices[name] ||
      comp_i >= fields.component_start_indices[name] +
                    fields.n_components_fields[name]) {
    return false;
  } else
    return true;
}

template <int dim>
void AbstractField<dim>::setup_dirichlet_boundary_condition(
    Controller<dim> &ctl) {
  // Dealing with dirichlet boundary conditions
  constraints_all.clear();
  constraints_all.reinit(locally_relevant_dofs);
  constraints_all.merge(constraints_hanging_nodes,
                        ConstraintMatrix::right_object_wins);
  for (auto &it : fields.dirichlet_boundary_info) {
    for (const std::tuple<unsigned int, std::string, unsigned int, double,
                          std::vector<double>> &info : it.second) {
      ctl.debug_dcout << "Setting dirichlet boundary" << std::endl;
      std::unique_ptr<Function<dim>> dirichlet_boundary =
          select_dirichlet_boundary<dim>(info, fields.n_components, ctl.time);
      VectorTools::interpolate_boundary_values(
          dof_handler, std::get<0>(info), *dirichlet_boundary, constraints_all,
          fields.component_masks[it.first + "_" +
                                 std::to_string(std::get<2>(info))]);
    }
  }
  constraints_all.close();
}

template <int dim>
void AbstractField<dim>::setup_neumann_boundary_condition(
    LA::MPI::BlockVector &neumann_rhs, Controller<dim> &ctl) {

  ctl.debug_dcout << "Setting neumann boundary" << std::endl;
  neumann_rhs = 0;

  const QGaussLobatto<dim - 1> face_quadrature_formula(ctl.params.poly_degree + 1);
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                       update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<Vector<double>> neumann_values(n_face_q_points);

  for (unsigned int i_field = 0; i_field < fields.n_fields; ++i_field) {
    std::string name = fields.names[i_field];
    for (unsigned int i_boundary = 0;
         i_boundary < fields.neumann_boundary_info[name].size(); ++i_boundary) {
      std::tuple<unsigned int, std::string, std::vector<double>,
                 std::vector<double>>
          neumann_info = fields.neumann_boundary_info[name][i_boundary];

      // Vector<double> cannot be automatically initialized like Tensor
      for (unsigned int j = 0; j < n_face_q_points; ++j) {
        neumann_values[j].reinit(fields.n_components_fields[name]);
      }

      unsigned int boundary_id = std::get<0>(neumann_info);

      std::unique_ptr<GeneralNeumannBoundary<dim>> neumann_boundary =
          select_neumann_boundary<dim>(
              neumann_info, fields.n_components_fields[name], ctl.time);

      for (const auto &cell : (this->dof_handler).active_cell_iterators())
        if (cell->is_locally_owned()) {
          for (const auto &face : cell->face_iterators()) {
            if (face->at_boundary() && face->boundary_id() == boundary_id) {
              cell_rhs = 0;
              fe_face_values.reinit(cell, face);
              neumann_boundary->vector_value_list(
                  fe_face_values.get_quadrature_points(), neumann_values);
              for (unsigned int q_point = 0; q_point < n_face_q_points;
                   ++q_point) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const unsigned int comp_i =
                      fe.system_to_component_index(i).first;
                  if (!this->dof_is_this_field(i, name)) {
                    continue;
                  }
                  cell_rhs(i) +=
                      (fe_face_values.shape_value(i, q_point) * // phi_i(x_q)
                       neumann_values[q_point][comp_i] *        // g(x_q)
                       fe_face_values.JxW(q_point));            // dx
                }
              }
              cell->get_dof_indices(local_dof_indices);
              constraints_all.distribute_local_to_global(
                  cell_rhs, local_dof_indices, neumann_rhs);
            }
          }
        }
    }
  }
  neumann_rhs.compress(VectorOperation::add);
}

template <int dim>
double AbstractField<dim>::update_linear_system(Controller<dim> &ctl) {
  // Cannot distribute constraints to parallel vectors with ghost dofs.
  LA::MPI::BlockVector distributed_solution(fields_locally_owned_dofs);
  distributed_solution = solution;

  ctl.debug_dcout << "Solve linear system - initialize" << std::endl;
  setup_dirichlet_boundary_condition(ctl);
  distribute_all_constraints(distributed_solution, ctl);
  solution = distributed_solution;
  ctl.debug_dcout << "Solve linear system - assemble" << std::endl;
  assemble_linear_system(ctl);

  ctl.debug_dcout << "Solve linear system - solve" << std::endl;
  double dummy = solve(ctl);
  ctl.debug_dcout << "Solve linear system - constraints" << std::endl;
  distributed_solution = system_solution;
  distribute_all_constraints(distributed_solution, ctl);
  solution = distributed_solution;

  return 0.0;
}

template <int dim>
void AbstractField<dim>::update_newton_residual(Controller<dim> &ctl) {
  // Cannot distribute constraints to parallel vectors with ghost dofs.
  LA::MPI::BlockVector distributed_solution(fields_locally_owned_dofs);
  distributed_solution = solution;

  // Application of the initial boundary conditions to the
  // variational equations:
  setup_dirichlet_boundary_condition(ctl);
  distribute_all_constraints(distributed_solution, ctl);
  solution = distributed_solution;
  neumann_rhs = 0;
  assemble_newton_system(true, neumann_rhs, ctl);
}

template <int dim>
double AbstractField<dim>::update_newton_system(Controller<dim> &ctl) {
  ctl.dcout << "It.\tResidual\tReduction\t#LinIts" << std::endl;

  ctl.debug_dcout
      << "Solve Newton system - Newton iteration - first residual assemble"
      << std::endl;
  update_newton_residual(ctl);

  LA::MPI::BlockVector distributed_solution(fields_locally_owned_dofs);
  distributed_solution = solution;

  newton_info.residual = get_norm(system_rhs, ctl.params.norm_type);
  newton_info.old_residual = newton_info.residual * 1e8;
  newton_info.i_step = 1;
  newton_info.iterative_solver_nonlinear_step = 0;
  newton_info.adjustment_step = 0;
  newton_info.new_residual = 0.0;
  newton_info.system_matrix_rebuilt = false;

  ctl.dcout << "0\t" << std::scientific << newton_info.residual << std::endl;

  while ((newton_info.residual > ctl.params.lower_bound_newton_residual &&
          newton_info.i_step < ctl.params.max_no_newton_steps) ||
         (newton_info.i_step == 1 &&
          !newton_ctl->allow_skip_first_iteration(newton_info, ctl))) {
    if (newton_ctl->quit_newton(newton_info, ctl) &&
        !(newton_info.i_step == 1 &&
          !newton_ctl->allow_skip_first_iteration(newton_info, ctl))) {
      ctl.dcout << '\t' << std::scientific << newton_info.residual << std::endl;
      break;
    }

    if (newton_ctl->rebuild_jacobian(newton_info, ctl)) {
      ctl.debug_dcout
          << "Solve Newton system - Newton iteration - system assemble"
          << std::endl;
      assemble_newton_system(false, neumann_rhs, ctl);
      newton_info.system_matrix_rebuilt = true;
    }

    // Solve Ax = b
    ctl.debug_dcout
        << "Solve Newton system - Newton iteration - solve linear system"
        << std::endl;
    newton_info.iterative_solver_nonlinear_step = solve(newton_info, ctl);
    newton_info.system_matrix_rebuilt = false;
    ctl.debug_dcout
        << "Solve Newton system - Newton iteration - solve linear system exit"
        << std::endl;
    newton_info.adjustment_step = 0;
    // Relaxation
    for (; newton_info.adjustment_step < ctl.params.max_adjustment_steps;) {
      ctl.debug_dcout
          << "Solve Newton system - Newton iteration - dealing solution"
          << std::endl;
      newton_info.new_residual = get_norm(system_rhs, ctl.params.norm_type);
      newton_ctl->apply_increment(system_solution, distributed_solution,
                                  this->system_matrix, this->system_rhs,
                                  neumann_rhs, newton_info, ctl);
      ctl.debug_dcout << "Solve Newton system - Newton iteration - distribute"
                      << std::endl;
      distribute_all_constraints(distributed_solution, ctl);
      solution = distributed_solution;
      ctl.debug_dcout << "Solve Newton system - Newton iteration - "
                         "residual assemble"
                      << std::endl;
      if (newton_ctl->re_solve(newton_info, ctl)) {
        if (newton_ctl->rebuild_jacobian(newton_info, ctl)) {
          ctl.debug_dcout << "Solve Newton system - Newton iteration - resolve "
                             "- system assemble"
                          << std::endl;
          assemble_newton_system(false, neumann_rhs, ctl);
          newton_info.system_matrix_rebuilt = true;
        }
        ctl.debug_dcout << "Solve Newton system - Newton iteration - resolve"
                        << std::endl;
        newton_info.iterative_solver_nonlinear_step = solve(newton_info, ctl);
        newton_info.system_matrix_rebuilt = false;
      } else {
        ctl.debug_dcout
            << "Solve Newton system - Newton iteration - residual assemble"
            << std::endl;
        assemble_newton_system(true, neumann_rhs, ctl);
      }
      newton_info.new_residual = get_norm(system_rhs, ctl.params.norm_type);

      if (newton_ctl->quit_adjustment(newton_info, ctl)) {
        ctl.debug_dcout
            << "Solve Newton system - Newton iteration - stop adjustment"
            << std::endl;
        break;
      } else {
        newton_info.adjustment_step += 1;
        ctl.debug_dcout
            << "Solve Newton system - Newton iteration - next adjustment"
            << std::endl;
        newton_ctl->prepare_next_adjustment(
            system_solution, distributed_solution, this->system_matrix,
            this->system_rhs, neumann_rhs, newton_info, ctl);
        distribute_all_constraints(distributed_solution, ctl);
        solution = distributed_solution;
      }
    }
    if (newton_info.i_step == 1 &&
        !newton_ctl->allow_skip_first_iteration(newton_info, ctl) &&
        newton_info.new_residual <= ctl.params.lower_bound_newton_residual) {
      // Do nothing, and jump out without triggering error if the residual
      // increases a little
    } else {
      newton_info.old_residual = newton_info.residual;
    }
    newton_info.residual = newton_info.new_residual;

    ctl.dcout << std::setprecision(-1) << std::defaultfloat
              << newton_info.i_step << '\t' << std::setprecision(5)
              << std::scientific << newton_info.residual;

    ctl.dcout << '\t' << std::scientific
              << newton_info.residual / newton_info.old_residual << '\t';

    ctl.dcout << newton_info.adjustment_step << '\t' << std::scientific
              << newton_info.iterative_solver_nonlinear_step << '\t'
              << std::scientific << std::endl;

    // Terminate if nothing is solved anymore. After this,
    // we cut the time step.
    if (newton_ctl->give_up(newton_info, ctl) || newton_info.residual > 1e50 ||
        newton_info.residual != newton_info.residual) {
      ctl.dcout << "Newton iteration did not converge in " << newton_info.i_step
                << " steps. Go to adaptive time stepping" << std::endl;
      throw SolverControl::NoConvergence(0, 0);
    }

    // Updates
    newton_info.i_step++;
  }

  solution = distributed_solution;
  return newton_info.residual / newton_info.old_residual;
}

template <int dim>
unsigned int AbstractField<dim>::solve(NewtonInformation<dim> &info,
                                       Controller<dim> &ctl) {
  SolverControl solver_control((this->dof_handler).n_dofs(),
                               1e-10 * this->system_rhs.l2_norm());
  ctl.debug_dcout << "Solve Newton system - Newton iteration - solve linear "
                     "system - preconditioner"
                  << std::endl;
  if (ctl.params.direct_solver) {
    std::vector<std::shared_ptr<LA::MPI::PreconditionAMG>> preconditioners;
    BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG> preconditioner(
        preconditioners);
    return AbstractField<dim>::solve_linear_system(info, ctl, solver_control,
                                                   preconditioner);
  } else {
    std::vector<std::shared_ptr<LA::MPI::PreconditionAMG>> preconditioners;
    for (unsigned int i = 0; i < fields.n_blocks; ++i) {
      LA::MPI::PreconditionAMG::AdditionalData data;
      data.constant_modes = fields_constant_modes[i];
      data.elliptic = true;
      data.higher_order_elements = true;
      data.smoother_sweeps = 2;
      data.aggregation_threshold = 0.02;
      std::shared_ptr<LA::MPI::PreconditionAMG> prec(
          new LA::MPI::PreconditionAMG);
      prec->initialize(system_matrix.block(i, i), data);
      preconditioners.push_back(prec);
    }
    BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG> preconditioner(
        preconditioners);
    return AbstractField<dim>::solve_linear_system(info, ctl, solver_control,
                                                   preconditioner);
  }
}

template <int dim>
unsigned int AbstractField<dim>::solve_linear_system(
    NewtonInformation<dim> &info, Controller<dim> &ctl,
    SolverControl &solver_control,
    BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG> &preconditioner) {
  if (ctl.params.direct_solver) {
    if (info.system_matrix_rebuilt) {
      ctl.debug_dcout
          << "Solve Newton system - Newton iteration - solve linear "
             "system - factorization"
          << std::endl;
      ctl.timer.enter_subsection("Factorization");
      ctl.computing_timer.enter_subsection("Factorization");
      direct_solver.initialize(system_matrix.block(0, 0));
      ctl.computing_timer.leave_subsection("Factorization");
      ctl.timer.leave_subsection("Factorization");
    }
    ctl.debug_dcout << "Solve Newton system - Newton iteration - solve linear "
                       "system - solve LUx=b"
                    << std::endl;
    ctl.timer.enter_subsection("Solve LUx=b");
    ctl.computing_timer.enter_subsection("Solve LUx=b");
    direct_solver.solve(system_solution.block(0), system_rhs.block(0));
    ctl.computing_timer.leave_subsection("Solve LUx=b");
    ctl.timer.leave_subsection("Solve LUx=b");
    return 1;
  } else {
    SolverGMRES<LA::MPI::BlockVector> solver(solver_control);
    ctl.debug_dcout << "Solve Newton system - Newton iteration - solve linear "
                       "system - solve"
                    << std::endl;
    solver.solve(system_matrix, system_solution, system_rhs, preconditioner);
    ctl.debug_dcout << "Solve Newton system - Newton iteration - solve linear "
                       "system - solve complete"
                    << std::endl;

    return solver_control.last_step();
  }
}

template <int dim>
void AbstractField<dim>::return_old_solution(Controller<dim> &ctl) {
  solution = old_solution;
}

template <int dim>
void AbstractField<dim>::record_old_solution(Controller<dim> &ctl) {
  old_solution = solution;
}

template <int dim>
void AbstractField<dim>::return_checkpoint(Controller<dim> &ctl) {
  solution = solution_checkpoint;
}

template <int dim>
void AbstractField<dim>::record_checkpoint(Controller<dim> &ctl) {
  solution_checkpoint = solution;
}

template <int dim>
void AbstractField<dim>::distribute_hanging_node_constraints(
    LA::MPI::BlockVector &vector, Controller<dim> &ctl) {
  constraints_hanging_nodes.distribute(vector);
}

template <int dim>
void AbstractField<dim>::distribute_all_constraints(
    LA::MPI::BlockVector &vector, Controller<dim> &ctl) {
  constraints_all.distribute(vector);
}

template <int dim>
parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector>
AbstractField<dim>::prepare_refine() {
  parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector> soltrans(
      dof_handler);
  soltrans.prepare_for_coarsening_and_refinement(solution);
  return soltrans;
}

template <int dim>
void AbstractField<dim>::post_refine(
    parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector>
        &soltrans,
    Controller<dim> &ctl) {
  LA::MPI::BlockVector interpolated_solution;
  interpolated_solution.reinit(fields_locally_owned_dofs);
  soltrans.interpolate(interpolated_solution);
  solution = interpolated_solution;
  record_old_solution(ctl);
}

#endif // CRACKS_ABSTRACT_FIELD_H
