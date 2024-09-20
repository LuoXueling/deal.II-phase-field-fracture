//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_PHASE_FIELD_H
#define CRACKS_PHASE_FIELD_H

#include "abstract_field.h"
#include "constitutive_law.h"
#include "dealii_includes.h"
#include "degradation.h"
#include "parameters.h"
#include "post_processors.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class PhaseField : public AbstractField<dim> {
public:
  PhaseField(std::string update_scheme, Controller<dim> &ctl);
  void assemble_newton_system(bool residual_only,
                              LA::MPI::BlockVector &neumann_rhs,
                              Controller<dim> &ctl) override;
  void assemble_linear_system(Controller<dim> &ctl) override;
  void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) override;

  void enforce_phase_field_limitation(Controller<dim> &ctl);

  std::unique_ptr<Degradation<dim>> degradation;
};

template <int dim>
PhaseField<dim>::PhaseField(std::string update_scheme, Controller<dim> &ctl)
    : AbstractField<dim>(std::vector<unsigned int>(1, 1),
                         std::vector<std::string>(1, "phasefield"),
                         std::vector<std::string>(1, "none"), update_scheme,
                         ctl) {
  degradation = select_degradation<dim>(ctl.params.degradation);
}

template <int dim>
void PhaseField<dim>::assemble_linear_system(Controller<dim> &ctl) {
  (this->system_rhs).block(this->block_id("phasefield")) = 0;
  (this->system_matrix)
      .block(this->block_id("phasefield"), this->block_id("phasefield")) = 0;

  FEValues<dim> fe_values((this->fe), ctl.quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = (this->fe).n_dofs_per_cell();
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  const FEValuesExtractors::Scalar extractor =
      (this->fields).extractors_scalar["phasefield"];

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Old Newton values
  std::vector<double> old_phasefield_values(n_q_points);
  // Old Newton grads
  std::vector<Tensor<1, dim>> old_phasefield_grads(n_q_points);

  std::vector<double> Nphi_kq(dofs_per_cell);
  std::vector<Tensor<1, dim>> Bphi_kq(dofs_per_cell);

  if (ctl.params.degradation != "quadratic") {
    AssertThrow(false,
                ExcInternalError("Cannot solve linear equations for phase "
                                 "field when degradation is not quadratic."))
  }

  for (const auto &cell : (this->dof_handler).active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      fe_values[extractor].get_function_values((this->solution),
                                               old_phasefield_values);
      fe_values[extractor].get_function_gradients((this->solution),
                                                  old_phasefield_grads);

      // Get history
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        double H = lqph[q]->get_latest("Driving force", 0.0);
        if (ctl.params.phasefield_model == "AT1") {
          H = std::max(H, 3.0 * ctl.params.Gc / (16.0 * ctl.params.l_phi));
        }
        lqph[q]->update("Phase field", old_phasefield_values[q]);
        lqph[q]->update("Phase field JxW",
                        old_phasefield_values[q] * fe_values.JxW(q));

        // Values of fields and their derivatives
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          Nphi_kq[k] = fe_values[extractor].value(k, q);
          Bphi_kq[k] = fe_values[extractor].gradient(k, q);
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            if (ctl.params.phasefield_model == "AT1") {
              cell_matrix(i, j) += (Nphi_kq[i] * Nphi_kq[j] * 2 * H +
                                    Bphi_kq[i] * Bphi_kq[j] * ctl.params.Gc *
                                        ctl.params.l_phi * 0.75) *
                                   fe_values.JxW(q);
            } else if (ctl.params.phasefield_model == "AT2") {
              cell_matrix(i, j) +=
                  (Nphi_kq[i] * Nphi_kq[j] * 2 * H +
                   Bphi_kq[i] * Bphi_kq[j] * ctl.params.Gc * ctl.params.l_phi +
                   Nphi_kq[i] * Nphi_kq[j] * ctl.params.Gc / ctl.params.l_phi) *
                  fe_values.JxW(q);
            } else {
              AssertThrow(
                  false, ExcNotImplemented("Phase field model not available."));
            }
          }
          if (ctl.params.phasefield_model == "AT1") {
            cell_rhs(i) += (Nphi_kq[i] * Mbracket(-3.0 / 8 * ctl.params.Gc /
                                                      ctl.params.l_phi +
                                                  2 * H)) *
                           fe_values.JxW(q);
          } else if (ctl.params.phasefield_model == "AT2") {
            cell_rhs(i) += (Nphi_kq[i] * Mbracket(2 * H)) * fe_values.JxW(q);
          } else {
            AssertThrow(false,
                        ExcNotImplemented("Phase field model not available."));
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      (this->constraints_all)
          .distribute_local_to_global(cell_matrix, local_dof_indices,
                                      (this->system_matrix));
      (this->constraints_all)
          .distribute_local_to_global(cell_rhs, local_dof_indices,
                                      (this->system_rhs));
    }

  (this->system_matrix).compress(VectorOperation::add);
  (this->system_rhs).compress(VectorOperation::add);
}

template <int dim>
void PhaseField<dim>::assemble_newton_system(bool residual_only,
                                             LA::MPI::BlockVector &neumann_rhs,
                                             Controller<dim> &ctl) {
  (this->system_rhs).block(this->block_id("phasefield")) = 0;
  if (!residual_only) {
    (this->system_matrix)
        .block(this->block_id("phasefield"), this->block_id("phasefield")) = 0;
  }

  FEValues<dim> fe_values((this->fe), ctl.quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = (this->fe).n_dofs_per_cell();
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  const FEValuesExtractors::Scalar extractor =
      (this->fields).extractors_scalar["phasefield"];

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Old Newton values
  std::vector<double> old_phasefield_values(n_q_points);
  // Old Newton grads
  std::vector<Tensor<1, dim>> old_phasefield_grads(n_q_points);

  std::vector<double> Nphi_kq(dofs_per_cell);
  std::vector<Tensor<1, dim>> Bphi_kq(dofs_per_cell);

  for (const auto &cell : (this->dof_handler).active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      fe_values[extractor].get_function_values((this->solution),
                                               old_phasefield_values);
      fe_values[extractor].get_function_gradients((this->solution),
                                                  old_phasefield_grads);
      // Get history
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // Values of fields and their derivatives
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          Nphi_kq[k] = fe_values[extractor].value(k, q);
          Bphi_kq[k] = fe_values[extractor].gradient(k, q);
        }

        double H = lqph[q]->get_latest("Driving force", 0.0);
        lqph[q]->update("Phase field", old_phasefield_values[q]);
        lqph[q]->update("Phase field JxW",
                        old_phasefield_values[q] * fe_values.JxW(q));
        double degrade = degradation->value(old_phasefield_values[q], ctl);
        double degrade_derivative =
            degradation->derivative(old_phasefield_values[q], ctl);
        double degrade_second_derivative =
            degradation->second_derivative(old_phasefield_values[q], ctl);
        double cw, w0, w1, w2;
        if (ctl.params.phasefield_model == "AT1") {
          cw = 2.0 / 3.0;
          w0 = old_phasefield_values[q];
          w1 = 1;
          w2 = 0;
        } else if (ctl.params.phasefield_model == "AT2") {
          cw = 0.5;
          w0 = old_phasefield_values[q] * old_phasefield_values[q];
          w1 = 2 * old_phasefield_values[q];
          w2 = 2;
        } else {
          AssertThrow(false,
                      ExcNotImplemented("Phase field model not available."));
        }
        lqph[q]->update(
            "Diffusion JxW",
            1 / (4 * cw) *
                (w0 / ctl.params.l_phi +
                 ctl.params.l_phi * old_phasefield_grads[q].norm_square()) *
                fe_values.JxW(q));

        if (ctl.params.phasefield_model == "AT1") {
          H = std::max(H, 3.0 * ctl.params.Gc / (16.0 * ctl.params.l_phi));
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          if (!this->dof_is_this_field(i, "phasefield")) {
            continue;
          }
          if (!residual_only) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              if (!this->dof_is_this_field(j, "phasefield")) {
                continue;
              }
              {
                cell_matrix(i, j) +=
                    (ctl.params.Gc / (2 * cw) *
                         ctl.params.l_phi * Bphi_kq[i] * Bphi_kq[j] +
                     Nphi_kq[i] * Nphi_kq[j] *
                         (degrade_second_derivative * H +
                          ctl.params.Gc / (2 * cw) /
                              (2 * ctl.params.l_phi) * w2)) *
                    fe_values.JxW(q);
              }
            }
          }

          cell_rhs(i) += (degrade_derivative * H * Nphi_kq[i] +
                          ctl.params.Gc / (2 * cw) *
                              (ctl.params.l_phi *
                                   old_phasefield_grads[q] * Bphi_kq[i] +
                               1 / (2 * ctl.params.l_phi) * w1 *
                                   Nphi_kq[i])) *
                         fe_values.JxW(q);
        }
      }

      cell->get_dof_indices(local_dof_indices);
      if (residual_only) {
        (this->constraints_all)
            .distribute_local_to_global(cell_rhs, local_dof_indices,
                                        (this->system_rhs));
      } else {
        (this->constraints_all)
            .distribute_local_to_global(cell_matrix, local_dof_indices,
                                        (this->system_matrix));
        (this->constraints_all)
            .distribute_local_to_global(cell_rhs, local_dof_indices,
                                        (this->system_rhs));
      }
    }

  (this->system_matrix).compress(VectorOperation::add);
  (this->system_rhs).compress(VectorOperation::add);
}

template <int dim>
void PhaseField<dim>::output_results(DataOut<dim> &data_out,
                                     Controller<dim> &ctl) {
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          1, DataComponentInterpretation::component_is_scalar);
  data_out.add_data_vector((this->dof_handler),
                           (this->solution).block(this->block_id("phasefield")),
                           std::vector<std::string>(1, "Phase_field"),
                           data_component_interpretation);
  PointHistoryProcessor<dim> hist_processor("Driving force", this->fields,
                                            this->fe, ctl);
  hist_processor.add_data_scalar(this->solution, this->fields, data_out,
                                 this->dof_handler, ctl);
}

template <int dim>
void PhaseField<dim>::enforce_phase_field_limitation(Controller<dim> &ctl) {
  typename DoFHandler<dim>::active_cell_iterator cell = (this->dof_handler)
                                                            .begin_active(),
                                                 endc =
                                                     (this->dof_handler).end();

  LA::MPI::BlockVector distributed_solution(this->fields_locally_owned_dofs);
  distributed_solution = this->solution;

  std::vector<types::global_dof_index> local_dof_indices(
      (this->fe).dofs_per_cell);
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < (this->fe).dofs_per_cell; ++i) {
        if (!this->dof_is_this_field(i, "phasefield")) {
          continue;
        }
        const types::global_dof_index idx = local_dof_indices[i];
        if (!(this->dof_handler).locally_owned_dofs().is_element(idx))
          continue;

        distributed_solution(idx) = std::max(
            0.0, std::min(static_cast<double>((this->solution)(idx)), 1.0));
      }
    }

  distributed_solution.compress(VectorOperation::insert);
  this->solution = distributed_solution;
}

#endif // CRACKS_PHASE_FIELD_H
