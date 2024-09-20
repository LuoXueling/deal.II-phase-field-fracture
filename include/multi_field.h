/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef MULTI_FIELD_H
#define MULTI_FIELD_H

#include "controller.h"
#include "dealii_includes.h"
#include "parameters.h"
using namespace dealii;

/**
 * @refitem tjhei/cracks
 *
 */
template <int dim> struct MultiFieldCfg {
  MultiFieldCfg(std::vector<unsigned int> n_components,
                std::vector<std::string> names,
                std::vector<std::string> boundary_from, Controller<dim> &ctl);
  void define_boundary_condition(const std::string boundary_from,
                                 const std::string name);

  std::vector<const FiniteElement<dim, dim> *> FE_Q_sequence;
  std::vector<unsigned int> FE_Q_dim_sequence;

  std::vector<unsigned int> block_component;
  unsigned int n_components;
  unsigned int n_fields;
  unsigned int n_blocks;

  std::map<std::string, unsigned int> n_components_fields;
  std::map<std::string, ComponentMask> component_masks;
  std::map<std::string, unsigned int> component_start_indices;
  std::map<std::string, FEValuesExtractors::Vector> extractors_vector;
  std::map<std::string, FEValuesExtractors::Scalar> extractors_scalar;
  std::vector<unsigned int> components_to_blocks;
  std::vector<std::string> names;

  std::map<std::string,
           std::vector<std::tuple<unsigned int, std::string, unsigned int,
                                  double, std::vector<double>>>>
      dirichlet_boundary_info;
  std::map<std::string,
           std::vector<std::tuple<unsigned int, std::string,
                                  std::vector<double>, std::vector<double>>>>
      neumann_boundary_info;
};

template <int dim>
MultiFieldCfg<dim>::MultiFieldCfg(std::vector<unsigned int> n_components_list,
                                  std::vector<std::string> names_in,
                                  std::vector<std::string> boundary_from,
                                  Controller<dim> &ctl)
    : n_components(0), n_fields(n_components_list.size()), names(names_in) {
  n_blocks = (ctl.params.direct_solver) ? 1 : n_fields;
  for (unsigned int i_field = 0; i_field < n_components_list.size();
       ++i_field) {
    n_components_fields[names[i_field]] = n_components_list[i_field];
    n_components += n_components_list[i_field];
  }
  block_component = std::vector<unsigned int>(n_components, 0);
  components_to_blocks.resize(n_components, 0);

  unsigned int processed_components = 0;
  for (unsigned int i_field = 0; i_field < n_components_list.size();
       ++i_field) {
    unsigned int n_component = n_components_list[i_field];
    std::string name = names[i_field];
    FE_Q_sequence.push_back(
        new FE_Q<dim>(QGaussLobatto<1>(ctl.params.poly_degree + 1)));
    FE_Q_dim_sequence.push_back(n_component);

    component_start_indices[name] = processed_components;
    component_masks[name] = ComponentMask(n_components, false);
    extractors_vector[name] =
        FEValuesExtractors::Vector(component_start_indices[name]);
    extractors_scalar[name] =
        FEValuesExtractors::Scalar(component_start_indices[name]);

    for (unsigned int i_comp = 0; i_comp < n_component; ++i_comp) {
      // block_component[i]=j means the ith component belongs to the jth field.
      block_component[processed_components + i_comp] = i_field;
      component_start_indices[name + "_" + std::to_string(i_comp)] = i_comp;
      component_masks[name + "_" + std::to_string(i_comp)] =
          ComponentMask(n_components, false);
      component_masks[name + "_" + std::to_string(i_comp)].set(
          processed_components + i_comp, true);
      component_masks[name].set(processed_components + i_comp, true);
      if (!ctl.params.direct_solver) {
        components_to_blocks[processed_components + i_comp] = i_field;
      }
    }

    define_boundary_condition(boundary_from[i_field], name);

    processed_components += n_component;
  }
}

template <int dim>
void MultiFieldCfg<dim>::define_boundary_condition(
    const std::string boundary_from, const std::string name) {
  if (boundary_from == "none") {
    return;
  }
  std::filebuf fb;
  if (fb.open(boundary_from, std::ios::in)) {
    std::istream is(&fb);
    std::string line;
    unsigned int boundary_id, constrained_dof;
    std::string constraint_type;
    double constraint_value;
    while (std::getline(is, line)) {
      if (line[0] == '#') {
        continue;
      }
      std::istringstream iss(line);
      iss >> boundary_id >> constraint_type;
      if (constraint_type == "velocity" || constraint_type == "dirichlet" ||
          constraint_type == "triangulardirichlet" ||
          constraint_type == "sinedirichlet") {
        std::vector<double> additional_info;
        if (constraint_type == "velocity" || constraint_type == "dirichlet") {
          iss >> constrained_dof >> constraint_value;
        } else {
          iss >> constrained_dof;
          constraint_value = 0.0;
          double temp_value;
          do {
            iss >> temp_value;
            additional_info.push_back(temp_value);
          } while (!iss.eof());
        }
        std::tuple<unsigned int, std::string, unsigned int, double,
                   std::vector<double>>
            info(boundary_id, constraint_type, constrained_dof,
                 constraint_value, additional_info);
        dirichlet_boundary_info[name].push_back(info);
      } else if (constraint_type == "neumann" ||
                 constraint_type == "neumannrate" ||
                 constraint_type == "sineneumann" ||
                 constraint_type == "triangularneumann") {
        std::vector<double> constraint_vector;
        std::vector<double> additional_info;
        double temp_value;
        do {
          iss >> temp_value;
          if (constraint_vector.size() < n_components_fields[name]) {
            constraint_vector.push_back(temp_value);
          } else {
            additional_info.push_back(temp_value);
          }
        } while (!iss.eof());
        std::tuple<unsigned int, std::string, std::vector<double>,
                   std::vector<double>>
            info(boundary_id, constraint_type, constraint_vector,
                 additional_info);
        neumann_boundary_info[name].push_back(info);
      } else {
        AssertThrow(false, ExcNotImplemented(constraint_type));
      }
    }
    fb.close();
  }
}

template <class Preconditioner> class BlockDiagonalPreconditioner {
public:
  BlockDiagonalPreconditioner(
      const std::vector<std::shared_ptr<Preconditioner>> &preconditioners)
      : prec(preconditioners) {}

  void vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
    for (unsigned int i = 0; i < prec.size(); ++i) {
      prec[i]->vmult(dst.block(i), src.block(i));
    }
  }

  const std::vector<std::shared_ptr<Preconditioner>> &prec;
};

#endif
