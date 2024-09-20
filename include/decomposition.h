//
// Created by xlluo on 24-8-1.
//

#ifndef CRACKS_DECOMPOSITION_H
#define CRACKS_DECOMPOSITION_H

#include "dealii_includes.h"
#include "utils.h"
using namespace dealii;

template <int dim> class Decomposition {
public:
  virtual void decompose_elasticity_tensor_stress_and_energy(
      const SymmetricTensor<2, dim> &strain,
      const SymmetricTensor<2, dim> &stress_0,
      const SymmetricTensor<4, dim> &elasticity_tensor, double &energy_positive,
      double &energy_negative, SymmetricTensor<2, dim> &stress_positive,
      SymmetricTensor<2, dim> &stress_negative,
      SymmetricTensor<4, dim> &elasticity_tensor_positive,
      SymmetricTensor<4, dim> &elasticity_tensor_negative,
      ConstitutiveLaw<dim> &constitutive_law) {
    elasticity_tensor_positive = elasticity_tensor;
    elasticity_tensor_negative = 0;
    decompose_stress_and_energy(strain, stress_0, energy_positive,
                                energy_negative, stress_positive,
                                stress_negative, constitutive_law);
  }
  virtual void
  decompose_stress_and_energy(const SymmetricTensor<2, dim> &strain,
                              const SymmetricTensor<2, dim> &stress_0,
                              double &energy_positive, double &energy_negative,
                              SymmetricTensor<2, dim> &stress_positive,
                              SymmetricTensor<2, dim> &stress_negative,
                              ConstitutiveLaw<dim> &constitutive_law) {
    stress_positive = stress_0;
    stress_negative = 0;
    energy_positive = 0.5 * scalar_product(stress_0, strain);
    energy_negative = 0;
  }
  virtual ~Decomposition(){};
};

template <int dim> class HybridEigenDecomposition : public Decomposition<dim> {
public:
  void decompose_elasticity_tensor_stress_and_energy(
      const SymmetricTensor<2, dim> &strain,
      const SymmetricTensor<2, dim> &stress_0,
      const SymmetricTensor<4, dim> &elasticity_tensor, double &energy_positive,
      double &energy_negative, SymmetricTensor<2, dim> &stress_positive,
      SymmetricTensor<2, dim> &stress_negative,
      SymmetricTensor<4, dim> &elasticity_tensor_positive,
      SymmetricTensor<4, dim> &elasticity_tensor_negative,
      ConstitutiveLaw<dim> &constitutive_law) override {
    elasticity_tensor_positive = elasticity_tensor;
    elasticity_tensor_negative = 0;
    decompose_stress_and_energy(strain, stress_0, energy_positive,
                                energy_negative, stress_positive,
                                stress_negative, constitutive_law);
  }
  void
  decompose_stress_and_energy(const SymmetricTensor<2, dim> &strain,
                              const SymmetricTensor<2, dim> &stress_0,
                              double &energy_positive, double &energy_negative,
                              SymmetricTensor<2, dim> &stress_positive,
                              SymmetricTensor<2, dim> &stress_negative,
                              ConstitutiveLaw<dim> &constitutive_law) override {
    double trE = trace(strain);
    double trE_pos = 0.5 * (trE + std::abs(trE));
    double trE_neg = 0.5 * (trE - std::abs(trE));
    std::array<std::pair<double, Tensor<1, dim, double>>,
               std::integral_constant<int, dim>::value>
        res = eigenvectors(strain);
    Tensor<2, dim> E_pos, E_neg;

    for (int i = 0; i < dim; ++i) {
      Tensor<2, dim> kron;
      double eig_val = std::get<0>(res[i]);
      Tensor<1, dim> eig_vec = std::get<1>(res[i]);
      Tensors::tensor_product<dim>(kron, eig_vec, eig_vec);
      E_pos += 0.5 * (eig_val + std::abs(eig_val)) * kron;
      E_neg += 0.5 * (eig_val - std::abs(eig_val)) * kron;
    }

    Tensor<2, dim> err = E_pos + E_neg - strain;
    AssertThrow(strain.norm() <= 1e-10 ||
                    (err.norm() / (strain.norm() + 1e-10) < 1e-10),
                ExcInternalError("Wrong strain decomposition"));

    energy_positive = 0.5 * constitutive_law.lambda * std::pow(trE_pos, 2) +
                      constitutive_law.mu * trace(E_pos * E_pos);
    energy_negative = 0.5 * constitutive_law.lambda * std::pow(trE_neg, 2) +
                      constitutive_law.mu * trace(E_neg * E_neg);

    double original_energy = 0.5 * scalar_product(stress_0, strain);
    AssertThrow(
        original_energy < 1e-10 ||
            (std::abs(original_energy - (energy_positive + energy_negative)) /
             (original_energy + 1e-10)) < 1e-10,
        ExcInternalError("Wrong energy decomposition"));

    stress_positive = stress_0;
    stress_negative = 0;
  }
};

template <int dim>
class HybridNoTensionDecomposition : public HybridEigenDecomposition<dim> {
public:
  void
  decompose_stress_and_energy(const SymmetricTensor<2, dim> &strain,
                              const SymmetricTensor<2, dim> &stress_0,
                              double &energy_positive, double &energy_negative,
                              SymmetricTensor<2, dim> &stress_positive,
                              SymmetricTensor<2, dim> &stress_negative,
                              ConstitutiveLaw<dim> &constitutive_law) override {
    double trE = trace(strain);
    double trE_pos = 0.5 * (trE + std::abs(trE));
    double trE_neg = 0.5 * (trE - std::abs(trE));
    double e1, e2, e3;
    std::array<double, dim> res = eigenvalues(strain);
    std::vector<double> v(res.begin(), res.end());
    if (dim == 2) {
      v.push_back(0);
    }
    std::sort(std::begin(v), std::end(v)); // ascending order

    e1 = v[2];
    e2 = v[1];
    e3 = v[0];

    double lambda = constitutive_law.lambda;
    double mu = constitutive_law.mu;
    double nu = constitutive_law.nu;
    double E = constitutive_law.E;
    if (e3 > 0) {
      energy_positive =
          0.5 * lambda * std::pow(e1 + e2 + e3, 2) +
          mu * (std::pow(e1, 2) + std::pow(e2, 2) + std::pow(e3, 2));
      energy_negative = 0;
    } else if ((e2 + nu * e3) > 0) {
      energy_positive =
          0.5 * lambda * std::pow(e1 + e2 + 2 * nu * e3, 2) +
          mu * (std::pow(e1 + nu * e3, 2) + std::pow(e2 + nu * e3, 2));
      energy_negative = 0.5 * E * std::pow(e3, 2);
    } else if (((1 - nu) * e1 + nu * (e2 + e3)) > 0) {
      energy_positive = 0.5 * lambda * (1 + nu) / (nu * (1 - nu * nu)) *
                        std::pow((1 - nu) * e1 + nu * e2 + nu * e3, 2);
      energy_negative = 0.5 * E / (1 - std::pow(nu, 2)) *
                        (std::pow(e2, 2) + std::pow(e3, 2) + 2 * nu * e2 * e3);
    } else {
      energy_positive = 0;
      energy_negative =
          0.5 * lambda * std::pow(e1 + e2 + e3, 2) +
          mu * (std::pow(e1, 2) + std::pow(e2, 2) + std::pow(e3, 2));
    }

    double original_energy = 0.5 * scalar_product(stress_0, strain);
    AssertThrow(
        original_energy < 1e-10 ||
            (std::abs(original_energy - (energy_positive + energy_negative)) /
             (original_energy + 1e-10)) < 1e-10,
        ExcInternalError("Wrong energy decomposition"));

    stress_positive = stress_0;
    stress_negative = 0;
  }
};

template <int dim> class SphericalDecomposition : public Decomposition<dim> {
public:
  void decompose_elasticity_tensor_stress_and_energy(
      const SymmetricTensor<2, dim> &strain,
      const SymmetricTensor<2, dim> &stress_0,
      const SymmetricTensor<4, dim> &elasticity_tensor, double &energy_positive,
      double &energy_negative, SymmetricTensor<2, dim> &stress_positive,
      SymmetricTensor<2, dim> &stress_negative,
      SymmetricTensor<4, dim> &elasticity_tensor_positive,
      SymmetricTensor<4, dim> &elasticity_tensor_negative,
      ConstitutiveLaw<dim> &constitutive_law) override {
    double trE = trace(strain);
    SymmetricTensor<4, dim> Identity = outer_product(
        unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());
    elasticity_tensor_positive =
        constitutive_law.kappa * (trE >= 0 ? 1 : 0) * Identity +
        2 * constitutive_law.mu * (identity_tensor<dim>() - Identity / dim);
    elasticity_tensor_negative =
        constitutive_law.kappa * (-trE > 0 ? 1 : 0) * Identity;

    SymmetricTensor<4, dim> err = elasticity_tensor_positive +
                                  elasticity_tensor_negative -
                                  elasticity_tensor;
    AssertThrow(err.norm() / elasticity_tensor.norm() < 1e-10,
                ExcInternalError("Wrong elasticity tensor decompostion"));

    decompose_stress_and_energy(strain, stress_0, energy_positive,
                                energy_negative, stress_positive,
                                stress_negative, constitutive_law);
  }

  void
  decompose_stress_and_energy(const SymmetricTensor<2, dim> &strain,
                              const SymmetricTensor<2, dim> &stress_0,
                              double &energy_positive, double &energy_negative,
                              SymmetricTensor<2, dim> &stress_positive,
                              SymmetricTensor<2, dim> &stress_negative,
                              ConstitutiveLaw<dim> &constitutive_law) override {
    double trE = trace(strain);
    double trE_pos = 0.5 * (trE + std::abs(trE));
    double trE_neg = 0.5 * (trE - std::abs(trE));
    SymmetricTensor<2, dim> E_dev = strain;
    SymmetricTensor<2, dim> Identity = Tensors::get_Identity<dim>();
    E_dev -= trE / dim * Identity;
    double E_dev_inner_prod = scalar_product(E_dev, E_dev);

    energy_positive = 0.5 * constitutive_law.kappa * trE_pos * trE_pos +
                      constitutive_law.mu * E_dev_inner_prod;
    energy_negative = 0.5 * constitutive_law.kappa * trE_neg * trE_neg;

    double original_energy = 0.5 * scalar_product(stress_0, strain);

    AssertThrow(
        original_energy < 1e-10 ||
            (std::abs(original_energy - (energy_positive + energy_negative)) /
             (original_energy + 1e-10)) < 1e-10,
        ExcInternalError("Wrong energy decomposition"));

    stress_positive = constitutive_law.kappa * trE_pos * Identity +
                      2 * constitutive_law.mu * E_dev;
    stress_negative = constitutive_law.kappa * trE_neg * Identity;

    Tensor<2, dim> err = stress_positive + stress_negative - stress_0;
    AssertThrow(stress_0.norm() <= 1e-10 ||
                    (err.norm() / (stress_0.norm() + 1e-10) < 1e-10),
                ExcInternalError("Wrong stress decomposition"));
  }
};

template <int dim>
class EigenDecomposition : public HybridEigenDecomposition<dim> {
  void decompose_elasticity_tensor_stress_and_energy(
      const SymmetricTensor<2, dim> &strain,
      const SymmetricTensor<2, dim> &stress_0,
      const SymmetricTensor<4, dim> &elasticity_tensor, double &energy_positive,
      double &energy_negative, SymmetricTensor<2, dim> &stress_positive,
      SymmetricTensor<2, dim> &stress_negative,
      SymmetricTensor<4, dim> &elasticity_tensor_positive,
      SymmetricTensor<4, dim> &elasticity_tensor_negative,
      ConstitutiveLaw<dim> &constitutive_law) override {
    // Hybrid formulation works pretty good for now.
    AssertThrow(false, ExcNotImplemented());
  }
};

template <int dim>
std::unique_ptr<Decomposition<dim>> select_decomposition(std::string method) {
  if (method == "none")
    return std::make_unique<Decomposition<dim>>();
  else if (method == "eigen")
    return std::make_unique<EigenDecomposition<dim>>();
  else if (method == "sphere")
    return std::make_unique<SphericalDecomposition<dim>>();
  else if (method == "hybrid")
    return std::make_unique<HybridEigenDecomposition<dim>>();
  else if (method == "hybridnotension")
    return std::make_unique<HybridNoTensionDecomposition<dim>>();
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_DECOMPOSITION_H
