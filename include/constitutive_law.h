//
// Created by xlluo on 24-7-29.
//

#ifndef CRACKS_CONSTITUTIVE_LAW_H
#define CRACKS_CONSTITUTIVE_LAW_H

#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

template <int dim> class ConstitutiveLaw {
public:
  ConstitutiveLaw(const double E, const double nu, std::string plane_state);

  void
  get_stress_strain_tensor(const Tensor<2, dim> &strain_tensor,
                           SymmetricTensor<2, dim> &E_symm,
                           SymmetricTensor<2, dim> &stress_tensor,
                           SymmetricTensor<4, dim> &elasticity_tensor) const;

  double E;
  double nu;
  double kappa;
  const double mu;
  double lambda;
  std::string plane_state;

private:
  SymmetricTensor<4, dim> stress_strain_tensor_kappa;
  SymmetricTensor<4, dim> stress_strain_tensor_mu;
};

template <int dim>
ConstitutiveLaw<dim>::ConstitutiveLaw(double E_in, double nu_in,
                                      std::string plane_state_in)
    : E(E_in), nu(nu_in), plane_state(plane_state_in), mu(E / (2 * (1 + nu))) {
  double kappa_3d = E / (3 * (1 - 2 * nu));
  lambda = E * nu / (1 + nu) / (1 - 2 * nu);
  if (dim == 3) {
    kappa = kappa_3d;
  } else if (dim == 2) {
    if (plane_state == "stress") {
      // https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
      lambda = 2 * lambda * mu / (lambda + 2 * mu);
      kappa = 9 * kappa_3d * mu /
              (3 * kappa_3d + 4 * mu); // kappa = E / (2 * (1 - nu))
    } else {
      kappa = kappa_3d + mu / 3;
    }
    //    kappa = E / (2 * (1 - nu));
  } else
    AssertThrow(false, ExcNotImplemented());
  stress_strain_tensor_kappa =
      kappa *
      outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());
  stress_strain_tensor_mu =
      2 * mu *
      (identity_tensor<dim>() - outer_product(unit_symmetric_tensor<dim>(),
                                              unit_symmetric_tensor<dim>()) /
                                    dim);
}

template <int dim>
void ConstitutiveLaw<dim>::get_stress_strain_tensor(
    const Tensor<2, dim> &strain_tensor, SymmetricTensor<2, dim> &E_symm,
    SymmetricTensor<2, dim> &stress_tensor,
    SymmetricTensor<4, dim> &elasticity_tensor) const {

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j <= i; ++j) {
      E_symm[i][j] = strain_tensor[i][j];
    }
  }
  elasticity_tensor = stress_strain_tensor_mu + stress_strain_tensor_kappa;
  stress_tensor = elasticity_tensor * E_symm;
}

#endif // CRACKS_CONSTITUTIVE_LAW_H
