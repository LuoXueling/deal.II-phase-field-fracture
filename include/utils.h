/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef SUPPORT_FUNCTIONS_H
#define SUPPORT_FUNCTIONS_H

#include "dealii_includes.h"
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

bool contains(const std::string str1, const std::string str2) {
  std::string::size_type idx = str1.find(str2);
  if (idx != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

// Define some tensors for cleaner notation later.
namespace Tensors {

template <int dim>
inline Tensor<1, dim> get_grad_pf(
    unsigned int q,
    const std::vector<std::vector<Tensor<1, dim>>> &old_solution_grads) {
  Tensor<1, dim> grad_pf;
  grad_pf[0] = old_solution_grads[q][dim][0];
  grad_pf[1] = old_solution_grads[q][dim][1];
  if (dim == 3)
    grad_pf[2] = old_solution_grads[q][dim][2];

  return grad_pf;
}

template <int dim>
inline Tensor<2, dim>
get_grad_u(unsigned int q,
           const std::vector<std::vector<Tensor<1, dim>>> &old_solution_grads) {
  Tensor<2, dim> grad_u;
  grad_u[0][0] = old_solution_grads[q][0][0];
  grad_u[0][1] = old_solution_grads[q][0][1];

  grad_u[1][0] = old_solution_grads[q][1][0];
  grad_u[1][1] = old_solution_grads[q][1][1];
  if (dim == 3) {
    grad_u[0][2] = old_solution_grads[q][0][2];

    grad_u[1][2] = old_solution_grads[q][1][2];

    grad_u[2][0] = old_solution_grads[q][2][0];
    grad_u[2][1] = old_solution_grads[q][2][1];
    grad_u[2][2] = old_solution_grads[q][2][2];
  }

  return grad_u;
}

template <int dim> inline SymmetricTensor<2, dim> get_Identity() {
  SymmetricTensor<2, dim> identity;
  identity[0][0] = 1.0;
  identity[1][1] = 1.0;
  if (dim == 3)
    identity[2][2] = 1.0;

  return identity;
}

template <int dim>
inline Tensor<1, dim>
get_u(unsigned int q, const std::vector<Vector<double>> &old_solution_values) {
  Tensor<1, dim> u;
  u[0] = old_solution_values[q](0);
  u[1] = old_solution_values[q](1);
  if (dim == 3)
    u[2] = old_solution_values[q](2);

  return u;
}

template <int dim>
inline Tensor<1, dim> get_u_LinU(const Tensor<1, dim> &phi_i_u) {
  Tensor<1, dim> tmp;
  tmp[0] = phi_i_u[0];
  tmp[1] = phi_i_u[1];
  if (dim == 3)
    tmp[2] = phi_i_u[2];
  return tmp;
}

template <int dim>
SymmetricTensor<4, dim> get_stress_strain_tensor(const double lambda,
                                                 const double mu) {
  SymmetricTensor<4, dim> tmp;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int k = 0; k < dim; ++k)
        for (unsigned int l = 0; l < dim; ++l)
          tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                             ((i == l) && (j == k) ? mu : 0.0) +
                             ((i == j) && (k == l) ? lambda : 0.0));
  return tmp;
}

template <int dim>
inline SymmetricTensor<2, dim> get_strain(const FEValues<dim> &fe_values,
                                          const unsigned int shape_func,
                                          const unsigned int q_point) {
  SymmetricTensor<2, dim> tmp;

  for (unsigned int i = 0; i < dim; ++i)
    tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      tmp[i][j] = (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
                   fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
                  2;

  return tmp;
}

template <int dim>
inline SymmetricTensor<2, dim>
get_strain(const std::vector<Tensor<1, dim>> &grad) {
  Assert(grad.size() == dim, ExcInternalError());

  SymmetricTensor<2, dim> strain;
  for (unsigned int i = 0; i < dim; ++i)
    strain[i][i] = grad[i][i];

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

  return strain;
}

template <int dim> inline double get_divergence_u(const Tensor<2, dim> grad_u) {
  double tmp;
  if (dim == 2) {
    tmp = grad_u[0][0] + grad_u[1][1];
  } else if (dim == 3) {
    tmp = grad_u[0][0] + grad_u[1][1] + grad_u[2][2];
  }

  return tmp;
}

template <int dim>
void to_voigt(const Tensor<2, dim> &in, Vector<double> &out) {
  for (unsigned int i_comp = 0; i_comp < dim; ++i_comp) {
    for (unsigned int j_comp = i_comp; j_comp < dim; ++j_comp) {
      if (i_comp == j_comp) {
        out[i_comp] = in[i_comp][j_comp];
      } else {
        out[dim - 1 + i_comp + j_comp] = in[i_comp][j_comp];
      }
    }
  }
}

template <int dim>
void tensor_product(Tensor<2, dim> &kronecker, const Tensor<1, dim> &x,
                    const Tensor<1, dim> &y) {
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      kronecker[i][j] = x[i] * y[j];
    }
  }
}

} // namespace Tensors

inline bool checkFileExsit(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

double triangular_wave(const double t, const double amplitude,
                       const double mean, const double frequency) {
  double T = 1 / frequency;
  double t_reduce = std::fmod(t, T);
  if (t_reduce <= 0.25 * T) {
    return mean + amplitude / (0.25 * T) * t_reduce;
  } else if (t_reduce <= 0.75 * T) {
    return mean + amplitude - amplitude / (0.25 * T) * (t_reduce - 0.25 * T);
  } else {
    return mean - amplitude + amplitude / (0.25 * T) * (t_reduce - 0.75 * T);
  }
}

double sine_wave(const double t, const double amplitude, const double mean,
                 const double frequency) {
  return amplitude * sin(2 * numbers::PI * frequency * t) + mean;
}

template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }

// https://stackoverflow.com/questions/13665090/trying-to-write-stdout-and-file-at-the-same-time
struct teebuf : std::streambuf {
  std::streambuf *sb1_;
  std::streambuf *sb2_;

  teebuf(std::streambuf *sb1, std::streambuf *sb2) : sb1_(sb1), sb2_(sb2) {}
  int overflow(int c) {
    typedef std::streambuf::traits_type traits;
    bool rc(true);
    if (!traits::eq_int_type(traits::eof(), c)) {
      traits::eq_int_type(this->sb1_->sputc(c), traits::eof()) && (rc = false);
      traits::eq_int_type(this->sb2_->sputc(c), traits::eof()) && (rc = false);
    }
    return rc ? traits::not_eof(c) : traits::eof();
  }
  int sync() {
    bool rc(true);
    this->sb1_->pubsync() != -1 || (rc = false);
    this->sb2_->pubsync() != -1 || (rc = false);
    return rc ? 0 : -1;
  }
};

class DebugConditionalOStream : public ConditionalOStream {
public:
  DebugConditionalOStream(std::ostream &stream, MPI_Comm *mpi_com,
                          const bool active = true)
      : ConditionalOStream(stream, active), my_mpi_com(mpi_com){};

  template <typename T>
  const DebugConditionalOStream &operator<<(const T &t) const;

  const DebugConditionalOStream &
  operator<<(std::ostream &(*p)(std::ostream &)) const;
  MPI_Comm *my_mpi_com;
};

template <class T>
inline const DebugConditionalOStream &
DebugConditionalOStream::operator<<(const T &t) const {
  if (is_active() == true) {
    get_stream() << std::to_string(
                        Utilities::MPI::this_mpi_process(*my_mpi_com)) +
                        ": ";
    get_stream() << t;
  }
  return *this;
}

inline const DebugConditionalOStream &
DebugConditionalOStream::operator<<(std::ostream &(*p)(std::ostream &)) const {
  if (is_active() == true) {
    get_stream() << p;
  }
  return *this;
}

double Mbracket(double x) { return (x + std::abs(x)) / 2; }

/**
 * PURPOSE:
 *
 *  Polynomial Regression aims to fit a non-linear relationship to a set of
 *  points. It approximates this by solving a series of linear equations using
 *  a least-squares approach.
 *
 *  We can model the expected value y as an nth degree polynomial, yielding
 *  the general polynomial regression model:
 *
 *  y = a0 + a1 * x + a2 * x^2 + ... + an * x^n
 *
 * LICENSE:
 *
 * MIT License
 *
 * Copyright (c) 2020 Chris Engelsma
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * @author Chris Engelsma
 */
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <stdexcept>

template <class TYPE>
class PolynomialRegression {
public:

  PolynomialRegression();
  virtual ~PolynomialRegression(){};

  bool fitIt(
      const std::vector<TYPE> & x,
      const std::vector<TYPE> & y,
      const int &             order,
      std::vector<TYPE> &     coeffs);
};

template <class TYPE>
PolynomialRegression<TYPE>::PolynomialRegression() {};

template <class TYPE>
bool PolynomialRegression<TYPE>::fitIt(
    const std::vector<TYPE> & x,
    const std::vector<TYPE> & y,
    const int &               order,
    std::vector<TYPE> &       coeffs)
{
  // The size of xValues and yValues should be same
  if (x.size() != y.size()) {
    throw std::runtime_error( "The size of x & y arrays are different" );
    return false;
  }
  // The size of xValues and yValues cannot be 0, should not happen
  if (x.size() == 0 || y.size() == 0) {
    throw std::runtime_error( "The size of x or y arrays is 0" );
    return false;
  }

  size_t N = x.size();
  int n = order;
  int np1 = n + 1;
  int np2 = n + 2;
  int tnp1 = 2 * n + 1;
  TYPE tmp;

  // X = vector that stores values of sigma(xi^2n)
  std::vector<TYPE> X(tnp1);
  for (int i = 0; i < tnp1; ++i) {
    X[i] = 0;
    for (int j = 0; j < N; ++j)
      X[i] += (TYPE)pow(x[j], i);
  }

  // a = vector to store final coefficients.
  std::vector<TYPE> a(np1);

  // B = normal augmented matrix that stores the equations.
  std::vector<std::vector<TYPE> > B(np1, std::vector<TYPE> (np2, 0));

  for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j)
      B[i][j] = X[i + j];

  // Y = vector to store values of sigma(xi^n * yi)
  std::vector<TYPE> Y(np1);
  for (int i = 0; i < np1; ++i) {
    Y[i] = (TYPE)0;
    for (int j = 0; j < N; ++j) {
      Y[i] += (TYPE)pow(x[j], i)*y[j];
    }
  }

  // Load values of Y as last column of B
  for (int i = 0; i <= n; ++i)
    B[i][np1] = Y[i];

  n += 1;
  int nm1 = n-1;

  // Pivotisation of the B matrix.
  for (int i = 0; i < n; ++i)
    for (int k = i+1; k < n; ++k)
      if (B[i][i] < B[k][i])
        for (int j = 0; j <= n; ++j) {
          tmp = B[i][j];
          B[i][j] = B[k][j];
          B[k][j] = tmp;
        }

  // Performs the Gaussian elimination.
  // (1) Make all elements below the pivot equals to zero
  //     or eliminate the variable.
  for (int i=0; i<nm1; ++i)
    for (int k =i+1; k<n; ++k) {
      TYPE t = B[k][i] / B[i][i];
      for (int j=0; j<=n; ++j)
        B[k][j] -= t*B[i][j];         // (1)
    }

  // Back substitution.
  // (1) Set the variable as the rhs of last equation
  // (2) Subtract all lhs values except the target coefficient.
  // (3) Divide rhs by coefficient of variable being calculated.
  for (int i=nm1; i >= 0; --i) {
    a[i] = B[i][n];                   // (1)
    for (int j = 0; j<n; ++j)
      if (j != i)
        a[i] -= B[i][j] * a[j];       // (2)
    a[i] /= B[i][i];                  // (3)
  }

  coeffs.resize(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    coeffs[i] = a[i];

  return true;
}

double get_norm(LA::MPI::BlockVector &x, std::string norm_type){
  double res = 0.0;
  if (norm_type == "l2"){
    res = x.l2_norm();
  } else if (norm_type == "l1"){
    res = x.l1_norm();
  } else if (norm_type == "linfty"){
    res = x.linfty_norm();
  } else {
    AssertThrow(false, ExcNotImplemented("Norm type not supported"));
  }
  return res;
}
#endif