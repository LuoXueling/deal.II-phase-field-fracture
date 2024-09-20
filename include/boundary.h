//
// Created by xlluo on 24-7-28.
//

#ifndef CRACKS_BOUNDARY_H
#define CRACKS_BOUNDARY_H

#include "dealii_includes.h"
#include "utils.h"
#include <cmath>

// Can only use apply AbstractBoundary condition on one dof
template <int dim> class AbstractBoundary : public Function<dim> {
public:
  AbstractBoundary(double present_time_inp, unsigned int n_components_inp);

  virtual double value(const Point<dim> &p,
                       unsigned int component) const override {
    return 0.0;
  };

  void vector_value(const Point<dim> &p, Vector<double> &values) const override;

  void
  vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>> &value_list) const override;

  const unsigned int n_components;
  const double present_time;
};

template <int dim>
AbstractBoundary<dim>::AbstractBoundary(const double present_time_inp,
                                        unsigned int n_components_inp)
    : Function<dim>(n_components_inp), present_time(present_time_inp),
      n_components(n_components_inp) {}

template <int dim>
void AbstractBoundary<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values[c] = value(p, c);
}

template <int dim>
void AbstractBoundary<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &value_list) const {
  const unsigned int n_points = points.size();

  AssertDimension(value_list.size(), n_points);

  for (unsigned int p = 0; p < n_points; ++p)
    vector_value(points[p], value_list[p]);
}

template <int dim>
class GeneralDirichletBoundary : public AbstractBoundary<dim> {
public:
  GeneralDirichletBoundary(double present_time_inp, double val,
                           unsigned int n_components_inp)
      : AbstractBoundary<dim>(present_time_inp, n_components_inp),
        constraint_value(val){};

  double value(const Point<dim> &p, unsigned int component) const override {
    return this->constraint_value;
  };

  double set_constraint(double val) { constraint_value = val; }

private:
  double constraint_value;
};

template <int dim>
class VelocityBoundary : public GeneralDirichletBoundary<dim> {
public:
  VelocityBoundary(double present_time_inp, double velocity_inp,
                   unsigned int n_components_inp)
      : GeneralDirichletBoundary<dim>(present_time_inp,
                                      velocity_inp * present_time_inp,
                                      n_components_inp){};
};

template <int dim>
class TriangularWaveDirichletBoundary : public GeneralDirichletBoundary<dim> {
public:
  TriangularWaveDirichletBoundary(double present_time_inp,
                                  std::vector<double> &frequency_mean_amplitude,
                                  unsigned int n_components_inp)
      : GeneralDirichletBoundary<dim>(
            present_time_inp,
            triangular_wave(
                present_time_inp - ((frequency_mean_amplitude.size() == 4)
                                        ? frequency_mean_amplitude[3]
                                        : 0.0),
                frequency_mean_amplitude[2], frequency_mean_amplitude[1],
                frequency_mean_amplitude[0]),
            n_components_inp){};
};

template <int dim>
class SineWaveDirichletBoundary : public GeneralDirichletBoundary<dim> {
public:
  SineWaveDirichletBoundary(double present_time_inp,
                            std::vector<double> &frequency_mean_amplitude,
                            unsigned int n_components_inp)
      : GeneralDirichletBoundary<dim>(
            present_time_inp,
            sine_wave(present_time_inp - ((frequency_mean_amplitude.size() == 4)
                                              ? frequency_mean_amplitude[3]
                                              : 0.0),
                      frequency_mean_amplitude[2], frequency_mean_amplitude[1],
                      frequency_mean_amplitude[0]),
            n_components_inp){};
};

template <int dim>
std::unique_ptr<Function<dim>>
select_dirichlet_boundary(std::tuple<unsigned int, std::string, unsigned int,
                                     double, std::vector<double>>
                              dirichlet_info,
                          unsigned int n_components, double time) {
  std::string boundary_type = std::get<1>(dirichlet_info);
  double constraint_value = std::get<3>(dirichlet_info);
  std::vector<double> additional_info = std::get<4>(dirichlet_info);
  if (boundary_type == "velocity") {
    return std::make_unique<VelocityBoundary<dim>>(time, constraint_value,
                                                   n_components);
  } else if (boundary_type == "dirichlet") {
    return std::make_unique<GeneralDirichletBoundary<dim>>(
        time, constraint_value, n_components);
  } else if (boundary_type == "triangulardirichlet") {
    return std::make_unique<TriangularWaveDirichletBoundary<dim>>(
        time, additional_info, n_components);
  } else if (boundary_type == "sinedirichlet") {
    return std::make_unique<SineWaveDirichletBoundary<dim>>(
        time, additional_info, n_components);
  } else {
    AssertThrow(false, ExcNotImplemented());
  }
}

template <int dim> class GeneralNeumannBoundary : public AbstractBoundary<dim> {
public:
  GeneralNeumannBoundary(double present_time_inp,
                         std::vector<double> &constraint_vector_inp,
                         unsigned int n_components_inp)
      : AbstractBoundary<dim>(present_time_inp, n_components_inp),
        constraint_vector(constraint_vector_inp),
        n_components(n_components_inp){};

  double value(const Point<dim> &p, unsigned int component) const override {
    return constraint_vector[component];
  };

  void multiply(double x) {
    for (unsigned int i = 0; i < constraint_vector.size(); ++i) {
      constraint_vector[i] *= x;
    }
  }

private:
  std::vector<double> constraint_vector;
  const unsigned int n_components;
};

template <int dim>
class NeumannRateBoundary : public GeneralNeumannBoundary<dim> {
public:
  NeumannRateBoundary(double present_time_inp,
                      std::vector<double> constraint_vector_inp,
                      unsigned int n_components_inp)
      : GeneralNeumannBoundary<dim>(present_time_inp, constraint_vector_inp,
                                    n_components_inp) {
    this->multiply(present_time_inp);
  };
};

template <int dim>
class TriangularWaveNeumannBoundary : public GeneralNeumannBoundary<dim> {
public:
  TriangularWaveNeumannBoundary(double present_time_inp,
                                std::vector<double> &constraint_vector_inp,
                                std::vector<double> &frequency_mean_amplitude,
                                unsigned int n_components_inp)
      : GeneralNeumannBoundary<dim>(present_time_inp, constraint_vector_inp,
                                    n_components_inp) {
    double phase = (frequency_mean_amplitude.size() == 4)
                       ? frequency_mean_amplitude[3]
                       : 0.0;
    double multiplier = triangular_wave(
        present_time_inp - phase, frequency_mean_amplitude[2],
        frequency_mean_amplitude[1], frequency_mean_amplitude[0]);
    ;
    this->multiply(multiplier);
  };
};

template <int dim>
class SineWaveNeumannBoundary : public GeneralNeumannBoundary<dim> {
public:
  SineWaveNeumannBoundary(double present_time_inp,
                          std::vector<double> &constraint_vector_inp,
                          std::vector<double> &frequency_mean_amplitude,
                          unsigned int n_components_inp)
      : GeneralNeumannBoundary<dim>(present_time_inp, constraint_vector_inp,
                                    n_components_inp) {
    double phase = (frequency_mean_amplitude.size() == 4)
                       ? frequency_mean_amplitude[3]
                       : 0.0;
    double multiplier =
        sine_wave(present_time_inp - phase, frequency_mean_amplitude[2],
                  frequency_mean_amplitude[1], frequency_mean_amplitude[0]);
    ;
    this->multiply(multiplier);
  };
};

template <int dim>
std::unique_ptr<GeneralNeumannBoundary<dim>>
select_neumann_boundary(std::tuple<unsigned int, std::string,
                                   std::vector<double>, std::vector<double>>
                            neumann_info,
                        unsigned int n_components, double time) {
  std::string boundary_type = std::get<1>(neumann_info);
  std::vector<double> vector = std::get<2>(neumann_info);
  std::vector<double> additional_info = std::get<3>(neumann_info);
  if (boundary_type == "neumann") {
    return std::make_unique<GeneralNeumannBoundary<dim>>(time, vector,
                                                         n_components);
  } else if (boundary_type == "neumannrate") {
    return std::make_unique<NeumannRateBoundary<dim>>(time, vector,
                                                      n_components);
  } else if (boundary_type == "sineneumann") {
    return std::make_unique<SineWaveNeumannBoundary<dim>>(
        time, vector, additional_info, n_components);
  } else if (boundary_type == "triangularneumann") {
    return std::make_unique<TriangularWaveNeumannBoundary<dim>>(
        time, vector, additional_info, n_components);
  } else {
    AssertThrow(false, ExcNotImplemented());
  }
}

#endif // CRACKS_DIRICHLET_BOUNDARY_H
