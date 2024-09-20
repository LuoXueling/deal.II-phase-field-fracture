//
// Created by xlluo on 24-8-4.
//

#ifndef CRACKS_DEGRADATION_H
#define CRACKS_DEGRADATION_H

#include "controller.h"
#include "dealii_includes.h"
using namespace dealii;

template <int dim> class Degradation {
public:
  Degradation() = default;
  virtual double value(double phi, Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented());
  };
  virtual double derivative(double phi, Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented());
  };
  virtual double second_derivative(double phi, Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented());
  };
};

template <int dim> class QuadraticDegradation : public Degradation<dim> {
public:
  double value(double phi, Controller<dim> &ctl) override {
    return pow(1 - phi, 2) + ctl.params.constant_k;
  };
  double derivative(double phi, Controller<dim> &ctl) override {
    return -2 * (1 - phi);
  };
  double second_derivative(double phi, Controller<dim> &ctl) override {
    return 2.0;
  };
};

template <int dim> class CubicDegradation : public Degradation<dim> {
public:
  double value(double phi, Controller<dim> &ctl) override {
    return pow(1 - phi, 3) + ctl.params.constant_k;
  };
  double derivative(double phi, Controller<dim> &ctl) override {
    return -3 * pow(1 - phi, 2);
  };
  double second_derivative(double phi, Controller<dim> &ctl) override {
    return 6 * (1 - phi);
  };
};

template <int dim>
std::unique_ptr<Degradation<dim>> select_degradation(std::string method) {
  if (method == "quadratic")
    return std::make_unique<QuadraticDegradation<dim>>();
  else if (method == "cubic")
    return std::make_unique<CubicDegradation<dim>>();
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_DEGRADATION_H
