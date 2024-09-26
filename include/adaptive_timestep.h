//
// Created by xlluo on 24-8-4.
//

#ifndef CRACKS_ADAPTIVE_TIMESTEP_H
#define CRACKS_ADAPTIVE_TIMESTEP_H

#include "controller.h"
#include "utils.h"
#include <fstream>
#include <iostream>

template <int dim> class AdaptiveTimeStep {
public:
  AdaptiveTimeStep(Controller<dim> &ctl)
      : last_time(0), count_reduction(0), save_results(false),
        new_timestep(ctl.current_timestep){};

  virtual void initialize_timestep(Controller<dim> &ctl) {};

  double get_timestep(Controller<dim> &ctl) {
    last_time = ctl.time;
    new_timestep = current_timestep(ctl);
    ctl.dt = new_timestep;
    return new_timestep;
  }

  virtual double current_timestep(Controller<dim> &ctl) {
    return ctl.current_timestep;
  }

  void execute_when_fail(Controller<dim> &ctl) {
    ctl.time = last_time;
    check_time(ctl);
    new_timestep = get_new_timestep_when_fail(ctl);
    failure_criteria(ctl);
    record(ctl);
    ctl.time += new_timestep;
    ctl.dt = new_timestep;
  }

  virtual bool fail(double newton_reduction, Controller<dim> &ctl) {
    return newton_reduction > ctl.params.upper_newton_rho;
  }

  virtual void after_step(Controller<dim> &ctl) {}

  virtual double get_new_timestep_when_fail(Controller<dim> &ctl) {
    return new_timestep * 0.1;
  }

  void check_time(Controller<dim> &ctl) {
    if (ctl.time != last_time) {
      last_time = ctl.time;
      count_reduction = 0;
      historical_timesteps.push_back(new_timestep);
    }
  }

  virtual bool save_checkpoint(Controller<dim> &ctl) { return false; }

  virtual std::string return_solution_or_checkpoint(Controller<dim> &ctl) {
    return "solution";
  }

  void record(Controller<dim> &ctl) { count_reduction++; }

  virtual void failure_criteria(Controller<dim> &ctl) {
    if (new_timestep < ctl.params.timestep * 1e-8) {
      AssertThrow(false, ExcInternalError("Step size too small"))
    }
  }
  virtual bool terminate(Controller<dim> &ctl) { return false; }

  std::vector<double> historical_timesteps;
  double last_time;
  int count_reduction;
  bool save_results;
  double new_timestep;
};

template <int dim> class ConstantTimeStep : public AdaptiveTimeStep<dim> {
public:
  ConstantTimeStep(Controller<dim> &ctl) : AdaptiveTimeStep<dim>(ctl){};
  void failure_criteria(Controller<dim> &ctl) override {
    throw std::runtime_error(
        "Staggered scheme does not converge, and ConstantTimeStep "
        "does not allow adaptive time stepping");
  }
};

template <int dim>
std::unique_ptr<AdaptiveTimeStep<dim>>
select_adaptive_timestep(std::string method, Controller<dim> &ctl) {
  if (method == "constant")
    return std::make_unique<ConstantTimeStep<dim>>(ctl);
  else if (method == "exponential")
    return std::make_unique<AdaptiveTimeStep<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_ADAPTIVE_TIMESTEP_H
