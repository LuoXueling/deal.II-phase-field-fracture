//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_CONTROLLER_H
#define CRACKS_CONTROLLER_H

#include "dealii_includes.h"
#include "parameters.h"
#include "utils.h"

class PointHistory : public TransferableQuadraturePointData {
public:
  void update(std::string name, double solution,
              std::string scheme = "latest") {
    _update(name, solution, solution_buffer, solution_old,
            solution_increment_buffer, scheme);
  };
  void update_independent(std::string name, double solution,
                          std::string scheme = "latest") {
    _update(name, solution, solution_independent_buffer,
            solution_independent_old, solution_independent_increment_buffer,
            scheme);
  };
  void _update(std::string name, double solution,
               std::map<std::string, double> &dict,
               std::map<std::string, double> &old_dict,
               std::map<std::string, double> &increment_dict,
               std::string scheme = "latest") {
    double res;
    if (scheme == "latest") {
      res = solution;
    } else if (scheme == "max") {
      res = std::max(solution, _get(name, old_dict, -1.0e20));
    } else if (scheme == "min") {
      res = std::min(solution, _get(name, old_dict, 1.0e20));
    } else if (scheme == "accumulate") {
      res = solution + _get(name, old_dict, 0.0);
    } else if (scheme == "multiplicative") {
      res = solution * _get(name, old_dict, 1.0);
    } else {
      AssertThrow(false,
                  ExcNotImplemented("Point history update scheme is illegal."));
    }
    increment_dict[name] = res - _get(name, old_dict, 0.0);
    dict[name] = res;
    finalize_scheme[name] = scheme;
  }
  double _get(std::string name, const std::map<std::string, double> &dict,
              double default_value = 0.0) const {
    // This function has to be const for pack_values so we cannot use
    // solution_dict[name]
    try {
      return _get_from_one(name, dict);
    } catch (...) {
      return default_value;
    }
  };
  double _get_either(std::string name,
                     const std::map<std::string, double> &dict1,
                     const std::map<std::string, double> &dict2,
                     double default_value = 0.0) const {
    // This function has to be const for pack_values so we cannot use
    // solution_dict[name]
    try {
      try {
        return _get_from_one(name, dict1);
      } catch (...) {
        return _get_from_one(name, dict2);
      }
    } catch (...) {
      return default_value;
    }
  };
  double _get_from_one(std::string name,
                       const std::map<std::string, double> &dict) const {
    // This function has to be const for pack_values so we cannot use
    // solution_dict[name]
    try {
      auto pos = dict.find(name);
      if (pos == dict.end()) {
        throw std::runtime_error("");
      } else
        return pos->second;
    } catch (...) {
      throw std::runtime_error("");
    }
  };
  double get_latest(std::string name, double default_value = 0.0) const {
    return _get(name, solution_buffer, default_value);
  };
  double get_initial(std::string name, double default_value = 0.0) const {
    return _get(name, solution_old, default_value);
  };
  double get_increment_latest(std::string name,
                              double default_value = 0.0) const {
    return _get(name, solution_increment_buffer, default_value);
  };
  double get_increment_initial(std::string name,
                               double default_value = 0.0) const {
    return _get(name, solution_increment_old, default_value);
  };
  double get_independent_latest(std::string name,
                                double default_value = 0.0) const {
    return _get(name, solution_independent_buffer, default_value);
  };
  double get_independent_initial(std::string name,
                                 double default_value = 0.0) const {
    return _get(name, solution_independent_old, default_value);
  };
  double get_independent_increment_latest(std::string name,
                                          double default_value = 0.0) const {
    return _get(name, solution_independent_increment_buffer, default_value);
  };
  double get_independent_increment_initial(std::string name,
                                           double default_value = 0.0) const {
    return _get(name, solution_independent_increment_old, default_value);
  };
  double get_either_latest(std::string name, double default_value = 0.0) const {
    return _get_either(name, solution_buffer, solution_independent_buffer,
                       default_value);
  };
  double get_either_initial(std::string name,
                            double default_value = 0.0) const {
    return _get_either(name, solution_old, solution_independent_old,
                       default_value);
  };
  double get_either_increment_latest(std::string name,
                                     double default_value = 0.0) const {
    return _get_either(name, solution_increment_buffer,
                       solution_independent_increment_buffer, default_value);
  };
  double get_either_increment_initial(std::string name,
                                      double default_value = 0.0) const {
    return _get_either(name, solution_increment_old,
                       solution_independent_increment_old, default_value);
  };
  void finalize() {
    typename std::map<std::string, double>::iterator it;
    for (it = solution_buffer.begin(); it != solution_buffer.end(); it++) {
      solution_old[it->first] = it->second;
    }
    for (it = solution_increment_buffer.begin();
         it != solution_increment_buffer.end(); it++) {
      solution_increment_old[it->first] = it->second;
    }
    for (it = solution_independent_buffer.begin();
         it != solution_independent_buffer.end(); it++) {
      solution_independent_old[it->first] = it->second;
    }
    for (it = solution_independent_increment_buffer.begin();
         it != solution_independent_increment_buffer.end(); it++) {
      solution_independent_increment_old[it->first] = it->second;
    }
  }

  unsigned int number_of_values() const override {
    return finalize_scheme.size() * 2;
  }

  void pack_values(std::vector<double> &values) const override {
    Assert(values.size() == finalize_scheme.size() * 2, ExcInternalError());
    std::vector<std::string> names = get_names();
    for (unsigned int i = 0; i < finalize_scheme.size() * 2; ++i) {
      values[i] = i < finalize_scheme.size()
                      ? get_either_initial(names[i], 0.0)
                      : get_either_increment_initial(
                            names[i - finalize_scheme.size()], 0.0);
    }
  }

  void unpack_values(const std::vector<double> &values) override {
    Assert(values.size() == finalize_scheme.size() * 2, ExcInternalError());
    std::vector<std::string> names = get_names();
    for (unsigned int i = 0; i < finalize_scheme.size() * 2; ++i) {
      if (i < finalize_scheme.size()) {
        if (solution_buffer.find(names[i]) != solution_buffer.end()) {
          solution_buffer[names[i]] = values[i];
          solution_old[names[i]] = values[i];
        } else {
          solution_independent_buffer[names[i]] = values[i];
          solution_independent_old[names[i]] = values[i];
        }
      } else {
        if (solution_buffer.find(names[i - finalize_scheme.size()]) !=
            solution_buffer.end()) {
          solution_increment_buffer[names[i - finalize_scheme.size()]] =
              values[i];
          solution_increment_old[names[i - finalize_scheme.size()]] = values[i];
        } else {
          solution_independent_increment_buffer[names[i -
                                                      finalize_scheme.size()]] =
              values[i];
          solution_independent_increment_old[names[i -
                                                   finalize_scheme.size()]] =
              values[i];
        }
      }
    }
  }

  std::vector<std::string> get_names() const {
    std::vector<std::string> names;
    for (auto it = finalize_scheme.begin(); it != finalize_scheme.end(); ++it) {
      names.push_back(it->first);
    }
    return names;
  }

  std::map<std::string, double> solution_buffer;
  std::map<std::string, double> solution_old;
  std::map<std::string, double> solution_increment_buffer;
  std::map<std::string, double> solution_increment_old;
  // The following two are not recorded in checkpoints
  std::map<std::string, double> solution_independent_buffer;
  std::map<std::string, double> solution_independent_old;
  std::map<std::string, double> solution_independent_increment_buffer;
  std::map<std::string, double> solution_independent_increment_old;

  inline static std::map<std::string, std::string> finalize_scheme;
};

template <int dim> class Controller {
public:
  explicit Controller(Parameters::AllParameters &prms);

  void finalize_point_history();
  void initialize_point_history();
  void record_point_history(
      CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory>
          &src,
      CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory>
          &dst);
  double get_info(std::string name, double default_value);
  void set_info(std::string name, double value);

  MPI_Comm mpi_com;

  parallel::distributed::Triangulation<dim> triangulation;
  QGaussLobatto<dim> quadrature_formula;
  Parameters::AllParameters params;

  ConditionalOStream dcout;
  DebugConditionalOStream debug_dcout;
  TimerOutput timer;
  TimerOutput computing_timer;
  std::ofstream fout;
  teebuf sbuf;
  std::ostream pout;

  double time;
  unsigned int timestep_number;
  int output_timestep_number;
  int last_refinement_timestep_number;
  double current_timestep;
  double old_timestep;

  TableHandler statistics;

  std::vector<int> boundary_ids;

  CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory>
      quadrature_point_history, old_quadrature_point_history,
      quadrature_point_history_checkpoint;

  std::map<std::string, double> info_center;
};

template <int dim>
Controller<dim>::Controller(Parameters::AllParameters &prms)
    : mpi_com(MPI_COMM_WORLD), params(prms),
      triangulation(mpi_com, typename Triangulation<dim>::MeshSmoothing(
                                 Triangulation<dim>::smoothing_on_refinement |
                                 Triangulation<dim>::smoothing_on_coarsening)),
      quadrature_formula(prms.poly_degree + 1),
      fout(prms.output_dir + "log.txt"), sbuf(fout.rdbuf(), std::cout.rdbuf()),
      pout(&sbuf),
      dcout(pout, (Utilities::MPI::this_mpi_process(mpi_com) == 0)),
      debug_dcout(std::cout, &mpi_com, prms.debug_output),
      timer(mpi_com, dcout, TimerOutput::never,
            TimerOutput::cpu_and_wall_times),
      computing_timer(mpi_com, dcout, TimerOutput::never,
                      TimerOutput::wall_times),
      time(0), timestep_number(0), output_timestep_number(0),
      current_timestep(0), old_timestep(0),
      last_refinement_timestep_number(-1) {
  statistics.set_auto_fill_mode(true);
}

template <int dim> void Controller<dim>::initialize_point_history() {
  // The original CellDataStorage.initialize use tria.begin_active() and
  // tria.end() and does not really loop over locally-owned cells
  // https://github.com/rezarastak/dealii/blob/381a8d3739e10a450b7efeb62fd2f74add7ee19c/tests/base/quadrature_point_data_04.cc#L101
  for (auto cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned()) {
      quadrature_point_history.template initialize<PointHistory>(
          cell, quadrature_formula.size());
      old_quadrature_point_history.template initialize<PointHistory>(
          cell, quadrature_formula.size());
      quadrature_point_history_checkpoint.template initialize<PointHistory>(
          cell, quadrature_formula.size());
    }
}

template <int dim> void Controller<dim>::finalize_point_history() {
  const unsigned int n_q_points = quadrature_formula.size();
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned()) {
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        lqph[q]->finalize();
      }
    }
}

template <int dim>
void Controller<dim>::record_point_history(
    CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory>
        &src,
    CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory>
        &dst) {
  const unsigned int n_q_points = quadrature_formula.size();
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned()) {
      const std::vector<std::shared_ptr<PointHistory>> lqph_src =
          src.get_data(cell);
      const std::vector<std::shared_ptr<PointHistory>> lqph_dst =
          dst.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        lqph_dst[q]->solution_buffer = lqph_src[q]->solution_buffer;
        lqph_dst[q]->solution_old = lqph_src[q]->solution_old;
        lqph_dst[q]->solution_increment_buffer =
            lqph_src[q]->solution_increment_buffer;
        lqph_dst[q]->solution_increment_old =
            lqph_src[q]->solution_increment_old;
        lqph_dst[q]->finalize_scheme = lqph_src[q]->finalize_scheme;
      }
    }
}

template <int dim>
double Controller<dim>::get_info(std::string name, double default_value) {
  try {
    auto pos = info_center.find(name);
    if (pos == info_center.end()) {
      return default_value;
    } else
      return pos->second;
  } catch (...) {
    return default_value;
  }
};

template <int dim>
void Controller<dim>::set_info(std::string name, double value) {
  info_center[name] = value;
}

#endif // CRACKS_CONTROLLER_H
