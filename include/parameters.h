/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "dealii_includes.h"
#include <ctime>

namespace Parameters {
struct Project {
  std::string mesh_from;
  std::string boundary_from;
  std::string project_name;
  std::string output_dir_top;
  std::string load_sequence_from;
  bool enable_phase_field;
  bool debug_output;

  static void subsection_declare_parameters(ParameterHandler &prm);

  void subsection_parse_parameters(ParameterHandler &prm);
};

void Project::subsection_declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Project");
  {
    prm.declare_entry("Mesh from", "script",
                      Patterns::FileName(Patterns::FileName::FileType::input));
    prm.declare_entry("Boundary from", "none",
                      Patterns::FileName(Patterns::FileName::FileType::input));
    prm.declare_entry("Project name", "Default project",
                      Patterns::FileName(Patterns::FileName::FileType::output));
    prm.declare_entry("Output directory", "../output/",
                      Patterns::FileName(Patterns::FileName::FileType::input));
    prm.declare_entry("Load sequence from", "script",
                      Patterns::FileName(Patterns::FileName::FileType::input));
    prm.declare_entry("Enable phase field", "true", Patterns::Bool());

    prm.declare_entry("Debug output", "false", Patterns::Bool());
  }
  prm.leave_subsection();
}

void Project::subsection_parse_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Project");
  {
    mesh_from = prm.get("Mesh from");
    boundary_from = prm.get("Boundary from");
    project_name = prm.get("Project name");
    output_dir_top = prm.get("Output directory");
    load_sequence_from = prm.get("Load sequence from");
    enable_phase_field = prm.get_bool("Enable phase field");
    debug_output = prm.get_bool("Debug output");
  }
  prm.leave_subsection();
}

struct Runtime {
  unsigned int max_no_timesteps;
  std::string adaptive_timestep;
  std::string adaptive_timestep_parameters;
  double timestep;
  double timestep_size_2;
  unsigned int switch_timestep;
  std::string norm_type;
  bool direct_solver;
  double lower_bound_newton_residual;
  unsigned int max_no_newton_steps;
  bool skip_first_iter;
  double upper_newton_rho;
  std::string adjustment_method;
  std::string adjustment_method_elasticity;
  std::string linesearch_parameters;
  std::string modified_newton_parameters;
  unsigned int max_adjustment_steps;
  bool is_monolithic;
  bool multipass_staggered;
  unsigned int max_multipass;
  double multipass_residual_tol;
  bool quit_multipass_if_increase;
  bool throw_if_multipass_increase;
  std::string phase_field_scheme;
  std::string decomposition;
  double constant_k;
  unsigned int save_vtk_per_step;

  static void subsection_declare_parameters(ParameterHandler &prm);

  void subsection_parse_parameters(ParameterHandler &prm);
};

void Runtime::subsection_declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Runtime");
  {
    prm.declare_entry("Max No of timesteps", "1", Patterns::Integer(0));

    prm.declare_entry(
        "Adaptive timestep", "exponential",
        Patterns::Selection(
            "exponential|constant|KristensenCLA|"
            "CojocaruCycleJump|JonasCycleJump|YangCycleJump|JacconCycleJump"));
    prm.declare_entry("Adaptive timestep parameters", "", Patterns::Anything());
    prm.declare_entry("Timestep size", "1.0", Patterns::Double(0));

    prm.declare_entry("Timestep size to switch to", "1.0", Patterns::Double(0));

    prm.declare_entry("Switch timestep after steps", "0", Patterns::Integer(0));
    prm.declare_entry("Use Direct Inner Solver", "false", Patterns::Bool());

    prm.declare_entry("Norm type", "linfty",
                      Patterns::Selection("linfty|l2|l1"));

    prm.declare_entry("Newton lower bound", "1.0e-10", Patterns::Double(0));

    prm.declare_entry("Newton maximum steps", "10", Patterns::Integer(0));

    prm.declare_entry("Upper Newton rho", "0.999", Patterns::Double(0));
    prm.declare_entry("Allow skip first Newton iteration", "true",
                      Patterns::Bool());

    prm.declare_entry(
        "Adjustment method", "linesearch",
        Patterns::Selection("none|linesearch|KristensenModifiedNewton"));

    prm.declare_entry(
        "Adjustment method for elasticity", "linesearch",
        Patterns::Selection(
            "none|linesearch|AndersonNewton|KristensenModifiedNewton"));

    prm.declare_entry("Parameters of line search", "0.1", Patterns::Anything());
    prm.declare_entry("Parameters of modified newton", "",
                      Patterns::Anything());

    prm.declare_entry("Maximum number of adjustment steps of Newton solution",
                      "5", Patterns::Integer(0));

    prm.declare_entry("Line search damping", "0.5", Patterns::Double(0));

    prm.declare_entry("Use monolithic", "false", Patterns::Bool());

    prm.declare_entry("Use multipass staggered", "false", Patterns::Bool());
    prm.declare_entry("Maximum number of multipass steps", "5",
                      Patterns::Integer(0));
    prm.declare_entry("Residual tolerance of multipass", "1e-8",
                      Patterns::Double(0));
    prm.declare_entry("Quit multipass if residual increasing", "true",
                      Patterns::Bool());
    prm.declare_entry("Throw if multipass residual increasing", "false",
                      Patterns::Bool());

    prm.declare_entry("Phase field update", "newton",
                      Patterns::Selection("newton|linear"));

    prm.declare_entry(
        "Decomposition", "hybrid",
        Patterns::Selection("none|hybrid|sphere|eigen|hybridnotension"));

    prm.declare_entry("Constant small quantity k", "1.0e-6",
                      Patterns::Double(0));
    prm.declare_entry("Save vtk per step", "1", Patterns::Integer(0));
  }
  prm.leave_subsection();
}

void Runtime::subsection_parse_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Runtime");
  {
    max_no_timesteps = prm.get_integer("Max No of timesteps");
    adaptive_timestep = prm.get("Adaptive timestep");
    adaptive_timestep_parameters = prm.get("Adaptive timestep parameters");
    timestep = prm.get_double("Timestep size");
    timestep_size_2 = prm.get_double("Timestep size to switch to");
    switch_timestep = prm.get_integer("Switch timestep after steps");

    direct_solver = prm.get_bool("Use Direct Inner Solver");

    norm_type = prm.get("Norm type");

    // Newton tolerances and maximum steps
    lower_bound_newton_residual = prm.get_double("Newton lower bound");
    max_no_newton_steps = prm.get_integer("Newton maximum steps");

    skip_first_iter = prm.get_bool("Allow skip first Newton iteration");

    // Criterion when time step should be cut
    // Higher number means: almost never
    // only used for simple penalization
    upper_newton_rho = prm.get_double("Upper Newton rho");

    adjustment_method = prm.get("Adjustment method");
    adjustment_method_elasticity = prm.get("Adjustment method for elasticity");
    max_adjustment_steps = prm.get_integer(
        "Maximum number of adjustment steps of Newton solution");
    linesearch_parameters = prm.get("Parameters of line search");
    modified_newton_parameters = prm.get("Parameters of modified newton");

    is_monolithic = prm.get_bool("Use monolithic");

    multipass_staggered = prm.get_bool("Use multipass staggered");
    max_multipass = prm.get_integer("Maximum number of multipass steps");
    multipass_residual_tol = prm.get_double("Residual tolerance of multipass");
    quit_multipass_if_increase =
        prm.get_bool("Quit multipass if residual increasing");
    throw_if_multipass_increase =
        prm.get_bool("Throw if multipass residual increasing");

    phase_field_scheme = prm.get("Phase field update");
    decomposition = prm.get("Decomposition");

    constant_k = prm.get_double("Constant small quantity k");

    save_vtk_per_step = prm.get_integer("Save vtk per step");
  }
  prm.leave_subsection();
}

struct Material {
  double E;
  double v;
  double Gc;
  double l_phi;
  std::string plane_state;
  std::string phasefield_model;
  std::string degradation;

  static void subsection_declare_parameters(ParameterHandler &prm);

  void subsection_parse_parameters(ParameterHandler &prm);
};

void Material::subsection_declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Material");
  {
    prm.declare_entry("Young's modulus", "1000", Patterns::Double(0));
    prm.declare_entry("Poisson's ratio", "0.3", Patterns::Double(0, 0.5));
    prm.declare_entry("Critical energy release rate", "1", Patterns::Double(0));
    prm.declare_entry("Phase field length scale", "0.01", Patterns::Double(0));
    prm.declare_entry("Plane state", "stress",
                      Patterns::Selection("stress|strain"));
    prm.declare_entry("Phase field model", "AT2",
                      Patterns::Selection("AT2|AT1"));
    prm.declare_entry("Degradation", "quadratic",
                      Patterns::Selection("quadratic|cubic"));
  }
  prm.leave_subsection();
}

void Material::subsection_parse_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Material");
  {
    E = prm.get_double("Young's modulus");
    v = prm.get_double("Poisson's ratio");
    Gc = prm.get_double("Critical energy release rate");
    l_phi = prm.get_double("Phase field length scale");
    plane_state = prm.get("Plane state");
    phasefield_model = prm.get("Phase field model");
    degradation = prm.get("Degradation");
  }
  prm.leave_subsection();
}

struct FESystemInfo {
  unsigned int dim;
  unsigned int poly_degree;
  bool refine;
  double refine_influence_initial;
  double refine_influence_final;
  double refine_minimum_size_ratio;

  static void subsection_declare_parameters(ParameterHandler &prm);

  void subsection_parse_parameters(ParameterHandler &prm);
};

void FESystemInfo::subsection_declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Finite element system");
  {
    prm.declare_entry("Physical dimension", "2", Patterns::Integer(0),
                      "Physical dimension");
    prm.declare_entry("Polynomial degree", "1", Patterns::Integer(0),
                      "Displacement system polynomial order");

    prm.declare_entry("Refine", "false", Patterns::Bool());

    prm.declare_entry("Phase field initial influential ratio (for refinement)",
                      "2", Patterns::Double(0));
    prm.declare_entry("Phase field final influential ratio (for refinement)",
                      "3", Patterns::Double(0));
    prm.declare_entry("Minimum relative size of refined cells w.r.t l_phi",
                      "0.2", Patterns::Double(0));
  }
  prm.leave_subsection();
}

void FESystemInfo::subsection_parse_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Finite element system");
  {
    dim = prm.get_integer("Physical dimension");
    poly_degree = prm.get_integer("Polynomial degree");
    refine = prm.get_bool("Refine");
    refine_influence_final =
        prm.get_double("Phase field final influential ratio (for refinement)");
    refine_influence_initial = prm.get_double(
        "Phase field initial influential ratio (for refinement)");
    refine_minimum_size_ratio =
        prm.get_double("Minimum relative size of refined cells w.r.t l_phi");
  }
  prm.leave_subsection();
}

struct AllParameters : public FESystemInfo,
                       public Project,
                       public Runtime,
                       public Material {
  AllParameters() = default;
  AllParameters(const std::string &input_file);

  static void declare_parameters(ParameterHandler &prm);

  void set_parameters(const std::string &input_file);
  void parse_parameters(ParameterHandler &prm);

  std::string output_dir;
  std::string param_dir;
};

AllParameters::AllParameters(const std::string &input_file) {
  set_parameters(input_file);
}

void AllParameters::set_parameters(const std::string &input_file) {
  ParameterHandler prm;
  declare_parameters(prm);
  prm.parse_input(input_file);
  parse_parameters(prm);

  // set output directory

  std::time_t currenttime = std::time(0);
  char tAll[255];
  std::strftime(tAll, sizeof(tAll), "%Y-%m-%d-%H-%M-%S",
                std::localtime(&currenttime));
  std::string stime;
  std::stringstream strtime;
  strtime << tAll;
  stime = strtime.str();

  output_dir = output_dir_top + this->project_name + "-" + stime + "/";

  param_dir = input_file;
}

void AllParameters::declare_parameters(ParameterHandler &prm) {
  FESystemInfo::subsection_declare_parameters(prm);
  Project::subsection_declare_parameters(prm);
  Runtime::subsection_declare_parameters(prm);
  Material::subsection_declare_parameters(prm);
}

void AllParameters::parse_parameters(ParameterHandler &prm) {
  FESystemInfo::subsection_parse_parameters(prm);
  Project::subsection_parse_parameters(prm);
  Runtime::subsection_parse_parameters(prm);
  Material::subsection_parse_parameters(prm);
}
} // namespace Parameters

#endif