## deal.II phase field fracture

A deal.II implementation of phase field fracture model with multiphysics interfaces, adaptive mesh, and MPI compatibility. 

### Install

This code is tested on the official deal.II [docker image (v9.4.0-focal)](https://hub.docker.com/r/dealii/dealii/tags).

### Features

* Multiphysics interfaces with the staggered scheme. For a new physics added to the existing system, one only needs to implement (1) how residual (or the Right Hand Side) and tangent stiffness matrix (or stiffness matrix) are calculated on Gaussian points, (2) calculating visualized fields for the physics, and (3) defining the staggered scheme.
* Problem definition with (1) an ABAQUS mesh, (2) a text file describing Dirichlet or Neumann boundary conditions (based on "Surface"s defined in Abaqus .inp file), and (3) a parameter file for computing setups, preferably without modifying any code.
* Adaptive mesh based on gradients of the phase field. 
* Parallelism with MPI, working smoothly with multiple nodes on HPC clusters. 

<img width="967" alt="image" src="https://github.com/user-attachments/assets/fd18a82e-bf44-4bec-b9fe-43c09105f8f6">

<img width="674" alt="image" src="https://github.com/user-attachments/assets/ceafb404-92bb-46bb-8e9e-b0e63221f4b7">


### Usage

* Mesh file: It uses meshes defined in ABAQUS-generated .inp files (see `meshes/singleNotchDense.inp` for example).
* Boundary conditions: First, define "Surfaces" in ABAQUS (in the Assembly module). Then define a boundary configuration file (see `meshes/singleNotchTension_boundary.txt` for example). The configuration is defined according to a specific format:

	* Each line defines: Surface-ID, type of constraint, constrained dof, and value(s).
	* For Dirichlet boundaries:
	  * For `velocity`, the fourth number is in mm/s (or the unit of the rate of the field variable)
	  * For `dirichlet`, the fourth number is in mm (or the unit of the field variable)
	  * For `sinedirichlet`/`triangulardirichlet`, the fourth part is frequency(Hz), mean(mm), and amplitude(mm), for example "20 1 2"
	* For Neumann boundaries: the third part is a series of floats denoting the vector of the neumann boundary (set 0 if is a scalar field).
	  * For `neumann`, the third part is in MPa (or the unit of gradient)
	  * For `neumannrate`, the third part is in MPa/s (or the unit of the rate of gradient)
	  * For `sineneumann`/`triangularneumann`, the third part is dimensionless. And add a fourth part being frequency (Hz), mean (MPa), and amplitude (MPa), for example "20 1 2"
* Parameters: See `parameters/singleNotchTension.prm` for an example of parameter definitions. All available parameters are shown in `include/parameters.h`. These parameters are straightforwardly named.

* Compile the code and execute a project using a parameter file:

  ```shell
  ./compile_and_run.sh -n 8 -f parameters/singleNotchTension.prm
  ```

  where  `-n` defines the number of MPI processes and  `-f` defines the parameter file

  There are some additional parameters:

  *  `-s`:  `true` or  `false`, meaning whether to use the previously compiled code.
  *  `-r`:  `debug` or  `release`. 

### Extensions

If you want to define a new field, ideally you have to (and only have to) imitate  `phase_field.h` or  `elasticity.h` (that inherits  `abstract_field.h`) to define operations for the field, and then imitate  `phase_field_fracture.h` (that inherits `abstract_multiphysics.h`) to define a regime of solving the multiphysical system (particularly the staggered scheme). 

  
