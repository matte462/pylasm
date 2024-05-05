2024-05-05 Matteo Costa <mattecosta48@gmail.com> 

	* Inputs/: This folder includes all the input files required to run the test routines implemented so far, and some reference files for structures and magnetic interactions. 

	* main.py (main): Here I have outlined the logic steps to achieve the purpose of the project. 

	* global_functions.py (print_logo): Just some aesthetics of the software.

    * global_functions.py (direct2cartesian): I have defined a function that converts a set of sites in the direct coordinate system to the cartesian one. Still it needs a good documentation and some testing routines.

    * global_functions.py (clean_line): I have defined a function that makes easier to read structure files, splitting the string of interest and removing useless empty spaces. Still it needs a good documentation and some testing routines.

    * global_functions.py (solve_by_lanczos): This function redirects the code execution to the proper solver function by the use of a customized mapping. Still it needs a good documentation and some testing routines.

    * spin_system.py (SpinSystem): This class defines the physical system whose low-energy properties need to be investigated. It includes 3 attributes (lattice vectors to apply the proper periodic boundary conditions, magnetic sites to find the NN vectors and the spin quantum number which is related to the Hilbert space' dimensions), some getter and setter methods, but documentation still needs to be written.

    * input_reader.py (InputReader): This class takes care of reading all input files if provided with a well-defined configuration file. Only the reading and getter methods are implemented and documented.

    * test_input_reader (test_read_*): These testing routines check whether the code behaves correctly when the configuration file is empty or at most contains the strictly-necessary information.