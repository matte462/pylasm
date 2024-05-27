2024-05-27 Matteo Costa <mattecosta48@gmail.com>

    * Inputs/: I have added some input files for testing the read_J_couplings() method of InputReader class.

    * global_functions.py (is_spin_acceptable): Improvements in the documentation.

    * global_functions.py (adapt_magintmatrix): I have defined a new function that implements a specific transformation to the Dipole-Dipole interaction matrices read in the J_couplings_file. Some documentation is also included.

    * input_reader.py (read_J_couplings_file): The method has been documented and properly modified in order to improve Exceptions' handling and to fix some bugs encountered during the testing phase.

    * input_reader.py (get_J_couplings & get_T_vectors): New getter methods are implemented to help the user to extract all the necessary info from the InputReader instance.

    * input_reader.py (print_summary): This method makes the user aware of what the Input Reader instance has been able to read from all the input files.

    * test_input_reader.py (test_read_J_couplings_file_*): I have added some testing function to make sure the read_J_couplings_file() method works correctly.

    * test_global_functions.py (test_adapt_magintmatrix_*): I have added some testing functions to check whether the implementation of adapt_magintmatrix() follows the expectations in the most general case.

2024-05-20 Matteo Costa <mattecosta48@gmail.com>

    * Inputs/: I have added some input files for testing the reading methods of InputReader class.

    * global_functions.py (is_spin_acceptable): New function to handle Exceptions about the choice of spin value.

    * input_reader.py (InputReader): I have introduced new Exceptions to be raised for an appropriate choice of the values of the numerical input variables (e.g. spin, max_NN_shell, n_iterations, energy_digits) and also implemented a new method to read the values in the POSCAR file. In order to pass the testing unctions successfully, I had to remove ast.literal_eval and exploit str.isdigit instead. A new method to read the J couplings file is implemented, but still needs a proper documentation and some test functions.

    * test_input_reader.py (test_read_*): New test functions are defined to check the correctness of read_config_file and  read_POSCAR methods of InputReader class.

    * test_global_functions.py (test_clean_line_* & test_is_spin_acceptable_*): Some basic test functions are defined to check the correctness of clean_line and is_spin_acceptable functions in some limit cases. 

2024-05-09 Matteo Costa <mattecosta48@gmail.com> 

    * Inputs/: I have added some input files for testing the reading methods of InputReader class.

    * global_functions.py (print_logo): Some improvement in the aesthetics of the software logo.

    * globals_functions.py (clean_line): Documentation is added.

    * spin_system.py (SpinSystem): Just added the documentation for all the methods implemented so far.

    * input_reader.py (InputReader): Main changes regard the documentation and some new Exception to instruct the user on how to deal with eventually-incoming errors, in particular read_POSCAR() method.

    * test_input_reader.py (test_read_*): New test functions are defined in such a way that the code properly behaves when provided with empty/problematic input files.

2024-05-05 Matteo Costa <mattecosta48@gmail.com> 

	* Inputs/: This folder includes all the input files required to run the test routines implemented so far, and some reference files for structures and magnetic interactions. 

	* main.py (main): Here I have outlined the logic steps to achieve the purpose of the project. 

	* global_functions.py (print_logo): Just some aesthetics of the software.

    * global_functions.py (direct2cartesian): I have defined a function that converts a set of sites in the direct coordinate system to the cartesian one. Still it needs a good documentation and some testing routines.

    * global_functions.py (clean_line): I have defined a function that makes easier to read structure files, splitting the string of interest and removing useless empty spaces. Still it needs a good documentation and some testing routines.

    * global_functions.py (solve_by_lanczos): This function redirects the code execution to the proper solver function by the use of a customized mapping. Still it needs a good documentation and some testing routines.

    * spin_system.py (SpinSystem): This class defines the physical system whose low-energy properties need to be investigated. It includes 3 attributes (lattice vectors to apply the proper periodic boundary conditions, magnetic sites to find the NN vectors and the spin quantum number which is related to the Hilbert space' dimensions), some getter and setter methods, but documentation still needs to be written.

    * input_reader.py (InputReader): This class takes care of reading all input files if provided with a well-defined configuration file. Only the reading and getter methods are implemented and documented.

    * test_input_reader.py (test_read_*): These testing routines check whether the code behaves correctly when the configuration file is empty or at most contains the strictly-necessary information.