2024-08-28 Matteo Costa <mattecosta48@gmail.com>

    * input_reader.py: New input variables are introduced in order to let the user be more aware of the process and the potentially incoming errors.

    * global_functions.py: Minor changes to the arguments of the functions implementing the Lanczos algorithm due to the new variables at play. I have also chosen not to return the Lanczos basis and directly express the eigenstates of the tridiagonal Hamiltonian matrix within the conventional spin-z basis.

    * spin_system.py: I have added three new methods for the SpinSystem class; they are called compute_spin_exp_value, compute_magnetization and compute_spin_correlation. Only the first two have been actually implemented and documented so far.

    * test_spin_system.py (test_compute_spin_exp_value_*): Some testing functions to check whether the compute_spin_exp_value method is well implemented. They only include two-sites S=1/2 systems and the ground state is assumed to be factorizable.

2024-08-26 Matteo Costa <mattecosta48@gmail.com>

    * spin_system.py (build_hamiltonian): I have noticed that the returned Hamiltonian matrix should not be constrained to be real-valued. So the return statement has been changed accordingly.

    * spin_system.py (find_NN_shell & compute_J_eff): The two methods have been slightly modified due to an incoming problem while running the testing routines in test_spin_system.py.

    * input_reader.py: Same as for the find_NN_shell and compute_J_eff methods of the SpinSystem class.

    * test_spin_system.py (test_build_hamiltonian_8): I have added a new testing function to check whether anisotropic contributions to the J tensors were properly treated within the build_hamiltonian method. This allowed me to notice the problem I mentioned above since complex values in the Hamiltonian matrix only arise from those terms.

    * global_functions.py (one_shot_lanczos_solver): A new implementation of the Lanczos algorithm is added. As a consequence of the latest changes to the build_hamiltonian method of the SpinSystem class, the initial random state requires the generation of two floating numbers between -1.0 and +1.0: one for the real part and the other for the imaginary part. The procedure and the name of the variables at play are also changed, but they can be easily understood by looking at the reference specified in the documentation. A re-orthogonalization step for the Lanczos vector obtained at each iteration is necessary to stabilize and improve the results.

    * test_global_functions.py (test_*_lanczos_solver_*): The new testing routines try to demonstrate that the functions for the Lanczos algorithm are capable to approximate reasonably well the lowest energy eigenvalue of an interacting spin system. They merely target Hamiltonian matrices of the form I expect. This suggests that some more attention while building/diagonalizing the Hamiltonian matrices is needed.

2024-08-14 Matteo Costa <mattecosta48@gmail.com>

    * global_functions.py: I have implemented and documented the functions to perform the Lanczos algorithm by a one-shot calculation or a self-consistent cycle. Still they need to be tested.

    * main.py: Minor modification due to the fact that now solve_by_lanczos only requires three prameters to be passed as arguments.


2024-08-13 Matteo Costa <mattecosta48@gmail.com>

    * test_spin_system.py (test_*): I have added several new testing functions for making sure the build_hamiltonian and compute_pair_interactions methods of the SpinSystem class behave exactly how I expect them to do. Due to the rapidly increasing complexity of these systems I could only include relatively trivial cases. However, they take into account the variation of all the parameters at play: the number of sites per unit cell, the number of spatial dimensions, the number of included NN shells and the spin quantum number.

2024-08-10 Matteo Costa <mattecosta48@gmail.com>

    * test_spin_system.py (test_*): I have added some testing routines for several methods of the SpinSystem class, namely find_NN_shell, build_hamiltonian and compute_J_eff. Still one could think about introducing further testing functions for systems with more complicated geometries and higher spin quantum numbers.

    * spin_system.py (find_NN_shell): The testing routines allowed to show that the application of the shifts at each iteration was poorly implemented.

    * spin_system.py (build_hamiltonian): The task of the method is now carried out by two smaller methods (compute_J_eff and compute_pair_interaction) in order to simplify the testing process. 

2024-08-08 Matteo Costa <mattecosta48@gmail.com>

    * input_reader.py: Introduction of two new tags (i.e. n_dim,shell_digits) to be readed from the configuration file. The associated getter methods are also implemented.

    * spin_system.py (find_NN_shell): The method now requires two more variables to be passed as arguments when called. A major modification to the implemetation was also necessary.

    * test_input_reader.py (test_*): Minor changes to the interested testing routines due to the new variables at play.

    * test_spin_system.py (test_find_NN_shell_*): Minor changes to the interested testing routines due to the new variables at play.


2024-06-02 Matteo Costa <mattecosta48@gmail.com>

    * global_functions.py: Some modifications in the way I import the SpinSystem class due to an incoming circular import error for the spin_system module.

    * input_reader.py (read_J_couplings_file): Just a small modification to the reading method to account for a possible string format for the T_vectors, as a consequence to a failed test.

    * spin_system.py (__init__): I have introduced some ValueError excpeptions to be raised when the SpinSystem instance does not follow the instructions precisely.

    * spin_system.py (find_NN_shell): I have implemented and documented a method for identifying the spin indices belonging to the chosen NN shell of the reference spin and the associated connecting vectors in Angstrom units. It only explore the 26 unit cells which are strictly adjacent to the one of interest. So one could think about a better implementation, but the present solution should be largely sufficient to the purpose of the code.

    * spin_system.py (build_spin_operator): This method computes the spin vector operator associated with all the sites in the system. The implementation is quite general, so it should work for most of the permitted spin quantum numbers.

    * spin_system.py (build_hamiltonian): I have implemented a method to calculate the hamiltonian matrix of the SpinSystem instance once the J tensors, the NN vectors and the maximum NN shell are given. It makes use of the previous methods (find_NN_shell & build_spin_operator) and several methods from NumPy library. Periodic boundary conditions, tensor products and anisotropic interaction terms should have been taken into account correctly, but still the method requires some test routines. 

    * test_spin_system.py (test_*): Some basic test functions are intended to check whether the SpinSystem constructor and its methods behave as expected for them. Still other testing functions for find_NN_shell are needed since only limit cases have been considered so far.

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