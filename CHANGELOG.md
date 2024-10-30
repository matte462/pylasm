# <h1 style="text-align: center; font-size: 3em;">Changelog</h1>

## 2024-10-29 Matteo Costa <mattecosta48@gmail.com>

- ### global_functions.py (map_spin_correlations):
> A ValueError exception is now raised if the given spin-spin correlation matrices are not symmetric within a fixed tolerance (1e-10). This feature derives from the fact that spin-spin correlation values must not depend on the order of the interested spins.

- ### spin_system.py (find_NN_shell):
> Two distinct parts of the code for the implementation of the find_NN_shell method are now included into other two methods of the SpinSystem class, namely find_shift_indices and update_shell_arrays. The main ideas behind the separation of the code blocks in question is to facilitate the readibility and to improve the effectiveness of the associated test functions.

- ### spin_system.py (find_shift_indices):
> This method allows the user to access all the sites within the unit cell and its adjacent raplica due to periodic buondary conditions. In particular, the shift vectors which lead the input sites to the associated replica are expressed in terms of the effective lattice vectors. The notation is outlined within the docstring at the beginning of the method itself, alongside with a practical example.

- ### spin_system.py (update_shell_arrays):
> This method exploits the list of integer tuples from find_shift_indices to access all the sites within the given NN unit cells. Due to the specific implementation of find_shift_indices, it also perform a preliminary shift to the "lowest" NN unit cell (the most left replica in the 1D case, the most low-left replica in the 2D case and the most low-left replica within the lowest a-b plane in the 3D case).

- ### test_global_functions.py:
> I have added new test functions to make sure adapt_magintmatrix, map_spin_correlations and solve_by_lanczos provide the expected results.

- ### test_spin_system.py:
> I have added new test functions to make sure find_shift_indices and update_shell_arrays provide the expected results.

## 2024-10-07 Matteo Costa <mattecosta48@gmail.com>

- ### README.md:
> The link to the folder Examples should now work in both the local and remote Git-hub repository.

- ### global_functions.py:
> The typing annotations and the parameters' layout are now improved for an optimal readability. In particular, I choose to list a parameter per line when the function requires more then 3 entries.

- ### input_reader.py:
> As for global_functions.py, I have decided to improve the typing annotations and the parameters' layout in order to facilitate the reading process. But I have also removed all the setters and those getters methods which allowed to access the attributes of the InputReader instance. Now, they are directly accessed by the dot operator.

- ### main.py:
> Improving the parameters' layout.

- ### spin_system.py:
> As in the other files, I have improved the typing annotations and the parameters' layout and I have also removed all the setters and those getters which allowed to access the attributes of the SpinSystem instance. Finally, I have fixed the logics behind the Exception raise statements within the definition of the constructor.

- ### test_global_functions.py & test_input_reader.py:
> The pytest.raises calls are now placed just before the line of interest for the specific testing function. Moreover, Numpy arrays are compared with each other by means of np.allclose, instead of checking np.linalg.norm(difference)<1e-10. A 1e-10 tolerance is chosen to ensure the equality condition.

- ### test_spin_system.py:
> In addition to the modifications I have mentioned for the other two testing file, I have also improved the efficiency of the testing functions in reporting failures by simplifying them and by targetting a single quantity per testing function. Unavoidably, the number of lines has increased a lot.

## 2024-10-02 Matteo Costa <mattecosta48@gmail.com>

- ### README.md:
> I have now added a README Markdown file in order to provide the user with a general guide on the project. It delves into the installation procedure, the available examples and so on.

- ### requirements.txt:
> Just a list of all the Python packages one has to install before running any PyLasm script.

- ### exec_*_run.ipynb:
> Set of python scripts for a more efficient usage of Pylasm code in the form of Jupyter Notebooks. Many details and useful suggestions concerning the task they perform are outlined within themselves.

- ### Examples/FM_Ising_Chain.md & Examples/AFM_Heisenberg_Chain.md:
> As suggested by the name of the two files in question, they show what PyLasm can achieve in two exemplary toy models. The whole procedure to reproduce the results is extensively described and the necessary input files are also given.

- ### Examples/Figures/*:
> Just some of the plots that one could obtain by reproducing the examples mentioned above.

- ### global_functions.py:
> Minor changes have been applied as a consequence of the occurrence of incorrect behaviour while running trials.

## 2024-09-30 Matteo Costa <mattecosta48@gmail.com>

- ### test_spin_system.py (test_compute_magnetization_*):
> I have added new test functions for the SpinSystem.compute_magnetization class. Basically, I check whether the magnetization modulus is correctly estimated in the same cases proposed for the functions test_compute_spin_correlation_*.

- ### exec_multiple_B_runs.py:
> This python script allows the user to perform multiple pylasm runs with different scaling factors for the initially given magnetic field B_field. Using the given configuration file as a template, it generates several init_config.ini with the chosen B_field values and copies them into subdirectories. Also the J_couplings file and the structure file are copied into the subdirectories. A pylasm single-run within each subdirectory is thus executed.

- ### exec_multiple_J_runs.py:
> This python script allows the user to perform multiple pylasm runs with different J_coupling files. Using the given configuration file as a template, it generates several init_config.ini with the chosen J_couplings files and copies them into subdirectories. Also the J_couplings files and the structure file are copied into the subdirectories. A pylasm single-run within each subdirectory is thus executed. I suggest the J_couplings to be somehow related to each other for post-processing purposes.

- ### clean_directory.py:
> This python script allows the user to remove useless files within the subdirectories of the working directory. It is meant to be used after a pylasm multiple-run, but please check that all calculations ended successfully.

- ### input_reader.py:
> A new input variable show_plot (bool) is added to the available keys to be assigned within the configuration file. It allows the user to choose whether the plot (spin-spin correlation vs distance) is going to be prepared at the end of the pylasm single-run.

- ### test_input_reader.py (test_read_config_file_*):
> Some of the test functions were affected by the introduction of the new input variable. So minor changes were necessary.

## 2024-09-29 Matteo Costa <mattecosta48@gmail.com>

- ### global_functions.py (log_exception):
> I have added a function to allow the message of the eventually raised Exceptions to be written into the report file named SPIN_REPORT.txt.

- ### global_functions.py (print_logo):
> Since the print statements are directed towards SPIN_REPORT.txt, the ANSI escape codes are now useless. So they are removed.

- ### global_functions.py (save_data):
> All the relevant data (i.e. Lanczos energies, spin-spin correlation values, magnetization) are now saved into a well-structured JSON file named SPIN_OUT.json, in case the results are needed for later use.

- ### global_functions.py (map_spin_correlations & plot_data):
> These functions provide the user with a plot of the just calculated spin-spin correlation values as a function of the intersite distance. This task requires to map the spin pair of interest with the associated NN shell, which is carried out by exploiting the previously defined SpinSystem.find_NN_shell method.

- ### input_reader.py (read_struct_file):
> The method did not cover the case in which there is no structure file with the specified string. Now the code raises an Exception if that occurs.

- ### test_spin_system.py (test_spin_correlation_*):
> I have now re-introduced the old testing functions for the SpinSystem.compute_spin_correlation method with all the appropriate modifications due to the recent changes. The functions investigate whether the method behaves correctly in some limit cases. Namely, they include: different GS degeneracies, distinct spin configurations and on-site/adjacent-sites/non-adjacent-sites correlation values.

- ### main.py:
> The import statement for the functions defined within global_functions.py is now explicit. I have also used the sys module to direct the stdout and stderr streams into the report file named SPIN_REPORT.txt.

- ### Scripts/exec_single_run.py:
> This new python script is meant to be copied and pasted into the working directory by the user. Once the path to main.py is fully specified, it allows the user to perform a signle-run Lanczos calculation. A clearer example/tutorial will be included soon.

## 2024-09-27 Matteo Costa <mattecosta48@gmail.com>

- ### MAJOR CHANGES TO THE CODE HAVE BEEN APPLIED

- ### spin_system.py (compute_*_spin_correlation*):
> Due to a non-trivial physical/numerical ambiguity concerning the computation of spin expectation values I have decided to return as output observables the spin-spin correlation values. They do not provide a precise description of the possible ground-state magnetic configurations (as the spin expectation values should have done), but just give a useful insight on the predicted magnetic order distinguishing between the short> and long-range domains.

- ### spin_system.py (compute_magnetization):
> I have further noticed that not always the full magnetization modulus (i.e. M = sqrt(Mx**2+My**2+Mz**2)) is the observable of interest, but sometimes one may just want to obtain the absolute value of its projection along the axes of the coordinate system (e.g. M = sqrt(Mz**2)). This fact will be probably discussed more deeply iwthin a tutorials or examples section of the project.

- ### spin_system.py (build_hamiltonian):
> New arguments tothe method have been introduced in order to check whether the Hamiltonian is Hermitian within a certain tolerance and to implement the application of an external homogeneous magnetic field. This feature may find utility in plotting magnetic phase diagrams and/or lifting ground-state degeneracies.

- ### global_functions.py (solve_by_lanczos):
> The focus of the project is now shifted towards the development of a python interface to treat composite spin systems with an already implemented version of the Lanczos algorithm. This version is called PyLanczos and published in the Git-Hub page [https://github.com/mrcdr/pylanczos.git]. It is indeed based on a efficient C++ library Lambda Lanczos [https://github.com/mrcdr/lambda-lanczos.git] also available in Git-Hub. The function I wrote allows to perform this version of the Lanczos algorithm iteratively until all the requested ground-state and excited-states manifolds are determined.

- ### input_reader.py (constructor):
> The configuration file name is now set to 'init_config.ini' by default.

- ### input_reader.py (read_config_file):
> Due to the logistic change mentioned above I have decided to modify the allowed sections' names into 'STRUCTURE', 'HAMILTONIAN' and 'OUTPUT'. At the same time, several input variables have been removed and substituted by new ones (e.g. B_field, n_excited, lanczos_digits, magn_output_mode), whose meaning can be easily understood by reading the paragraphs above. I have also adjusted the raise statements accordingly.

- ### input_reader.py (get_*):
> Getters method have slightly changed to compensate the modification of the name of the sections and the input variables.

- ### main.py (main):
> The steps within the overall procedure slightly differs, but the structure remains basically the same:
> Reading, Summary, System Definition, Hamiltonian Building, Lanczos Algorithm and finally Output.

- ### test_input_reader.py & test_spin_system.py:
> The old testing functions are now adjusted to the new features of the code.

- ### test_global_functions.py:
> The number of current testing functions is now drastically reduced as a result of the smaller number of the global functions.

- ### Inputs/TestFiles/config_file_* & init_config.ini:
> The old input configuration files are now adjusted to the new features of the code.

## 2024-08-31 Matteo Costa <mattecosta48@gmail.com>

- ### global_functions.py (plot_data):
> This new function allows the user to plot the most relevant data for the just performed Lanczos calculation by reading the content of SPIN_OUT.json file. The first graph shows how the Lanczos energies converge to a finite value while increasing the number of iterations (only in SCF mode), and the second one represents a 3D visualization of the just approximated spin ground-state in order to easily identify the magnetic state in question.

- ### test_spin_system.py (test_compute_magnetization_*):
> I have added some test functions for the compute_magnetization method.

- ### spin_system.py & test_global_functions.py:
> Minor changes due the failure of some test functions.

## 2024-08-29 Matteo Costa <mattecosta48@gmail.com>

- ### spin_system.py (compute_spin_correlation):
> I have added implementation and documentation to the method in question, as well as minor changes to the compute_spin_exp_value and compute_magnetization methods.

- ### global_functions.py (save_data):
> The function allows the user to save all the relevant outcomes from the Lanczos calculation into a well-structured JSON file. Further suggestions about the conventional structure will be later specified. 

## 2024-08-28 Matteo Costa <mattecosta48@gmail.com>

- ### input_reader.py:
> New input variables are introduced in order to let the user be more aware of the process and the potentially incoming errors.

- ### global_functions.py:
> Minor changes to the arguments of the functions implementing the Lanczos algorithm due to the new variables at play. I have also chosen not to return the Lanczos basis and directly express the eigenstates of the tridiagonal Hamiltonian matrix within the conventional spin-z basis.

- ### spin_system.py:
> I have added three new methods for the SpinSystem class; they are called compute_spin_exp_value, compute_magnetization and compute_spin_correlation. Only the first two have been actually implemented and documented so far.

- ### test_spin_system.py (test_compute_spin_exp_value_*):
> Some testing functions to check whether the compute_spin_exp_value method is well implemented. They only include two-sites S=1/2 systems and the ground state is assumed to be factorizable.

## 2024-08-26 Matteo Costa <mattecosta48@gmail.com>

- ### spin_system.py (build_hamiltonian):
> I have noticed that the returned Hamiltonian matrix should not be constrained to be real-valued. So the return statement has been changed accordingly.

- ### spin_system.py (find_NN_shell & compute_J_eff):
> The two methods have been slightly modified due to an incoming problem while running the testing routines in test_spin_system.py.

- ### input_reader.py:
> Same as for the find_NN_shell and compute_J_eff methods of the SpinSystem class.

- ### test_spin_system.py (test_build_hamiltonian_8):
> I have added a new testing function to check whether anisotropic contributions to the J tensors were properly treated within the build_hamiltonian method. This allowed me to notice the problem I mentioned above since complex values in the Hamiltonian matrix only arise from those terms.

- ### global_functions.py (one_shot_lanczos_solver):
> A new implementation of the Lanczos algorithm is added. As a consequence of the latest changes to the build_hamiltonian method of the SpinSystem class, the initial random state requires the generation of two floating numbers between -1.0 and +1.0: one for the real part and the other for the imaginary part. The procedure and the name of the variables at play are also changed, but they can be easily understood by looking at the reference specified in the documentation. A re-orthogonalization step for the Lanczos vector obtained at each iteration is necessary to stabilize and improve the results.

- ### test_global_functions.py (test_*_lanczos_solver_*):
> The new testing routines try to demonstrate that the functions for the Lanczos algorithm are capable to approximate reasonably well the lowest energy eigenvalue of an interacting spin system. They merely target Hamiltonian matrices of the form I expect. This suggests that some more attention while building/diagonalizing the Hamiltonian matrices is needed.

## 2024-08-14 Matteo Costa <mattecosta48@gmail.com>

- ### global_functions.py:
> I have implemented and documented the functions to perform the Lanczos algorithm by a one-shot calculation or a self-consistent cycle. Still they need to be tested.

- ### main.py:
> Minor modification due to the fact that now solve_by_lanczos only requires three prameters to be passed as arguments.


## 2024-08-13 Matteo Costa <mattecosta48@gmail.com>

- ### test_spin_system.py (test_*):
> I have added several new testing functions for making sure the build_hamiltonian and compute_pair_interactions methods of the SpinSystem class behave exactly how I expect them to do. Due to the rapidly increasing complexity of these systems I could only include relatively trivial cases. However, they take into account the variation of all the parameters at play: the number of sites per unit cell, the number of spatial dimensions, the number of included NN shells and the spin quantum number.

## 2024-08-10 Matteo Costa <mattecosta48@gmail.com>

- ### test_spin_system.py (test_*):
> I have added some testing routines for several methods of the SpinSystem class, namely find_NN_shell, build_hamiltonian and compute_J_eff. Still one could think about introducing further testing functions for systems with more complicated geometries and higher spin quantum numbers.

- ### spin_system.py (find_NN_shell):
> The testing routines allowed to show that the application of the shifts at each iteration was poorly implemented.

- ### spin_system.py (build_hamiltonian):
> The task of the method is now carried out by two smaller methods (compute_J_eff and compute_pair_interaction) in order to simplify the testing process. 

## 2024-08-08 Matteo Costa <mattecosta48@gmail.com>

- ### input_reader.py:
> Introduction of two new tags (i.e. n_dim,shell_digits) to be readed from the configuration file. The associated getter methods are also implemented.

- ### spin_system.py (find_NN_shell):
> The method now requires two more variables to be passed as arguments when called. A major modification to the implemetation was also necessary.

- ### test_input_reader.py (test_*):
> Minor changes to the interested testing routines due to the new variables at play.

- ### test_spin_system.py (test_find_NN_shell_*):
> Minor changes to the interested testing routines due to the new variables at play.


## 2024-06-02 Matteo Costa <mattecosta48@gmail.com>

- ### global_functions.py:
> Some modifications in the way I import the SpinSystem class due to an incoming circular import error for the spin_system module.

- ### input_reader.py (read_J_couplings_file):
> Just a small modification to the reading method to account for a possible string format for the T_vectors, as a consequence to a failed test.

- ### spin_system.py (__init__):
> I have introduced some ValueError excpeptions to be raised when the SpinSystem instance does not follow the instructions precisely.

- ### spin_system.py (find_NN_shell):
> I have implemented and documented a method for identifying the spin indices belonging to the chosen NN shell of the reference spin and the associated connecting vectors in Angstrom units. It only explore the 26 unit cells which are strictly adjacent to the one of interest. So one could think about a better implementation, but the present solution should be largely sufficient to the purpose of the code.

- ### spin_system.py (build_spin_operator):
> This method computes the spin vector operator associated with all the sites in the system. The implementation is quite general, so it should work for most of the permitted spin quantum numbers.

- ### spin_system.py (build_hamiltonian):
> I have implemented a method to calculate the hamiltonian matrix of the SpinSystem instance once the J tensors, the NN vectors and the maximum NN shell are given. It makes use of the previous methods (find_NN_shell & build_spin_operator) and several methods from NumPy library. Periodic boundary conditions, tensor products and anisotropic interaction terms should have been taken into account correctly, but still the method requires some test routines. 

- ### test_spin_system.py (test_*):
> Some basic test functions are intended to check whether the SpinSystem constructor and its methods behave as expected for them. Still other testing functions for find_NN_shell are needed since only limit cases have been considered so far.

## 2024-05-27 Matteo Costa <mattecosta48@gmail.com>

- ### Inputs/:
> I have added some input files for testing the read_J_couplings() method of InputReader class.

- ### global_functions.py (is_spin_acceptable):
> Improvements in the documentation.

- ### global_functions.py (adapt_magintmatrix):
> I have defined a new function that implements a specific transformation to the Dipole-Dipole interaction matrices read in the J_couplings_file. Some documentation is also included.

- ### input_reader.py (read_J_couplings_file):
> The method has been documented and properly modified in order to improve Exceptions' handling and to fix some bugs encountered during the testing phase.

- ### input_reader.py (get_J_couplings & get_T_vectors):
> New getter methods are implemented to help the user to extract all the necessary info from the InputReader instance.

- ### input_reader.py (print_summary):
> This method makes the user aware of what the Input Reader instance has been able to read from all the input files.

- ### test_input_reader.py (test_read_J_couplings_file_*):
> I have added some testing function to make sure the read_J_couplings_file() method works correctly.

- ### test_global_functions.py (test_adapt_magintmatrix_*):
> I have added some testing functions to check whether the implementation of adapt_magintmatrix() follows the expectations in the most general case.

## 2024-05-20 Matteo Costa <mattecosta48@gmail.com>

- ### Inputs/:
> I have added some input files for testing the reading methods of InputReader class.

- ### global_functions.py (is_spin_acceptable):
> New function to handle Exceptions about the choice of spin value.

- ### input_reader.py (InputReader):
> I have introduced new Exceptions to be raised for an appropriate choice of the values of the numerical input variables (e.g. spin, max_NN_shell, n_iterations, energy_digits) and also implemented a new method to read the values in the POSCAR file. In order to pass the testing unctions successfully, I had to remove ast.literal_eval and exploit str.isdigit instead. A new method to read the J couplings file is implemented, but still needs a proper documentation and some test functions.

- ### test_input_reader.py (test_read_*):
> New test functions are defined to check the correctness of read_config_file and  read_POSCAR methods of InputReader class.

- ### test_global_functions.py (test_clean_line_* & test_is_spin_acceptable_*):
> Some basic test functions are defined to check the correctness of clean_line and is_spin_acceptable functions in some limit cases. 

## 2024-05-09 Matteo Costa <mattecosta48@gmail.com> 

- ### Inputs/:
> I have added some input files for testing the reading methods of InputReader class.

- ### global_functions.py (print_logo):
> Some improvement in the aesthetics of the software logo.

- ### globals_functions.py (clean_line):
> Documentation is added.

- ### spin_system.py (SpinSystem):
> Just added the documentation for all the methods implemented so far.

- ### input_reader.py (InputReader):
> Main changes regard the documentation and some new Exception to instruct the user on how to deal with eventually-incoming errors, in particular read_POSCAR() method.

- ### test_input_reader.py (test_read_*):
> New test functions are defined in such a way that the code properly behaves when provided with empty/problematic input files.

## 2024-05-05 Matteo Costa <mattecosta48@gmail.com> 

- ### Inputs/:
> This folder includes all the input files required to run the test routines implemented so far, and some reference files for structures and magnetic interactions. 

- ### main.py (main):
> Here I have outlined the logic steps to achieve the purpose of the project. 

- ### global_functions.py (print_logo):
> Just some aesthetics of the software.

- ### global_functions.py (direct2cartesian):
> I have defined a function that converts a set of sites in the direct coordinate system to the cartesian one. Still it needs a good documentation and some testing routines.

- ### global_functions.py (clean_line):
> I have defined a function that makes easier to read structure files, splitting the string of interest and removing useless empty spaces. Still it needs a good documentation and some testing routines.

- ### global_functions.py (solve_by_lanczos):
> This function redirects the code execution to the proper solver function by the use of a customized mapping. Still it needs a good documentation and some testing routines.

- ### spin_system.py (SpinSystem):
> This class defines the physical system whose low-energy properties need to be investigated. It includes 3 attributes (lattice vectors to apply the proper periodic boundary conditions, magnetic sites to find the NN vectors and the spin quantum number which is related to the Hilbert space' dimensions), some getter and setter methods, but documentation still needs to be written.

- ### input_reader.py (InputReader):
> This class takes care of reading all input files if provided with a well-defined configuration file. Only the reading and getter methods are implemented and documented.

- ### test_input_reader.py (test_read_*):
> These testing routines check whether the code behaves correctly when the configuration file is empty or at most contains the strictly-necessary information.