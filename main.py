from global_functions import *
from input_reader import *
from spin_system import *

import numpy as np

def main() :
    # Print software logo
    print_logo()

    # Read all input files
    reader = InputReader('init_config.ini')
    reader.print_summary()
    
    # Initialize the spin system
    lattice_vectors = reader.get_lattice_vectors()
    mag_ions_pos = reader.get_mag_ions_pos()
    spin = reader.get_spin()
    n_dim = reader.get_n_dim()
    system = SpinSystem(lattice_vectors,mag_ions_pos,spin)
    
    # Build the associated Hamiltonian matrix
    J_couplings = reader.get_J_couplings()
    T_vectors = reader.get_T_vectors()
    max_NN_shell = reader.get_max_NN_shell()
    shell_digits = reader.get_shell_digits()
    hamiltonian = system.build_hamiltonian(J_couplings,T_vectors,max_NN_shell,shell_digits,n_dim)
    
    np.random.seed(187)
    lanczos_mode = reader.get_lanczos_mode()
    lanczos_par = reader.get_lanczos_par()
    results = solve_by_lanczos(hamiltonian,lanczos_mode,lanczos_par)
    print_conclusions(results)


if __name__ == '__main__':
    main()