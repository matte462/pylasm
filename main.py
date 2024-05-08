from global_functions import *
from input_reader import *
from spin_system import *

import numpy as np

def main() :
    # Print software logo
    print_logo()

    # Read all input files
    reader = InputReader('./Inputs/Tests/config_file_1.ini')
    reader.print_summary()
    
    # Initialize the spin system
    lattice_vectors = reader.get_lattice_vectors()
    mag_ions_pos = reader.get_mag_ions_pos()
    spin = reader.get_spin()
    system = SpinSystem(lattice_vectors,mag_ions_pos,spin)
    
    # Build the associated Hamiltonian matrix
    J_couplings = reader.get_J_couplings()
    NN_vectors = reader.get_NN_vectors()
    max_NN_shell = reader.get_max_NN_shell()
    hamiltonian = system.build_hamiltonian(J_couplings,NN_vectors,max_NN_shell)
    
    np.random.seed(187)
    lanczos_mode = reader.get_lanczos_mode()
    lanczos_par = reader.get_lanczos_par()
    results = solve_by_lanczos(system,hamiltonian,lanczos_mode,lanczos_par)
    print_conclusions(results)


if __name__ == '__main__':
    main()