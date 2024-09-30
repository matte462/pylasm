from global_functions import log_exception, print_logo, solve_by_lanczos, save_data, plot_data
from input_reader import InputReader
from spin_system import SpinSystem

import sys

def main() :
    
    with open('SPIN_REPORT.txt','w') as file :
        sys.stdout = file
        sys.excepthook = log_exception
        
        # Print software logo
        print_logo()

        # Read all input files
        reader = InputReader()
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
        B_field = reader.get_B_field()
        tol_imag = reader.get_tol_imag()
        hamiltonian = system.build_hamiltonian(J_couplings,T_vectors,max_NN_shell,shell_digits,B_field,n_dim,tol_imag)
        
        # Perform the Lanczos algorithm
        n_excited = reader.get_n_excited()
        lanczos_digits = reader.get_lanczos_digits()
        states, energies, GS_deg = solve_by_lanczos(hamiltonian,n_excited,lanczos_digits)
        
        # Extract the requested output
        magn_output_mode = reader.get_magn_output_mode()
        spin_correlations = system.compute_all_spin_correlations(states,GS_deg)
        magnetization = system.compute_magnetization(states,GS_deg,magn_output_mode)
        
        # Save and plot all the relevant data
        show_plot = reader.get_show_plot()
        save_data(energies,spin_correlations,magnetization)
        if show_plot :
            plot_data(system,spin_correlations,magnetization,shell_digits,n_dim)

if __name__ == '__main__':
    main()