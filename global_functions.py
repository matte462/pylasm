from spin_system import SpinSystem
from types import TracebackType
import numpy as np
import matplotlib.pyplot as plt
import pylanczos as pl
import traceback
import json

# Some useful global functions
def log_exception(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType) -> None :
    '''
    Logs the details of raised Exceptions to a TXT file named SPIN_REPORT.txt.
    
    Args:
        exc_type (type[BaseException): Type of the just raised Exception;
        exc_value (BaseException): Instance of the just raised Exception;
        exc_traceback (traceback.TracebackType): Traceback of the just raised Exception.
    '''
    with open('SPIN_REPORT.txt', 'a') as error_file :
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=error_file)

def print_logo() -> None :
    '''
    Prints the software's logo, to use at the beginning.
    '''
    print("###############################################################")
    print(r" \    __________  ___",end=" ")
    print("___       __        _________   ___",end=" ")
    print("   / ")
    print(r"  \   |*|  || \*\/*/ ",end=" ")
    print("|*|      /**"+r"\ "+"     /**___)|**"+r"\ "+"/**|",end=" ")
    print("  /  ")
    print(r"  /   |*|__||  \**/  ",end=" ")
    print("|*|__   /____"+r"\ "+"   |__**|  |*|*V*|*|",end=" ")
    print(r"  \  ")
    print(r" /    |_|      |__|  ",end=" ")
    print("|____| /_/  "+r"\_"+r"\ "+"(_____/   |_|   |_|",end=" ")
    print(r"   \ ")
    print("###############################################################")

def clean_line(raw_line: str) -> list[str] :
    '''
    Removes all the empty spaces and the newline character, useful when reading text files.

    Args:
        raw_line (str): String containing the info to be extracted.
    '''
    line_content = raw_line.split(sep=' ')

    # Last element may include the newline character
    # So replace it with an empty character to be removed later
    line_content[-1] = line_content[-1].replace('\n','')
    empty_counts = line_content.count('')
    for c in range(empty_counts) : line_content.remove('')

    return line_content

def is_spin_acceptable(spin_trial: float) -> bool :
    '''
    Returns True if spin_trial is a positive integer or a positive half-integer, False otherwise.

    Args:
        spin_trial (float): Candidate real number for the spin quantum number of the system.
    '''
    double_trial = 2*spin_trial
    double_trial_int = np.abs(np.floor(double_trial))
    if double_trial-double_trial_int==0.0 and double_trial!=0.0 :
        return True
    else :
        return False
    
def adapt_magintmatrix(matrix: np.ndarray) -> np.ndarray :
    '''
    Adapts the input matrix from the MagInt conventional representation to the stamdard one.
    Example:    [Jyy, Jyz, Jyx]             [Jxx, Jxy, Jxz]
            J = [Jzy, Jzz, Jzx]  -->    J'= [Jyx, Jyy, Jyz]
                [Jxy, Jxz, Jxx]             [Jzx, Jzy, Jzz]
    
    Args:
        matrix (np.ndarray): Interaction matrix as written into the J couplings file.
    '''
    if matrix.shape!=(3,3) :
        raise ValueError('The adapt_magintmatrix function only accepts 3x3 square matrices as argument.')
    
    new_matrix = np.zeros((3,3))
    for r in range(-1,matrix.shape[0]-1) :
        for c in range(-1,matrix.shape[1]-1) :
            new_matrix[r+1][c+1] = matrix[r][c]
    
    return new_matrix

def solve_by_lanczos(hamiltonian: np.ndarray, n_excited: int, lanczos_digits: int) -> tuple :
    '''
    Performs Lanczos algorithm until the whole ground-state eigenspace and the requested excited eigenstates are identified,
    meaning both the eigenvectors and the associated eigenenergies. The excited states' counting takes into account degeneracies.
    
    Args:
        hamiltonian (np.ndarray): Hermitian Hamiltonian matrix of the composite spin system;
        n_excited (int): Number of excited manifolds to be computed;
        lanczos_digits (int): Number of Lanczos energy digits to be taken into account.
    '''
    print('\nPerforming the Lanczos algorithm...')
    
    # Ground-state degeneracy is initialized to 0 since unknown
    GS_deg = 0
    
    # Loop over the number of eigenstates to be determined by Lanczos algorithm
    n = 2
    excited_count = 0
    while excited_count<=n_excited :
        
        # Perform Lanczos algorithm to determine the lowest n energy eigenstates
        engine = pl.PyLanczos(hamiltonian,False,n)
        energies, states = engine.run()
        energies = np.round(energies,lanczos_digits)
        
        # Increment the number of estimated excited states
        if np.abs(energies[-1]-energies[-2])!=0.0 :
            excited_count += 1
            
            # Save the ground-state degeneracy
            if GS_deg==0 :
                GS_deg = n-1
        
        # Stop the loop if the Lanczos algorithm has estimated all the eigenstates of the highest-energy eigenspace
        if excited_count==n_excited+1 :
            print('The Lanczos algorithm has converged successfully.')
            return states[:,:-1], energies[:-1], GS_deg

        n += 1

def save_data(energies: np.ndarray, spin_correlations: tuple[np.ndarray], magnetization: float) -> None :
    '''
    Saves all the relevant data about the just performed Lanczos aloìgorithm, as well as
    the spin-spin correlation values and the magnetization.
    All the items I have just mentioned are stored into a well-structured JSON file, 
    called SPIN_OUT.json.
    
    Args:
        energies (np.ndarray): Tuple containing the eigenenergies approximated by the Lanczos algorithm;
        spin_correlations (tuple): Tuple of the spin-spin correlation matrices, the indices specify the associated spin pair;
        magnetization (float): Modulus of the magnetization vector.
    '''
    print('\nSaving data...')
    
    # Initialize a dictionary collecting all the relevant data
    data_dict = {
        'Lanczos Energies': [float(np.real(e)) for e in energies.tolist()],
        'Spin-Spin Corr. Matrices': {
            'X components': {},
            'Y components': {},
            'Z components': {},
            'Total': {}
        },
        'Magnetization Modulus': float(magnetization)
    }
    
    # Fill in the spin-spin correlation values
    for n in range(spin_correlations[0].shape[0]) :
        for m in range(spin_correlations[0].shape[0]) :
            i = 0
            for key in data_dict['Spin-Spin Corr. Matrices'].keys() :
                data_dict['Spin-Spin Corr. Matrices'][key][f'Spins {n}-{m}'] = spin_correlations[i][n][m]
                i += 1
    
    # Write the data into a JSON file
    with open('SPIN_OUT.json', 'w') as file :
        json.dump(data_dict, file, indent=4)
        print('All the relevant data is finally saved.')

def map_spin_correlations(
        system: SpinSystem,
        spin_correlations: tuple[np.ndarray],
        shell_digits: int,
        n_dim: int
    ) -> tuple[list] :
    '''
    Maps the just obtained spin-spin correlation values to the associated NN shell, exploiting the find_NN_shell
    method of the SpinSystem class.
    
    Args:
        system (SpinSystem): Instance of the spin system, including its main properties;
        spin_correlations (tuple[np.ndarray]): Tuple of the spin-spin correlation matrices, the indices specify the associated spin pair;
        shell_digits (int): Number of digits to be considered during the identification of the NN shells by distance;
        n_dim (int): Number of spatial dimensions of the spin system under study.
    '''
    for ssc_mat in spin_correlations :
        if not np.allclose(ssc_mat, ssc_mat.T, atol=1e-6, rtol=1e-6) :
            raise ValueError('The given spin-spin correlation matrices are not symmetric as expected.')
    
    # Target quantities
    spin_corr_data = [[],[],[],[]]
    distances = []
    
    # First element represents the "on-site" spin-spin correlation
    for i in range(4) :
        spin_corr_data[i].append(spin_correlations[i][0][0])
    distances.append(0.0)
    
    # Take spin 0 as a reference and
    # Loop over the NN shells until any replica of spin 0 is found
    nn = 1
    is_replica_found = False
    while not is_replica_found :
        shell_indices, shell_vectors = system.find_NN_shell(0, nn, shell_digits, n_dim)
        
        # Store the spin-spin correlation of the current NN shell
        for i in range(4) :
            spin_corr_data[i].append(spin_correlations[i][0][shell_indices[0]])
        distances.append(np.linalg.norm(shell_vectors[0]))
        
        # Break the cycle just after any replica of spin 0 is found
        if shell_indices.count(0)!=0 :
            is_replica_found = True
        
        nn += 1
    
    return spin_corr_data, distances

def plot_data(
        system: SpinSystem,
        spin_correlations: tuple[np.ndarray],
        magnetization: float,
        shell_digits: int,
        n_dim: int
    ) -> None :
    '''
    Use matplotlib tools for plotting the just obtained spin-spin correlation values as a function of the intersite distance
    of the spin pair they are refered to. The resulting magnetization modulus is also reported as an horizontal line.
      
    Args:
        system (SpinSystem): Physical system under study, in order to have an easier access to its properties;
        spin_correlations (tuple[np.ndarray]): Tuple of the spin-spin correlation matrices, the indices specify the associated spin pair;
        magnetization (float): Modulus of the magnetization vector;
        shell_digits (int): Number of digits to be considered during the identification of the NN shells by distance;
        n_dim (int): Number of spatial dimensions of the spin system under study.
    '''
    print('\nPlotting data...')
    
    # Set the default font family and size
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    # Initialize the figure and add some aesthetics
    fig, axs = plt.subplots(figsize=(10,6))
    axs.set_title('Spin-spin correlation function', fontdict={'fontsize': 20})
    axs.set_xlabel(r'Intersite distance |$\vec{R}_{ij}$| (Å)', fontdict={'fontsize': 16})
    axs.set_ylabel('Exp. value', fontdict={'fontsize': 16})
    styles = ['o-','s-','P-','d-']
    labels = [r'$S^{(i)}_x \cdot S^{(j)}_x$',
              r'$S^{(i)}_y \cdot S^{(j)}_y$',
              r'$S^{(i)}_z \cdot S^{(j)}_z$',
              r'$\vec{S}^{(i)} \cdot \vec{S}^{(i)}$']
    
    # Sort the spin-spin correlations by the associated intersite distance
    spin_corr_data, distances = map_spin_correlations(system, spin_correlations, shell_digits, n_dim)
    x_min = -0.01*distances[-1]
    x_max = 1.01*distances[-1]
    axs.set_xlim(x_min,x_max)
    for i in range(4) :
        axs.plot(distances, spin_corr_data[i], styles[i], markersize=8, markerfacecolor='white', label=labels[i])
    axs.plot(np.linspace(0.0,distances[-1],2), [magnetization for i in range(2)], 'k--', label=r'|$\vec{M}$|')
    
    # Just some final aesthetics and save
    plt.grid(linestyle='--', linewidth=0.5)
    plt.fill_between(distances, spin_corr_data[3], color='gray', alpha=0.3)
    plt.legend(loc='best')
    plt.savefig('SPIN_CORRS.png')
    print('All the relevant data is finally plotted.')
    plt.show()