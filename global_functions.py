import numpy as np

from spin_system import SpinSystem

# Some useful global functions
def print_logo() -> None :
    '''
    Prints the software's logo, to use at the beginning.
    '''
    print("###############################################################")
    print(r" \    __________  ___",end=" ")
    print("\033[91m___       __        _________   ___\033[00m",end=" ")
    print("   / ")
    print(r"  \   |*|  || \*\/*/ ",end=" ")
    print("\033[91m|*|      /**"+r"\ "+"     /**___)|**"+r"\ "+"/**|\033[00m",end=" ")
    print("  /  ")
    print(r"  /   |*|__||  \**/  ",end=" ")
    print("\033[91m|*|__   /____"+r"\ "+"   |__**|  |*|*V*|*|\033[00m",end=" ")
    print(r"  \  ")
    print(r" /    |_|      |__|  ",end=" ")
    print("\033[91m|____| /_/  "+r"\_"+r"\ "+"(_____/   |_|   |_|\033[00m",end=" ")
    print(r"   \ ")
    print("###############################################################")

def clean_line(raw_line: str) -> list :
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
    
def adapt_magintmatrix(matrix: 'np.ndarray') -> 'np.ndarray' :
    '''
    Adapts the input matrix from the MagInt conventional representation to the stamdard one.
    Example:    [Jyy, Jyz, Jyx]             [Jxx, Jxy, Jxz]
            J = [Jzy, Jzz, Jzx]  -->    J'= [Jyx, Jyy, Jyz]
                [Jxy, Jxz, Jxx]             [Jzx, Jzy, Jzz]
    
    Args:
        matrix (np.ndarray): Interaction matrix as written into the J couplings file.
    '''
    new_matrix = np.zeros((3,3))
    if matrix.shape!=(3,3) :
        raise ValueError('The adapt_magintmatrix function only accepts 3x3 square matrices as argument.')
    for r in range(-1,matrix.shape[0]-1) :
        for c in range(-1,matrix.shape[1]-1) :
            new_matrix[r+1][c+1] = matrix[r][c]
    return new_matrix

def solve_by_lanczos(hamiltonian: 'np.ndarray',lanczos_mode: str,lanczos_par: int) :
    '''
    Calls the proper function to perform the Lanczos algorithm for the exact diagonalization 
    of the Spin Hamiltonian of the system by either a one-shot calculation or a 
    self-consistent (SCF) cycle with convergence criterion on the ground-state (GS) energy.

    Args:
        hamiltonian (np.ndarray): Spin Hamiltonian matrix;
        lanczos_mode (str): String indicating the chosen method for Lanczos algorithm (i.e. one-shot or SCF);
        lanczos_par (int): The meaning of this parameter depends on lanczos_mode. 
            If lanczos_mode="one_shot", it sets the number of iterations to perform; 
            while if lanczos_mode="scf", it specifies the number of GS energy digits 
            to consider when checking for the convergence criterion. 
    '''
    label = 'One-shot'*(lanczos_mode=='one_shot')+'SCF'*(lanczos_mode=='scf')
    print(f'\n{label} Lanczos Algorithm for Exact Diagonalization')
    
    # Select the function to call depending on lanczos_mode
    lanczos_mapping = {
        'one_shot' : one_shot_lanczos_solver,
        'scf' : scf_lanczos_solver
    }
    results = lanczos_mapping[lanczos_mode](hamiltonian,lanczos_par)

    return results

def one_shot_lanczos_solver(hamiltonian: 'np.ndarray',n_iterations: int) -> list :
    '''
    Implements the Lanczos algorithm by a one-shot calculation with a precise number 
    of iterations.

    Args:
        hamiltonian (np.ndarray): Spin Hamiltonian matrix.
        n_iterations (int): Number of iterations to perform. 
    '''
    # Initial random state for the spin system
    dim = hamiltonian.shape[0]
    psi0 = np.random.rand(dim)
    norm0 = np.linalg.norm(psi0)
    psi0 = (1.0/norm0)*psi0

    # Initalization of the Lanczos basis and coefficients
    f_basis = []
    alpha_s = []
    beta_s = []

    # Initial set-up for the Lanczos algorithm 
    # (f_nm stands for f_n-1, while f_np for f_n+1)
    f_nm = np.array([1.0])
    f_n = psi0

    print(f'\nN° iterations: {n_iterations}')
    for n in range(1,n_iterations+1) :

        # Definition of the current Lanczos coefficients
        Hf_n = np.dot(hamiltonian,f_n)
        a_n = np.dot(f_n,Hf_n)/np.dot(f_n,f_n)
        b_n2 = np.dot(f_n,f_n)/np.dot(f_nm,f_nm)

        # Report the (Non-)Orthogonality of the current Lanczos vectors
        is_fn_orthogonal = True
        for f in f_basis :
            is_fn_orthogonal = is_fn_orthogonal and (np.dot(f_n,f)<1e-5)
        response = ''*(is_fn_orthogonal)+'NOT '*(not is_fn_orthogonal)
        print(f'{n}: New Lanczos vector is {response}orthogonal the previous ones.')

        # Save all items of interest
        f_basis.append(f_n)
        alpha_s.append(a_n)
        if n==1 : b_n2 = 0.0
        else : beta_s.append(np.sqrt(b_n2))

        # Definition of the next Lanczos vector
        an_f_n = a_n*f_n
        bn2_f_nm = b_n2*f_nm
        f_np = Hf_n-an_f_n-bn2_f_nm

        # Set-up for the next iteration
        f_nm = f_n
        f_n = f_np
    
    # Definition and Diagonalization of the final Tridiagonal matrix
    tridiag = np.diag(beta_s, -1) + np.diag(alpha_s, 0) + np.diag(beta_s, 1)
    energies, states = np.linalg.eigh(tridiag)

    # Sorting Eigenvectors by the associated energy Eigenvalue
    ord_indices = sorted(range(len(energies)), key=lambda k: energies[k])
    ord_energies = [energies[i] for i in ord_indices]
    ord_states = [states[i] for i in ord_indices]

    return [(n_iterations, ord_energies, ord_states, f_basis)]

def scf_lanczos_solver(hamiltonian: 'np.ndarray',energy_digits: int) -> list :
    '''
    Implements the Lanczos algorithm by a SCF cycle with increasing number
    of iterations. The loop is terminated as soon as the GS energy difference
    between two subsequent one-shot calculations is lower than 10^{-energy_digits} eV.

    Args:
        hamiltonian (np.ndarray): Spin Hamiltonian matrix;
        energy_digits (int): Number of GS energy digits to consider when checking for
            the convergence criterion.
    '''
    # Quantities to be stored
    GS_energies = []
    results = []

    # SCF cycle with convergence criterion on the GS energy
    n = 2
    is_GS_converged = False
    while not is_GS_converged :

        # Perform Lanczos with the current number of iterations 
        nth_result = one_shot_lanczos_solver(hamiltonian,n)

        # Save the results
        GS_energies.append(nth_result[n-2][1][0])
        results.append(nth_result)

        # Check whether convergence is achieved
        if n>=3 :
            is_GS_converged = (np.abs(GS_energies[-1]-GS_energies[-2])<10**(-energy_digits))

        # Increment the number of iterations
        n += 1
    
    return results

def print_conclusions(results) -> None :
    pass