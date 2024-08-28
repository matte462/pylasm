import numpy as np
from scipy.linalg import ishermitian, eigh_tridiagonal

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

def solve_by_lanczos(hamiltonian: 'np.ndarray',lanczos_mode: str,n_iterations: int,energy_res: float,tol_imag: float,tol_ortho: float,n_states: int) -> list :
    '''
    Calls the proper function to perform the Lanczos algorithm for the exact diagonalization 
    of the Spin Hamiltonian of the system. Only two modes are available: a one-shot calculation or a 
    self-consistent (SCF) cycle with convergence criterion on the ground-state (GS) energy.

    Args:
        hamiltonian (np.ndarray): Spin Hamiltonian matrix;
        lanczos_mode (str): String indicating the chosen method for Lanczos algorithm (i.e. one-shot or SCF);
        n_iterations (int): Number of iterations to perform in a one-shot Lanczos calculation;
        energy_res (float): Resolution on the GS energy for the SCF mode;
        tol_imag (float): Tolerance on the imaginary part of Hamiltonian matrix elements;
        tol_ortho (float): Tolerance on the non-orthogonality of the Lanczos vectors;
        n_states (int): Number of eigenvector-eigenvalue pairs to be stored as outputs.
    '''
    label = 'One-shot'*(lanczos_mode=='one_shot')+'SCF'*(lanczos_mode=='scf')
    print(f'\n{label} Lanczos Algorithm for Exact Diagonalization')
    
    # Select the function to call depending on lanczos_mode
    if lanczos_mode=='scf' :
        print('\nSCF Lanczos Algorithm for Exact Diagonalization')
        results = scf_lanczos_solver(hamiltonian,energy_res,tol_imag,tol_ortho,n_states)
        return results
    elif lanczos_mode=='one_shot' :
        print('\nOne-shot Lanczos Algorithm for Exact Diagonalization')
        results = one_shot_lanczos_solver(hamiltonian,n_iterations,tol_imag,tol_ortho,n_states)
        return results

def one_shot_lanczos_solver(hamiltonian: 'np.ndarray',n_iterations: int,tol_imag: float,tol_ortho: float,n_states: int) -> list :
    '''
    Implements the Lanczos algorithm by a one-shot calculation with a precise number 
    of iterations. Details about the procedure and the nomenclature can be found in the 
    lecture notes (Chapter 10) of the ETH course "Numerical Methods for Solving Large Scale 
    Eigenvalue Problems" available on the following website.
    [https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf]

    Args:
        hamiltonian (np.ndarray): Spin Hamiltonian matrix;
        n_iterations (int): Number of iterations to perform;
        tol_imag (float): Tolerance on the imaginary part of Hamiltonian matrix elements;
        tol_ortho (float): Tolerance on the non-orthogonality of the Lanczos vectors;
        n_states (int): Number of eigenvector-eigenvalue pairs to be stored as outputs.
    '''
    if not ishermitian(hamiltonian,atol=tol_imag) :
        raise ValueError('The Spin Hamiltonian matrix is not Hermitian.')
    
    # Initial random state for the spin system
    dim = hamiltonian.shape[0]
    x = np.random.uniform(-1,1,dim)+np.random.uniform(-1,1,dim)*1.0j
    x_norm = np.linalg.norm(x)

    # First iteration
    q = (1.0/x_norm)*x
    r = np.dot(hamiltonian,q)
    a_1 = np.real_if_close(np.vdot(q,r),tol=tol_imag)
    r = r-a_1*q
    b_1 = np.linalg.norm(r)
    
    # Save the Lanczos vector and coefficients from the first iteration
    Q_basis = [q]
    alphas = [a_1]
    betas = [b_1]

    print(f'\nN° iterations: {n_iterations}')
    for j in range(2,n_iterations+1) :

        # Current iteration
        v = q
        q = (1/betas[-1])*r
        r = np.dot(hamiltonian,q)-betas[-1]*v
        a_j = np.real_if_close(np.vdot(q,r),tol=tol_imag)
        r = r-a_j*q
        for q_prime in Q_basis : r = r-np.vdot(q_prime,r)*q_prime # re-orthogonalization step
        b_j = np.linalg.norm(r)
        
        # Save the Lanczos vector and coefficients from the current iteration
        Q_basis.append(q)
        alphas.append(a_j)
        if j<=n_iterations-1 :
            betas.append(b_j)

        # Report the (Non-)Orthogonality of the current Lanczos basis
        is_q_orthogonal = True
        for q_prime in Q_basis :
            is_q_orthogonal = is_q_orthogonal and bool(np.vdot(q,q_prime)<tol_ortho)
        response = ''*(is_q_orthogonal)+'NOT '*(not is_q_orthogonal)
        print(f'{j}: New Lanczos vector is {response}orthogonal the previous ones.')
    
    # Diagonalization of the approximated Tridiagonal matrix
    energies, states = eigh_tridiagonal(alphas,betas,select='i',select_range=(0,n_states-1))
    spin_states = np.matmul(states.T,np.array(Q_basis))
    print(f'Ground-state energy: {energies[0]} eV')

    return [(n_iterations, energies, spin_states)]

def scf_lanczos_solver(hamiltonian: 'np.ndarray',energy_res: float,tol_imag: float,tol_ortho: float,n_states: float) -> list :
    '''
    Implements the Lanczos algorithm by a SCF cycle with increasing number
    of iterations. The loop is terminated as soon as the GS energy difference
    between two subsequent one-shot calculations is lower than the chosen energy 
    resolution.

    Args:
        hamiltonian (np.ndarray): Spin Hamiltonian matrix;
        energy_res (float): Resolution on the GS energy for the SCF mode;
        tol_imag (float): Tolerance on the imaginary part of Hamiltonian matrix elements;
        tol_ortho (float): Tolerance on the non-orthogonality of the Lanczos vectors;
        n_states (int): Number of eigenvector-eigenvalue pairs to be stored as outputs.
    '''
    # Quantities to be stored
    GS_energies = []
    results = []

    # SCF cycle with convergence criterion on the GS energy
    n = 2
    is_GS_converged = False
    while not is_GS_converged :

        # Perform Lanczos with the current number of iterations 
        nth_result = one_shot_lanczos_solver(hamiltonian,n,tol_imag,tol_ortho,n_states)

        # Save the results
        GS_energies.append(nth_result[0][1][0])
        results.append(nth_result[0])

        # Check whether convergence is achieved
        if n>=3 :
            is_GS_converged = (np.abs(GS_energies[-1]-GS_energies[-2])<energy_res)

        # Increment the number of iterations
        n += 1
    
    return results