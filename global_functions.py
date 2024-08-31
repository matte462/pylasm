import numpy as np
import json
from scipy.linalg import ishermitian, eigh_tridiagonal
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def save_data(system: SpinSystem,lanczos_results: list) -> None :
    '''
    Saves all the relevant data about the Lanczos SCF cycle or one-shot calculation, as well as
    the spin expectation value for each site, the magnetization vector and the spin-spin 
    correlation matrices always with respect to the just approximated GS.
    All the items I have just mentioned are stored into a well-structured JSON file, 
    called SPIN_OUT.json.
    
    Args:
        system (SpinSystem): Physical system under study, in order to have an easier access to its properties;
        lanczos_relults (list): Set of tuples containing the number of Lanczos iterations, the eigenenergies 
            and the eigenstates of the approximated tridiagonal Hamiltonian matrix. 
    '''
    Nspins = system.get_Nspins()
    ground_state = lanczos_results[-1][2][0]
    
    # Initialize a dictionary collecting all the relevant data
    data_dict = {
        'Lanczos': {},
        'Spin Exp. Values': {},
        'Magnetization': system.compute_magnetization(ground_state),
        'Spin-Spin Corr. Matrices': {}
    }
    
    # Loop over all Lanczos calculations (one or more for 'one_shot' or 'scf' modes respectively)
    for item in lanczos_results :
        data_dict['Lanczos'][f'Iteration {item[0]}'] = {'Energies': item[1],'States': item[2]}
    
    # Loop over all the spin indices
    for n in range(Nspins) :
        data_dict['Spin Exp. Values'][f'Spin {n}'] = system.compute_spin_exp_value(ground_state,n)
        for m in range(n+1,Nspins) :
            data_dict['Spin-Spin Corr. Matrices'][f'Spin_{n}-Spin_{m}'] = system.compute_spin_correlation(ground_state,n,m)
    
    with open('SPIN_OUT.json','w') as file :
        json.dump(data_dict,file,indent=4)

def plot_data(system: SpinSystem) -> None :
    '''
    Shows the user two plots: one for the convergence of the just obtained Lanczos energies 
    with respect to the number of iterations (only if more than 1 Lanczos calculation was previously
    stored in SPIN_OUT.json) and the other for the approximated spin Ground-State (GS).
    
    Args:
        system (SpinSystem): Physical system under study, in order to have an easier access to its properties.
    '''
    # Font Settings
    font_dict_title = {'fontname': 'serif', 'fontweight': 'bold', 'size': 18}
    font_dict_legend = {'family': 'serif', 'size': 13}
    font_dict_ticks = {'fontname': 'serif', 'size': 10}
    font_dict_labels = {'fontname': 'serif', 'size': 15}
    
    with open('SPIN_OUT.json','r') as file :
        data = json.load(file)
        
        if len(list(data['Lanczos'].keys()))>1 :
            
            # Prepare the window for the Lanczos energies' plot
            fig1, ax1 = plt.subplots()
            ax1.set_title('Lanczos Eigenenergies',fontdict=font_dict_title)
            ax1.set_xlabel('Iterations',fontdict=font_dict_labels)
            ax1.set_ylabel('Energy (eV)',fontdict=font_dict_labels)
            for texts in ax1.get_xticklabels() :
                texts.set(fontfamily=font_dict_ticks['fontname'],fontsize=font_dict_ticks['size'])
            for texts in ax1.get_yticklabels() :
                texts.set(fontfamily=font_dict_ticks['fontname'],fontsize=font_dict_ticks['size'])
            ax1.grid()
        
            # Adapt the data of interest for the plot
            last_lanczos_key = list(data['Lanczos'].keys())[-1]
            n_states = len(data['Lanczos'][last_lanczos_key]['States'])
            energies = [[] for n in range(n_states)]
            iterations = [[] for n in range(n_states)]
            for key in data['Lanczos'].keys() :
                for n in range(n_states) :
                    if n<len(data['Lanczos'][key]['Energies']) :
                        energies[n].append(data['Lanczos'][key]['Energies'][n])
                        iterations[n].append(int(key[-1]))
                    
            # Plot the Lanczos energies vs the number of iterations
            for n in range(n_states) :
                ax1.plot(iterations[n],energies[n],label=f'State {n}')
            ax1.legend(loc='best',prop=font_dict_legend)
            plt.show()
            fig1.savefig('LANCZOS_ENERGIES.png')
        
        # Prepare the window for the 3D Visualization of the Spin GS
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111,projection='3d')
        ax2.set_title('Spin Ground-state',fontdict=font_dict_title)
        ax2.set_xlabel('x (Å)',fontdict=font_dict_labels)
        ax2.set_ylabel('y (Å)',fontdict=font_dict_labels)
        ax2.set_zlabel('z (Å)',fontdict=font_dict_labels)
        for texts in ax2.get_xticklabels() :
            texts.set(fontfamily=font_dict_ticks['fontname'],fontsize=font_dict_ticks['size'])
        for texts in ax2.get_yticklabels() :
            texts.set(fontfamily=font_dict_ticks['fontname'],fontsize=font_dict_ticks['size'])
        for texts in ax2.get_zticklabels() :
            texts.set(fontfamily=font_dict_ticks['fontname'],fontsize=font_dict_ticks['size'])
        
        # Plot the edges of the input unit cell
        a1, a2, a3 = system.get_latt_vecs()
        vertices = np.array([[0.0,0.0,0.0],a1,a2,a3,a1+a2,a1+a3,a2+a3,a1+a2+a3])
        edges = [[vertices[0], vertices[1]],
                 [vertices[0], vertices[2]],
                 [vertices[0], vertices[3]],
                 [vertices[1], vertices[4]],
                 [vertices[1], vertices[5]],
                 [vertices[2], vertices[4]],
                 [vertices[2], vertices[6]],
                 [vertices[3], vertices[5]],
                 [vertices[3], vertices[6]],
                 [vertices[4], vertices[7]],
                 [vertices[5], vertices[7]],
                 [vertices[6], vertices[7]]]
        for edge in edges:
            ax2.plot3D(*zip(*edge), color='black',alpha=0.4)
        
        # Plot the Spin exp. values
        Nspins = len(list(data['Spin Exp. Values'].keys()))
        for n in range(Nspins) :
            site = system.get_sites()[n]
            spin = system.get_spin()
            spin_vec = np.array(data['Spin Exp. Values'][f'Spin {n}'])
            color = plt.cm.viridis(spin_vec[2])
            spin_vec = (1.0/spin)*spin_vec
            ppu.plot_vector(ax2, start=(site-1.5*spin_vec),
                            direction=spin_vec, s=3.0,
                            color=color)
        
        # Plot the Colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(vmin=-spin,vmax=+spin))
        sm.set_array([])
        cbar = plt.colorbar(sm,ax=ax2,pad=0.1,shrink=0.6,aspect=10)
        cbar.ax.set_ylabel(r'$S_z$',fontdict=font_dict_labels,rotation='horizontal')
        cbar.ax.set_yticks([-spin,+spin])
        cbar.ax.set_yticklabels([str(-spin),str(+spin)],fontdict=font_dict_ticks)
        plt.show()
        fig2.savefig('SPIN_GS.png')