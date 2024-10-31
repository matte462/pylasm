from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
from scipy.linalg import ishermitian

class SpinSystem() :
    '''
    This class implements the physical system under study, starting from basic information 
    (i.e. lattice vectors, atomic positions and spin quantum number).
    If the magnetic interactions and the NN vectors are also provided, one can define the associated
    hamiltonian matrix so as to proceed with the exact diagonalization (ED) algorithm. 
    '''

    def __init__(self, latt_vecs: NDArray[np.float64], sites: NDArray[np.float64], spin: float) :
        '''
        Initializes a new instance of SpinSystem.
        
        Args:
            latt_vecs (NDArray[np.float64]):
                Set of 3 lattice vectors defining the periodic boundary conditions;
            sites (NDArray[np.float64]):
                Finite set of atomic positions of the spins in the system;
            spin (float): 
                Spin quantum number.
        '''
        from global_functions import is_spin_acceptable
        if latt_vecs.shape[0]!=3 :
            raise ValueError(f'{latt_vecs.shape[0]} lattice vectors are given, while 3 are expected.')
        if sites.shape[0]<=1 :
            raise ValueError(f'Only {sites.shape[0]} site is given, while they should be 2 or more.')
        if not is_spin_acceptable(spin) :
            raise ValueError(f'{spin} is not a valid spin quantum number. Only integer or half-integer values are accepted.')
        
        self.latt_vecs = latt_vecs
        self.sites = sites
        self.spin = spin

    # Getters methods
    def get_Nspins(self) -> int :
        '''
        Returns the total number of spins in the unit cell.
        '''
        return self.sites.shape[0]

    def get_spin_mult(self) -> int :
        '''
        Returns the spin multiplicity: the dimension of the 1-spin Hilbert space.
        '''
        return int(2*self.spin+1)
    
    def find_shift_indices(self, eff_latt_vecs: NDArray[np.float64], n_uc_shell: int) -> List[Tuple[int, ...]] :
        '''
        Allows the user to access the NN unit cells of the system. In order to identify the translational operation that brings
        the (arbitrary) central unit cell into the NN ones of interest, the method returns tuples of integers specifying the how many 
        times the associated lattice vector should be added while computing the translational vector.
        
        Args:
            eff_latt_vecs (NDArray[np.float64]): 
                Array for the effective lattice vectors of the system, and their number depends on the dimensionality;
            n_uc_shell (int): 
                Index for the NN unit cells of interest.
        
        Example:
            Consider the shift_indices tuple (n1, n2) and the effective lattice vectors a1, a2. The associated translational vector reads as
            T = n1*a1 + n2*a2.
        '''
        # System's dimensionality
        n_dim = min(eff_latt_vecs.shape)
        
        # Target quantity
        shift_indices = []
        
        # Loop over all possible unit cells along the 1° lattice vector
        for i in range(2*n_uc_shell+1) :
                is_i_extremal = (i==0 or i==2*n_uc_shell)
                if n_dim==1 :
                    if is_i_extremal : 
                        shift_indices.append(tuple([i]))
                    continue
                
                # Loop over all possible unit cells along the 2° lattice vector
                for j in range(2*n_uc_shell+1) :
                    is_j_extremal = (j==0 or j==2*n_uc_shell)
                    if n_dim==2 :
                        if (is_i_extremal or is_j_extremal) : 
                            shift_indices.append((i,j))
                        continue
                    
                    # Loop over all possible unit cells along the 3° lattice vector
                    for k in range(2*n_uc_shell+1) :
                        is_k_extremal = (k==0 or k==2*n_uc_shell)
                        if n_dim==3 :
                            if (is_i_extremal or is_j_extremal or is_k_extremal) :
                                shift_indices.append((i,j,k))
        
        return shift_indices

    def update_shell_arrays(
            self, 
            ref_spin: int, 
            eff_latt_vecs: NDArray[np.float64], 
            n_uc_shell: int, 
            shift_indices: List[Tuple[int, ...]],
            shell_digits: int
        ) -> Tuple[List[int], List[NDArray[np.float64]], List[float]] :
        '''
        Allows to identify and characterize the NN spins of a reference spin by applying all possible shift vectors 
        within a given set of adjacent unit cells to all the magnetic sites of the system. For each NN bond, it stores: 
        the indices of the involved spins, the connecting vector and the relative distance.
        
        |. Pictorial representation of how the method works:\n
        |. \n
        |. [o] [x] [-]  (1D case)\n
        |. → → →\n
        |.\n
        |. [-] [-] [-]\n       
        |. [-] [x] [-] ↑  (2D case)\n
        |. [o] [-] [-]\n
        |. → → →\n   
        |.\n
        |. [-] [-] [-]\n
        |. [-] [-] [-]    (upper a-b plane)\n
        |. [-] [-] [-]\n
        |.\n
        |. [-] [-] [-]\n
        |. [-] [x] [-]    (mid a-b plane)\n
        |. [-] [-] [-]\n
        |.\n
        |. [-] [-] [-] ↑ ʘ\n
        |. [-] [-] [-] ↑  (lower a-b plane)  (3D case)\n
        |. [o] [-] [-] ↑\n
        |. → → →\n
        |.\n
        |. with x = reference unit cell, o = first investigated NN unit cell and all the left NN unit cells are indicated by -.\n
        |. Moreover, the order of the sequential investigation of the NN unit cells is specified by the arrows (→ ↑ ʘ from lowest to highest priority).\n
        |. ʘ = arrow towards higher a-b planes
         
        Args:
            ref_spin (int): 
                Index of the reference spin, whose NN shells are the target of this method;
            eff_latt_vecs (NDArray[np.float64]): 
                Array for the effective lattice vectors of the system, and their number depends on the dimensionality;
            n_uc_shell (int): 
                Index for the NN unit cells of interest;
            shift_indices (List[Tuple[int, ...]]): 
                List of all the shift indices to access the NN unit cells of interest;
            shell_digits (int): 
                Number of digits to be considered during the identification of the NN shells by distanc.
        
        Note:
            The elements of the lists within the returning tuple are univocally associated to a NN bond, so their order is relevant.
        '''
        Nspins = self.get_Nspins()
        n_dim = min(eff_latt_vecs.shape)
        ref_site = self.sites[ref_spin]
        
        # Apply the initial shift
        initial_shift = np.zeros(3)
        for d in range(n_dim) :
            initial_shift += np.dot(eff_latt_vecs[d], -n_uc_shell)
            
        # Target quantities
        new_shell_indices = []
        new_shell_vectors = []
        new_shell_distances = []
            
        # Loop over all other spins
        for jth_spin in range(Nspins) :
            other_site = self.sites[jth_spin]
            replica_site = other_site+initial_shift
            
            # Loop over all the given shifts (or replica sites)
            for s_ind in shift_indices :
                    
                # Apply the current shift
                for d in range(n_dim) :
                    replica_site += np.dot(eff_latt_vecs[d],s_ind[d])
                distance_vec = np.array(ref_site-replica_site, dtype=float)
                distance = float(np.linalg.norm(distance_vec))

                # Save the results if the relative distance is significantly non-vanishing
                if np.round(distance, shell_digits)!=0.0 :
                    new_shell_indices.append(jth_spin)
                    new_shell_vectors.append(distance_vec)
                    new_shell_distances.append(distance)
                        
                # Undo the current shift
                for d in range(n_dim) :
                    replica_site += np.dot(eff_latt_vecs[d],-s_ind[d])
        
        return new_shell_indices, new_shell_vectors, new_shell_distances
    
    def find_NN_shell(
            self,
            ref_spin: int,
            n_shell: int,
            shell_digits: int,
            n_dim: int
        ) -> Tuple[List[int], List[NDArray[np.float64]]] :
        '''
        Returns a tuple of two arrays, which contain the indices of all the spins belonging to the specified NN shell
        of the reference spin and the associated connecting vectors respectively. The basic idea is to take the 
        coordinates of all magnetic sites in the unit cell and to shift them into the outer unit cells if necessary. 
        They are finally sorted according to the distance from the position of the reference spin so as to identify
        the members of the desired NN shell.
        In case of low-dimensional systems, only the smallest lattice vectors are used to perform the shifts.
        
        Args:
            ref_spin (int): 
                Index of the reference spin, whose NN shells are the target of this method;
            n_shell (int): 
                Indicator of the NN shell of interest;
            shell_digits (int): 
                Number of digits to be considered during the identification of the NN shells by distance;
            n_dim (int): 
                Number of spatial dimensions of the spin system under study.
        '''
        Nspins = self.get_Nspins()
        latt_vecs = self.latt_vecs
        if ref_spin<0 or ref_spin>=Nspins :
            raise ValueError(rf'{ref_spin} is not a valid index for the spins of the system under study.')
        if n_shell<1 :
            raise ValueError(f'{n_shell} is not a valid value for the NN shell to be studied. Only positive integer values are accepted.')

        # Ordering lattice vectors by their norms
        latt_vecs_norms = np.linalg.norm(latt_vecs, axis=1)
        ord_latt_vecs_inds = sorted(range(len(latt_vecs_norms)), key=lambda k: latt_vecs_norms[k])
        ord_latt_vecs = np.array([latt_vecs[i] for i in ord_latt_vecs_inds])
        eff_latt_vecs = np.array([ord_latt_vecs[d] for d in range(n_dim)])

        # Quantities to be computed
        shell_indices = []
        shell_vectors = []
        shell_distances = []

        # Loop over adjacent unit cells (outwardly with respect to the one of the reference spin)
        n_uc_shell = 0
        is_shell_found = False
        while not is_shell_found :

            # Preparation of all possible shift vectors to the (n_uc_shell)-th adjacent unit cells 
            shift_indices = self.find_shift_indices(eff_latt_vecs, n_uc_shell)
            
            # Add new shell data to the lists
            new_shell_indices, new_shell_vectors, new_shell_distances = self.update_shell_arrays(ref_spin, eff_latt_vecs, n_uc_shell, shift_indices, shell_digits)
            shell_indices.extend(new_shell_indices)
            shell_vectors.extend(new_shell_vectors)
            shell_distances.extend(new_shell_distances)

            # Sort the so obtained lists by the shell_distance value
            sorted_indices = sorted(range(len(shell_distances)), key=lambda k: shell_distances[k])
            shell_distances = [np.round(shell_distances[i], shell_digits) for i in sorted_indices]
            shell_indices = [shell_indices[i] for i in sorted_indices]
            shell_vectors = [shell_vectors[i] for i in sorted_indices]

            # Identify the elements which are associated with the desired NN shell
            distances_aux = [shell_distances[0]]
            for d in shell_distances :
                if d!=distances_aux[-1] :
                    distances_aux.append(d)
            
            # Accept the results only if the length scale of the initial shift is higher than
            # the presumed shell distance
            if len(distances_aux)>=n_shell :
                length_scales = np.array([np.linalg.norm(np.dot(vec, n_uc_shell)) for vec in eff_latt_vecs])
                min_length_scale = length_scales.min()
                if float(min_length_scale)>distances_aux[n_shell-1] :
                    shell_distance = distances_aux[n_shell-1]
                    is_shell_found = True
            
            n_uc_shell += 1
        
        # Focus on the NN shell of interest
        final_shell_indices = []
        final_shell_vectors = []
        for d in range(len(shell_distances)) :
            if shell_distances[d]==shell_distance :
                final_shell_indices.append(shell_indices[d])
                final_shell_vectors.append(shell_vectors[d])
        
        return final_shell_indices, final_shell_vectors

    def build_spin_operator(self) -> NDArray[np.complex128] :
        '''
        Returns the local spin vector operator associated with the magnetic sites of the system.
        It is strictly related to the chosen spin quantum number, which indeed sets the dimension
        of the matrix representation to (2S+1)x(2S+1).
        '''
        spin = self.spin
        spin_mult = self.get_spin_mult()

        # Construction of spin operators
        S_x = np.zeros((spin_mult, spin_mult), dtype=np.complex128) 
        S_y = np.zeros((spin_mult, spin_mult), dtype=np.complex128)
        S_z = np.zeros((spin_mult, spin_mult), dtype=np.complex128)
        for a in range(1,spin_mult+1) :
            for b in range(1,spin_mult+1) :
                S_x[a-1][b-1] = complex(0.5*(a==b+1 or a+1==b) * np.sqrt((spin+1)*(a+b-1) - a*b))
                S_y[a-1][b-1] = complex((0.5j*(a==b+1)-0.5j*(a+1==b)) * np.sqrt((spin+1)*(a+b-1) - a*b))
                S_z[a-1][b-1] = complex((spin+1-a)*(a==b))
        
        return np.array([S_x,S_y,S_z])

    def build_hamiltonian(
            self,
            J_couplings: List[List[NDArray[np.float64]]],
            NN_vectors: List[List[NDArray[np.float64]]],
            max_NN_shell: int,
            shell_digits: int,
            B_field: NDArray[np.float64],
            n_dim: int,
            tol_imag: float
        ) -> NDArray[np.complex128] :
        '''
        Returns the Hamiltonian matrix for the interacting spin system.
        As mentioned below, the order of the elements in the provided arrays is important,
        since the i-th J matrix specifically corresponds to the i-th NN vector.

        Args:
            J_couplings (List[List[NDArray[np.float64]]]): 
                Ordered set of 3x3 intersite exchange matrices;
            NN_vectors (List[List[NDArray[np.float64]]]): 
                Ordered set of 3D NN vectors;
            max_NN_shell (int): 
                The highest NN shell to be considered when computing the spin-spin interactions;
            shell_digits (int): 
                Number of digits to be considered during the identification of the NN shells by distance;
            B_field (NDArray[np.float64]): 
                3D vector for the external homogeneous magnetic field;
            n_dim (int): 
                Number of spatial dimensions of the spin system under study;
            tol_imag (float): 
                Tolerance on the imaginary part of Hamiltonian matrix elements.
        '''
        print('\nThe spin Hamiltonian is being defined...')
        
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()
        S_vec = self.build_spin_operator()

        # Initialize the Spin Hamiltonian matrix
        dim = spin_mult**Nspins
        H = np.zeros((dim, dim), dtype=np.complex128)

        # Loop over all permitted NN Shells
        for nn in range(1, max_NN_shell+1) :

            # Loop over all spins
            for i in range(Nspins) :

                # Identify the NN shell of interest for the first spin
                ith_NN_spins, ith_NN_vecs = self.find_NN_shell(i, nn, shell_digits, n_dim)

                # Inner cycle over the second spins
                for j in range(len(ith_NN_spins)) :
                    if ith_NN_spins[j]<i : 
                        continue
                    elif ith_NN_spins[j]==i :
                        raise ValueError(f'The {nn}° NN shell for spin {i} includes some/all of its replica. Consider decreasing the max_NN_shell value or taking a larger unit cell.')
                        
                    # Compute the effective J tensor between the two spins
                    J_eff = self.compute_J_eff(J_couplings[nn-1],NN_vectors[nn-1],ith_NN_vecs[j],shell_digits)

                    # Compute the interaction term and update the Spin Hamiltonian
                    interaction_term = self.compute_pair_interaction(i,ith_NN_spins[j],J_eff)
                    H += interaction_term
        
        # Add Zeeman term if B_field is significantly non-vanishing
        if not np.allclose(B_field, np.zeros(3), atol=1e-10, rtol=1e-10) :
            BS_product = B_field[0]*S_vec[0] + B_field[1]*S_vec[1] + B_field[2]*S_vec[2]
            for i in range(Nspins) :
                H += np.kron(np.kron(np.eye(spin_mult**i), BS_product), np.eye(spin_mult**(Nspins-i-1)))
        
        # Check if the final Hamiltonian matrix is hermitian
        if not ishermitian(H, atol=tol_imag) :
            raise ValueError('The Spin Hamiltonian matrix is not Hermitian.')
        print('The spin Hamiltonian is finally computed.')
        
        return H
    
    def compute_J_eff(
            self,
            J_couplings: List[NDArray[np.float64]],
            NN_vectors: List[NDArray[np.float64]],
            vector: NDArray[np.float64],
            shell_digits: int
        ) -> NDArray[np.float64] :
        '''
        Returns the effective exchange interaction tensor associated with a specific NN vector.

        Args:
            J_couplings (List[NDArray[np.float64]]): 
                Ordered set of 3x3 intersite exchange matrices (only for a specific NN shell);
            NN_vectors (List[NDArray[np.float64]]): 
                Ordered set of 3D NN vectors (only for a specific NN shell);
            vector (NDArray[np.float64]): 
                3D vector to be found among the NN vectors;
            shell_digits (int): 
                Number of digits to be considered during the identification of the NN shells by distance.
        '''
        # Initialize the J matrix
        J_eff = np.zeros((3,3))

        # Find the input J matrix by correspondence with the NN vectors
        is_vector_found = False
        for i in range(len(NN_vectors)) :
            tol = 10.0**(-shell_digits)
            if np.allclose(vector, NN_vectors[i], atol=tol, rtol=tol) :
                is_vector_found = True
                J_eff += J_couplings[i]
        
        if not is_vector_found :
            raise ValueError(f'{vector} could not be found among the input NN vectors.')
        
        return J_eff
    
    def compute_pair_interaction(self, first: int, second: int, J_eff: NDArray[np.float64]) -> NDArray[np.complex128] :
        '''
        Returns the contribution of a single pair to the global spin Hamiltonian matrix.

        Args:
            first (int): 
                Index for the first spin within the pair;
            second (int): 
                Index for the second spin within the pair;
            J_eff (NDArray[np.float64]): 
                3x3 real matrix for the effective intersite exchange interaction between the pair.
        '''
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()
        S_vec = self.build_spin_operator()

        # Initialize the interaction term between the two spins
        matrix_size = spin_mult**(second-first+1)
        interaction_term = np.zeros((matrix_size, matrix_size), dtype=np.complex128)

        # Loop over all the spatial coordinates (twice)
        for a in range(3) :
            for b in range(3) :
                if second>first : 
                    aux_size = spin_mult**(second-first-1)
                    interaction_term += J_eff[a][b] * np.kron(np.kron(S_vec[a], np.eye(aux_size)), S_vec[b])
                else :
                    raise ValueError(f'The interaction term between spins {first} and {second} is not allowed.')

        # Adjust the shape by the proper tensor products  
        aux_size = spin_mult**(Nspins-second-1)
        final_term = np.kron(np.kron(np.eye(spin_mult**first), interaction_term), np.eye(aux_size))
        
        return final_term
    
    def compute_spin_correlation(
            self,
            states: NDArray[np.complex128],
            GS_deg: int,
            first: int,
            second: int
        ) -> NDArray[np.float64] :
        '''
        Returns the spin-spin correlation between the two chosen magnetic sites. It is indeed obtained 
        by calculating the expectation value of the scalar product of the associated spin operators.
        Tensor products with identity matrices are also included in order to account for the order of
        the two spins within the sequence.
        
        Args:
            states (NDArray[np.complex128]): 
                Orthonormal column eigenstates of the spin Hamiltonian;
            GS_deg (int): 
                Ground-state degeneracy;
            first (int): 
                Index of the first magnetic site;
            second (int): 
                Index of the second magnetic site.
        
        Note:
            It actually returns a 1D array of size 3 with the expectation values of the Sx*Sx, Sy*Sy and Sz*Sz
            product operators. This may help the user to have the full picture of spin-spin relative orientation.
        '''
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()
        S_vec = self.build_spin_operator()
        states = states.T
        
        # Target quantity
        corr_vec = np.zeros(3)
        
        # Loop over all the spatial coordinates
        for i in range(3) :
            
            # Initialize the spin-spin scalar product operator
            aux_size = spin_mult**(abs(second-first) + 1)
            initial_product = np.zeros((aux_size, aux_size), dtype=np.complex128)
            if second!=first :
                aux_size = spin_mult**(abs(second-first) - 1)
                initial_product = np.kron(np.kron(S_vec[i], np.eye(aux_size)), S_vec[i])
            else :
                initial_product = S_vec[i] @ S_vec[i]
            
            # Adjust its shape of the spin-spin scalar product operator
            aux_size = spin_mult**(Nspins - max(first,second) - 1)
            spins_product = np.kron(np.kron(np.eye(spin_mult**min(first,second)), initial_product), np.eye(aux_size))
            
            # Compute its expectation value with respect to the given GS states
            for n in range(GS_deg) :
                corr_vec[i] += np.real(states[n].conj() @ (spins_product) @ states[n].T) / GS_deg
            
        return corr_vec

    def compute_all_spin_correlations(
            self, 
            states: NDArray[np.complex128], 
            GS_deg: int
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] :
        '''
        Returns all the inequivalent spin-spin correlation values of the system in question
        by cycling over all the available spin pairs. 
        No single-spin "self-correlation" is taken into account since that would
        reduce to an expectation value of the squared spin operator (prop. to
        identity), which is thus a mere constant (i.e. S*(S+1)).
        
        Args:
            states (NDArray[np.float64]): 
                Orthonormal column eigenstates of the spin Hamiltonian;
            GS_deg (int): 
                Ground-state degeneracy.
        '''
        Nspins = self.get_Nspins()
        spin = self.spin
        
        # Initialize the spin-spin correlation matrix and the contribution from Sx, Sy and Sz components
        spin_corr_vals = spin*(spin+1) * np.eye(Nspins)
        spin_corr_xs = np.zeros((Nspins, Nspins))
        spin_corr_ys = np.zeros((Nspins, Nspins))
        spin_corr_zs = np.zeros((Nspins, Nspins))
        
        # Cycle over all possible spin pairs
        for n in range(Nspins) :
            for m in range(n, Nspins) :
                spin_corr_vec = self.compute_spin_correlation(states, GS_deg, n, m)
                spin_corr = spin_corr_vec.sum()
                
                if n!=m :
                    spin_corr_xs[n][m] = spin_corr_vec[0]
                    spin_corr_xs[m][n] = spin_corr_vec[0]
                    spin_corr_ys[n][m] = spin_corr_vec[1]
                    spin_corr_ys[m][n] = spin_corr_vec[1]
                    spin_corr_zs[n][m] = spin_corr_vec[2]
                    spin_corr_zs[m][n] = spin_corr_vec[2]
                    spin_corr_vals[n][m] = spin_corr
                    spin_corr_vals[m][n] = spin_corr
                else :
                    spin_corr_xs[n][n] = spin_corr_vec[0]
                    spin_corr_ys[n][n] = spin_corr_vec[1]
                    spin_corr_zs[n][n] = spin_corr_vec[2]
        
        return spin_corr_xs, spin_corr_ys, spin_corr_zs, spin_corr_vals

    def compute_magnetization(self, states: NDArray[np.complex128], GS_deg: int, magn_output_mode: str) -> float :
        '''
        Returns the magnetization modulus of the composite spin system as proportional to the expectation value of
        the sum of all the spin-spin scalar product operators. The definition of such scalar products depends on the chosen
        string for the magn_output_mode variable. See example for clarity.
        
        Args:
            states (NDArray[np.complex128]): 
                Orthonormal column eigenstates of the spin Hamiltonian;
            GS_deg (int): 
                Ground-state degeneracy;
            magn_output_mode (str): 
                String literal to select the definiton of the output magnetization.
        
        Example:
            If magn_output_mode='M_z', then only the Sz-Sz correlation values are actually targeted
            and thus the magnetization modulus operator is defined as
            M = sqrt( sum_{i,j}( Sz_i @ Sz_j ) )
            where Sz_i stands for the z component of the spin operator of the i-th magnetic site.
            The cases for magn_output_mode='M_x' or 'M_y' are similar to the latter, except for taking
            the x or y components of the spin operators respectively.
            Finally, if magn_output_mode='M_full', all the spin-spin correlation vales are accessed,
            leading to
            M = sqrt( sum_{i,j}( S_i * S_j ) )
            with S_i * S_j as the scalar product of the two spin operators in question.
        '''
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()
        S_vec = self.build_spin_operator()
        states = states.T
        
        # Target quantity
        M2_operator = np.zeros((spin_mult**Nspins, spin_mult**Nspins), dtype=np.complex128)
        
        # Choose which spin-spin correlations will contribute to the computation of the magnetization
        selected_coords = {'M_x': [0],'M_y': [1],'M_z': [2],'M_full': [0,1,2]}
        
        # Loop over all the selected spatial coordinates
        for i in selected_coords[magn_output_mode] :
            
            # Loop over all spin pairs (order is not important)
            for n in range(Nspins) :
                for m in range(n, Nspins) :
            
                    # Initialize the spin-spin scalar product operator
                    aux_size = spin_mult**(m-n+1)
                    initial_product = np.zeros((aux_size, aux_size), dtype=np.complex128)
                    if n!=m :
                        aux_size = spin_mult**(m-n-1)
                        initial_product = np.complex128(2.0)*np.kron(np.kron(S_vec[i], np.eye(aux_size)), S_vec[i])
                    else :
                        initial_product = S_vec[i] @ S_vec[i]
            
                    # Adjust its shape of the spin-spin scalar product operator
                    aux_size = spin_mult**(Nspins-m-1)
                    spins_product = np.kron(np.kron(np.eye(spin_mult**n), initial_product), np.eye(aux_size))
                    
                    # Add it to the squared magnetization operator
                    M2_operator += spins_product
        
        # Compute its expectation value with respect to the given GS states
        M2 = 0.0
        for n in range(GS_deg) :
            M2 += abs(np.real(states[n].conj() @ M2_operator @ states[n].T) / GS_deg)
        M2 = np.sqrt(M2) / Nspins
        
        return M2