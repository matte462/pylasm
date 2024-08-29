import numpy as np

class SpinSystem() :
    '''
    This class implements the physical system under study, starting from basic information 
    (i.e. lattice vectors, atomic positions and spin quantum number).
    If the magnetic interactions and the NN vectors are also provided, one can define the associated
    hamiltonian matrix so as to proceed with the exact diagonalization (ED) algorithm. 
    '''

    def __init__(self,latt_vecs: 'np.ndarray',sites: 'np.ndarray',spin: float) :
        '''
        Initializes a new instance of SpinSystem.
        
        Args:
            latt_vecs (np.ndarray): Set of 3 lattice vectors defining the periodic boundary conditions;
            sites (np.ndarray): Finite set of atomic positions of the spins in the system;
            spin (float): Spin quantum number.
        '''
        if latt_vecs.shape[0]==3 :
            self.latt_vecs = latt_vecs
        else :
            raise ValueError(f'{latt_vecs.shape[0]} lattice vectors are given, while 3 are expected.')
        if sites.shape[0]>1 :
            self.sites = sites
        else :
            raise ValueError(f'Only {sites.shape[0]} site is given, while they should be 2 or more.')
        
        from global_functions import is_spin_acceptable
        if is_spin_acceptable(spin) :
            self.spin = spin
        else :
            raise ValueError(f'{spin} is not a valid spin quantum number. Only integer or half-integer values are accepted.')

    # Getters methods
    def get_latt_vecs(self) -> 'np.ndarray' :
        '''
        Returns the lattice vectors.
        '''
        return self.latt_vecs

    def get_sites(self) -> 'np.ndarray':
        '''
        Returns the atomic positions within the Carthesian coordinate system.
        '''
        return self.sites
    
    def get_spin(self) -> float :
        '''
        Returns the spin quantum number.
        '''
        return self.spin
    
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
    
    # Setters methods
    def set_latt_vecs(self,new_latt_vecs: 'np.ndarray') -> None :
        '''
        Sets a new set of 3 lattice vectors, if the old one needs to be changed.

        Args:
            new_latt_vecs (np.ndarray): New set of 3 lattice vectors.
        '''
        self.latt_vecs = new_latt_vecs

    def set_sites(self,new_sites: 'np.ndarray') -> None :
        '''
        Sets a new set of atomic positions, if the old one needs to be changed.

        Args:
            new_sites (np.ndarray): New set of atomic positions.
        '''
        self.sites = new_sites

    def set_spin(self,new_spin: float) -> None :
        '''
        Sets a new value for the spin quantum number.

        Args:
            new_spin (float): New spin quantum number.
        '''
        self.spin = new_spin

    def set_ith_site(self,new_ith_site: 'np.ndarray',i: int) -> None :
        '''
        Changes the i-th spin position to a new one.

        Args:
            new_ith_site (np.ndarray): New atomic position for the i-th spin;
            i (int): The index of the spin, whose position needs to be changed,
                     consistently to the previously defined spin system.
        '''
        self.sites[i] = new_ith_site

    def find_NN_shell(self,ref_spin: int,n_shell: int,shell_digits: int,n_dim: int) -> tuple :
        '''
        Returns a tuple of two arrays, which contain the indices of all the spins belonging to the specified NN shell
        of the reference spin and the associated connecting vectors respectively. The basic idea is to take the 
        coordinates of all magnetic sites in the unit cell and to shift them into the outer unit cells if necessary. 
        They are finally sorted according to the distance from the position of the reference spin so as to identify
        the members of the desired NN shell.
        In case of low-dimensional systems, only the smallest lattice vectors are used to perform the shifts.
        
        Args:
            ref_spin (int): The index of the reference spin, whose NN shells are the target of this method;
            n_shell (int): Indicator of the NN shell of interest;
            shell_digits (int): Number of digits to be considered during the identification of the NN shells by distance;
            n_dim (int): Number of spatial dimensions of the spin system under study.
        '''
        Nspins = self.get_Nspins()
        if ref_spin<0 or ref_spin>=Nspins :
            raise ValueError(rf'{ref_spin} is not a valid index for the spins of the system under study.')
        if n_shell<1 :
            raise ValueError(f'{n_shell} is not a valid value for the NN shell to be studied. Only positive integer values are accepted.')
        ref_site = self.get_sites()[ref_spin]
        latt_vecs = self.get_latt_vecs()

        # Ordering lattice vectors by their norms
        latt_vecs_norms = np.linalg.norm(latt_vecs,axis=1)
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

            # Initial shift
            initial_shift = np.zeros(3)
            for d in range(n_dim) :
                initial_shift += np.dot(eff_latt_vecs[d],-n_uc_shell)

            # Preparation of all possible shift vectors to the (n_uc_shell)-th adjacent unit cells 
            shift_indices = []
            for i in range(2*n_uc_shell+1) :
                is_i_extremal = (i==0 or i==2*n_uc_shell)
                if n_dim==1 :
                    if is_i_extremal : 
                        shift_indices.append(tuple([i]))
                    continue
                for j in range(2*n_uc_shell+1) :
                    is_j_extremal = (j==0 or j==2*n_uc_shell)
                    if n_dim==2 :
                        if (is_i_extremal or is_j_extremal) : 
                            shift_indices.append((i,j))
                        continue
                    for k in range(2*n_uc_shell+1) :
                        is_k_extremal = (k==0 or k==2*n_uc_shell)
                        if n_dim==3 :
                            if (is_i_extremal or is_j_extremal or is_k_extremal) :
                                shift_indices.append((i,j,k))
            
            # Loop over all other spins
            for jth_spin in range(Nspins) :
                other_site = self.get_sites()[jth_spin]
                replica_site = other_site+initial_shift
                for s in range(len(shift_indices)) :
                    
                    # Apply the current shift
                    for d in range(n_dim) :
                        replica_site += np.dot(eff_latt_vecs[d],shift_indices[s][d])
                    distance_vec = ref_site-replica_site
                    distance = np.linalg.norm(distance_vec)

                    # Save the results
                    if np.round(distance,shell_digits)!=0.0 :
                        shell_indices.append(jth_spin)
                        shell_vectors.append(distance_vec)
                        shell_distances.append(distance)
                        
                    # Undo the current shift
                    for d in range(n_dim) :
                        replica_site += np.dot(eff_latt_vecs[d],-shift_indices[s][d])

            # Sort the so obtained lists
            sorted_indices = sorted(range(len(shell_distances)), key=lambda k: shell_distances[k])
            shell_distances = [np.round(shell_distances[i],shell_digits) for i in sorted_indices]
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
                if min([np.linalg.norm(np.dot(vec,n_uc_shell)) for vec in eff_latt_vecs])>distances_aux[n_shell-1] :
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

    def build_spin_operator(self) -> 'np.ndarray' :
        '''
        Returns the local spin vector operator associated with the magnetic sites of the system.
        It is strictly related to the chosen spin quantum number, which indeed sets the dimension
        of the matrix representation to (2S+1)x(2S+1).
        '''
        spin = self.get_spin()
        spin_mult = self.get_spin_mult()

        # Construction of spin operators
        S_x = np.zeros((spin_mult,spin_mult),dtype=complex) 
        S_y = np.zeros((spin_mult,spin_mult),dtype=complex)
        S_z = np.zeros((spin_mult,spin_mult),dtype=complex)
        for a in range(1,spin_mult+1) :
            for b in range(1,spin_mult+1) :
                S_x[a-1][b-1] = complex(0.5*(a==b+1 or a+1==b)*np.sqrt((spin+1)*(a+b-1)-a*b))
                S_y[a-1][b-1] = complex((0.5j*(a==b+1)-0.5j*(a+1==b))*np.sqrt((spin+1)*(a+b-1)-a*b))
                S_z[a-1][b-1] = complex((spin+1-a)*(a==b))
        return np.array([S_x,S_y,S_z])

    def build_hamiltonian(self,J_couplings: list,NN_vectors: list,max_NN_shell: int,shell_digits: int,n_dim: int) -> 'np.ndarray' :
        '''
        Returns the Hamiltonian matrix for the interacting spin system.
        As mentioned below, the order of the elements in the provided arrays is important,
        since the i-th J matrix specifically corresponds to the i-th NN vector.

        Args:
            J_couplings (list): Ordered set of 3x3 intersite exchange matrices;
            NN_vectors (list): Ordered set of 3D NN vectors;
            max_NN_shell (int): The highest NN shell to be considered when computing the spin-spin interactions;
            shell_digits (int): Number of digits to be considered during the identification of the NN shells by distance;
            n_dim (int): Number of spatial dimensions of the spin system under study. 
        '''
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()

        # Initialize the Spin Hamiltonian matrix
        dim = spin_mult**Nspins
        H = np.zeros((dim,dim),dtype=complex)

        # Loop over all permitted NN Shells
        for nn in range(1,max_NN_shell+1) :

            # Loop over all spins
            for i in range(Nspins) :

                # Identify the NN shell of interest for the first spin
                ith_NN_spins, ith_NN_vecs = self.find_NN_shell(i,nn,shell_digits,n_dim)

                # Inner cycle over the second spins
                for j in range(len(ith_NN_spins)) :
                    if ith_NN_spins[j]<i : continue
                        
                    # Compute the effective J tensor between the two spins
                    J_eff = self.compute_J_eff(J_couplings[nn-1],NN_vectors[nn-1],ith_NN_vecs[j],shell_digits)

                    # Compute the interaction term and update the Spin Hamiltonian
                    interaction_term = self.compute_pair_interaction(i,ith_NN_spins[j],J_eff)
                    H += interaction_term
        return H
    
    def compute_J_eff(self,J_couplings: list,NN_vectors: list,vector: 'np.ndarray',shell_digits: int) -> 'np.ndarray' :
        '''
        Returns the effective exchange interaction tensor associated with a specific NN vector.

        Args:
            J_couplings (list): Ordered set of 3x3 intersite exchange matrices (only for a specific NN shell);
            NN_vectors (list): Ordered set of 3D NN vectors (only for a specific NN shell);
            vector (np.ndarray): 3D vector to be found among the NN vectors;
            shell_digits (int): Number of digits to be considered during the identification of the NN shells by distance.
        '''
        # Initialize the J matrix
        J_eff = np.zeros((3,3))

        # Find the input J matrix by correspondence with the NN vectors
        is_vector_found = False
        for i in range(len(NN_vectors)) :
            distance = np.linalg.norm(NN_vectors[i]-vector)
            if np.round(distance,shell_digits)==0.0 :
                is_vector_found = True
                J_eff += J_couplings[i]
            if i==len(NN_vectors)-1 and not is_vector_found :
                raise ValueError(f'{vector} could not be found among the input NN vectors.')
        return J_eff
    
    def compute_pair_interaction(self,first: int,second: int,J_eff: 'np.ndarray') -> 'np.ndarray' :
        '''
        Returns the contribution of a single pair to the global spin Hamiltonian matrix.

        Args:
            first (int): Index for the first spin within the pair;
            second (int): Index for the second spin within the pair;
            J_eff (np.ndarray): 3x3 real matrix for the effective intersite exchange interaction between the pair.
        '''
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()
        S_vec = self.build_spin_operator()

        # Initialize the interaction term between the two spins
        matrix_size = spin_mult**(second-first+1)
        interaction_term = np.zeros((matrix_size, matrix_size),dtype=complex)

        # Loop over all the spatial coordinates (twice)
        for a in range(3) :
            for b in range(3) :
                if second>first : 
                    aux_size = spin_mult**(second-first-1)
                    interaction_term += J_eff[a][b] * np.kron(np.kron(S_vec[a],np.eye(aux_size)), S_vec[b])
                else :
                    interaction_term += J_eff[a][b] * np.matmul(S_vec[a], S_vec[b])

        # Adjust the shape by the proper tensor products  
        aux_size = spin_mult**(Nspins-second-1)
        final_term = np.kron(np.kron(np.eye(spin_mult**first), interaction_term), np.eye(aux_size))
        return final_term
    
    def compute_spin_exp_value(self,state: 'np.ndarray',ref_spin: int) -> 'np.ndarray' :
        '''
        Returns the expectation value of the vector spin operator of the chosen magnetic site with respect
        to the state passed as first argument.
        
        Args:
            state (np.ndarray): Spinor state belonging to the Hilbert space of the composite spin system;
            ref_spin (int): Index of the interested magnetic site. 
        '''
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()
        S_vec = self.build_spin_operator()
        
        # Initialize the spin expectation value
        exp_value = np.zeros(3)
        
        # Loop over all the spatial coordinates
        for a in range(3) :
            Sa_extended = np.kron(np.kron(np.eye(spin_mult**ref_spin), S_vec[a]), np.eye(Nspins-ref_spin))
            Sa_expected = np.dot(np.matmul(state.conj(),Sa_extended),state.T)[0][0]
            exp_value[a] = np.real(Sa_expected)
        return exp_value
    
    def compute_magnetization(self,state: 'np.ndarray') -> 'np.ndarray' :
        '''
        Returns the magnetization vector of the composite spin system as proportional to the sum
        of the spin expectation values associated with all the available magnetic sites. 
        
        Args:
            state (np.ndarray): Spinor state belonging to the Hilbert space of the composite spin system.
        '''
        Nspins = self.get_Nspins()
        M_vec = np.zeros(3)
        
        # Loop over all spin indices
        for spin in range(Nspins) :
            M_vec += (1.0/Nspins)*self.compute_spin_exp_value(state,spin)
            
        return M_vec
    
    def compute_spin_correlation(self,state: 'np.ndarray',first: int,second: int) -> 'np.ndarray' :
        '''
        Returns the spin-spin correlation matrix between the two chosen magnetic sites. Each element
        is labelled by the components of the two spin operators in question and it is thus obtained 
        by calculating the expectation value of their product.
        Tensor products with identity matrices are also included in order to keep the order of the two
        spins within the sequence.
        
        Args:
            state (np.ndarray): Spinor state belonging to the Hilbert space of the composite spin system;
            first (int): Index of the first magnetic site;
            second (int): Index of the second magnetic site.
        '''
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()
        S_vec = self.build_spin_operator()
        
        # Initialize the spin-spin correlation matrix
        corr_matrix = np.zeros((3,3))

        # Loop over all the spatial coordinates (twice)
        for a in range(3) :
            for b in range(3) :
                aux_size1 = spin_mult**(second-first-1)
                spin_product = np.kron(np.kron(S_vec[a],np.eye(aux_size1)), S_vec[b])

                # Adjust the shape by the proper tensor products  
                aux_size2 = spin_mult**(Nspins-second-1)
                final_product = np.kron(np.kron(np.eye(spin_mult**first), spin_product), np.eye(aux_size2))
                
                # Perform the products with the states
                corr_matrix[a][b] = np.dot(np.matmul(state.conj(),final_product),state.T)[0][0]
                
        return corr_matrix