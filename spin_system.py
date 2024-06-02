import numpy as np
from global_functions import is_spin_acceptable

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

    def find_NN_shell(self,ref_spin: int,n_shell: int) -> tuple :
        '''
        Returns a tuple of two lists, which contain the indices of all the spins belonging to the specified NN shell
        of the reference spin and the associated connecting vectors respectively. The basic idea is to generate the 
        coordinates of all sites in the unit cell and their 26 replicae in the adjacent ones. 
        They are finally sorted according to the distance from the position of the reference spin so as to identify
        the members of all NN shells.
        
        Args:
            ref_spin (int): The index of the reference spin, whose NN shells are the target of this method;
            n_shell (int): Indicator of which NN shell is to be computed.
        '''
        Nspins = self.get_Nspins()
        if ref_spin<0 or ref_spin>=Nspins :
            raise ValueError(rf'{ref_spin} is not a valid index for the spins of the system under study.')
        if n_shell<1 :
            raise ValueError(f'{n_shell} is not a valid value for the NN shell to be studied. Only positive integer values are accepted.')
        ref_site = self.get_sites()[ref_spin]
        latt_vecs = self.get_latt_vecs()

        # Quantities to be computed
        shell_indices = []
        shell_vectors = []
        distances = []

        # Loop over all other spins
        for jth_spin in range(Nspins) :
            if jth_spin!=ref_spin :
                other_site = self.get_sites()[jth_spin]

                # Loop over all possible replicae of the second spin
                # Consider to introduce a new N_LowD just to reduce the work for this loop
                for a in [-1,0,1] :
                    for b in [-1,0,1] :
                        for c in [-1,0,1] :
                            a_shift = np.dot(latt_vecs[0],a)
                            b_shift = np.dot(latt_vecs[1],b)
                            c_shift = np.dot(latt_vecs[2],c)
                            replica_site = other_site+a_shift+b_shift+c_shift
                            distance_vec = ref_site-replica_site
                            distance = np.sqrt(np.dot(distance_vec,distance_vec))
                            shell_indices.append(jth_spin)
                            shell_vectors.append([np.round(el,3) for el in distance_vec])
                            distances.append(distance)
                            # number of digits for the the shell vectors could be introduced as parameter in the configuration file

        # Sort the so obtained lists so as to identify the NN shells
        sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        distances = [np.round(distances[i],3) for i in sorted_indices]
        shell_indices = [shell_indices[i] for i in sorted_indices]
        shell_vectors = [shell_vectors[i] for i in sorted_indices]

        # Find the desired shell
        shell_count = 1 # First shell is automatically found at the beginning of distances list
        is_shell_found = False
        old_distance = distances[0]
        for d in range(1,len(distances)) :
            new_distance = distances[d]
            if shell_count==n_shell :
                is_shell_found = True
            if new_distance!=old_distance :
                shell_count += 1
            
            # Properly modify the elements of shell_indices and shell_vectors that do not belong to the desired shell so as to remove them later
            if shell_count!=n_shell :
                shell_indices[d] = -1
                shell_vectors[d] = np.zeros(3)
        if is_shell_found==False :
            raise ValueError(f'The desired NN shell was not found. Consider to decrement the n_shell value {n_shell} to less than {shell_count}.')
        
        # Remove those elements
        shell_indices = [ind for ind in shell_indices if ind!=-1]
        shell_vectors = [vec for vec in shell_vectors if not np.all(vec==np.zeros(3))]
        return shell_indices, shell_vectors

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

    def build_hamiltonian(self,J_couplings: 'np.ndarray',NN_vectors: 'np.ndarray',max_NN_shell: int) -> 'np.ndarray' : # STILL TO BE IMPLEMENTED
        '''
        Returns the Hamiltonian matrix for the interacting spin system.
        As mentioned below, the order of the elements in the provided arrays is important,
        since the i-th J matrix specifically corresponds to the i-th NN vector.

        Args:
            J_couplings (np.ndarray): Ordered set of 3x3 intersite exchange matrices;
            NN_vectors (np.ndarray): Ordered set of 3D NN vectors;
            max_NN_shell (int): The highest NN shell to be considered when computing the spi-spin interactions.
        '''
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()

        # Construction of spin operators
        S_vec = self.build_spin_operator()

        # Initialize the Spin Hamiltonian matrix
        dim = spin_mult**Nspins
        H = np.zeros((dim,dim),dtype=complex)

        # Loop over all permitted NN Shells
        for nn in range(1,max_NN_shell+1) :

            # Loop over all possible Spin pairs (double-counting is avoided)
            for i in range(Nspins) :
                ith_NN_spins, ith_NN_vecs = self.find_NN_shell(i,nn)

                # Inner cycle avoids double-counting
                for j in range(i+1,Nspins) :
                    if j in ith_NN_spins :

                        # Initialize the interaction term between i & j spins
                        interaction_term = np.zeros((spin_mult**(j-i+1), spin_mult**(j-i+1)),dtype=complex)
                        J_tensor = np.zeros((3,3),dtype=float)

                        # i & j spins could have more than one J matrix (or NN bond) due to periodic boundary conditions 
                        j_indices = np.where(np.array(ith_NN_spins)==j)[0]
                        jth_NN_vecs = np.array([ith_NN_vecs[v] for v in range(len(ith_NN_vecs)) if v in j_indices])
                        for vec in jth_NN_vecs :
                            vec_index = np.where(NN_vectors[nn-1]==vec)[0][0]
                            J_tensor += J_couplings[nn-1][vec_index]

                        # Loop over all the spatial coordinates
                        for a in range(3) :
                            for b in range(3) :
                                interaction_term += J_tensor[a][b] * np.kron(np.kron(S_vec[a],np.eye(spin_mult**(j-i-1))), S_vec[b])
                        H += np.kron(np.kron(np.eye(spin_mult**i), interaction_term), np.eye(spin_mult**(Nspins-j-1)))
        return H

    def init_random_state(self) -> 'np.ndarray' :
        '''
        Initializes the spin system to a random state in the N-spins Hilbert space,
        which means that returns an arbitrary linear combination of the associated
        N-spins basis states.
        Note that the dimension of the N-spins Hilbert space (D) is related to the 
        one of the 1-spin Hilbert space (d) by
                D = d**N
        '''
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()
        dim = spin_mult**Nspins
        return np.random.rand(dim)