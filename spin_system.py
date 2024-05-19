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
        self.latt_vecs = latt_vecs
        self.sites = sites
        self.spin = spin

    # Getters methods
    def get_latt_vecs(self) -> 'np.ndarray' :
        '''
        Returns the lattice vectors.
        '''
        return self.latt_vecs

    def get_sites(self) -> 'np.ndarray':
        '''
        Returns the atomic positions.
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
        pass 

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