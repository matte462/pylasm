import numpy as np

class SpinSystem():
    def __init__(self,latt_vecs: 'np.ndarray',sites: 'np.ndarray',spin: float) :
        self.latt_vecs = latt_vecs
        self.sites = sites
        self.spin = spin

    # Getters methods
    def get_latt_vecs(self) -> 'np.ndarray' :
        return self.latt_vecs

    def get_sites(self) -> 'np.ndarray':
        return self.sites
    
    def get_spin(self) -> float :
        return self.spin
    
    def get_Nspins(self) -> int :
        return len(self.sites)

    def get_spin_mult(self) -> int :
        return int(2*self.spin+1)
    
    # Setters methods
    def set_latt_vecs(self,new_latt_vecs: 'np.ndarray') -> None :
        self.latt_vecs = new_latt_vecs

    def set_sites(self,new_sites: 'np.ndarray') -> None :
        self.sites = new_sites

    def set_spin(self,new_spin: float) -> None :
        self.spin = new_spin

    def set_ith_site(self,new_ith_site: 'np.ndarray',i: int) -> None :
        self.sites[i] = new_ith_site

    def get_hamiltonian(self,J_couplings: 'np.ndarray',NN_vectors: 'np.ndarray',max_NN_shell: int) -> 'np.ndarray' : # STILL TO BE IMPLEMENTED
        pass 

    def init_random_state(self) -> 'np.ndarray' :
        Nspins = self.get_Nspins()
        spin_mult = self.get_spin_mult()
        dim = spin_mult**Nspins
        return np.random.rand(dim)