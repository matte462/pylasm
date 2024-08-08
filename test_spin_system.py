from spin_system import SpinSystem
import numpy as np
import pytest

def test_constructor_0() -> None :
    '''
    Tests that the proper Exception is raised when the SpinSystem instance does not include 
    an appropriate number of lattice vectors.
    '''
    with pytest.raises(ValueError, match='1 lattice vectors are given, while 3 are expected.') :
        latt_vecs = np.array([[1.0,1.0,1.0]])
        sites = np.array([[0.0,0.0,0.0],[0.25,0.25,0.25]])
        spin = 1.5
        system = SpinSystem(latt_vecs,sites,spin)

def test_constructor_1() -> None :
    '''
    Tests that the proper Exception is raised when the SpinSystem instance does not include 
    an appropriate number of atomic sites.
    '''
    with pytest.raises(ValueError, match='Only 1 site is given, while they should be 2 or more.') :
        latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        sites = np.array([[0.0,0.0,0.0]])
        spin = 1.5
        system = SpinSystem(latt_vecs,sites,spin)

def test_constructor_2() -> None :
    '''
    Tests that the proper Exception is raised when the SpinSystem instance does not include 
    an valid spin quantum number.
    '''
    with pytest.raises(ValueError, match='0.2 is not a valid spin quantum number. Only integer or half-integer values are accepted.') :
        latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
        spin = 0.2
        system = SpinSystem(latt_vecs,sites,spin)

def test_find_NN_shell_0() -> None :
    '''
    Tests that the proper Exception is raised when the find_NN_shell method is provided with 
    an invalid spin index, namely larger or equal than the total number of spins in the system.
    '''
    with pytest.raises(ValueError, match='2 is not a valid index for the spins of the system under study.') :
        latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
        spin = 0.5
        system = SpinSystem(latt_vecs,sites,spin)
        shell_indices, shell_vectors = system.find_NN_shell(2,1,3,3)

def test_find_NN_shell_1() -> None :
    '''
    Tests that the proper Exception is raised when the find_NN_shell method is provided with
    an invalid NN shell indicator, namely a non-positive integer.
    '''
    with pytest.raises(ValueError, match='0 is not a valid value for the NN shell to be studied. Only positive integer values are accepted.') :
        latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
        spin = 0.5
        system = SpinSystem(latt_vecs,sites,spin)
        shell_indices, shell_vectors = system.find_NN_shell(0,0,3,3)

def test_build_spin_operator_0() -> None :
    '''
    Tests that the spin operator is computed correctly for S=1/2 as a representative case of half-integer spin quantum number.
    Its components are expected to be proportional to the well known Pauli matrices.
    '''
    # Initialize a cubic system with 2 sites as an example
    latt_vecs = np.eye(3)
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # Expected spin vector operator
    Sx_exp = 0.5*np.array([[0.0,1.0],
                           [1.0,0.0]],dtype=complex)
    Sy_exp = 0.5*np.array([[0.0,-1.0j],
                           [1.0j,0.0]],dtype=complex)
    Sz_exp = 0.5*np.array([[1.0,0.0],
                           [0.0,-1.0]],dtype=complex)
    S_exp = np.array([Sx_exp,Sy_exp,Sz_exp])

    # Effective spin vector operator
    S_eff = system.build_spin_operator()
    spin_mult = system.get_spin_mult()

    is_S_ok = True
    for x in range(len(S_eff)) :
        for i in range(spin_mult) :
            for j in range(spin_mult) :
                is_S_ok = is_S_ok and (S_eff[x][i][j]==S_exp[x][i][j])
    assert is_S_ok

def test_build_spin_operator_1() -> None :
    '''
    Tests that the spin operator is computed correctly for S=1, as a representative case of integer spin quantum number.
    Its components are expected to be linear combinations of the well known Gell-Mann matrices.
    '''
    # Initialize a cubic system with 2 sites as an example
    latt_vecs = np.eye(3)
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,1.0)

    # Expected spin vector operator
    Sx_exp = (1/np.sqrt(2))*np.array([[0.0,1.0,0.0],
                                      [1.0,0.0,1.0],
                                      [0.0,1.0,0.0]],dtype=complex)
    Sy_exp = (1/np.sqrt(2))*np.array([[0.0,-1.0j,0.0],
                                      [1.0j,0.0,-1.0j],
                                      [0.0,1.0j,0.0]],dtype=complex)
    Sz_exp = np.array([[1.0,0.0,0.0],
                       [0.0,0.0,0.0],
                       [0.0,0.0,-1.0]],dtype=complex)
    S_exp = np.array([Sx_exp,Sy_exp,Sz_exp])

    # Effective spin vector operator
    S_eff = system.build_spin_operator()
    spin_mult = system.get_spin_mult()

    is_S_ok = True
    for x in range(len(S_eff)) :
        for i in range(spin_mult) :
            for j in range(spin_mult) :
                is_S_ok = is_S_ok and (np.round(S_eff[x][i][j],10)==np.round(S_exp[x][i][j],10))
    assert is_S_ok