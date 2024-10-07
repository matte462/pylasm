from spin_system import SpinSystem
import numpy as np
import pytest

def test_constructor_0() -> None :
    '''
    Tests that the proper Exception is raised when the SpinSystem instance does not include 
    an appropriate number of lattice vectors.
    '''
    latt_vecs = np.array([[1.0,1.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.25,0.25]])
    spin = 1.5
    
    with pytest.raises(ValueError, match='1 lattice vectors are given, while 3 are expected.') :
        system = SpinSystem(latt_vecs,sites,spin)

def test_constructor_1() -> None :
    '''
    Tests that the proper Exception is raised when the SpinSystem instance does not include 
    an appropriate number of atomic sites.
    '''
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0]])
    spin = 1.5
    
    with pytest.raises(ValueError, match='Only 1 site is given, while they should be 2 or more.') :
        system = SpinSystem(latt_vecs,sites,spin)

def test_constructor_2() -> None :
    '''
    Tests that the proper Exception is raised when the SpinSystem instance does not include 
    an valid spin quantum number.
    '''
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    spin = 0.2
    
    with pytest.raises(ValueError, match='0.2 is not a valid spin quantum number. Only integer or half-integer values are accepted.') :
        system = SpinSystem(latt_vecs,sites,spin)

def test_find_NN_shell_0() -> None :
    '''
    Tests that the proper Exception is raised when the find_NN_shell method is provided with 
    an invalid spin index, namely larger or equal than the total number of spins in the system.
    '''
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    with pytest.raises(ValueError, match='2 is not a valid index for the spins of the system under study.') :
        shell_indices, shell_vectors = system.find_NN_shell(2,1,3,3)

def test_find_NN_shell_1() -> None :
    '''
    Tests that the proper Exception is raised when the find_NN_shell method is provided with
    an invalid NN shell indicator, namely a non-positive integer.
    '''
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    with pytest.raises(ValueError, match='0 is not a valid value for the NN shell to be studied. Only positive integer values are accepted.') :
        shell_indices, shell_vectors = system.find_NN_shell(0,0,3,3)

def test_find_NN_shell_2() -> None :
    '''
    Tests that the method correctly identifies the indices for the first NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = system.find_NN_shell(0,1,3,1)
    shell1_indices_exp = np.array([1,1])
    assert np.allclose(shell1_indices, shell1_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_3() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the first NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = system.find_NN_shell(0,1,3,1)
    shell1_vectors = np.array([list(el) for el in shell1_vectors])
    shell1_vectors_exp = np.array([[-0.5,0.0,0.0],[0.5,0.0,0.0]])
    assert np.allclose(shell1_vectors, shell1_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_4() -> None :
    '''
    Tests that the method correctly identifies the indices for the second NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = system.find_NN_shell(0,2,3,1)
    shell2_indices_exp = np.array([0,0])
    assert np.allclose(shell2_indices, shell2_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_5() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the second NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = system.find_NN_shell(0,2,3,1)
    shell2_vectors = np.array([list(el) for el in shell2_vectors])
    shell2_vectors_exp = np.array([[1.0,0.0,0.0],[-1.0,0.0,0.0]])
    assert np.allclose(shell2_vectors, shell2_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_6() -> None :
    '''
    Tests that the method correctly identifies the indices for the third NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = system.find_NN_shell(0,3,3,1)
    shell3_indices_exp = np.array([1,1])
    assert np.allclose(shell3_indices, shell3_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_7() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the third NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = system.find_NN_shell(0,3,3,1)
    shell3_vectors = np.array([list(el) for el in shell3_vectors])
    shell3_vectors_exp = np.array([[-1.5,0.0,0.0],[1.5,0.0,0.0]])
    assert np.allclose(shell3_vectors, shell3_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_8() -> None :
    '''
    Tests that the method correctly identifies the indices for the first NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = system.find_NN_shell(0,1,3,2)
    shell1_indices_exp = np.array([1,2,1,2])
    assert np.allclose(shell1_indices, shell1_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_9() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the first NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = system.find_NN_shell(0,1,3,2)
    shell1_vectors = np.array([list(el) for el in shell1_vectors])
    shell1_vectors_exp = np.array([[-0.5,0.0,0.0],[0.0,-0.5,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0]])
    assert np.allclose(shell1_vectors, shell1_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_10() -> None :
    '''
    Tests that the method correctly identifies the indices for the second NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = system.find_NN_shell(0,2,3,2)
    shell2_indices_exp = np.array([3,3,3,3])
    assert np.allclose(shell2_indices, shell2_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_11() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the second NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = system.find_NN_shell(0,2,3,2)
    shell2_vectors = np.array([list(el) for el in shell2_vectors])
    shell2_vectors_exp = np.array([[-0.5,-0.5,0.0],
                                   [0.5,0.5,0.0],
                                   [0.5,-0.5,0.0],
                                   [-0.5,0.5,0.0]])
    assert np.allclose(shell2_vectors, shell2_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_12() -> None :
    '''
    Tests that the method correctly identifies the indices for the third NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = system.find_NN_shell(0,3,3,2)
    shell3_indices_exp = np.array([0,0,0,0])
    assert np.allclose(shell3_indices, shell3_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_13() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the third NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = system.find_NN_shell(0,3,3,2)
    shell3_vectors = np.array([list(el) for el in shell3_vectors])
    shell3_vectors_exp = np.array([[1.0,0.0,0.0],
                                   [0.0,1.0,0.0],
                                   [0.0,-1.0,0.0],
                                   [-1.0,0.0,0.0]])
    assert np.allclose(shell3_vectors, shell3_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_14() -> None :
    '''
    Tests that the method correctly identifies the indices for the first NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = system.find_NN_shell(0,1,3,3)
    shell1_indices_exp = np.array([1,1,1,1,1,1,1,1])
    assert np.allclose(shell1_indices, shell1_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_15() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the first NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = system.find_NN_shell(0,1,3,3)
    shell1_vectors = np.array([list(el) for el in shell1_vectors])
    shell1_vectors_exp = np.array([[-0.5,-0.5,-0.5],
                                   [0.5,0.5,0.5],
                                   [0.5,0.5,-0.5],
                                   [0.5,-0.5,0.5],
                                   [0.5,-0.5,-0.5],
                                   [-0.5,0.5,0.5],
                                   [-0.5,0.5,-0.5],
                                   [-0.5,-0.5,0.5]])
    assert np.allclose(shell1_vectors, shell1_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_16() -> None :
    '''
    Tests that the method correctly identifies the indices for the second NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = system.find_NN_shell(0,2,3,3)
    shell2_indices_exp = np.array([0,0,0,0,0,0])
    assert np.allclose(shell2_indices, shell2_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_17() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the second NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = system.find_NN_shell(0,2,3,3)
    shell2_vectors = np.array([list(el) for el in shell2_vectors])
    shell2_vectors_exp = np.array([[1.0,0.0,0.0],
                                   [0.0,1.0,0.0],
                                   [0.0,0.0,1.0],
                                   [0.0,0.0,-1.0],
                                   [0.0,-1.0,0.0],
                                   [-1.0,0.0,0.0]])
    assert np.allclose(shell2_vectors, shell2_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_18() -> None :
    '''
    Tests that the method correctly identifies the indices for the third NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = system.find_NN_shell(0,3,3,3)
    shell3_indices_exp = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    assert np.allclose(shell3_indices, shell3_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_19() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the third NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = system.find_NN_shell(0,3,3,3)
    shell3_vectors = np.array([list(el) for el in shell3_vectors])
    shell3_vectors_exp = np.array([[1.0,1.0,0.0],
                                   [1.0,0.0,1.0],
                                   [1.0,0.0,-1.0],
                                   [1.0,-1.0,0.0],
                                   [0.0,1.0,1.0],
                                   [0.0,1.0,-1.0],
                                   [0.0,-1.0,1.0],
                                   [0.0,-1.0,-1.0],
                                   [-1.0,1.0,0.0],
                                   [-1.0,0.0,1.0],
                                   [-1.0,0.0,-1.0],
                                   [-1.0,-1.0,0.0]])
    assert np.allclose(shell3_vectors, shell3_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_20() -> None :
    '''
    Tests that the method correctly identifies the indices for the first NN spins in a 3D rhombohedral lattice,
    where all the first NN sites do not lie within the same unit cell.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,6.0],[-0.5,0.5*np.sqrt(3),6.0],[-0.5,-0.5*np.sqrt(3),6.0]])
    sites = np.array([[0.0,0.0,6.0],[0.0,0.0,12.0]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = system.find_NN_shell(0,1,4,3)
    shell1_indices_exp = np.array([1,1,1])
    assert np.allclose(shell1_indices, shell1_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_21() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the first NN spins in a 3D rhombohedral lattice,
    where all the first NN sites do not lie within the same unit cell.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,6.0],[-0.5,0.5*np.sqrt(3),6.0],[-0.5,-0.5*np.sqrt(3),6.0]])
    sites = np.array([[0.0,0.0,6.0],[0.0,0.0,12.0]])
    system = SpinSystem(latt_vecs,sites,0.5)

    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = system.find_NN_shell(0,1,4,3)
    shell1_vectors = np.array([list(el) for el in shell1_vectors])
    shell1_vectors_exp = np.array([[-0.5,0.5*np.sqrt(3),0.0],[-0.5,-0.5*np.sqrt(3),0.0],[1.0,0.0,0.0]])
    assert np.allclose(shell1_vectors, shell1_vectors_exp, atol=1e-10, rtol=1e-10)

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
                           [1.0,0.0]], dtype=complex)
    Sy_exp = 0.5*np.array([[0.0,-1.0j],
                           [1.0j,0.0]], dtype=complex)
    Sz_exp = 0.5*np.array([[1.0,0.0],
                           [0.0,-1.0]], dtype=complex)
    S_exp = np.array([Sx_exp, Sy_exp, Sz_exp])

    # Effective spin vector operator
    S_eff = system.build_spin_operator()
    
    assert np.allclose(S_eff, S_exp, atol=1e-10, rtol=1e-10)

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

    assert np.allclose(S_eff, S_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_0() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 1D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only the first NN shell is taken into account.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [[np.eye(3),np.eye(3)]]
    NN_vectors = [[[-0.5,0.0,0.0],[0.5,0.0,0.0]]]

    # Expected vs Computed Spin Hamiltonian (only first NN shell)
    H_1_exp = np.array([[0.5,0.0,0.0,0.0],
                        [0.0,-0.5,1.0,0.0],
                        [0.0,1.0,-0.5,0.0],
                        [0.0,0.0,0.0,0.5]])
    H_1 = system.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),1,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_1() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 1D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only first and second NN shells are taken into account.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.5,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [[np.eye(3),np.eye(3)],[np.eye(3),np.eye(3)]]
    NN_vectors = [[[-0.5,0.0,0.0],[0.5,0.0,0.0]],[[-1.0,0.0,0.0],[1.0,0.0,0.0]]]

    # Expected vs Computed Spin Hamiltonian (up to second NN shell)
    H_2_exp = np.array([[1.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,-0.5,1.0,0.0,1.0,0.0,0.0,0.0],
                        [0.0,1.0,-0.5,0.0,1.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,-0.5,0.0,1.0,1.0,0.0],
                        [0.0,1.0,1.0,0.0,-0.5,0.0,0.0,0.0],
                        [0.0,0.0,0.0,1.0,0.0,-0.5,1.0,0.0],
                        [0.0,0.0,0.0,1.0,0.0,1.0,-0.5,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.5]])
    H_2 = system.build_hamiltonian(J_couplings,NN_vectors,2,3,np.zeros(3),1,1e-4)

    assert np.allclose(H_2, H_2_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_2() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 2D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only the first NN shell is taken into account.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [[np.eye(3),np.eye(3),np.eye(3),np.eye(3)]]
    NN_vectors = [[[0.5,0.5,0.0],[0.5,-0.5,0.0],[-0.5,0.5,0.0],[-0.5,-0.5,0.0]]]

    # Expected vs Computed Spin Hamiltonian (only first NN shell)
    H_1_exp = np.array([[1.0,0.0,0.0,0.0],
                        [0.0,-1.0,2.0,0.0],
                        [0.0,2.0,-1.0,0.0],
                        [0.0,0.0,0.0,1.0]])
    H_1 = system.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),2,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_3() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    three-sites 2D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only first and second NN shells are taken into account.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.5,0.0,0.0],[0.0,1.5,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.0],[1.0,1.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [[np.eye(3) for i in range(4)],[np.eye(3) for i in range(4)]]
    NN_vectors = [[[0.5,0.5,0.0],[0.5,-0.5,0.0],[-0.5,0.5,0.0],[-0.5,-0.5,0.0]],
                  [[1.0,-0.5,0.0],[-1.0,0.5,0.0],[-0.5,1.0,0.0],[0.5,-1.0,0.0]]]

    # Expected vs Computed Spin Hamiltonian (up to second NN shell)
    H_2_exp = np.array([[2.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,-0.75,1.5,0.0,1.5,0.0,0.0,0.0],
                        [0.0,1.5,-0.75,0.0,1.5,0.0,0.0,0.0],
                        [0.0,0.0,0.0,-0.75,0.0,1.5,1.5,0.0],
                        [0.0,1.5,1.5,0.0,-0.75,0.0,0.0,0.0],
                        [0.0,0.0,0.0,1.5,0.0,-0.75,1.5,0.0],
                        [0.0,0.0,0.0,1.5,0.0,1.5,-0.75,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.25]])
    H_2 = system.build_hamiltonian(J_couplings,NN_vectors,2,3,np.zeros(3),2,1e-4)

    assert np.allclose(H_2, H_2_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_4() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    four-sites 2D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only the first NN shells are taken into account.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [[np.eye(3),np.eye(3),np.eye(3),np.eye(3)]]
    NN_vectors = [[[0.5,0.0,0.0],[-0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,-0.5,0.0]]]

    # Expected vs Computed Spin Hamiltonian (only first NN shell)
    H_1_exp = np.array([[2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,1.0,0.0,1.0,-2.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0],
                        [0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,-2.0,1.0,0.0,1.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0]])
    H_1 = system.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),2,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_5() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 3D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only the first NN shell is taken into account.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [[np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3)]]
    NN_vectors = [[[0.5,0.5,0.5],[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,0.5],[-0.5,-0.5,-0.5]]]

    # Expected vs Computed Spin Hamiltonian (only first NN shell)
    H_1_exp = np.array([[2.0,0.0,0.0,0.0],
                        [0.0,-2.0,4.0,0.0],
                        [0.0,4.0,-2.0,0.0],
                        [0.0,0.0,0.0,2.0]])
    H_1 = system.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),3,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_6() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 3D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only first and second NN shells are taken into account.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.5,0.0,0.0],[0.0,1.5,0.0],[0.0,0.0,1.5]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5],[1.0,1.0,1.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [[np.eye(3) for i in range(8)],
                   [np.eye(3) for i in range(6)]]
    NN_vectors = [[[0.5,0.5,0.5],[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,0.5],[-0.5,-0.5,-0.5]],
                  [[1.0,-0.5,-0.5],[-0.5,1.0,-0.5],[-0.5,-0.5,1.0],[0.5,0.5,-1.0],[0.5,-1.0,0.5],[-1.0,0.5,0.5]]]

    # Expected vs Computed Spin Hamiltonian (up to second NN shell)
    H_2_exp = np.array([[3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,-1.0,2.0,0.0,2.0,0.0,0.0,0.0],
                        [0.0,2.0,-1.0,0.0,2.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,-1.0,0.0,2.0,2.0,0.0],
                        [0.0,2.0,2.0,0.0,-1.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,2.0,0.0,-1.0,2.0,0.0],
                        [0.0,0.0,0.0,2.0,0.0,2.0,-1.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.0]])
    H_2 = system.build_hamiltonian(J_couplings,NN_vectors,2,3,np.zeros(3),3,1e-4)

    assert np.allclose(H_2, H_2_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_7() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 1D S=1 system and Heisenberg-like exchange interaction matrices. 
    Only the first NN shell is taken into account.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[10.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.0,1.0,0.0]])
    spin = 1.0
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [[np.eye(3),np.eye(3)]]
    NN_vectors = [[[0.0,1.0,0.0],[0.0,-1.0,0.0]]]

    # Expected vs Computed Spin Hamiltonian (only first NN shell)
    H_1_exp = np.array([[2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,2.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,-2.0,0.0,2.0,0.0,0.0,0.0,0.0],
                        [0.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,2.0,0.0,0.0,0.0,2.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0],
                        [0.0,0.0,0.0,0.0,2.0,0.0,-2.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,2.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0]])
    H_1 = system.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),1,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)
    
def test_build_hamiltonian_8() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 1D S=1/2 system and exchange interaction matrices that also include 
    anisotropic contributions. 
    Only the first NN shell is taken into account.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[10.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.0,1.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [[np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]),
                    np.array([[1.0,1.0,-1.0],[1.0,1.0,-1.0],[-1.0,-1.0,1.0]])]]
    NN_vectors = [[[0.0,1.0,0.0],[0.0,-1.0,0.0]]]

    # Expected vs Computed Spin Hamiltonian (only first NN shell)
    H_1_exp = np.array([[0.5,0.0,0.0,-1.0j],
                        [0.0,-0.5,1.0,0.0],
                        [0.0,1.0,-0.5,0.0],
                        [1.0j,0.0,0.0,0.5]])
    H_1 = system.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),1,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_9() -> None :
    '''
    Tests that the proper Exception is raised when the method is about to include the pair interaction term
    between the reference spin and one of its replica.
    No (quadratic) pair interaction is allowed when the spins in question are associated to the same label.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)
        
    # Set the J couplings and the associated NN vectors
    J_couplings = [[np.eye(3) for i in range(8)],[np.eye(3) for i in range(6)]]
    NN_vecs = [[np.array([-0.5,-0.5,-0.5]),
                np.array([0.5,0.5,0.5]),
                np.array([0.5,0.5,-0.5]),
                np.array([0.5,-0.5,0.5]),
                np.array([0.5,-0.5,-0.5]),
                np.array([-0.5,0.5,0.5]),
                np.array([-0.5,0.5,-0.5]),
                np.array([-0.5,-0.5,0.5])],
               [np.array([1.0,0.0,0.0]),
                np.array([0.0,1.0,0.0]),
                np.array([0.0,0.0,1.0]),
                np.array([0.0,0.0,-1.0]),
                np.array([0.0,-1.0,0.0]),
                np.array([-1.0,0.0,0.0])]]

    with pytest.raises(ValueError, match='The 2° NN shell for spin 0 includes some/all of its replica. Consider decreasing the max_NN_shell value or taking a larger unit cell.') :
        H = system.build_hamiltonian(J_couplings,NN_vecs,2,3,np.zeros(3),3,1e-4)

def test_compute_J_eff_0() -> None :
    '''
    Tests that the proper Exception is raised when the vector passed as agument
    to the method of interest is not included in the available NN vectors.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_couplings = [np.eye(3),np.eye(3)]
    NN_vectors = [np.array([0.0,0.5,0.0]),np.array([0.0,0.0,0.5])]
    vector = np.array([0.5,0.0,0.0])
    
    with pytest.raises(ValueError, match='\\[0.5 0.  0. \\] could not be found among the input NN vectors.') :
        J_eff = system.compute_J_eff(J_couplings, NN_vectors, vector, 3)

def test_compute_J_eff_1() -> None :
    '''
    Tests that the method correctly returns the effective intersite exchange tensor 
    in its most general form.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the J couplings
    J_exp = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
    J_couplings = [J_exp, np.eye(3), np.eye(3)]
    NN_vectors = [np.array([0.5,0.0,0.0]), np.array([0.0,0.5,0.0]), np.array([0.0,0.0,0.5])]
    vector = np.array([0.5,0.0,0.0])
    J_eff = system.compute_J_eff(J_couplings, NN_vectors, vector, 3)

    assert np.allclose(J_eff, J_exp, atol=1e-10, rtol=1e-10)

def test_compute_pair_interaction_0() -> None :
    '''
    Tests that the contribution to the spin Hamiltonian from the exchange interaction 
    between two NN spins in a three-sites 1D S=1/2 system is properly computed.
    In particular, the indices of the chosen spins are adjacent and one of them 
    represents the last spin in the sequence.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[3.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the effective J coupling
    J_eff = 4.0*np.eye(3)

    # Expected vs Computed contribution to the Spin Hamiltonian
    H_term_exp = np.array([[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                           [0.0,-1.0,2.0,0.0,0.0,0.0,0.0,0.0],
                           [0.0,2.0,-1.0,0.0,0.0,0.0,0.0,0.0],
                           [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
                           [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],
                           [0.0,0.0,0.0,0.0,0.0,-1.0,2.0,0.0],
                           [0.0,0.0,0.0,0.0,0.0,2.0,-1.0,0.0],
                           [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]])
    H_term = system.compute_pair_interaction(1,2,J_eff)

    assert np.allclose(H_term, H_term_exp, atol=1e-10, rtol=1e-10)

def test_compute_pair_interaction_1() -> None :
    '''
    Tests that the contribution to the spin Hamiltonian from the exchange interaction 
    between two NN spins in a three-sites 1D S=1/2 system is properly computed.
    In particular, the indices of the chosen spins are adjacent and one of them 
    represents the first spin in the sequence.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[3.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the effective J coupling
    J_eff = 4.0*np.eye(3)

    # Expected vs Computed contribution to the Spin Hamiltonian
    H_term_exp = np.array([[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                           [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                           [0.0,0.0,-1.0,0.0,2.0,0.0,0.0,0.0],
                           [0.0,0.0,0.0,-1.0,0.0,2.0,0.0,0.0],
                           [0.0,0.0,2.0,0.0,-1.0,0.0,0.0,0.0],
                           [0.0,0.0,0.0,2.0,0.0,-1.0,0.0,0.0],
                           [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                           [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]])
    H_term = system.compute_pair_interaction(0,1,J_eff)

    assert np.allclose(H_term, H_term_exp, atol=1e-10, rtol=1e-10)

def test_compute_pair_interaction_2() -> None :
    '''
    Tests that the contribution to the spin Hamiltonian from the exchange interaction 
    between two NN spins in a three-sites 1D S=1/2 system is properly computed.
    In particular, the indices of the chosen spins are not adjacent and they represent
    the first and the last spin in the sequence respectively.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[3.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the effective J coupling
    J_eff = 4.0*np.eye(3)

    # Expected vs Computed contribution to the Spin Hamiltonian
    H_term_exp = np.array([[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                           [0.0,-1.0,0.0,0.0,2.0,0.0,0.0,0.0],
                           [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
                           [0.0,0.0,0.0,-1.0,0.0,0.0,2.0,0.0],
                           [0.0,2.0,0.0,0.0,-1.0,0.0,0.0,0.0],
                           [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
                           [0.0,0.0,0.0,2.0,0.0,0.0,-1.0,0.0],
                           [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]])
    H_term = system.compute_pair_interaction(0,2,J_eff)

    assert np.allclose(H_term, H_term_exp, atol=1e-10, rtol=1e-10)

def test_compute_pair_interaction_3() -> None :
    '''
    Tests that the contribution to the spin Hamiltonian from the exchange interaction 
    between two NN spins in a three-sites 1D S=1/2 system is not allowed since their indices 
    coincide. An Exception is thus raised.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[3.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)

    # Setting the effective J coupling
    J_eff = np.eye(3)

    with pytest.raises(ValueError, match='The interaction term between spins 0 and 0 is not allowed.') :
        H_term = system.compute_pair_interaction(0,0,J_eff)

def test_compute_spin_correlation_0() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T
    
    # Spin-spin Correlation values vs Expectations
    SS_val = system.compute_spin_correlation(GS_vec,1,0,1)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_1() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with non-adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,1,0,2)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_2() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T
    
    # Spin-spin Correlation values vs Expectations
    SS_val = system.compute_spin_correlation(GS_vec,1,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_3() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)],
                       [1.0*(i==15) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,2,0,1)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_4() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with non-adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)],
                       [1.0*(i==15) for i in range(16)]]).T

    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,2,0,2)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_5() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)],
                       [1.0*(i==15) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,2,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_6() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T
    
    # Spin-spin Correlation values vs Expectations
    SS_val = system.compute_spin_correlation(GS_vec,1,0,1)
    SS_exp = np.array([0.0,0.0,-0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_7() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with non-adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,1,0,2)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_8() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # Spin-spin Correlation values vs Expectations
    SS_val = system.compute_spin_correlation(GS_vec,1,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_9() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==5) for i in range(16)],
                       [1.0*(i==10) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,2,0,1)
    SS_exp = np.array([0.0,0.0,-0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_10() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with non-adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==5) for i in range(16)],
                       [1.0*(i==10) for i in range(16)]]).T

    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,2,0,2)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_11() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==5) for i in range(16)],
                       [1.0*(i==10) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,2,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)
    
def test_compute_spin_correlation_12() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T

    # Spin-spin Correlation matrix vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,1,0,1)
    SS_exp = np.array([[0.0,0.0,0.0]])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_13() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,1,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_14() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # Spin-spin Correlation matrix vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,4,0,1)
    SS_exp = np.array([[0.0,0.0,0.0]])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_15() -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # Spin-spin Correlation value vs Expectation
    SS_val = system.compute_spin_correlation(GS_vec,4,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_magnetization_0() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T
    
    # M_x value vs Expectation
    M_x = system.compute_magnetization(GS_vec,1,'M_x')
    M_x_exp = 0.25
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_1() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T

    # M_y value vs Expectation
    M_y = system.compute_magnetization(GS_vec,1,'M_y')
    M_y_exp = 0.25
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_2() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T

    # M_z value vs Expectation
    M_z = system.compute_magnetization(GS_vec,1,'M_z')
    M_z_exp = 0.5
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_3() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T

    # M_full value vs Expectation
    M_full = system.compute_magnetization(GS_vec,1,'M_full')
    M_full_exp = np.sqrt(1.5)/2
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_4() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==0)+1.0*(i==15) for i in range(16)],
                                      [1.0*(i==0)+1.0*(i==15) for i in range(16)]]).T
    
    # M_x value vs Expectation
    M_x = system.compute_magnetization(GS_vec,2,'M_x')
    M_x_exp = 0.25
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_5() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==0)+1.0*(i==15) for i in range(16)],
                                      [1.0*(i==0)+1.0*(i==15) for i in range(16)]]).T

    # M_y value vs Expectation
    M_y = system.compute_magnetization(GS_vec,2,'M_y')
    M_y_exp = 0.25
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_6() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==0)+1.0*(i==15) for i in range(16)],
                                      [1.0*(i==0)+1.0*(i==15) for i in range(16)]]).T

    # M_z value vs Expectation
    M_z = system.compute_magnetization(GS_vec,2,'M_z')
    M_z_exp = 0.5
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_7() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==0)+1.0*(i==15) for i in range(16)],
                                      [1.0*(i==0)+1.0*(i==15) for i in range(16)]]).T

    # M_full value vs Expectation
    M_full = system.compute_magnetization(GS_vec,2,'M_full')
    M_full_exp = np.sqrt(1.5)/2
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_8() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T
    
    # M_x value vs Expectation
    M_x = system.compute_magnetization(GS_vec,1,'M_x')
    M_x_exp = 0.25
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_9() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # M_y value vs Expectation
    M_y = system.compute_magnetization(GS_vec,1,'M_y')
    M_y_exp = 0.25
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_10() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # M_z value vs Expectation
    M_z = system.compute_magnetization(GS_vec,1,'M_z')
    M_z_exp = 0.0
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_11() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # M_full value vs Expectation
    M_full = system.compute_magnetization(GS_vec,1,'M_full')
    M_full_exp = 0.5/np.sqrt(2)
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_12() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==5)+1.0*(i==10) for i in range(16)],
                                      [1.0*(i==5)-1.0*(i==10) for i in range(16)]]).T
    
    # M_x value vs Expectation
    M_x = system.compute_magnetization(GS_vec,2,'M_x')
    M_x_exp = 0.25
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_13() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==5)+1.0*(i==10) for i in range(16)],
                                      [1.0*(i==5)-1.0*(i==10) for i in range(16)]]).T

    # M_y value vs Expectation
    M_y = system.compute_magnetization(GS_vec,2,'M_y')
    M_y_exp = 0.25
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_14() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==5)+1.0*(i==10) for i in range(16)],
                                      [1.0*(i==5)-1.0*(i==10) for i in range(16)]]).T

    # M_z value vs Expectation
    M_z = system.compute_magnetization(GS_vec,2,'M_z')
    M_z_exp = 0.0
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_15() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==5)+1.0*(i==10) for i in range(16)],
                                      [1.0*(i==5)-1.0*(i==10) for i in range(16)]]).T

    # M_full value vs Expectation
    M_full = system.compute_magnetization(GS_vec,2,'M_full')
    M_full_exp = 0.5/np.sqrt(2)
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_16() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T
    
    # M_x value vs Expectation
    M_x = system.compute_magnetization(GS_vec,1,'M_x')
    M_x_exp = 1/np.sqrt(8)
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_17() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T

    # M_y value vs Expectation
    M_y = system.compute_magnetization(GS_vec,1,'M_y')
    M_y_exp = 1/np.sqrt(8)
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_18() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T

    # M_z value vs Expectation
    M_z = system.compute_magnetization(GS_vec,1,'M_z')
    M_z_exp = 1/np.sqrt(8)
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_19() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T

    # M_full value vs Expectation
    M_full = system.compute_magnetization(GS_vec,1,'M_full')
    M_full_exp = np.sqrt(1.5)/2
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_20() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T
    
    # M_x value vs Expectation
    M_x = system.compute_magnetization(GS_vec,4,'M_x')
    M_x_exp = 1/np.sqrt(8)
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_21() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # M_y value vs Expectation
    M_y = system.compute_magnetization(GS_vec,4,'M_y')
    M_y_exp = 1/np.sqrt(8)
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_22() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # M_z value vs Expectation
    M_z = system.compute_magnetization(GS_vec,4,'M_z')
    M_z_exp = 1/np.sqrt(8)
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_23() -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[2.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
    spin = 0.5
    system = SpinSystem(latt_vecs,sites,spin)
    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # M_full value vs Expectation
    M_full = system.compute_magnetization(GS_vec,4,'M_full')
    M_full_exp = np.sqrt(1.5)/2
    assert np.abs(M_full-M_full_exp)<1e-10