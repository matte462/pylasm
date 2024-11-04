from spin_system import SpinSystem
import numpy as np
import pytest

@pytest.fixture
def two_sites_1D_chain() -> SpinSystem :
    '''
    Defines a SpinSystem instance with two equally-spaced sites along the first lattice vector.
    '''
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    return SpinSystem(latt_vecs,sites,0.5)

@pytest.fixture
def three_sites_1D_chain() -> SpinSystem :
    '''
    Defines a SpinSystem instance with three equally-spaced sites along the first lattice vector.
    '''
    latt_vecs = np.array([[3.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0]])
    return SpinSystem(latt_vecs,sites,0.5)

@pytest.fixture
def four_sites_1D_chain() -> SpinSystem :
    '''
    Defines a SpinSystem instance with four equally-spaced sites along the first lattice vector.
    '''
    latt_vecs = np.array([[4.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]])
    return SpinSystem(latt_vecs,sites,0.5)

@pytest.fixture
def four_sites_2D_square() -> SpinSystem :
    '''
    Defines a SpinSystem instance with four sites, which form a square and lie on the a-b plane.
    '''
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    return SpinSystem(latt_vecs,sites,0.5)

@pytest.fixture
def two_sites_3D_cube() -> SpinSystem :
    '''
    Defines a SpinSystem instance with two equally-spaced sites along the inner diagonal of the cubic unit cell.
    '''
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    return SpinSystem(latt_vecs,sites,0.5)

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

def test_find_NN_shell_2(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the indices for the first NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = two_sites_1D_chain.find_NN_shell(0,1,3,1)
    shell1_indices_exp = np.array([1,1])
    assert np.allclose(shell1_indices, shell1_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_3(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the first NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    '''    
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = two_sites_1D_chain.find_NN_shell(0,1,3,1)
    shell1_vectors = np.array([list(el) for el in shell1_vectors])
    shell1_vectors_exp = np.array([[-0.5,0.0,0.0],[0.5,0.0,0.0]])
    assert np.allclose(shell1_vectors, shell1_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_4(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the indices for the second NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    '''    
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = two_sites_1D_chain.find_NN_shell(0,2,3,1)
    shell2_indices_exp = np.array([0,0])
    assert np.allclose(shell2_indices, shell2_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_5(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the second NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    '''    
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = two_sites_1D_chain.find_NN_shell(0,2,3,1)
    shell2_vectors = np.array([list(el) for el in shell2_vectors])
    shell2_vectors_exp = np.array([[1.0,0.0,0.0],[-1.0,0.0,0.0]])
    assert np.allclose(shell2_vectors, shell2_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_6(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the indices for the third NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    '''    
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = two_sites_1D_chain.find_NN_shell(0,3,3,1)
    shell3_indices_exp = np.array([1,1])
    assert np.allclose(shell3_indices, shell3_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_7(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the third NN spins
    in a standard 1D lattice with two equally spaced magnetic sites.
    '''
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    '''    
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = two_sites_1D_chain.find_NN_shell(0,3,3,1)
    shell3_vectors = np.array([list(el) for el in shell3_vectors])
    shell3_vectors_exp = np.array([[-1.5,0.0,0.0],[1.5,0.0,0.0]])
    assert np.allclose(shell3_vectors, shell3_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_8(four_sites_2D_square) -> None :
    '''
    Tests that the method correctly identifies the indices for the first NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = four_sites_2D_square.find_NN_shell(0,1,3,2)
    shell1_indices_exp = np.array([1,2,1,2])
    assert np.allclose(shell1_indices, shell1_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_9(four_sites_2D_square) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the first NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = four_sites_2D_square.find_NN_shell(0,1,3,2)
    shell1_vectors = np.array([list(el) for el in shell1_vectors])
    shell1_vectors_exp = np.array([[-0.5,0.0,0.0],[0.0,-0.5,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0]])
    assert np.allclose(shell1_vectors, shell1_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_10(four_sites_2D_square) -> None :
    '''
    Tests that the method correctly identifies the indices for the second NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = four_sites_2D_square.find_NN_shell(0,2,3,2)
    shell2_indices_exp = np.array([3,3,3,3])
    assert np.allclose(shell2_indices, shell2_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_11(four_sites_2D_square) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the second NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = four_sites_2D_square.find_NN_shell(0,2,3,2)
    shell2_vectors = np.array([list(el) for el in shell2_vectors])
    shell2_vectors_exp = np.array([[-0.5,-0.5,0.0],
                                   [0.5,0.5,0.0],
                                   [0.5,-0.5,0.0],
                                   [-0.5,0.5,0.0]])
    assert np.allclose(shell2_vectors, shell2_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_12(four_sites_2D_square) -> None :
    '''
    Tests that the method correctly identifies the indices for the third NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    '''    
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = four_sites_2D_square.find_NN_shell(0,3,3,2)
    shell3_indices_exp = np.array([0,0,0,0])
    assert np.allclose(shell3_indices, shell3_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_13(four_sites_2D_square) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the third NN spins
    in a standard 2D square lattice with four magnetic sites.
    '''
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    '''    
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = four_sites_2D_square.find_NN_shell(0,3,3,2)
    shell3_vectors = np.array([list(el) for el in shell3_vectors])
    shell3_vectors_exp = np.array([[1.0,0.0,0.0],
                                   [0.0,1.0,0.0],
                                   [0.0,-1.0,0.0],
                                   [-1.0,0.0,0.0]])
    assert np.allclose(shell3_vectors, shell3_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_14(two_sites_3D_cube) -> None :
    '''
    Tests that the method correctly identifies the indices for the first NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = two_sites_3D_cube.find_NN_shell(0,1,3,3)
    shell1_indices_exp = np.array([1,1,1,1,1,1,1,1])
    assert np.allclose(shell1_indices, shell1_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_15(two_sites_3D_cube) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the first NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # First NN shell is compared to the expectations
    shell1_indices, shell1_vectors = two_sites_3D_cube.find_NN_shell(0,1,3,3)
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

def test_find_NN_shell_16(two_sites_3D_cube) -> None :
    '''
    Tests that the method correctly identifies the indices for the second NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = two_sites_3D_cube.find_NN_shell(0,2,3,3)
    shell2_indices_exp = np.array([0,0,0,0,0,0])
    assert np.allclose(shell2_indices, shell2_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_17(two_sites_3D_cube) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the second NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    '''    
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Second NN shell is compared to the expectations
    shell2_indices, shell2_vectors = two_sites_3D_cube.find_NN_shell(0,2,3,3)
    shell2_vectors = np.array([list(el) for el in shell2_vectors])
    shell2_vectors_exp = np.array([[1.0,0.0,0.0],
                                   [0.0,1.0,0.0],
                                   [0.0,0.0,1.0],
                                   [0.0,0.0,-1.0],
                                   [0.0,-1.0,0.0],
                                   [-1.0,0.0,0.0]])
    assert np.allclose(shell2_vectors, shell2_vectors_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_18(two_sites_3D_cube) -> None :
    '''
    Tests that the method correctly identifies the indices for the third NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = two_sites_3D_cube.find_NN_shell(0,3,3,3)
    shell3_indices_exp = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    assert np.allclose(shell3_indices, shell3_indices_exp, atol=1e-10, rtol=1e-10)

def test_find_NN_shell_19(two_sites_3D_cube) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors for the third NN spins
    in a standard 3D cubic lattice with two magnetic sites.
    '''
    # Third NN shell is compared to the expectations
    shell3_indices, shell3_vectors = two_sites_3D_cube.find_NN_shell(0,3,3,3)
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

def test_build_hamiltonian_0(two_sites_1D_chain) -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 1D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only the first NN shell is taken into account.
    '''
    # Setting the J couplings
    J_couplings = [[np.eye(3),np.eye(3)]]
    NN_vectors = [[[-0.5,0.0,0.0],[0.5,0.0,0.0]]]

    # Expected vs Computed Spin Hamiltonian (only first NN shell)
    H_1_exp = np.array([[0.5,0.0,0.0,0.0],
                        [0.0,-0.5,1.0,0.0],
                        [0.0,1.0,-0.5,0.0],
                        [0.0,0.0,0.0,0.5]])
    H_1 = two_sites_1D_chain.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),1,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_1() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    three-sites 1D S=1/2 system and Heisenberg-like exchange interaction matrices. 
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

def test_build_hamiltonian_4(four_sites_2D_square) -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    four-sites 2D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only the first NN shells are taken into account.
    '''
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
    H_1 = four_sites_2D_square.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),2,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_5(two_sites_3D_cube) -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 3D S=1/2 system and Heisenberg-like exchange interaction matrices. 
    Only the first NN shell is taken into account.
    '''
    # Setting the J couplings
    J_couplings = [[np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3)]]
    NN_vectors = [[[0.5,0.5,0.5],[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,0.5],[-0.5,-0.5,-0.5]]]

    # Expected vs Computed Spin Hamiltonian (only first NN shell)
    H_1_exp = np.array([[2.0,0.0,0.0,0.0],
                        [0.0,-2.0,4.0,0.0],
                        [0.0,4.0,-2.0,0.0],
                        [0.0,0.0,0.0,2.0]])
    H_1 = two_sites_3D_cube.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),3,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_6() -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    three-sites 3D S=1/2 system and Heisenberg-like exchange interaction matrices. 
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
    
def test_build_hamiltonian_8(two_sites_1D_chain) -> None :
    '''
    Tests that the method returns the proper hamiltonian matrix when provided with a 
    two-sites 1D S=1/2 system and exchange interaction matrices that also include 
    anisotropic contributions. 
    Only the first NN shell is taken into account.
    '''
    # Setting the J couplings
    J_couplings = [[np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]),
                    np.array([[1.0,1.0,-1.0],[1.0,1.0,-1.0],[-1.0,-1.0,1.0]])]]
    NN_vectors = [[[-0.5,0.0,0.0],[0.5,0.0,0.0]]]

    # Expected vs Computed Spin Hamiltonian (only first NN shell)
    H_1_exp = np.array([[0.5,0.0,0.0,-1.0j],
                        [0.0,-0.5,1.0,0.0],
                        [0.0,1.0,-0.5,0.0],
                        [1.0j,0.0,0.0,0.5]])
    H_1 = two_sites_1D_chain.build_hamiltonian(J_couplings,NN_vectors,1,3,np.zeros(3),1,1e-4)

    assert np.allclose(H_1, H_1_exp, atol=1e-10, rtol=1e-10)

def test_build_hamiltonian_9(two_sites_3D_cube) -> None :
    '''
    Tests that the proper Exception is raised when the method is about to include the pair interaction term
    between the reference spin and one of its replica.
    No (quadratic) pair interaction is allowed when the spins in question are associated to the same label.
    '''        
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
        H = two_sites_3D_cube.build_hamiltonian(J_couplings,NN_vecs,2,3,np.zeros(3),3,1e-4)

def test_compute_J_eff_0(two_sites_1D_chain) -> None :
    '''
    Tests that the proper Exception is raised when the vector passed as agument
    to the method of interest is not included in the available NN vectors.
    '''
    # Setting the J couplings
    J_couplings = [np.eye(3),np.eye(3)]
    NN_vectors = [np.array([0.0,0.5,0.0]),np.array([0.0,0.0,0.5])]
    vector = np.array([0.5,0.0,0.0])
    
    with pytest.raises(ValueError, match='\\[0.5 0.  0. \\] could not be found among the input NN vectors.') :
        J_eff = two_sites_1D_chain.compute_J_eff(J_couplings, NN_vectors, vector, 3)

def test_compute_J_eff_1(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly returns the effective intersite exchange tensor 
    in its most general form.
    '''
    # Setting the J couplings
    J_exp = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
    J_couplings = [J_exp, np.eye(3), np.eye(3)]
    NN_vectors = [np.array([0.5,0.0,0.0]), np.array([0.0,0.5,0.0]), np.array([0.0,0.0,0.5])]
    vector = np.array([0.5,0.0,0.0])
    J_eff = two_sites_1D_chain.compute_J_eff(J_couplings, NN_vectors, vector, 3)

    assert np.allclose(J_eff, J_exp, atol=1e-10, rtol=1e-10)

def test_compute_pair_interaction_0(three_sites_1D_chain) -> None :
    '''
    Tests that the contribution to the spin Hamiltonian from the exchange interaction 
    between two NN spins in a three-sites 1D S=1/2 system is properly computed.
    In particular, the indices of the chosen spins are adjacent and one of them 
    represents the last spin in the sequence.
    '''
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
    H_term = three_sites_1D_chain.compute_pair_interaction(1,2,J_eff)

    assert np.allclose(H_term, H_term_exp, atol=1e-10, rtol=1e-10)

def test_compute_pair_interaction_1(three_sites_1D_chain) -> None :
    '''
    Tests that the contribution to the spin Hamiltonian from the exchange interaction 
    between two NN spins in a three-sites 1D S=1/2 system is properly computed.
    In particular, the indices of the chosen spins are adjacent and one of them 
    represents the first spin in the sequence.
    '''
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
    H_term = three_sites_1D_chain.compute_pair_interaction(0,1,J_eff)

    assert np.allclose(H_term, H_term_exp, atol=1e-10, rtol=1e-10)

def test_compute_pair_interaction_2(three_sites_1D_chain) -> None :
    '''
    Tests that the contribution to the spin Hamiltonian from the exchange interaction 
    between two NN spins in a three-sites 1D S=1/2 system is properly computed.
    In particular, the indices of the chosen spins are not adjacent and they represent
    the first and the last spin in the sequence respectively.
    '''
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
    H_term = three_sites_1D_chain.compute_pair_interaction(0,2,J_eff)

    assert np.allclose(H_term, H_term_exp, atol=1e-10, rtol=1e-10)

def test_compute_pair_interaction_3(three_sites_1D_chain) -> None :
    '''
    Tests that the contribution to the spin Hamiltonian from the exchange interaction 
    between two NN spins in a three-sites 1D S=1/2 system is not allowed since their indices 
    coincide. An Exception is thus raised.
    '''
    # Setting the effective J coupling
    J_eff = np.eye(3)

    with pytest.raises(ValueError, match='The interaction term between spins 0 and 0 is not allowed.') :
        H_term = three_sites_1D_chain.compute_pair_interaction(0,0,J_eff)

def test_compute_spin_correlation_0(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T
    
    # Spin-spin Correlation values vs Expectations
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,1,0,1)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_1(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with non-adjacent indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,1,0,2)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_2(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T
    
    # Spin-spin Correlation values vs Expectations
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,1,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_3(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)],
                       [1.0*(i==15) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,2,0,1)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_4(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with non-adjacent indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)],
                       [1.0*(i==15) for i in range(16)]]).T

    # Spin-spin Correlation value vs Expectation
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,2,0,2)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_5(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)],
                       [1.0*(i==15) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,2,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_6(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T
    
    # Spin-spin Correlation values vs Expectations
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,1,0,1)
    SS_exp = np.array([0.0,0.0,-0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_7(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with non-adjacent indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # Spin-spin Correlation value vs Expectation
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,1,0,2)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_8(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # Spin-spin Correlation values vs Expectations
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,1,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_9(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==5) for i in range(16)],
                       [1.0*(i==10) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,2,0,1)
    SS_exp = np.array([0.0,0.0,-0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_10(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with non-adjacent indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==5) for i in range(16)],
                       [1.0*(i==10) for i in range(16)]]).T

    # Spin-spin Correlation value vs Expectation
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,2,0,2)
    SS_exp = np.array([0.0,0.0,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_11(four_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==5) for i in range(16)],
                       [1.0*(i==10) for i in range(16)]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = four_sites_1D_chain.compute_spin_correlation(GS_vec,2,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)
    
def test_compute_spin_correlation_12(two_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T

    # Spin-spin Correlation matrix vs Expectation
    SS_val = two_sites_1D_chain.compute_spin_correlation(GS_vec,1,0,1)
    SS_exp = np.array([[0.0,0.0,0.0]])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_13(two_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T
    
    # Spin-spin Correlation value vs Expectation
    SS_val = two_sites_1D_chain.compute_spin_correlation(GS_vec,1,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_14(two_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with adjacent indices.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # Spin-spin Correlation matrix vs Expectation
    SS_val = two_sites_1D_chain.compute_spin_correlation(GS_vec,4,0,1)
    SS_exp = np.array([[0.0,0.0,0.0]])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_spin_correlation_15(two_sites_1D_chain) -> None :
    '''
    Tests that the method in question allows a correct estimation of the spin-spin correlation values
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case involves spins with same indices.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # Spin-spin Correlation value vs Expectation
    SS_val = two_sites_1D_chain.compute_spin_correlation(GS_vec,4,1,1)
    SS_exp = np.array([0.25,0.25,0.25])
    assert np.allclose(SS_val, SS_exp, atol=1e-10, rtol=1e-10)

def test_compute_magnetization_0(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T
    
    # M_x value vs Expectation
    M_x = four_sites_1D_chain.compute_magnetization(GS_vec,1,'M_x')
    M_x_exp = 0.25
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_1(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T

    # M_y value vs Expectation
    M_y = four_sites_1D_chain.compute_magnetization(GS_vec,1,'M_y')
    M_y_exp = 0.25
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_2(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T

    # M_z value vs Expectation
    M_z = four_sites_1D_chain.compute_magnetization(GS_vec,1,'M_z')
    M_z_exp = 0.5
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_3(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==0) for i in range(16)]]).T

    # M_full value vs Expectation
    M_full = four_sites_1D_chain.compute_magnetization(GS_vec,1,'M_full')
    M_full_exp = np.sqrt(1.5)/2
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_4(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==0)+1.0*(i==15) for i in range(16)],
                                      [1.0*(i==0)+1.0*(i==15) for i in range(16)]]).T
    
    # M_x value vs Expectation
    M_x = four_sites_1D_chain.compute_magnetization(GS_vec,2,'M_x')
    M_x_exp = 0.25
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_5(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==0)+1.0*(i==15) for i in range(16)],
                                      [1.0*(i==0)+1.0*(i==15) for i in range(16)]]).T

    # M_y value vs Expectation
    M_y = four_sites_1D_chain.compute_magnetization(GS_vec,2,'M_y')
    M_y_exp = 0.25
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_6(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==0)+1.0*(i==15) for i in range(16)],
                                      [1.0*(i==0)+1.0*(i==15) for i in range(16)]]).T

    # M_z value vs Expectation
    M_z = four_sites_1D_chain.compute_magnetization(GS_vec,2,'M_z')
    M_z_exp = 0.5
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_7(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==0)+1.0*(i==15) for i in range(16)],
                                      [1.0*(i==0)+1.0*(i==15) for i in range(16)]]).T

    # M_full value vs Expectation
    M_full = four_sites_1D_chain.compute_magnetization(GS_vec,2,'M_full')
    M_full_exp = np.sqrt(1.5)/2
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_8(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T
    
    # M_x value vs Expectation
    M_x = four_sites_1D_chain.compute_magnetization(GS_vec,1,'M_x')
    M_x_exp = 0.25
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_9(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # M_y value vs Expectation
    M_y = four_sites_1D_chain.compute_magnetization(GS_vec,1,'M_y')
    M_y_exp = 0.25
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_10(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # M_z value vs Expectation
    M_z = four_sites_1D_chain.compute_magnetization(GS_vec,1,'M_z')
    M_z_exp = 0.0
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_11(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''    
    # GS definition
    GS_vec = np.array([[1.0*(i==10) for i in range(16)]]).T

    # M_full value vs Expectation
    M_full = four_sites_1D_chain.compute_magnetization(GS_vec,1,'M_full')
    M_full_exp = 0.5/np.sqrt(2)
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_12(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==5)+1.0*(i==10) for i in range(16)],
                                      [1.0*(i==5)-1.0*(i==10) for i in range(16)]]).T
    
    # M_x value vs Expectation
    M_x = four_sites_1D_chain.compute_magnetization(GS_vec,2,'M_x')
    M_x_exp = 0.25
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_13(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==5)+1.0*(i==10) for i in range(16)],
                                      [1.0*(i==5)-1.0*(i==10) for i in range(16)]]).T

    # M_y value vs Expectation
    M_y = four_sites_1D_chain.compute_magnetization(GS_vec,2,'M_y')
    M_y_exp = 0.25
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_14(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==5)+1.0*(i==10) for i in range(16)],
                                      [1.0*(i==5)-1.0*(i==10) for i in range(16)]]).T

    # M_z value vs Expectation
    M_z = four_sites_1D_chain.compute_magnetization(GS_vec,2,'M_z')
    M_z_exp = 0.0
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_15(four_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is doubly-degenerate and consists of four anti-parallel spins aligned to the z axis,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0*(i==5)+1.0*(i==10) for i in range(16)],
                                      [1.0*(i==5)-1.0*(i==10) for i in range(16)]]).T

    # M_full value vs Expectation
    M_full = four_sites_1D_chain.compute_magnetization(GS_vec,2,'M_full')
    M_full_exp = 0.5/np.sqrt(2)
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_16(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T
    
    # M_x value vs Expectation
    M_x = two_sites_1D_chain.compute_magnetization(GS_vec,1,'M_x')
    M_x_exp = 1/np.sqrt(8)
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_17(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T

    # M_y value vs Expectation
    M_y = two_sites_1D_chain.compute_magnetization(GS_vec,1,'M_y')
    M_y_exp = 1/np.sqrt(8)
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_18(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T

    # M_z value vs Expectation
    M_z = two_sites_1D_chain.compute_magnetization(GS_vec,1,'M_z')
    M_z_exp = 1/np.sqrt(8)
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_19(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is non-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0]]).T

    # M_full value vs Expectation
    M_full = two_sites_1D_chain.compute_magnetization(GS_vec,1,'M_full')
    M_full_exp = np.sqrt(1.5)/2
    assert np.abs(M_full-M_full_exp)<1e-10

def test_compute_magnetization_20(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sx-Sx contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T
    
    # M_x value vs Expectation
    M_x = two_sites_1D_chain.compute_magnetization(GS_vec,4,'M_x')
    M_x_exp = 1/np.sqrt(8)
    assert np.abs(M_x-M_x_exp)<1e-10

def test_compute_magnetization_21(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sy-Sy contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # M_y value vs Expectation
    M_y = two_sites_1D_chain.compute_magnetization(GS_vec,4,'M_y')
    M_y_exp = 1/np.sqrt(8)
    assert np.abs(M_y-M_y_exp)<1e-10

def test_compute_magnetization_22(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the Sz-Sz contributions to the magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # M_z value vs Expectation
    M_z = two_sites_1D_chain.compute_magnetization(GS_vec,4,'M_z')
    M_z_exp = 1/np.sqrt(8)
    assert np.abs(M_z-M_z_exp)<1e-10

def test_compute_magnetization_23(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly estimates the magnetization strength as requested by the user
    when the ground state (GS) is 4-fold-degenerate and consists of two perpendicular spins,
    according to Bloch representation of spin-1/2 states.
    This case targets the overall magnetization strength.
    '''    
    # GS definition
    GS_vec = (1/np.sqrt(2))*np.array([[1.0,1.0j,0.0,0.0],
                                      [1.0,-1.0j,0.0,0.0],
                                      [0.0,0.0,1.0,1.0j],
                                      [0.0,0.0,1.0,-1.0j]]).T

    # M_full value vs Expectation
    M_full = two_sites_1D_chain.compute_magnetization(GS_vec,4,'M_full')
    M_full_exp = np.sqrt(1.5)/2
    assert np.abs(M_full-M_full_exp)<1e-10

def test_find_shift_indices_0(two_sites_1D_chain) -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 1D spin system with equally-spaced sites.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = two_sites_1D_chain.find_shift_indices(eff_latt_vecs, 0)
    assert shift_indices==[(0,)]

def test_find_shift_indices_1(two_sites_1D_chain) -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 1D spin system with equally-spaced sites.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = two_sites_1D_chain.find_shift_indices(eff_latt_vecs, 1)
    assert shift_indices==[(0,),(2,)]

def test_find_shift_indices_2(two_sites_1D_chain) -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 1D spin system with equally-spaced sites.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = two_sites_1D_chain.find_shift_indices(eff_latt_vecs, 2)
    assert shift_indices==[(0,),(4,)]

def test_find_shift_indices_3() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 1D spin system with longitudinal distortions.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.4,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 0)
    assert shift_indices==[(0,)]

def test_find_shift_indices_4() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 1D spin system with longitudinal distortions.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.4,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 1)
    assert shift_indices==[(0,),(2,)]

def test_find_shift_indices_5() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 1D spin system with longitudinal distortions.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.4,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 2)
    assert shift_indices==[(0,),(4,)]

def test_find_shift_indices_6() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D square system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 0)
    assert shift_indices==[(0,0)]

def test_find_shift_indices_7() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D square system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 1)
    assert shift_indices==[(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]

def test_find_shift_indices_8() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D square system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 2)
    assert shift_indices==[(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,4),(2,0),(2,4),(3,0),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]

def test_find_shift_indices_9() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D rectangular system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 0)
    assert shift_indices==[(0,0)]

def test_find_shift_indices_10() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D rectangular system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 1)
    assert shift_indices==[(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]

def test_find_shift_indices_11() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D rectangular system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 2)
    assert shift_indices==[(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,4),(2,0),(2,4),(3,0),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]

def test_find_shift_indices_12() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D triangular system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[np.cos(np.pi/3),np.sin(np.pi/3),0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[np.cos(np.pi/3),np.sin(np.pi/3),0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 0)
    assert shift_indices==[(0,0)]

def test_find_shift_indices_13() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D triangular system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[np.cos(np.pi/3),np.sin(np.pi/3),0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[np.cos(np.pi/3),np.sin(np.pi/3),0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 1)
    assert shift_indices==[(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]

def test_find_shift_indices_14() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D triangular system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[np.cos(np.pi/3),np.sin(np.pi/3),0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[np.cos(np.pi/3),np.sin(np.pi/3),0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 2)
    assert shift_indices==[(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,4),(2,0),(2,4),(3,0),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]

def test_find_shift_indices_15() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D hexagonal system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[np.cos(np.pi/6),np.sin(np.pi/6),0.0],[-np.cos(np.pi/6),np.sin(np.pi/6),0.0],[0.0,0.0,10.0]])
    sites = np.array([[-0.2,0.5,0.0],[0.2,0.5,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[np.cos(np.pi/6),np.sin(np.pi/6),0.0],[-np.cos(np.pi/6),np.sin(np.pi/6),0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 0)
    assert shift_indices==[(0,0)]

def test_find_shift_indices_16() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D hexagonal system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[np.cos(np.pi/6),np.sin(np.pi/6),0.0],[-np.cos(np.pi/6),np.sin(np.pi/6),0.0],[0.0,0.0,10.0]])
    sites = np.array([[-0.2,0.5,0.0],[0.2,0.5,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[np.cos(np.pi/6),np.sin(np.pi/6),0.0],[-np.cos(np.pi/6),np.sin(np.pi/6),0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 1)
    assert shift_indices==[(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]

def test_find_shift_indices_17() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 2D hexagonal system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[np.cos(np.pi/6),np.sin(np.pi/6),0.0],[-np.cos(np.pi/6),np.sin(np.pi/6),0.0],[0.0,0.0,10.0]])
    sites = np.array([[-0.2,0.5,0.0],[0.2,0.5,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[np.cos(np.pi/6),np.sin(np.pi/6),0.0],[-np.cos(np.pi/6),np.sin(np.pi/6),0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 2)
    assert shift_indices==[(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,4),(2,0),(2,4),(3,0),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]

def test_find_shift_indices_18() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D triclinic system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[-0.2,0.8,0.0],[0.2,0.3,0.7]])
    sites = np.array([[0.0,0.0,0.0],[0.1,0.1,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 0)
    assert shift_indices==[(0,0,0)]

def test_find_shift_indices_19() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D triclinic system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[-0.2,0.8,0.0],[0.2,0.3,0.7]])
    sites = np.array([[0.0,0.0,0.0],[0.1,0.1,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 1)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
                           (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,2),(1,2,0),(1,2,1),(1,2,2),(2,0,0),
                           (2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2)]

def test_find_shift_indices_20() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D triclinic system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[-0.2,0.8,0.0],[0.2,0.3,0.7]])
    sites = np.array([[0.0,0.0,0.0],[0.1,0.1,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 2)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,1,0),(0,1,1),(0,1,2),(0,1,3),
                           (0,1,4),(0,2,0),(0,2,1),(0,2,2),(0,2,3),(0,2,4),(0,3,0),(0,3,1),(0,3,2),
                           (0,3,3),(0,3,4),(0,4,0),(0,4,1),(0,4,2),(0,4,3),(0,4,4),(1,0,0),(1,0,1),
                           (1,0,2),(1,0,3),(1,0,4),(1,1,0),(1,1,4),(1,2,0),(1,2,4),(1,3,0),(1,3,4),
                           (1,4,0),(1,4,1),(1,4,2),(1,4,3),(1,4,4),(2,0,0),(2,0,1),(2,0,2),(2,0,3),
                           (2,0,4),(2,1,0),(2,1,4),(2,2,0),(2,2,4),(2,3,0),(2,3,4),(2,4,0),(2,4,1),
                           (2,4,2),(2,4,3),(2,4,4),(3,0,0),(3,0,1),(3,0,2),(3,0,3),(3,0,4),(3,1,0),
                           (3,1,4),(3,2,0),(3,2,4),(3,3,0),(3,3,4),(3,4,0),(3,4,1),(3,4,2),(3,4,3),
                           (3,4,4),(4,0,0),(4,0,1),(4,0,2),(4,0,3),(4,0,4),(4,1,0),(4,1,1),(4,1,2),
                           (4,1,3),(4,1,4),(4,2,0),(4,2,1),(4,2,2),(4,2,3),(4,2,4),(4,3,0),(4,3,1),
                           (4,3,2),(4,3,3),(4,3,4),(4,4,0),(4,4,1),(4,4,2),(4,4,3),(4,4,4)]

def test_find_shift_indices_21() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D monoclinic system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.3,0.7]])
    sites = np.array([[0.0,0.0,0.0],[0.3,0.6,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 0)
    assert shift_indices==[(0,0,0)]

def test_find_shift_indices_22() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D monoclinic system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.3,0.7]])
    sites = np.array([[0.0,0.0,0.0],[0.3,0.6,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 1)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
                           (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,2),(1,2,0),(1,2,1),(1,2,2),(2,0,0),
                           (2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2)]

def test_find_shift_indices_23() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D monoclinic system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.3,0.7]])
    sites = np.array([[0.0,0.0,0.0],[0.3,0.6,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 2)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,1,0),(0,1,1),(0,1,2),(0,1,3),
                           (0,1,4),(0,2,0),(0,2,1),(0,2,2),(0,2,3),(0,2,4),(0,3,0),(0,3,1),(0,3,2),
                           (0,3,3),(0,3,4),(0,4,0),(0,4,1),(0,4,2),(0,4,3),(0,4,4),(1,0,0),(1,0,1),
                           (1,0,2),(1,0,3),(1,0,4),(1,1,0),(1,1,4),(1,2,0),(1,2,4),(1,3,0),(1,3,4),
                           (1,4,0),(1,4,1),(1,4,2),(1,4,3),(1,4,4),(2,0,0),(2,0,1),(2,0,2),(2,0,3),
                           (2,0,4),(2,1,0),(2,1,4),(2,2,0),(2,2,4),(2,3,0),(2,3,4),(2,4,0),(2,4,1),
                           (2,4,2),(2,4,3),(2,4,4),(3,0,0),(3,0,1),(3,0,2),(3,0,3),(3,0,4),(3,1,0),
                           (3,1,4),(3,2,0),(3,2,4),(3,3,0),(3,3,4),(3,4,0),(3,4,1),(3,4,2),(3,4,3),
                           (3,4,4),(4,0,0),(4,0,1),(4,0,2),(4,0,3),(4,0,4),(4,1,0),(4,1,1),(4,1,2),
                           (4,1,3),(4,1,4),(4,2,0),(4,2,1),(4,2,2),(4,2,3),(4,2,4),(4,3,0),(4,3,1),
                           (4,3,2),(4,3,3),(4,3,4),(4,4,0),(4,4,1),(4,4,2),(4,4,3),(4,4,4)]

def test_find_shift_indices_24() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D orthorhombic system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,3.0]])
    sites = np.array([[0.0,0.0,0.0],[0.3,0.6,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 0)
    assert shift_indices==[(0,0,0)]

def test_find_shift_indices_25() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D orthorhombic system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,3.0]])
    sites = np.array([[0.0,0.0,0.0],[0.3,0.6,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 1)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
                           (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,2),(1,2,0),(1,2,1),(1,2,2),(2,0,0),
                           (2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2)]

def test_find_shift_indices_26() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D orthorhombic system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,3.0]])
    sites = np.array([[0.0,0.0,0.0],[0.3,0.6,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 2)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,1,0),(0,1,1),(0,1,2),(0,1,3),
                           (0,1,4),(0,2,0),(0,2,1),(0,2,2),(0,2,3),(0,2,4),(0,3,0),(0,3,1),(0,3,2),
                           (0,3,3),(0,3,4),(0,4,0),(0,4,1),(0,4,2),(0,4,3),(0,4,4),(1,0,0),(1,0,1),
                           (1,0,2),(1,0,3),(1,0,4),(1,1,0),(1,1,4),(1,2,0),(1,2,4),(1,3,0),(1,3,4),
                           (1,4,0),(1,4,1),(1,4,2),(1,4,3),(1,4,4),(2,0,0),(2,0,1),(2,0,2),(2,0,3),
                           (2,0,4),(2,1,0),(2,1,4),(2,2,0),(2,2,4),(2,3,0),(2,3,4),(2,4,0),(2,4,1),
                           (2,4,2),(2,4,3),(2,4,4),(3,0,0),(3,0,1),(3,0,2),(3,0,3),(3,0,4),(3,1,0),
                           (3,1,4),(3,2,0),(3,2,4),(3,3,0),(3,3,4),(3,4,0),(3,4,1),(3,4,2),(3,4,3),
                           (3,4,4),(4,0,0),(4,0,1),(4,0,2),(4,0,3),(4,0,4),(4,1,0),(4,1,1),(4,1,2),
                           (4,1,3),(4,1,4),(4,2,0),(4,2,1),(4,2,2),(4,2,3),(4,2,4),(4,3,0),(4,3,1),
                           (4,3,2),(4,3,3),(4,3,4),(4,4,0),(4,4,1),(4,4,2),(4,4,3),(4,4,4)]

def test_find_shift_indices_27() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D tetragonal system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,3.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 0)
    assert shift_indices==[(0,0,0)]

def test_find_shift_indices_28() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D tetragonal system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,3.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 1)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
                           (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,2),(1,2,0),(1,2,1),(1,2,2),(2,0,0),
                           (2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2)]

def test_find_shift_indices_29() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D tetragonal system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,3.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.5,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 2)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,1,0),(0,1,1),(0,1,2),(0,1,3),
                           (0,1,4),(0,2,0),(0,2,1),(0,2,2),(0,2,3),(0,2,4),(0,3,0),(0,3,1),(0,3,2),
                           (0,3,3),(0,3,4),(0,4,0),(0,4,1),(0,4,2),(0,4,3),(0,4,4),(1,0,0),(1,0,1),
                           (1,0,2),(1,0,3),(1,0,4),(1,1,0),(1,1,4),(1,2,0),(1,2,4),(1,3,0),(1,3,4),
                           (1,4,0),(1,4,1),(1,4,2),(1,4,3),(1,4,4),(2,0,0),(2,0,1),(2,0,2),(2,0,3),
                           (2,0,4),(2,1,0),(2,1,4),(2,2,0),(2,2,4),(2,3,0),(2,3,4),(2,4,0),(2,4,1),
                           (2,4,2),(2,4,3),(2,4,4),(3,0,0),(3,0,1),(3,0,2),(3,0,3),(3,0,4),(3,1,0),
                           (3,1,4),(3,2,0),(3,2,4),(3,3,0),(3,3,4),(3,4,0),(3,4,1),(3,4,2),(3,4,3),
                           (3,4,4),(4,0,0),(4,0,1),(4,0,2),(4,0,3),(4,0,4),(4,1,0),(4,1,1),(4,1,2),
                           (4,1,3),(4,1,4),(4,2,0),(4,2,1),(4,2,2),(4,2,3),(4,2,4),(4,3,0),(4,3,1),
                           (4,3,2),(4,3,3),(4,3,4),(4,4,0),(4,4,1),(4,4,2),(4,4,3),(4,4,4)]

def test_find_shift_indices_30() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D fcc cubic system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.25,0.25]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 0)
    assert shift_indices==[(0,0,0)]

def test_find_shift_indices_31() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D fcc cubic system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.25,0.25]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 1)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
                           (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,2),(1,2,0),(1,2,1),(1,2,2),(2,0,0),
                           (2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2)]

def test_find_shift_indices_32() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D fcc cubic system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.25,0.25]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 2)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,1,0),(0,1,1),(0,1,2),
                           (0,1,3),(0,1,4),(0,2,0),(0,2,1),(0,2,2),(0,2,3),(0,2,4),(0,3,0),
                           (0,3,1),(0,3,2),(0,3,3),(0,3,4),(0,4,0),(0,4,1),(0,4,2),(0,4,3),
                           (0,4,4),(1,0,0),(1,0,1),(1,0,2),(1,0,3),(1,0,4),(1,1,0),(1,1,4),
                           (1,2,0),(1,2,4),(1,3,0),(1,3,4),(1,4,0),(1,4,1),(1,4,2),(1,4,3),
                           (1,4,4),(2,0,0),(2,0,1),(2,0,2),(2,0,3),(2,0,4),(2,1,0),(2,1,4),
                           (2,2,0),(2,2,4),(2,3,0),(2,3,4),(2,4,0),(2,4,1),(2,4,2),(2,4,3),
                           (2,4,4),(3,0,0),(3,0,1),(3,0,2),(3,0,3),(3,0,4),(3,1,0),(3,1,4),
                           (3,2,0),(3,2,4),(3,3,0),(3,3,4),(3,4,0),(3,4,1),(3,4,2),(3,4,3),
                           (3,4,4),(4,0,0),(4,0,1),(4,0,2),(4,0,3),(4,0,4),(4,1,0),(4,1,1),
                           (4,1,2),(4,1,3),(4,1,4),(4,2,0),(4,2,1),(4,2,2),(4,2,3),(4,2,4),
                           (4,3,0),(4,3,1),(4,3,2),(4,3,3),(4,3,4),(4,4,0),(4,4,1),(4,4,2),
                           (4,4,3),(4,4,4)]

def test_find_shift_indices_33() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D bcc cubic system.
    This case focuses on those shift indices that lead to the unit cell itself.
    '''
    # Definition of the system
    latt_vecs = np.array([[0.5,0.5,-0.5],[-0.5,0.5,0.5],[0.5,-0.5,0.5]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.25,0.25]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 0)
    assert shift_indices==[(0,0,0)]

def test_find_shift_indices_34() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D bcc cubic system.
    This case focuses on those shift indices that lead to the 1NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[0.5,0.5,-0.5],[-0.5,0.5,0.5],[0.5,-0.5,0.5]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.25,0.25]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 1)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),
                           (0,2,1),(0,2,2),(1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,2),
                           (1,2,0),(1,2,1),(1,2,2),(2,0,0),(2,0,1),(2,0,2),(2,1,0),
                           (2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2)]

def test_find_shift_indices_35() -> None :
    '''
    Tests that the method in question correctly provides the shift indices for a 3D bcc cubic system.
    This case focuses on those shift indices that lead to the 2NN unit cells.
    '''
    # Definition of the system
    latt_vecs = np.array([[0.5,0.5,-0.5],[-0.5,0.5,0.5],[0.5,-0.5,0.5]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.25,0.25]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 2)
    assert shift_indices==[(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,1,0),(0,1,1),(0,1,2),
                           (0,1,3),(0,1,4),(0,2,0),(0,2,1),(0,2,2),(0,2,3),(0,2,4),(0,3,0),
                           (0,3,1),(0,3,2),(0,3,3),(0,3,4),(0,4,0),(0,4,1),(0,4,2),(0,4,3),
                           (0,4,4),(1,0,0),(1,0,1),(1,0,2),(1,0,3),(1,0,4),(1,1,0),(1,1,4),
                           (1,2,0),(1,2,4),(1,3,0),(1,3,4),(1,4,0),(1,4,1),(1,4,2),(1,4,3),
                           (1,4,4),(2,0,0),(2,0,1),(2,0,2),(2,0,3),(2,0,4),(2,1,0),(2,1,4),
                           (2,2,0),(2,2,4),(2,3,0),(2,3,4),(2,4,0),(2,4,1),(2,4,2),(2,4,3),
                           (2,4,4),(3,0,0),(3,0,1),(3,0,2),(3,0,3),(3,0,4),(3,1,0),(3,1,4),
                           (3,2,0),(3,2,4),(3,3,0),(3,3,4),(3,4,0),(3,4,1),(3,4,2),(3,4,3),
                           (3,4,4),(4,0,0),(4,0,1),(4,0,2),(4,0,3),(4,0,4),(4,1,0),(4,1,1),
                           (4,1,2),(4,1,3),(4,1,4),(4,2,0),(4,2,1),(4,2,2),(4,2,3),(4,2,4),
                           (4,3,0),(4,3,1),(4,3,2),(4,3,3),(4,3,4),(4,4,0),(4,4,1),(4,4,2),
                           (4,4,3),(4,4,4)]

def test_update_shell_arrays_0(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the indices of the spins that are included to the same unit cell
    as the reference spin. This case involves a 1D system with two sites.
    '''    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = two_sites_1D_chain.find_shift_indices(eff_latt_vecs, 0)
    shell_indices, shell_vectors, shell_distances = two_sites_1D_chain.update_shell_arrays(0, eff_latt_vecs, 0, shift_indices, 5)
    assert shell_indices==[1]

def test_update_shell_arrays_1(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors to the spins that are included to the same unit cell
    as the reference spin. This case involves a 1D system with two sites.
    '''    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = two_sites_1D_chain.find_shift_indices(eff_latt_vecs, 0)
    shell_indices, shell_vectors, shell_distances = two_sites_1D_chain.update_shell_arrays(0, eff_latt_vecs, 0, shift_indices, 5)
    assert np.allclose(shell_vectors, np.array([[-0.5,0.0,0.0]]), atol=1e-10, rtol=1e-10)
    
def test_update_shell_arrays_2(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the relative distances to the spins that are included to the same unit cell
    as the reference spin. This case involves a 1D system with two sites.
    '''    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = two_sites_1D_chain.find_shift_indices(eff_latt_vecs, 0)
    shell_indices, shell_vectors, shell_distances = two_sites_1D_chain.update_shell_arrays(0, eff_latt_vecs, 0, shift_indices, 5)
    assert np.allclose(np.array(shell_distances), np.array([0.5]), atol=1e-10, rtol=1e-10)

def test_update_shell_arrays_3(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the indices of the spins that are included to the 1NN unit cells
    as the reference spin. This case involves a 1D system with two sites.
    '''    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = two_sites_1D_chain.find_shift_indices(eff_latt_vecs, 1)
    shell_indices, shell_vectors, shell_distances = two_sites_1D_chain.update_shell_arrays(0, eff_latt_vecs, 1, shift_indices, 5)
    assert shell_indices==[0,0,1,1]

def test_update_shell_arrays_4(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the connecting vectors to the spins that are included to the 1NN unit cells
    as the reference spin. This case involves a 1D system with two sites.
    '''    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = two_sites_1D_chain.find_shift_indices(eff_latt_vecs, 1)
    shell_indices, shell_vectors, shell_distances = two_sites_1D_chain.update_shell_arrays(0, eff_latt_vecs, 1, shift_indices, 5)
    assert np.allclose(np.array(shell_vectors), 
                       np.array([[1.0,0.0,0.0],[-1.0,0.0,0.0],
                                 [0.5,0.0,0.0],[-1.5,0.0,0.0]]), 
                       atol=1e-10, rtol=1e-10)
    
def test_update_shell_arrays_5(two_sites_1D_chain) -> None :
    '''
    Tests that the method correctly identifies the relative distances to the spins that are included to the 1NN unit cells
    as the reference spin. This case involves a 1D system with two sites.
    '''    
    eff_latt_vecs = np.array([[1.0,0.0,0.0]])
    shift_indices = two_sites_1D_chain.find_shift_indices(eff_latt_vecs, 1)
    shell_indices, shell_vectors, shell_distances = two_sites_1D_chain.update_shell_arrays(0, eff_latt_vecs, 1, shift_indices, 5)
    assert np.allclose(np.array(shell_distances), np.array([1.0,1.0,0.5,1.5]), atol=1e-10, rtol=1e-10)

def test_update_shell_arrays_6() -> None :
    '''
    Tests that the method correctly identifies the indices of the spins that are included to the same unit cell
    as the reference spin. This case involves a 2D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 0)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, eff_latt_vecs, 0, shift_indices, 5)
    assert shell_indices==[1]

def test_update_shell_arrays_7() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors to the spins that are included to the same unit cell
    as the reference spin. This case involves a 2D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 0)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, eff_latt_vecs, 0, shift_indices, 5)
    assert np.allclose(shell_vectors, np.array([[-0.5,0.0,0.0]]), atol=1e-10, rtol=1e-10)
    
def test_update_shell_arrays_8() -> None :
    '''
    Tests that the method correctly identifies the relative distances to the spins that are included to the same unit cell
    as the reference spin. This case involves a 2D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 0)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, eff_latt_vecs, 0, shift_indices, 5)
    assert np.allclose(np.array(shell_distances), np.array([0.5]), atol=1e-10, rtol=1e-10)

def test_update_shell_arrays_9() -> None :
    '''
    Tests that the method correctly identifies the indices of the spins that are included to the 1NN unit cells
    as the reference spin. This case involves a 2D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 1)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, eff_latt_vecs, 1, shift_indices, 5)
    assert shell_indices==[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

def test_update_shell_arrays_10() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors to the spins that are included to the 1NN unit cells
    as the reference spin. This case involves a 2D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 1)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, eff_latt_vecs, 1, shift_indices, 5)
    assert np.allclose(np.array(shell_vectors), 
                       np.array([[1.0,1.0,0.0],[1.0,0.0,0.0],[1.0,-1.0,0.0],[0.0,1.0,0.0],
                                 [0.0,-1.0,0.0],[-1.0,1.0,0.0],[-1.0,0.0,0.0],[-1.0,-1.0,0.0],
                                 [0.5,1.0,0.0],[0.5,0.0,0.0],[0.5,-1.0,0.0],[-0.5,1.0,0.0],
                                 [-0.5,-1.0,0.0],[-1.5,1.0,0.0],[-1.5,0.0,0.0],[-1.5,-1.0,0.0]]),
                       atol=1e-10, rtol=1e-10)
    
def test_update_shell_arrays_11() -> None :
    '''
    Tests that the method correctly identifies the relative distances to the spins that are included to the 1NN unit cells
    as the reference spin. This case involves a 2D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    eff_latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    shift_indices = system.find_shift_indices(eff_latt_vecs, 1)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, eff_latt_vecs, 1, shift_indices, 5)
    assert np.allclose(np.array(shell_distances),
                       np.array([np.sqrt(2),1.0,np.sqrt(2),1.0,1.0,np.sqrt(2),1.0,np.sqrt(2),
                                 np.sqrt(1.25),0.5,np.sqrt(1.25),np.sqrt(1.25),np.sqrt(1.25),
                                 np.sqrt(3.25),1.5,np.sqrt(3.25)]),
                       atol=1e-10, rtol=1e-10)

def test_update_shell_arrays_12() -> None :
    '''
    Tests that the method correctly identifies the indices of the spins that are included to the same unit cell
    as the reference spin. This case involves a 3D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 0)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, latt_vecs, 0, shift_indices, 5)
    assert shell_indices==[1]

def test_update_shell_arrays_13() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors to the spins that are included to the same unit cell
    as the reference spin. This case involves a 3D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 0)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, latt_vecs, 0, shift_indices, 5)
    assert np.allclose(shell_vectors, np.array([[-0.5,0.0,0.0]]), atol=1e-10, rtol=1e-10)
    
def test_update_shell_arrays_14() -> None :
    '''
    Tests that the method correctly identifies the relative distances to the spins that are included to the same unit cell
    as the reference spin. This case involves a 3D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 0)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, latt_vecs, 0, shift_indices, 5)
    assert np.allclose(np.array(shell_distances), np.array([0.5]), atol=1e-10, rtol=1e-10)

def test_update_shell_arrays_15() -> None :
    '''
    Tests that the method correctly identifies the indices of the spins that are included to the 1NN unit cells
    as the reference spin. This case involves a 3D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 1)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, latt_vecs, 1, shift_indices, 5)
    assert shell_indices==[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

def test_update_shell_arrays_16() -> None :
    '''
    Tests that the method correctly identifies the connecting vectors to the spins that are included to the 1NN unit cells
    as the reference spin. This case involves a 3D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 1)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, latt_vecs, 1, shift_indices, 5)
    assert np.allclose(shell_vectors, 
                       np.array([[1.0,1.0,1.0],[1.0,1.0,0.0],[1.0,1.0,-1.0],[1.0,0.0,1.0],
                                 [1.0,0.0,0.0],[1.0,0.0,-1.0],[1.0,-1.0,1.0],[1.0,-1.0,0.0],
                                 [1.0,-1.0,-1.0],[0.0,1.0,1.0],[0.0,1.0,0.0],[0.0,1.0,-1.0],
                                 [0.0,0.0,1.0],[0.0,0.0,-1.0],[0.0,-1.0,1.0],[0.0,-1.0,0.0],
                                 [0.0,-1.0,-1.0],[-1.0,1.0,1.0],[-1.0,1.0,0.0],[-1.0,1.0,-1.0],
                                 [-1.0,0.0,1.0],[-1.0,0.0,0.0],[-1.0,0.0,-1.0],[-1.0,-1.0,1.0],
                                 [-1.0,-1.0,0.0],[-1.0,-1.0,-1.0],[0.5,1.0,1.0],[0.5,1.0,0.0],
                                 [0.5,1.0,-1.0],[0.5,0.0,1.0],[0.5,0.0,0.0],[0.5,0.0,-1.0],
                                 [0.5,-1.0,1.0],[0.5,-1.0,0.0],[0.5,-1.0,-1.0],[-0.5,1.0,1.0],
                                 [-0.5,1.0,0.0],[-0.5,1.0,-1.0],[-0.5,0.0,1.0],[-0.5,0.0,-1.0],
                                 [-0.5,-1.0,1.0],[-0.5,-1.0,0.0],[-0.5,-1.0,-1.0],[-1.5,1.0,1.0],
                                 [-1.5,1.0,0.0],[-1.5,1.0,-1.0],[-1.5,0.0,1.0],[-1.5,0.0,0.0],
                                 [-1.5,0.0,-1.0],[-1.5,-1.0,1.0],[-1.5,-1.0,0.0],[-1.5,-1.0,-1.0]]),
                       atol=1e-10, rtol=1e-10)
    
def test_update_shell_arrays_17() -> None :
    '''
    Tests that the method correctly identifies the relative distances to the spins that are included to the 1NN unit cells
    as the reference spin. This case involves a 3D system with two sites.
    '''
    # Definition of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs, sites, 0.5)
    
    shift_indices = system.find_shift_indices(latt_vecs, 1)
    shell_indices, shell_vectors, shell_distances = system.update_shell_arrays(0, latt_vecs, 1, shift_indices, 5)
    assert np.allclose(np.array(shell_distances), 
                       np.array([np.sqrt(3),np.sqrt(2),np.sqrt(3),np.sqrt(2),
                                 1.0,np.sqrt(2),np.sqrt(3),np.sqrt(2),
                                 np.sqrt(3),np.sqrt(2),1.0,np.sqrt(2),
                                 1.0,1.0,np.sqrt(2),1.0,
                                 np.sqrt(2),np.sqrt(3),np.sqrt(2),np.sqrt(3),
                                 np.sqrt(2),1.0,np.sqrt(2),np.sqrt(3),
                                 np.sqrt(2),np.sqrt(3),np.sqrt(2.25),np.sqrt(1.25),
                                 np.sqrt(2.25),np.sqrt(1.25),np.sqrt(0.25),np.sqrt(1.25),
                                 np.sqrt(2.25),np.sqrt(1.25),np.sqrt(2.25),np.sqrt(2.25),
                                 np.sqrt(1.25),np.sqrt(2.25),np.sqrt(1.25),np.sqrt(1.25),
                                 np.sqrt(2.25),np.sqrt(1.25),np.sqrt(2.25),np.sqrt(4.25),
                                 np.sqrt(3.25),np.sqrt(4.25),np.sqrt(3.25),1.5,
                                 np.sqrt(3.25),np.sqrt(4.25),np.sqrt(3.25),np.sqrt(4.25)]), 
                       atol=1e-10, rtol=1e-10)