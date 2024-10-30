from global_functions import *
from spin_system import *
import pytest

def test_clean_line_0() -> None :
    '''
    Tests that the clean_line() function returns an empty list if provided with an empty string.
    '''
    vector = clean_line('')
    assert len(vector)==0

def test_clean_line_1() -> None :
    '''
    Tests that the clean_line() function returns an empty list if provided with a string consisting 
    of empty spaces and a new line character.
    '''
    vector = clean_line('   \n')
    assert len(vector)==0

def test_clean_line_2() -> None :
    '''
    Tests that the clean_line() function returns the proper list of strings if provided with a string
    whose last word finishes with the new line character (common case).
    '''
    vector = clean_line('2.3 is a real number\n')
    exp_vector = ['2.3','is','a','real','number']
    assert len(vector)==len(exp_vector)
    for i in range(len(vector)) :
        assert vector[i]==exp_vector[i]

def test_is_spin_acceptable_0() -> None :
    '''
    Tests that the is_spin_acceptable() function returns False if provided with a negative half-integer
    floating number.
    '''
    assert is_spin_acceptable(-1.5)==False

def test_is_spin_acceptable_1() -> None :
    '''
    Tests that the is_spin_acceptable() function returns False if provided with a negative integer number.
    '''
    assert is_spin_acceptable(-2)==False

def test_is_spin_acceptable_2() -> None :
    '''
    Tests that the is_spin_acceptable() function returns False if provided with zero.
    '''
    assert is_spin_acceptable(0)==False

def test_is_spin_acceptable_3() -> None :
    '''
    Tests that the is_spin_acceptable() function returns False if provided with a positive floating number
    which is neither integer nor half-integer.
    '''
    assert is_spin_acceptable(0.7)==False

def test_adapt_magintmatrix_0() -> None :
    '''
    Tests that the proper Exception is raised when the adapt_magintmatrix() function is provided with 
    a 2D array whose shape is not (3,3).
    '''
    with pytest.raises(ValueError, match='The adapt_magintmatrix function only accepts 3x3 square matrices as argument.') :
        new_matrix = adapt_magintmatrix(np.array([[1.0]]))

def test_adapt_magintmatrix_1() -> None :
    '''
    Tests that the proper Exception is raised when the adapt_magintmatrix() function is provided with 
    a 1D array whose shape is not (3,3).
    '''
    with pytest.raises(ValueError, match='The adapt_magintmatrix function only accepts 3x3 square matrices as argument.') :
        new_matrix = adapt_magintmatrix(np.array([1.0]))

def test_adapt_magintmatrix_2() -> None :
    '''
    Tests that the adapt_magintmatrix() function returns the correct transformed matrix when provided with 
    a 3x3 matrix in the most general form.
    '''
    # Matrix vs its expected transformation
    matrix = np.array([[1.0,2.0,3.0],
                       [4.0,5.0,6.0],
                       [7.0,8.0,9.0]])
    exp_matrix = np.array([[9.0,7.0,8.0],
                           [3.0,1.0,2.0],
                           [6.0,4.0,5.0]])
    new_matrix = adapt_magintmatrix(matrix)

    assert np.allclose(new_matrix, exp_matrix, atol=1e-10, rtol=1e-10)

def test_adapt_magintmatrix_3() -> None :
    '''
    Tests that the adapt_magintmatrix() function returns the correct transformed matrix when provided with 
    a 3x3 diagonal matrix.
    '''
    # Matrix vs its expected transformation
    matrix = np.array([[1.0,0.0,0.0],
                       [0.0,5.0,0.0],
                       [0.0,0.0,9.0]])
    exp_matrix = np.array([[9.0,0.0,0.0],
                           [0.0,1.0,0.0],
                           [0.0,0.0,5.0]])
    new_matrix = adapt_magintmatrix(matrix)

    assert np.allclose(new_matrix, exp_matrix, atol=1e-10, rtol=1e-10)

def test_map_spin_correlations_0() -> None :
    '''
    Tests that the function in question correctly provides the length scale for all the NN shells
    of the system, starting from the "on-site bond" (e.g. links spin 0 to itself = null distance)
    to the closest reproducible "replica bond" (e.g. links spin 0 to its closest replica).
    This case involves a 2-sites system and null NN spin-spin correlation values.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Spin-spin correlation values in the expected format
    SSC_xs = np.eye(2)
    SSC_ys = np.eye(2)
    SSC_zs = np.eye(2)
    SSC_tots = np.eye(2)
    spin_correlations = (SSC_xs, SSC_ys, SSC_zs, SSC_tots)
    
    # Final arrangement of the spin-spin correlation values vs expectation
    final_SSCs, NN_distances = map_spin_correlations(system,spin_correlations,4,1)
    NN_distances = np.array(NN_distances)
    exp_NN_distances = np.array([0.0,0.5,1.0])
    
    assert np.allclose(NN_distances, exp_NN_distances, atol=1e-10, rtol=1e-10)

def test_map_spin_correlations_1() -> None :
    '''
    Tests that the function in question correctly arranges the spin-spin correlation values according to 
    the associated NN shell of the system, starting from the "on-site correlation" (e.g. spin0-spin0
    correlation) to the closest reproducible "replica correlation" (e.g. spin0-replica correlation).
    This case involves a 2-sites system and null NN spin-spin correlation values.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Spin-spin correlation values in the expected format
    SSC_xs = np.eye(2)
    SSC_ys = np.eye(2)
    SSC_zs = np.eye(2)
    SSC_tots = np.eye(2)
    spin_correlations = (SSC_xs, SSC_ys, SSC_zs, SSC_tots)
    
    # Final arrangement of the spin-spin correlation values vs expectation
    final_SSCs, NN_distances = map_spin_correlations(system,spin_correlations,4,1)
    exp_SSCs = np.array([[1.0,0.0,1.0],
                         [1.0,0.0,1.0],
                         [1.0,0.0,1.0],
                         [1.0,0.0,1.0]])
    
    assert np.allclose(final_SSCs, exp_SSCs, atol=1e-10, rtol=1e-10)

def test_map_spin_correlations_2() -> None :
    '''
    Tests that the function in question correctly provides the length scale for all the NN shells
    of the system, starting from the "on-site bond" (e.g. links spin 0 to itself = null distance)
    to the closest reproducible "replica bond" (e.g. links spin 0 to its closest replica).
    This case involves a 4-sites system and the only non vanishing spin-spin correlation values
    are those of the 1°NN shell.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.0,0.0],[0.5,0.0,0.0],[0.75,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Spin-spin correlation values in the expected format
    SSC_xs = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))
    SSC_ys = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))
    SSC_zs = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))
    SSC_tots = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))
    spin_correlations = (SSC_xs, SSC_ys, SSC_zs, SSC_tots)
    
    # Final arrangement of the spin-spin correlation values vs expectation
    final_SSCs, NN_distances = map_spin_correlations(system,spin_correlations,4,1)
    NN_distances = np.array(NN_distances)
    exp_NN_distances = np.array([0.0,0.25,0.5,0.75,1.0])
    
    assert np.allclose(NN_distances, exp_NN_distances, atol=1e-10, rtol=1e-10)

def test_map_spin_correlations_3() -> None :
    '''
    Tests that the function in question correctly arranges the spin-spin correlation values according to 
    the associated NN shell of the system, starting from the "on-site correlation" (e.g. spin0-spin0
    correlation) to the closest reproducible "replica correlation" (e.g. spin0-replica correlation).
    This case involves a 4-sites system and the only non vanishing spin-spin correlation values
    are those of the 1°NN shell.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.0,0.0],[0.5,0.0,0.0],[0.75,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Spin-spin correlation values in the expected format
    SSC_xs = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))
    SSC_ys = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))
    SSC_zs = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))
    SSC_tots = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))
    spin_correlations = (SSC_xs, SSC_ys, SSC_zs, SSC_tots)
    
    # Final arrangement of the spin-spin correlation values vs expectation
    final_SSCs, NN_distances = map_spin_correlations(system,spin_correlations,4,1)
    exp_SSCs = np.array([[1.0,0.5,0.0,0.5,1.0],
                         [1.0,0.5,0.0,0.5,1.0],
                         [1.0,0.5,0.0,0.5,1.0],
                         [1.0,0.5,0.0,0.5,1.0]])
    
    assert np.allclose(final_SSCs, exp_SSCs, atol=1e-10, rtol=1e-10)

def test_map_spin_correlations_4() -> None :
    '''
    Tests that the function in question correctly provides the length scale for all the NN shells
    of the system, starting from the "on-site bond" (e.g. links spin 0 to itself = null distance)
    to the closest reproducible "replica bond" (e.g. links spin 0 to its closest replica).
    This case involves a 4-sites system and fully non vanishing spin-spin correlation values.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.0,0.0],[0.5,0.0,0.0],[0.75,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Spin-spin correlation values in the expected format
    SSC_xs = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))-0.25*(np.eye(4, k=2)+np.eye(4, k=-2))
    SSC_ys = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))-0.25*(np.eye(4, k=2)+np.eye(4, k=-2))
    SSC_zs = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))-0.25*(np.eye(4, k=2)+np.eye(4, k=-2))
    SSC_tots = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))-0.25*(np.eye(4, k=2)+np.eye(4, k=-2))
    spin_correlations = (SSC_xs, SSC_ys, SSC_zs, SSC_tots)
    
    # Final arrangement of the spin-spin correlation values vs expectation
    final_SSCs, NN_distances = map_spin_correlations(system,spin_correlations,4,1)
    NN_distances = np.array(NN_distances)
    exp_NN_distances = np.array([0.0,0.25,0.5,0.75,1.0])
    
    assert np.allclose(NN_distances, exp_NN_distances, atol=1e-10, rtol=1e-10)

def test_map_spin_correlations_5() -> None :
    '''
    Tests that the function in question correctly arranges the spin-spin correlation values according to 
    the associated NN shell of the system, starting from the "on-site correlation" (e.g. spin0-spin0
    correlation) to the closest reproducible "replica correlation" (e.g. spin0-replica correlation).
    This case involves a 4-sites system and the only non vanishing spin-spin correlation values
    are those of the 1°NN shell.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.0,0.0],[0.5,0.0,0.0],[0.75,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Spin-spin correlation values in the expected format
    SSC_xs = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))-0.25*(np.eye(4, k=2)+np.eye(4, k=-2))
    SSC_ys = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))-0.25*(np.eye(4, k=2)+np.eye(4, k=-2))
    SSC_zs = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))-0.25*(np.eye(4, k=2)+np.eye(4, k=-2))
    SSC_tots = np.eye(4)+0.5*(np.eye(4, k=1)+np.eye(4, k=-1))+0.5*(np.eye(4, k=3)+np.eye(4, k=-3))-0.25*(np.eye(4, k=2)+np.eye(4, k=-2))
    spin_correlations = (SSC_xs, SSC_ys, SSC_zs, SSC_tots)
    
    # Final arrangement of the spin-spin correlation values vs expectation
    final_SSCs, NN_distances = map_spin_correlations(system,spin_correlations,4,1)
    exp_SSCs = np.array([[1.0,0.5,-0.25,0.5,1.0],
                         [1.0,0.5,-0.25,0.5,1.0],
                         [1.0,0.5,-0.25,0.5,1.0],
                         [1.0,0.5,-0.25,0.5,1.0]])
    
    assert np.allclose(final_SSCs, exp_SSCs, atol=1e-10, rtol=1e-10)

def test_map_spin_correlations_6() -> None :
    '''
    Tests that the function in question correctly raises a ValueError exception when provided 
    with non-symmetric spin-spin correlation matrices.
    '''
    # Structural properties of the system
    latt_vecs = np.array([[1.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    sites = np.array([[0.0,0.0,0.0],[0.25,0.0,0.0],[0.5,0.0,0.0],[0.75,0.0,0.0]])
    system = SpinSystem(latt_vecs,sites,0.5)
    
    # Spin-spin correlation values in the expected format
    SSC_xs = np.eye(4)+0.5*(np.eye(4, k=1)-np.eye(4, k=-1))
    SSC_ys = np.eye(4)+0.5*(np.eye(4, k=1)-np.eye(4, k=-1))
    SSC_zs = np.eye(4)+0.5*(np.eye(4, k=1)-np.eye(4, k=-1))
    SSC_tots = np.eye(4)+0.5*(np.eye(4, k=1)-np.eye(4, k=-1))
    spin_correlations = (SSC_xs, SSC_ys, SSC_zs, SSC_tots)
    
    with pytest.raises(ValueError, match='The given spin-spin correlation matrices are not symmetric as expected.') :
        final_SSCs, NN_distances = map_spin_correlations(system,spin_correlations,4,1)

def test_solve_by_lanczos_0() -> None :
    '''
    Tests that the function leads to a correct estimation of the ground-state eigenvectors when the given Hamiltonian
    consists of a 4x4 diagonal matrix and does not admit any degeneracy.
    '''
    H = np.array([[1.0,0.0,0.0,0.0],
                  [0.0,2.3,0.0,0.0],
                  [0.0,0.0,2.1,0.0],
                  [0.0,0.0,0.0,11.1]])
    
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 4)
    approx_eigvecs = np.abs(approx_eigvecs)
    assert np.allclose(approx_eigvecs, np.array([[1.0,0.0,0.0,0.0]]).T, atol=1e-10, rtol=1e-10)

def test_solve_by_lanczos_1() -> None :
    '''
    Tests that the function leads to a correct estimation of the ground-state energy eigenvalue when 
    the given Hamiltonian consists of a 4x4 diagonal matrix and does not admit any degeneracy.
    '''
    H = np.array([[1.0,0.0,0.0,0.0],
                  [0.0,2.3,0.0,0.0],
                  [0.0,0.0,2.1,0.0],
                  [0.0,0.0,0.0,11.1]])
    
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 4)
    assert np.allclose(approx_eigvals, np.array([1.0]), atol=1e-10, rtol=1e-10)

def test_solve_by_lanczos_2() -> None :
    '''
    Tests that the function leads to a correct estimation of the ground-state degeneracy when 
    the given Hamiltonian consists of a 4x4 diagonal matrix and does not admit any degeneracy.
    '''
    H = np.array([[1.0,0.0,0.0,0.0],
                  [0.0,2.3,0.0,0.0],
                  [0.0,0.0,2.1,0.0],
                  [0.0,0.0,0.0,11.1]])
    
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 4)
    assert GS_deg==1

def test_solve_by_lanczos_3() -> None :
    '''
    Tests that the function leads to a correct estimation of the lowest energy eigenvalue when 
    the given Hamiltonian consists of a 4x4 diagonal matrix and the ground-state is also doubly-degenerate.
    '''
    H = np.array([[1.0,0.0,0.0,0.0],
                  [0.0,2.3,0.0,0.0],
                  [0.0,0.0,1.0,0.0],
                  [0.0,0.0,0.0,11.1]])
    
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 4)
    assert np.allclose(approx_eigvals, np.array([1.0]), atol=1e-10, rtol=1e-10)

def test_solve_by_lanczos_4() -> None :
    '''
    Tests that the function leads to a correct estimation of the ground-state degeneracy when 
    the given Hamiltonian consists of a 4x4 diagonal matrix and the ground-state is also doubly-degenerate.
    '''
    H = np.array([[1.0,0.0,0.0,0.0],
                  [0.0,2.3,0.0,0.0],
                  [0.0,0.0,1.0,0.0],
                  [0.0,0.0,0.0,11.1]])
    
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 4)
    assert GS_deg==2

def test_solve_by_lanczos_5() -> None :
    '''
    Tests that the function leads to a correct estimation of the ground-state eigenvectors when the given Hamiltonian
    consists of a generic 4x4 hermitian matrix and does not admit any degeneracy.
    '''
    H = np.array([[ 1.0,      0.2+1.5j, 1.1-2.2j, -2.1+1.0j],
                  [ 0.2-1.5j, 2.3,      0.4+0.1j, 4.4+0.1j],
                  [ 1.1+2.2j, 0.4-0.1j, 2.1,      1.2-0.4j],
                  [-2.1-1.0j, 4.4-0.1j, 1.2+0.4j, 11.1]], dtype=complex)
    
    # Compare eigenvectors from standard diagonalization vs Lanczos algorithm
    exact_eigvals, exact_eigvecs = np.linalg.eigh(H)
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 4)
    
    # Adjust the (arbitrary) phase factor
    first_angle = np.angle(approx_eigvecs[0][0])
    approx_eigvecs = np.array([el*np.exp(-first_angle*1.0j) for el in approx_eigvecs])
    
    assert np.allclose(approx_eigvecs.T, exact_eigvecs.T[0], atol=1e-10, rtol=1e-10)

def test_solve_by_lanczos_6() -> None :
    '''
    Tests that the function leads to a correct estimation of the lowest energy eigenvalue when the given Hamiltonian
    consists of a generic 4x4 hermitian matrix and admits a 1D ground-state eigenspace.
    '''
    H = np.array([[ 1.0,      0.2+1.5j, 1.1-2.2j, -2.1+1.0j],
                  [ 0.2-1.5j, 2.3,      0.4+0.1j, 4.4+0.1j],
                  [ 1.1+2.2j, 0.4-0.1j, 2.1,      1.2-0.4j],
                  [-2.1-1.0j, 4.4-0.1j, 1.2+0.4j, 11.1]], dtype=complex)
    
    # Compare eigenvalues from standard diagonalization vs Lanczos algorithm
    exact_eigvals, exact_eigvecs = np.linalg.eigh(H)
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 10)
    
    assert np.allclose(approx_eigvals, np.array([min(exact_eigvals)]), atol=1e-10, rtol=1e-10)

def test_solve_by_lanczos_7() -> None :
    '''
    Tests that the function leads to a correct estimation of the ground-state degeneracy when the given Hamiltonian
    consists of a generic 4x4 hermitian matrix and admits a 1D ground-state eigenspace.
    '''
    H = np.array([[ 1.0,      0.2+1.5j, 1.1-2.2j, -2.1+1.0j],
                  [ 0.2-1.5j, 2.3,      0.4+0.1j, 4.4+0.1j],
                  [ 1.1+2.2j, 0.4-0.1j, 2.1,      1.2-0.4j],
                  [-2.1-1.0j, 4.4-0.1j, 1.2+0.4j, 11.1]], dtype=complex)
    
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 4)
    assert GS_deg==1

def test_solve_by_lanczos_8() -> None :
    '''
    Tests that the function leads to a correct estimation of the lowest energy eigenvalue when the given Hamiltonian
    consists of a generic 4x4 hermitian matrix and admits a 2D ground-state eigenspace.
    '''
    H = np.array([[ 1.0,      0.0,      0.0,      0.0],
                  [ 0.0,      1.0,      0.0,      0.0],
                  [ 0.0,      0.0,     12.1, 1.2-0.4j],
                  [ 0.0,      0.0, 1.2+0.4j,     11.1]], dtype=complex)
    
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 10)
    assert np.allclose(approx_eigvals, np.array([1.0]), atol=1e-10, rtol=1e-10)

def test_solve_by_lanczos_9() -> None :
    '''
    Tests that the function leads to a correct estimation of the ground-state degeneracy when the given Hamiltonian
    consists of a generic 4x4 hermitian matrix and admits a 2D ground-state eigenspace.
    '''
    H = np.array([[ 1.0,      0.0,      0.0,      0.0],
                  [ 0.0,      1.0,      0.0,      0.0],
                  [ 0.0,      0.0,     12.1, 1.2-0.4j],
                  [ 0.0,      0.0, 1.2+0.4j,     11.1]], dtype=complex)
    
    approx_eigvecs, approx_eigvals, GS_deg = solve_by_lanczos(H, 0, 4)
    assert GS_deg==2
