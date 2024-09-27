from global_functions import *
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
    a matrix whose shape is not (3,3).
    '''
    with pytest.raises(ValueError, match='The adapt_magintmatrix function only accepts 3x3 square matrices as argument.') :
        new_matrix = adapt_magintmatrix(np.array([[1.0]]))

def test_adapt_magintmatrix_1() -> None :
    '''
    Tests that the  adapt_magintmatrix() function returns the correct transformed matrix when provided with 
    a 3x3 matrix in the most general form.
    '''
    matrix = np.array([[1.0,2.0,3.0],
                       [4.0,5.0,6.0],
                       [7.0,8.0,9.0]])
    exp_matrix = np.array([[9.0,7.0,8.0],
                           [3.0,1.0,2.0],
                           [6.0,4.0,5.0]])
    new_matrix = adapt_magintmatrix(matrix)

    is_new_matrix_ok = True
    for r in range(new_matrix.shape[0]) :
        is_new_matrix_ok = is_new_matrix_ok and np.array(exp_matrix[r]==new_matrix[r]).all()
    assert is_new_matrix_ok==True