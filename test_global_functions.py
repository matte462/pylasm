from global_functions import *

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