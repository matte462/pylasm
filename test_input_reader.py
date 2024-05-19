from input_reader import InputReader
import numpy as np
import pytest

def test_read_config_file_0() -> None :
    '''
    Tests that the default configuration is correctly set, if the provided input file
    only specifies the strictly necessary keys (i.e. struct_file_name, mag_ion and
    J_couplings_file). So, technically this function checks whether the value of all
    other keys is initialized properly or not.
    '''
    # Expected default configuration
    default_settings = {}
    default_settings['SYSTEM'] = {'struct_file_name': 'NOT SPECIFIED','struct_file_type': 'POSCAR','mag_ion': 'NOT SPECIFIED','spin': 0.5,'J_couplings_file': 'NOT SPECIFIED','max_NN_shell': 1}
    default_settings['LANCZOS'] = {'lanczos_mode': 'scf','n_iterations': 20,'energy_digits': 6}

    # The values of struct_file_name, mag_ion, J_couplings_file keys still need to be fixed
    default_settings['SYSTEM']['struct_file_name'] = './Inputs/StructureFiles/POSCAR_Ce_2x1x1.vasp'
    default_settings['SYSTEM']['mag_ion'] = 'Ce'
    default_settings['SYSTEM']['J_couplings_file'] = './Inputs/MagIntFiles/V_Mult_Ce0-Ce0_1NN_chain.dat'

    # Effective configuration to be tested
    reader = InputReader('./Inputs/TestFiles/config_file_3.ini')
    read_settings = reader.get_config_info()
    assert read_settings==default_settings

def test_read_config_file_1() -> None :
    '''
    Tests that the proper Exception is raised when the name of a section is mistyped or not allowed.
    '''
    with pytest.raises(ValueError, match='SSTEM is not a valid section name.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_4.ini')

def test_read_config_file_2() -> None :
    '''
    Tests that the proper Exception is raised when a key is mistyped or not allowed.
    '''
    with pytest.raises(KeyError, match='ma_ion is not a valid SYSTEM key.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_5.ini')

def test_read_config_file_3() -> None :
    '''
    Tests that the proper Exception is raised when both the name of a section and 
    a key are mistyped or not allowed.
    The code is written so as to give higher priority to the first error.
    '''
    with pytest.raises(ValueError, match='SSTEM is not a valid section name.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_6.ini')

def test_read_config_file_4() -> None :
    '''
    Tests that the proper Exception is raised when the type of a variable in the
    configuration file differs from what is expected.
    ''' 
    with pytest.raises(TypeError, match=f'max_NN_shell value is not compatible with the expected type.') :
        reader = InputReader('./Inputs/TestFiles/config_file_11.ini')

def test_read_config_file_5() -> None :
    '''
    Tests that the proper Exception is raised when the spin value is not physical, namely neither
    positive integer nor positive half-integer.
    ''' 
    with pytest.raises(ValueError, match='2.05 is not acceptable as a spin quantum number.') :
        reader = InputReader('./Inputs/TestFiles/config_file_12.ini')

def test_read_config_file_6() -> None :
    '''
    Tests that the proper Exception is raised when the n_iterations value is non-positive.
    Same behaviour should be valid for max_NN_shell and energy_digits.
    ''' 
    with pytest.raises(ValueError, match='Non-positive values for n_iterations key are not permitted.') :
        reader = InputReader('./Inputs/TestFiles/config_file_13.ini')

def test_read_config_file_7() -> None :
    '''
    Tests that the code properly reads the configuration file in a standard situation
    when all keys are specified.
    '''
    # Expected configuration
    exp_settings = {}
    exp_settings['SYSTEM'] = {
        'struct_file_name': './Inputs/TestFiles/POSCAR_test_2.vasp',
        'struct_file_type': 'POSCAR',
        'mag_ion': 'Ag',
        'spin': 1.5,
        'J_couplings_file': './Inputs/MagIntFiles/V_Mult_Ce0-Ce0_1NN_chain.dat',
        'max_NN_shell': 2
    }
    exp_settings['LANCZOS'] = {
        'lanczos_mode': 'one_shot',
        'n_iterations': 12,
        'energy_digits': 8
    }

    # Effective configuration to be tested
    reader = InputReader('./Inputs/TestFiles/config_file_14.ini')
    read_settings = reader.get_config_info()
    assert read_settings==exp_settings

def test_read_struct_file_0() -> None :
    '''
    Tests that the proper Exception is raised when the configuration file does not
    include the structure file name.
    '''
    with pytest.raises(ValueError, match='The struct_file_name value is NOT SPECIFIED in ./Inputs/TestFiles/config_file_0.ini.\nSo no structure is actually read.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_0.ini')

def test_read_struct_file_1() -> None :
    '''
    Tests that the proper Exception is raised when the configuration file does not
    include the symbol of the magnetic ion.
    '''
    with pytest.raises(ValueError, match='The mag_ion value is NOT SPECIFIED in ./Inputs/TestFiles/config_file_1.ini.\nSo no structure is actually read.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_1.ini')

def test_read_struct_file_2() -> None :
    '''
    Tests that the proper Exception is raised when the value of the mag_ion key is
    not compatible with the usual symbols for elements.
    '''
    with pytest.raises(ValueError, match='foo does not belong to the periodic table of elements.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_7.ini')

def test_read_POSCAR_0() -> None :
    '''
    Tests that the proper Exception is raised when the POSCAR file is empty.
    '''
    with pytest.raises(IOError, match='./Inputs/TestFiles/POSCAR_test_0.vasp is empty.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_8.ini')

def test_read_POSCAR_1() -> None :
    '''
    Tests that the proper Exception is raised when the POSCAR file does not follow the
    standard conventions for POSCAR files.
    '''
    with pytest.raises(ValueError, match='The content of the second line in ./Inputs/TestFiles/POSCAR_test_1.vasp differs from what is expected.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_9.ini')

def test_read_POSCAR_2() -> None :
    '''
    Tests that the proper Exception is raised when the symbol of the magnetic ion is not
    found in the list of elements within the POSCAR file.
    '''
    with pytest.raises(ValueError, match='No Ce atoms are found in ./Inputs/TestFiles/POSCAR_test_2.vasp.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_10.ini')

def test_read_POSCAR_3() -> None :
    '''
    Tests that the code properly reads the lattice vectors and the atomic positions when
    the POSCAR file takes the most basic form.
    '''
    # Expected lattice vectors and atomic positions
    exp_latt_vecs = np.array([[3.0,0.0,0.0],
                              [0.0,6.0,0.0],
                              [0.0,1.0,4.0]])
    exp_mag_ions_pos = np.array([[0.0,0.0,0.0],
                                 [1.0,0.0,0.0]])

    # Effective lattice vectors and atomic positions to be tested
    reader = InputReader('./Inputs/TestFiles/config_file_15.ini')
    latt_vecs = reader.get_lattice_vectors()
    mag_ions_pos = reader.get_mag_ions_pos()

    # Exploit AND logic operation to check if the vectors are all the same
    are_latt_vecs_ok = True
    for v in range(latt_vecs.shape[0]) :
        are_latt_vecs_ok = are_latt_vecs_ok and np.array(exp_latt_vecs[v]==latt_vecs[v]).all()

    are_mag_ions_pos_ok = True
    for v in range(mag_ions_pos.shape[0]) :
        are_mag_ions_pos_ok = are_mag_ions_pos_ok and np.array(exp_mag_ions_pos[v]==mag_ions_pos[v]).all()
    assert are_latt_vecs_ok and are_mag_ions_pos_ok

def test_read_POSCAR_4() -> None :
    '''
    Tests that the code properly reads the atomic positions of the elements under study
    when they are preceded by the ones of other elements, which shall not be read.
    '''
    # Expected atomic positions
    exp_mag_ions_pos = np.array([[1.0,0.0,0.0],
                                 [1.5,0.0,0.0]])

    # Effective atomic positions to be tested
    reader = InputReader('./Inputs/TestFiles/config_file_16.ini')
    mag_ions_pos = reader.get_mag_ions_pos()

    # Exploit AND logic operation to check if the vectors are all the same
    are_mag_ions_pos_ok = True
    for v in range(mag_ions_pos.shape[0]) :
        are_mag_ions_pos_ok = are_mag_ions_pos_ok and np.array(exp_mag_ions_pos[v]==mag_ions_pos[v]).all()
    assert are_mag_ions_pos_ok

def test_read_POSCAR_5() -> None :
    '''
    Tests that the code properly reads the atomic positions of the elements under study
    when Selective dynamics are written into the POSCAR file.
    '''
    # Expected atomic positions
    exp_mag_ions_pos = np.array([[1.0,0.0,0.0],
                                 [1.5,0.0,0.0]])

    # Effective atomic positions to be tested
    reader = InputReader('./Inputs/TestFiles/config_file_17.ini')
    mag_ions_pos = reader.get_mag_ions_pos()

    # Exploit AND logic operation to check if the vectors are all the same
    are_mag_ions_pos_ok = True
    for v in range(mag_ions_pos.shape[0]) :
        are_mag_ions_pos_ok = are_mag_ions_pos_ok and np.array(exp_mag_ions_pos[v]==mag_ions_pos[v]).all()
    assert are_mag_ions_pos_ok

def test_read_POSCAR_6() -> None :
    '''
    Tests that the proper Exception is raised when the number of atomic positons is not enough 
    to read the magnetic sites of interest.
    '''
    with pytest.raises(IOError, match='End of ./Inputs/TestFiles/POSCAR_test_5.vasp is reached before all magnetic sites could be read.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_18.ini')

def test_read_POSCAR_7() -> None :
    '''
    Tests that the proper Exception is raised when the number of lattice vectors is lower 
    than 3 as expected.
    '''
    with pytest.raises(ValueError, match='The content of the 5-th line in ./Inputs/TestFiles/POSCAR_test_6.vasp differs from what is expected.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_19.ini')

def test_read_POSCAR_8() -> None :
    '''
    Tests that the proper Exception is raised when the number of lattice vectors is lower
    than 3 as expected, and at the same time exactly 3 symbols of elements are listed just below.
    '''
    with pytest.raises(TypeError, match='Be is read in the 5-th line of ./Inputs/TestFiles/POSCAR_test_7.vasp, while a lattice vector is expected.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_20.ini')

def test_read_POSCAR_9() -> None :
    '''
    Tests that the proper Exception is raised when the number of atomic positons is not enough 
    to read the sites of interest, but the last line is empty.
    '''
    with pytest.raises(ValueError, match='The content of the 12-th line in ./Inputs/TestFiles/POSCAR_test_8.vasp differs from what is expected.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_21.ini')

def test_read_J_couplings_file_0() -> None :
    '''
    Tests that the proper Exception is raised when the configuration file does not
    include the input file for the magnetic interactions.
    '''
    with pytest.raises(ValueError, match='The J_couplings_file value is NOT SPECIFIED in ./Inputs/TestFiles/config_file_2.ini.\nSo no interaction matrices are actually read.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_2.ini')

