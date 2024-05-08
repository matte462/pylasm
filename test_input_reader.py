from input_reader import InputReader
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
    default_settings['LANCZOS'] = {'lanczos_mode': 'scf','n_iterations': 0,'energy_digits': 6}

    # Effective configuration to be tested
    reader = InputReader('./Inputs/TestFiles/config_file_3.ini')
    read_settings = reader.get_config_info()

    # The values of struct_file_name, mag_ion, J_couplings_file keys still needs to be updated
    eff_struct_file_name = reader.get_struct_file_name()
    eff_mag_ion = reader.get_mag_ion()
    eff_J_couplings_file = reader.get_J_couplings_file()
    default_settings['SYSTEM']['struct_file_name'] = eff_struct_file_name
    default_settings['SYSTEM']['mag_ion'] = eff_mag_ion
    default_settings['SYSTEM']['J_couplings_file'] = eff_J_couplings_file

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
    
    with pytest.raises(ValueError, match='foo unlikely belongs to the periodic table of elements.') : 
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

def test_read_J_couplings_file_0() -> None :
    '''
    Tests that the proper Exception is raised when the configuration file does not
    include the input file for the magnetic interactions.
    '''

    with pytest.raises(ValueError, match='The J_couplings_file value is NOT SPECIFIED in ./Inputs/TestFiles/config_file_2.ini.\nSo no interaction matrices are actually read.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_2.ini')

    

