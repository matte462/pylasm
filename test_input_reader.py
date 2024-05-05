from input_reader import InputReader
import pytest

def test_read_config_file_0() -> None :
    reader = InputReader('./Inputs/TestFiles/config_file_3.ini')
    read_settings = reader.get_config_info()
    eff_struct_file_name = reader.get_struct_file_name()
    eff_mag_ion = reader.get_mag_ion()
    eff_J_couplings_file = reader.get_J_couplings_file()
    default_settings = {}
    default_settings['SYSTEM'] = {'struct_file_name': 'NOT SPECIFIED','struct_file_type': 'POSCAR','mag_ion': 'NOT SPECIFIED','spin': 0.5,'J_couplings_file': 'NOT SPECIFIED','max_NN_shell': 1}
    default_settings['LANCZOS'] = {'lanczos_mode': 'scf','n_iterations': 0,'energy_digits': 6}
    default_settings['SYSTEM']['struct_file_name'] = eff_struct_file_name
    default_settings['SYSTEM']['mag_ion'] = eff_mag_ion
    default_settings['SYSTEM']['J_couplings_file'] = eff_J_couplings_file
    assert read_settings==default_settings

def test_read_struct_file_0() -> None :
    with pytest.raises(ValueError, match='The struct_file_name value is NOT SPECIFIED in ./Inputs/TestFiles/config_file_0.ini.\nSo no structure is actually read.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_0.ini')

def test_read_struct_file_1() -> None :
    with pytest.raises(ValueError, match='The mag_ion value is NOT SPECIFIED in ./Inputs/TestFiles/config_file_1.ini.\nSo no structure is actually read.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_1.ini')

def test_read_J_couplings_file_0() -> None :
    with pytest.raises(ValueError, match='The J_couplings_file value is NOT SPECIFIED in ./Inputs/TestFiles/config_file_2.ini.\nSo no interaction matrices are actually read.') : 
        reader = InputReader('./Inputs/TestFiles/config_file_2.ini')

    

