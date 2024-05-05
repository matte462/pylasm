from global_functions import clean_line, direct2cartesian

import configparser
import ast
import numpy as np

class InputReader:
    '''
    This class implements all the necessary reading procedures 
    and provides all the parameters to the user.
    '''

    def __init__(self,config_file: str) :
        '''
        Initialize a new instance of InputReader 
        and directly stores the parameters from the input files 
        into its dict attributes.

        Args:
            config_file (str):   The path to the configuration file name to be read
        '''

        self.config_file = config_file
        self.config_info = {}
        self.struct_info = {}
        self.Js_info = {}
        self.read_config_file(config_file)
        self.read_struct_file()
        self.read_J_couplings_file()

    # Reading methods
    def read_config_file(self,config_file: str) -> None :
        '''
        Read the configuration file using the configparser library
        and updates the attribute self.config_info.

        Args:
            config_file (str): The path to the configuration file name to be read
        '''

        # Define permitted sections
        allowed_sections = ['SYSTEM','LANCZOS']

        # Define permitted keys per each section and the reference dictionary
        system_dict = {'struct_file_name': str,'struct_file_type': str,'mag_ion': str,'spin': float,'J_couplings_file': str,'max_NN_shell': int}
        lanczos_dict = {'lanczos_mode': str,'n_iterations': int,'energy_digits': int}
        ref_dict = {'SYSTEM': system_dict,'LANCZOS': lanczos_dict}

        # Define the default configuration
        default_config = configparser.ConfigParser()
        default_config.optionxform = str # Keys are case-sensitive
        default_config['SYSTEM'] = {'struct_file_name': "'NOT SPECIFIED'",'struct_file_type': "'POSCAR'",'mag_ion': "'NOT SPECIFIED'",'spin': "0.5",'J_couplings_file': "'NOT SPECIFIED'",'max_NN_shell': "1"}
        default_config['LANCZOS'] = {'lanczos_mode': "'scf'",'n_iterations': "0",'energy_digits': "6"}
    
        # Initialize the effective configuration
        config = configparser.ConfigParser()
        config.optionxform = str # Keys are case-sensitive
        config.read(config_file)
    
        # Raise exception if section in config_file is not allowed
        for section in config.sections() :
            if section not in allowed_sections :
                raise ValueError(f'{section} is not a valid section name')

        # Set default configuration as a starting point
        for section in default_config.sections() :
            if config.has_section(section) :
                for key in default_config[section].keys() :
                    if key not in config[section] :
                        config[section][key] = default_config[section][key]
            else :
                config.add_section(section)
                config[section] = default_config[section]

        # Fill the reference dictionary with the input data (translated into python literals)
        for section in allowed_sections :
            if section in config.sections() :
                for key, value in config[section].items() :
                    if key not in ref_dict[section].keys() :
                        raise KeyError(f'{key} is not a valid {section} key')
                    ref_dict[section][key] = ast.literal_eval(value)
            
        # Update the InputReader attribute
        self.config_info = ref_dict

    def read_struct_file(self) -> None : # EVENTUALLY IMPLEMENT OTHER STRUCTURE FILE FORMATS
        '''
        Reads the structural properties of the spin system under study
        depending on the format declared in the configuration file.
        '''
    
        struct_file_name = self.get_struct_file_name()
        if struct_file_name=='NOT SPECIFIED' :
            raise ValueError(f'The struct_file_name value is {struct_file_name} in {self.get_config_file()}.\nSo no structure is actually read.')
        mag_ion = self.get_mag_ion()
        if mag_ion=='NOT SPECIFIED' :
            raise ValueError(f'The mag_ion value is {mag_ion} in {self.get_config_file()}.\nSo no structure is actually read.')
        struct_file_type = self.get_struct_file_type()
        struct_mapping = {
            'POSCAR': InputReader.read_POSCAR,
            'STRUCT': InputReader.read_STRUCT,
            'CIF': InputReader.read_CIF,
            'PWI': InputReader.read_PWI
        }
        lattice_vectors, mag_ions_pos = struct_mapping[struct_file_type](self)
        self.struct_info = {
            'lattice_vectors': lattice_vectors,
            'mag_ions_pos': mag_ions_pos
        }

    def read_POSCAR(self) -> tuple :
        '''
        Returns the lattice vectors and the sites of the magnetic ions
        in case the format follows the standard conventions for POSCAR files. 
        '''

        struct_file_name = self.get_struct_file_name()
        mag_ion = self.get_mag_ion()

        # Quantities to be read from the POSCAR file
        scaling = 0.0
        lattice_vectors = []
        elements = []
        n_per_element = []
        is_direct = False
        mag_ions_pos = []
        is_selective = False

        # Read the content line-by-line (since POSCAR have a pre-determined structure)
        # CHECK :   - is the format compatible with the expectations?
        #           - is the file empty?
        with open(struct_file_name,'r') as f :
            content = f.readlines()
            for k in range(len(content)) :
                # First line (k=0) is ignored

                # Second line (k=1) provides the scaling factor
                if k==1 : scaling = float(content[k])

                # Third/Fourth/Fifth lines (k=2/3/4) provide the lattice vectors
                if k==2 or k==3 or k==4 : 
                    vector = clean_line(content[k])
                    vector = [float(el) for el in vector]
                    lattice_vectors.append(vector)

                # Sixth line (k=5) provides a list of the elements in the system
                if k==5 :
                    vector = clean_line(content[k])
                    vector[-1].replace('\n','')
                    for el in vector : elements.append(el)
                
                # Seventh line (k=6) provides how many atoms per element in the previous list there are
                if k==6 : 
                    vector = clean_line(content[k])
                    for el in vector : n_per_element.append(int(el))
                
                # Eighth line (k=7) provides info about whether Selective dynamics were set or not:
                # if yes --> content[k]='Selective dynamics' and the rest of the file is shifted down by one line
                # if no  --> content[k]='Direct' or 'Cartesian'
                if k==7 and content[k].count('S')!=0 : is_selective = True
                
                # Eighth/Nineth line (k=7/8) provides info about the chosen coordinate system
                # Important to know for properly reading the sites of the magnetic ions 
                if (k==7 or k==8) and content[k].count('D')!=0 : # D stands for Direct (coordinate system for atomic positions)
                    is_direct = True
            
            lattice_vectors = [np.dot(vec,scaling) for vec in lattice_vectors]
            
            # Loop over the following lines to capture the sites
            # The total number of sites should coincide with the sum of all elements in n_per_element
            # and they are ordered as in the list of elements 
            n_per_element.insert(0,0) # just to properly cycle over the sites
            starting_row = 8*(is_selective==False)+9*(is_selective==True)
            for i in range(len(elements)) :

                # If the element of interest does not come first in the list of elements
                # one should skip all the sites associated with the previous elements
                n_cumulated = 0
                for j in range(0,i+1) :
                    n_cumulated += n_per_element[j]

                # Only the sites of the magnetic ions are stored
                if mag_ion==elements[i] :
                    for k in range(starting_row+n_cumulated,starting_row+n_cumulated+n_per_element[i+1]) :
                        vector = clean_line(content[k])

                        # Selective dynamics requires the user to put 
                        # 3 boolean values at the end of each line
                        # So they should be removed
                        if is_selective==True :
                            for i in range(3) : vector.remove(vector[-1])
                        
                        vector = [float(el) for el in vector]
                        mag_ions_pos.append(vector)
            
            # Cartesian coordinate system is better than Direct one
            # So in case trasform one to the other
            if is_direct==True : mag_ions_pos = direct2cartesian(mag_ions_pos,lattice_vectors)
            n_per_element.remove(0)
        return lattice_vectors, mag_ions_pos
    
    def read_STRUCT(self) -> tuple : # STILL TO BE IMPLEMENTED
        pass
    
    def read_CIF(self) -> tuple : # STILL TO BE IMPLEMENTED
        pass
    
    def read_PWI(self) -> tuple : # STILL TO BE IMPLEMENTED
        pass

    def read_J_couplings_file(self) -> None : # STILL TO BE IMPLEMENTED
        J_couplings_file = self.get_J_couplings_file()
        if J_couplings_file!='NOT SPECIFIED' :
            # Reading procedure ...
            pass
        else :
            raise ValueError(f'The J_couplings_file value is {J_couplings_file} in {self.get_config_file()}.\nSo no interaction matrices are actually read.')

    # Getters methods
    def get_config_file(self) -> str :
        return self.config_file

    def get_config_info(self) -> dict :
        return self.config_info
    
    def get_struct_info(self) -> dict :
        return self.struct_info
    
    def get_Js_info(self) -> dict :
        return self.Js_info

    def get_struct_file_name(self) -> str :
        return self.config_info['SYSTEM']['struct_file_name']
    
    def get_struct_file_type(self) -> str :
        return self.config_info['SYSTEM']['struct_file_type']
    
    def get_mag_ion(self) -> str :
        return self.config_info['SYSTEM']['mag_ion']
    
    def get_spin(self) -> float :
        return self.config_info['SYSTEM']['spin']
    
    def get_J_couplings_file(self) -> str :
        return self.config_info['SYSTEM']['J_couplings_file']
    
    def get_max_NN_shell(self) -> int :
        return self.config_info['SYSTEM']['max_NN_shell']
    
    def get_lanczos_mode(self) -> str :
        return self.config_info['LANCZOS']['lanczos_mode']
    
    def get_lanczos_par(self) -> float :
        lanczos_mode = self.get_lanczos_mode()
        if lanczos_mode=='one_shot' : return self.config_info['LANCZOS']['n_iterations']
        elif lanczos_mode=='scf' : return self.config_info['LANCZOS']['energy_digits']
    
    def get_lattice_vectors(self) -> 'np.ndarray' :
        return self.struct_info['lattice_vectors']
    
    def get_mag_ions_pos(self) -> 'np.ndarray' :
        return self.struct_info['mag_ions_pos']
    
    def get_J_couplings(self) -> 'np.ndarray' : # STILL TO BE IMPLEMENTED
        pass

    def get_NN_vectors(self) -> 'np.ndarray' : # STILL TO BE IMPLEMENTED
        pass

    def print_summary(self) : # STILL TO BE IMPLEMENTED
        pass