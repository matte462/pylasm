from global_functions import *

import configparser
import ast
import numpy as np

class InputReader :
    '''
    This class implements all the necessary reading procedures 
    and provides all the parameters to the user.
    '''

    def __init__(self,config_file: str) :
        '''
        Initializes a new instance of InputReader 
        and directly stores the parameters from the input files 
        into its dict attributes.

        Args:
            config_file (str):   The path to the configuration file name to be read.
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
            config_file (str): The path to the configuration file name to be read.
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
        default_config['LANCZOS'] = {'lanczos_mode': "'scf'",'n_iterations': "20",'energy_digits': "6"}
    
        # Initialize the effective configuration
        config = configparser.ConfigParser()
        config.optionxform = str # Keys are case-sensitive
        config.read(config_file)
    
        # Raise exception if section in config_file is not allowed
        for section in config.sections() :
            if section not in allowed_sections :
                raise ValueError(f'{section} is not a valid section name.')

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
                        raise KeyError(f'{key} is not a valid {section} key.')
                    expected_type = ref_dict[section][key]
                    if type(ast.literal_eval(value))!=expected_type :
                        raise TypeError(f'{key} value is not compatible with the expected type {expected_type.__name__}.')
                    ref_dict[section][key] = ast.literal_eval(value)
        
        # Some exceptions to help the use to choose appropriate values for spin, max_NN_shell, n_iterations and energy_digits keys
        if not is_spin_acceptable(ref_dict['SYSTEM']['spin']) : 
            raise ValueError(f"{ref_dict['SYSTEM']['spin']} is not acceptable as a spin quantum number.")
        if ref_dict['SYSTEM']['max_NN_shell']<=0 :
            raise ValueError(f'Non-positive values for max_NN_shell key are not permitted.')
        if ref_dict['LANCZOS']['n_iterations']<=0 :
            raise ValueError(f'Non-positive values for n_iterations key are not permitted.')
        if ref_dict['LANCZOS']['energy_digits']<=0 :
            raise ValueError(f'Non-positive values for energy_digits key are not permitted.')
            
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
        # Symbols for the elements do not contain more than 2 characters
        # The only exceptions to this rule are the heaviest elements, but they are
        # unlikely to be used in condensed matter physics' applications or studies. 
        elif len(mag_ion)>2 or len(mag_ion)==0 :
            raise ValueError(f'{mag_ion} does not belong to the periodic table of elements.')
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
        # Some of them are initialized as lists since their size in unknown a-priori
        scaling = 0.0
        lattice_vectors = np.zeros((3,3))
        elements = []
        n_per_element = []
        is_direct = False
        mag_ions_pos = np.array([])
        is_selective = False

        # Read the content line-by-line (since POSCAR have a pre-determined structure)
        # CHECK :   - is the format compatible with the expectations?
        #           - is the file empty?
        with open(struct_file_name,'r') as f :
            content = f.readlines()
            if len(content)==0 :
                raise IOError(f'{struct_file_name} is empty.')
            for k in range(1,len(content)) :
                vector = clean_line(content[k])
                # First line (k=0) is ignored

                # Second line (k=1) provides the scaling factor
                if k==1 : 
                    if len(vector)!=1 :
                        raise ValueError(f'The content of the second line in {struct_file_name} differs from what is expected.')
                    if vector[0].replace('.','').isdigit() :
                        scaling = float(vector[0])
                    else :
                        raise TypeError(f'{vector[0]} is read in the {k+1}-th line of {struct_file_name}, while the scaling factor is expected.')

                # Third/Fourth/Fifth lines (k=2/3/4) provide the lattice vectors
                if k==2 or k==3 or k==4 : 
                    if len(vector)!=3 :
                        raise ValueError(f'The content of the {k+1}-th line in {struct_file_name} differs from what is expected.')
                    for el in vector :
                        if el.replace('.','').isdigit() : 
                            el = float(el)
                        else :
                            raise TypeError(f'{el} is read in the {k+1}-th line of {struct_file_name}, while a lattice vector is expected.')
                    lattice_vectors[k-2] = np.array(vector)

                # Sixth line (k=5) provides a list of the elements in the system
                if k==5 :
                    for el in vector : 
                        # Same exception is raised for the mag_ion variable in the method read_struct_file()
                        if len(el)>2 or len(el)==0 :
                            raise ValueError(f'{el} unlikely belongs to te periodic table of elements.')
                        else :
                            elements.append(el)
                
                # Seventh line (k=6) provides how many atoms per element in the previous list there are
                if k==6 : 
                    if len(vector)!=len(elements) :
                        raise ValueError(f'The content of the {k+1}-th line in {struct_file_name} differs from what is expected.')
                    for el in vector : 
                        if el.isdigit() :
                            el = int(el)
                            n_per_element.append(el)
                        else :
                            raise TypeError(f'{el} is read in {k+1}-th line in {struct_file_name}, while a positive integer is expected.') 
                
                # Eighth line (k=7) provides info about whether Selective dynamics were set or not:
                # if yes    --> the first non-empty character in content[k] is either 'S' or 's'
                #               and the rest of the file is shifted down by one line
                # if no     --> the first non-empty character in content[k] either 'D'/'d' or 'C'/'c'
                if k==7 and vector[0][0].upper()=='S' :
                    if len(vector)>2 or len(vector)==0 :
                        raise ValueError(f'The content of the {k+1}-th line in {struct_file_name} differs from what is expected.')
                    is_selective = True
                
                # Eighth/Nineth line (k=7/8) provides info about the chosen coordinate system
                # Important to know for properly reading the sites of the magnetic ions 
                if (k==7 or k==8) and vector[0][0].upper()=='D' : # D stands for Direct (coordinate system for atomic positions)
                    if len(vector)!=1 :
                        raise ValueError(f'The content of the {k+1}-th line in {struct_file_name} differs from what is expected.')
                    is_direct = True
                
                if k==8 : break
            
            lattice_vectors = np.multiply(lattice_vectors,scaling)
            if elements.count(mag_ion)==0 :
                raise ValueError(f'No {mag_ion} atoms are found in {struct_file_name}.')
            
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
                    mag_ions_pos = np.zeros((n_per_element[i+1],3))
                    for k in range(starting_row+n_cumulated,starting_row+n_cumulated+n_per_element[i+1]) :
                        if k==len(content) :
                            raise IOError(f'End of {struct_file_name} is reached before all magnetic sites could be read.')
                        vector = clean_line(content[k])
                        if len(vector)<3 :
                            raise ValueError(f'The content of the {k+1}-th line in {struct_file_name} differs from what is expected.')

                        # Removing last elements until the first 3 remain allows to deal with the Selective dynamics case
                        # and eventually the presence of comments at the end of each line simultaneously 
                        while len(vector)>3 :
                            vector.remove(vector[-1])

                        for el in vector :
                            if el.replace('.','').isdigit() :
                                el = float(el)
                            else :
                                raise TypeError(f'{el} is read in the {k+1}-th line in {struct_file_name}, while a coordinate is expected.')
                        mag_ions_pos[k-starting_row-n_cumulated] = np.array(vector)
            mag_ions_pos = np.array(mag_ions_pos)
            
            # Cartesian coordinate system is better than Direct one
            # So in case trasform one to the other
            if is_direct==True :
                mag_ions_pos = np.dot(mag_ions_pos,lattice_vectors)
            n_per_element.remove(0)
            return lattice_vectors, mag_ions_pos
    
    def read_STRUCT(self) -> tuple : # STILL TO BE IMPLEMENTED
        pass
    
    def read_CIF(self) -> tuple : # STILL TO BE IMPLEMENTED
        pass
    
    def read_PWI(self) -> tuple : # STILL TO BE IMPLEMENTED
        pass

    def read_J_couplings_file(self) -> None :
        J_couplings_file = self.get_J_couplings_file()
        if J_couplings_file!='NOT SPECIFIED' :
            # Reading procedure ...
            Js_info = {}

            magint_matrices = []
            T_vectors = []
            distances = []
            involved_atoms = []
            shells = []
            coor_nums = []
            with open(J_couplings_file,'r') as Vf :
                content = Vf.readlines()
                for k in range(len(content)) :
                    if content[k].find('INTERACTION')!=-1 :
                        how_many_matrices = len(magint_matrices)
                        cleaned_line = clean_line(content[k])
                        cleaned_line.remove('-----INTERACTION:')
                        cleaned_line = cleaned_line[0]+cleaned_line[1]+cleaned_line[2] 
                        involved_atoms.append([cleaned_line,how_many_matrices])
                    if content[k].find('Shell=')!=-1 :
                        cleaned_line = content[k].replace('Shell=','')
                        shells.append(int(cleaned_line))
                    if content[k].find('Coor_num')!=-1 :
                        cleaned_line = clean_line(content[k])
                        cleaned_line.remove('Coor_num')
                        cleaned_line.remove('=')
                        coor_nums.append(int(cleaned_line[0]))
                    if content[k].find('Dipole-Dipole interactions')!=-1 and k+2<len(content) :
                        if content[k+2].find('interactions')==-1 and content[k+2].find('INTERACTION')==-1 :
                            for i in range(coor_nums[-1]) :
                                vector = clean_line(content[k+3+8*i])
                                if len(vector)>1 and vector[0]=='T' :
                                    vector.remove('T')
                                    vector.remove('=')
                                    vector[0] = vector[0].replace('[','')
                                    vector[-1] = vector[-1].replace(']','')
                                    vector[-1] = vector[-1].replace('\n','')
                                    if vector.count('\n')!=0 : vector.remove('\n')
                                    if vector.count('')!=0 : 
                                        for n in range(vector.count('')) : vector.remove('')
                                    T_vectors.append([float(el) for el in vector])
                                    vector = [float(el)**2 for el in vector]
                                    distance = np.sqrt(np.array(vector).sum())
                                    distances.append(distance)

                                    matrix = []
                                    n_rows = 3
                                    for n in range(n_rows) :
                                        row = clean_line(content[k+6+8*i+n])
                                        row.remove(row[0])
                                        #row.remove('\n')
                                        row = [float(el) for el in row]
                                        matrix.append(row)
                                    magint_matrices.append(matrix)
                sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
                distances = [round(distances[i],3) for i in sorted_indices]
                T_vectors = [np.array(T_vectors[i]) for i in sorted_indices]
                magint_matrices = [np.mat(magint_matrices[i]) for i in sorted_indices]

                #j = 0
                NN_index = 0
                shell_count = 0
                #for i in range(1,max_NN_shell+1) :
                aux_magint_dict = {}
                aux_J_matrices = {}
                aux_T_vectors = {}
                for j in range(1,len(distances)) :
                    if distances[j-1]!=distances[j] : NN_index=0
                    aux_T_vectors[f'{NN_index}'] = [round(el,3) for el in np.dot(T_vectors[j-1],0.529177249)]
                    aux_J_matrices[f'{NN_index}'] = magint_matrices[j-1]
                    if NN_index==0 : 
                        aux_magint_dict['NN_shell_radius'] = distances[j-1]
                        aux_magint_dict['T_vectors'] = aux_T_vectors
                        aux_magint_dict['J_matrices'] = aux_J_matrices
                        shell_count += 1
                        Js_info[f'{shell_count}° NN shell'] = aux_magint_dict
                    NN_index += 1
                self.Js_info = Js_info
        else :
            raise ValueError(f'The J_couplings_file value is {J_couplings_file} in {self.get_config_file()}.\nSo no interaction matrices are actually read.')

    # Getters methods
    def get_config_file(self) -> str :
        '''
        Returns the name of the configuration file.
        '''
        return self.config_file

    def get_config_info(self) -> dict :
        '''
        Returns all the information read from the configuration file as a dictionary.
        '''
        return self.config_info
    
    def get_struct_info(self) -> dict :
        '''
        Returns lattice vectors and sites read from the structure file as a dictionary.
        '''
        return self.struct_info
    
    def get_Js_info(self) -> dict :
        '''
        Returns all the interaction matrices read from the MagInt output file as a dictionary.
        '''
        return self.Js_info

    def get_struct_file_name(self) -> str :
        '''
        Returns the name of the structure file.
        '''
        return self.config_info['SYSTEM']['struct_file_name']
    
    def get_struct_file_type(self) -> str :
        '''
        Returns the chosen structure file format if specified in the configuration file or simply POSCAR otherwise.
        '''
        return self.config_info['SYSTEM']['struct_file_type']
    
    def get_mag_ion(self) -> str :
        '''
        Returns the symbol for the magnetic element of interest.
        '''
        return self.config_info['SYSTEM']['mag_ion']
    
    def get_spin(self) -> float :
        '''
        Returns the spin quantum number read from the configuration file.
        '''
        return self.config_info['SYSTEM']['spin']
    
    def get_J_couplings_file(self) -> str :
        '''
        Returns the name of the MagInt output file.
        '''
        return self.config_info['SYSTEM']['J_couplings_file']
    
    def get_max_NN_shell(self) -> int :
        '''
        Returns the chosen number of NN shells if specified in the configuration file or directly 1 otherwise.
        '''
        return self.config_info['SYSTEM']['max_NN_shell']
    
    def get_lanczos_mode(self) -> str :
        '''
        Returns the chosen execution mode of the Lanczos algorithm for Exact Diagonalization.
        '''
        return self.config_info['LANCZOS']['lanczos_mode']
    
    def get_lanczos_par(self) -> float :
        '''
        Returns a fundamental integer for the execution of the Lanczos algorithm.
        Its meaning depends on the choice of the lanczos_mode value.
        In particular, if lanczos_mode='one_shot', it stands for the exact number of Lanczos iterations to be performed;
        while if lanczos_mode='scf', it determines the resolution on the ground-state energy as follows
                epsilon = 10**(-energy_digits).
        This parameter sets the convergence criterion for the self-consistent cycle of Lanczos iterations.
        '''
        lanczos_mode = self.get_lanczos_mode()
        if lanczos_mode=='one_shot' : 
            return self.config_info['LANCZOS']['n_iterations']
        elif lanczos_mode=='scf' : 
            return self.config_info['LANCZOS']['energy_digits']
    
    def get_lattice_vectors(self) -> 'np.ndarray' :
        '''
        Returns the lattice vectors read from the structure file.
        '''
        return self.struct_info['lattice_vectors']
    
    def get_mag_ions_pos(self) -> 'np.ndarray' :
        '''
        Returns the atomic positions of the magnetic elements in the system read from the structure file.
        '''
        return self.struct_info['mag_ions_pos']
    
    def get_J_couplings(self) -> 'np.ndarray' : # STILL TO BE IMPLEMENTED
        pass

    def get_NN_vectors(self) -> 'np.ndarray' : # STILL TO BE IMPLEMENTED
        pass

    def print_summary(self) : # STILL TO BE IMPLEMENTED
        pass