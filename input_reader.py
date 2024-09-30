from global_functions import *

import os
import configparser
import ast
import numpy as np

class InputReader :
    '''
    This class implements all the necessary reading procedures 
    and provides all the parameters to the user.
    '''

    def __init__(self,config_file: str='init_config.ini') :
        '''
        Initializes a new instance of InputReader 
        and directly stores the parameters from the input files 
        into its dict attributes.
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
        allowed_sections = ['STRUCTURE','HAMILTONIAN','OUTPUT']

        # Define permitted keys per each section and the reference dictionary
        struct_dict = {'struct_file_name': str,'struct_file_type': str,'mag_ion': str,'spin': float,'n_dim': int}
        ham_dict = {'J_couplings_file': str,'max_NN_shell': int,'shell_digits': int,'B_field': list,'tol_imag': float}
        out_dict = {'n_excited': int,'lanczos_digits': int,'magn_output_mode': str,'show_plot': bool}
        ref_dict = {'STRUCTURE': struct_dict,'HAMILTONIAN': ham_dict,'OUTPUT': out_dict}
        
        # Define the default configuration
        default_config = configparser.ConfigParser()
        default_config.optionxform = str # Keys are case-sensitive
        default_config['STRUCTURE'] = {'struct_file_name': "'NOT SPECIFIED'",'struct_file_type': "'POSCAR'",'mag_ion': "'NOT SPECIFIED'",'spin': "0.5",'n_dim': "1"}
        default_config['HAMILTONIAN'] = {'J_couplings_file': "'NOT SPECIFIED'",'max_NN_shell': "1",'shell_digits': "3",'B_field': "[0.0,0.0,0.0]",'tol_imag': "1e-6"}
        default_config['OUTPUT'] = {'n_excited': "0",'lanczos_digits': "10",'magn_output_mode': "'M_z'",'show_plot': "True"}
    
        # Initialize the effective configuration
        config = configparser.ConfigParser()
        config.optionxform = str # Keys are case-sensitive
        if os.path.exists(config_file) :
            config.read(config_file)
        else :
            raise FileNotFoundError('No configuration file called init_config.ini was found within the working directory.')
    
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
        
        # Some exceptions to help the use to choose appropriate values for the input keys
        if not is_spin_acceptable(ref_dict['STRUCTURE']['spin']) : 
            raise ValueError(f"{ref_dict['STRUCTURE']['spin']} is not acceptable as a spin quantum number.")
        if ref_dict['STRUCTURE']['n_dim']<=0 :
            raise ValueError('Non-positive values for n_dim key are not permitted.')
        for key in ['max_NN_shell','shell_digits','tol_imag'] :
            if ref_dict['HAMILTONIAN'][key]<=0 :
                raise ValueError('Non-positive values for '+key+' key are not permitted.')
        if ref_dict['OUTPUT']['lanczos_digits']<=0 :
            raise ValueError('Non-positive values for lanczos_digits key are not permitted.')
        if ref_dict['OUTPUT']['n_excited']<0 :
            raise ValueError('Negative values for n_excited key are not permitted.')
        if ref_dict['OUTPUT']['magn_output_mode'] not in ['M_x','M_y','M_z','M_full'] :
            raise ValueError(ref_dict['OUTPUT']['magn_output_mode']+' is not an allowed value for magn_output_mode key\\. Choose among M_x, M_y, M_z or M_full according to the desired definition of the magnetization.')
            
        # Update the InputReader attribute
        self.config_info = ref_dict

    def read_struct_file(self) -> None : # EVENTUALLY IMPLEMENT OTHER STRUCTURE FILE FORMATS
        '''
        Reads the structural properties of the spin system under study
        depending on the format declared in the configuration file.
        '''
        struct_file_name = self.get_struct_file_name()
        mag_ion = self.get_mag_ion()
        
        # Raise Exceptions when needed
        if struct_file_name=='NOT SPECIFIED' :
            raise ValueError(f'The struct_file_name value is {struct_file_name} in {self.get_config_file()}.\nSo no structure is actually read.')
        elif not os.path.exists(struct_file_name) :
            raise FileNotFoundError(f'No structure file called {struct_file_name} was found.')
        if mag_ion=='NOT SPECIFIED' :
            raise ValueError(f'The mag_ion value is {mag_ion} in {self.get_config_file()}.\nSo no structure is actually read.')
        # Symbols for the elements do not contain more than 2 characters
        # The only exceptions to this rule are the heaviest elements, but they are
        # unlikely to be used in condensed matter physics' applications or studies. 
        elif len(mag_ion)>2 or len(mag_ion)==0 :
            raise ValueError(f'{mag_ion} does not belong to the periodic table of elements.')
        
        # Select the proper reading function according to the declared structure file type
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
                        el_tmp = el.replace('.','').replace('-','')
                        if not el_tmp.isdigit() :
                            raise TypeError(f'{el} is read in the {k+1}-th line of {struct_file_name}, while a lattice vector is expected.')
                    lattice_vectors[k-2] = np.array([float(el) for el in vector])

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
                        el_tmp = el.replace('.','').replace('-','')
                        if not el_tmp.isdigit() :
                            raise TypeError(f'{el} is read in {k+1}-th line in {struct_file_name}, while a positive integer is expected.') 
                        n_per_element.append(int(el))
                
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
                            el_tmp = el.replace('.','').replace('-','')
                            if not el_tmp.isdigit() :
                                raise TypeError(f'{el} is read in the {k+1}-th line in {struct_file_name}, while a coordinate is expected.')
                        mag_ions_pos[k-starting_row-n_cumulated] = np.array([float(el) for el in vector])
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
        '''
        Reads the content of J couplings file specified in the configuration file, following 
        the standard format conventions of MagInt output file. Also updates the Js_info attribute
        accordingly.

        Note:
            Only Dipole-Dipole interactions are actually taken into account, so we often assume to
            deal with 3x3 square matrices.
        '''
        shell_digits = self.get_shell_digits()
        J_couplings_file = self.get_J_couplings_file()
        
        # Raise Exceptions when needed
        if J_couplings_file=='NOT SPECIFIED' :
            raise ValueError(f'The J_couplings_file value is {J_couplings_file} in {self.get_config_file()}.\nSo no interaction matrices are actually read.')

        Js_info = {}
        magint_matrices = []
        T_vectors = []
        distances = []
        coor_nums = []
        with open(J_couplings_file,'r') as Vf :
            content = Vf.readlines()
            if len(content)==0 :
                raise IOError(f'{J_couplings_file} is empty.')
            
            for k in range(len(content)) :
                    
                # Coor_num gives the number of interaction matrices to be read per each shell
                # Expected string: ' Coor_num = 3' 
                if content[k].find('Coor_num')!=-1 :
                    vector = clean_line(content[k])
                    if len(vector)!=3 :
                        raise IOError(f'The {k+1}-th line in {J_couplings_file} is longer/shorter than expected.')
                    if vector[0]!='Coor_num' or not vector[2].isdigit() :
                        raise ValueError(f'The {k+1}-th line in {J_couplings_file} does not provide the coordination number as expected.')
                    coor_nums.append(int(vector[2]))

                # Only Dipole-Dipole interaction matrices are to be read
                if content[k].find('Dipole-Dipole interactions')!=-1 and k+2<len(content) :

                    # The fastest way to make sure the D-D section is empty is to check whether the second row
                    # after the string 'Dipole-Dipole interactions':
                    # 1) is after the end of file
                    # 2) includes the word 'interactions' or 'INTERACTION'
                    if content[k+2].find('interactions')==-1 and content[k+2].find('INTERACTION')==-1 :

                        # Loop over the next rows
                        # The n° of interaction matrices in the current NN shell should coincide with the last integer in coor_nums
                        # Otherwise the matrix is filled with zeros
                        for i in range(coor_nums[-1]) :
                            current_row = k+3+8*i
                            if current_row>len(content)-1 :
                                raise IOError(f'End of {J_couplings_file} is reached before all T vectors and J matrices could be read.')
                            vector = clean_line(content[current_row])

                            # Expected format 'T = [1.0 0.0 2.5]\n' or 'T = [1.0 0.0 2.5] \n' or 'T = [1.0 0.0 2.5 ] \n'
                            if len(vector)==0 or vector[0]!='T' :
                                raise ValueError(f'The {current_row+1}-th line in {J_couplings_file} does not include a T vector.')
                            if len(vector)>1 and vector[0]=='T' :

                                # Remove useless info
                                vector.remove('T')
                                vector.remove('=')
                                if vector.count('[')==1 : vector.remove('[')
                                if vector.count('\n')==1 : vector.remove('\n')
                                if vector.count(']')==1 : vector.remove(']')
                                if vector.count(']\n')==1 : vector.remove(']\n')
                                vector[0] = vector[0].replace('[','')
                                vector[-1] = vector[-1].replace(']','')
                                vector[-1] = vector[-1].replace(']\n','')
                                
                                # Save the current T vector
                                for el in vector :
                                    el_tmp = el.replace('.','').replace('-','')
                                    if not el_tmp.isdigit() :
                                        raise TypeError(f'{el} is read in the {current_row+1}-th line in {J_couplings_file}, while a coordinate is expected.')
                                
                                # Rescale T vectors into Angstrom units of length
                                vector = np.dot(np.array([float(el) for el in vector]),0.529177249)
                                T_vectors.append(vector)

                                # Save the length of the current T vector
                                squares = [el**2 for el in vector]
                                norm = np.sqrt(np.array(squares).sum())
                                distances.append(norm)

                                # Read the associated 3x3 interaction matrix
                                n_rows = 3
                                matrix = np.zeros((n_rows,n_rows))
                                for n in range(n_rows) :

                                    # Expected format '  y 0.33 0.22 0.11'
                                    row = clean_line(content[current_row+3+n])
                                    row.remove(row[0])
                                    if len(row)!=3 :
                                        raise ValueError(f'The content of {current_row+3+n+1}-th line in {J_couplings_file} differs from what is expected.')
                                    for j in range(len(row)) :
                                        if row[j].replace('.','').replace('-','').isdigit() :
                                            matrix[n][j] = float(row[j])
                                        else :
                                            raise TypeError(f'{row[j]} is read in the {current_row+3+n+1}-th line in {J_couplings_file}, while a J coupling constant is expected.')
                                            
                                # Save the interaction matrix after readjusting the order of rows and columns
                                matrix = adapt_magintmatrix(matrix)
                                magint_matrices.append(matrix)
                
            # Sort T vectors and J matrices in increasing order of distance
            sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
            distances = [round(distances[i],shell_digits) for i in sorted_indices]
            T_vectors = [T_vectors[i] for i in sorted_indices]
            magint_matrices = [magint_matrices[i] for i in sorted_indices]

            # Collect the T vectors and the associated J matrices into auxiliary dictionaries 
            # to finally update self.Js_info attribute 
            NN_index = 0
            shell_count = 0
            aux_magint_dict = {}
            aux_J_matrices = {}
            aux_T_vectors = {}

            # Note that distances, T_vectors & magint_matrices have the same length by construction
            # Each NN shell may contain one or more T vectors (& thus J matrices) 
            # as long as they are associated roughly with the same distance
            last_distance = distances[-1]
            distances.insert(len(distances),last_distance)
            for j in range(1,len(distances)) :

                # Reset NN_index to the initial value when the NN shell distance changes
                if distances[j-1]!=distances[j] : NN_index=0
                
                aux_T_vectors[f'{NN_index}'] = np.array([round(el,shell_digits) for el in T_vectors[j-1]])
                aux_J_matrices[f'{NN_index}'] = magint_matrices[j-1]

                # When NN_index is reset to 0, the last bond in the current shell is being stored
                # since we always work with the (j-1)-th element in the lists
                if NN_index==0 :
                    aux_magint_dict['NN_shell_radius'] = distances[j-1]
                    aux_magint_dict['T_vectors'] = aux_T_vectors
                    aux_magint_dict['J_matrices'] = aux_J_matrices
                    shell_count += 1
                    Js_info[f'{shell_count}° NN shell'] = aux_magint_dict
                NN_index += 1
            self.Js_info = Js_info

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
        return self.config_info['STRUCTURE']['struct_file_name']
    
    def get_struct_file_type(self) -> str :
        '''
        Returns the chosen structure file format if specified in the configuration file or simply POSCAR otherwise.
        '''
        return self.config_info['STRUCTURE']['struct_file_type']
    
    def get_mag_ion(self) -> str :
        '''
        Returns the symbol for the magnetic element of interest.
        '''
        return self.config_info['STRUCTURE']['mag_ion']
    
    def get_spin(self) -> float :
        '''
        Returns the spin quantum number read from the configuration file.
        '''
        return self.config_info['STRUCTURE']['spin']
    
    def get_n_dim(self) -> int :
        '''
        Returns the number of spatial dimensions of the system in question if specified
        in the configuration file or directly 3 otherwise.
        '''
        return self.config_info['STRUCTURE']['n_dim']
    
    def get_J_couplings_file(self) -> str :
        '''
        Returns the name of the MagInt output file.
        '''
        return self.config_info['HAMILTONIAN']['J_couplings_file']
    
    def get_max_NN_shell(self) -> int :
        '''
        Returns the chosen number of NN shells if specified in the configuration file
        or directly 1 otherwise.
        '''
        return self.config_info['HAMILTONIAN']['max_NN_shell']
    
    def get_shell_digits(self) -> int :
        '''
        Returns the number of digits to identify the NN shells by distance if specified
        in the configuration file or directly 3 otherwise.
        '''
        return self.config_info['HAMILTONIAN']['shell_digits']
    
    def get_B_field(self) -> 'np.ndarray' :
        '''
        Returns the 3D vector for the applied magnetic field B
        within the coordinate system of the spin quantization axis.
        '''
        return np.array(self.config_info['HAMILTONIAN']['B_field'])
    
    def get_tol_imag(self) -> float :
        '''
        Returns the tolerance on the imaginary parts of Hamiltonian matrix elements in general.
        Such an assessment process is applied to both the initial Spin Hamiltonian matrix and 
        the approximated tridiagonal one.
        '''
        return self.config_info['HAMILTONIAN']['tol_imag']
    
    def get_n_excited(self) -> int :
        '''
        Returns the number of excited non-degenarate eigenstates  to be estimated by the Lanczos algorithm. 
        '''
        return self.config_info['OUTPUT']['n_excited']
    
    def get_lanczos_digits(self) -> int :
        '''
        Returns the number of digits of the Lanczos eigenvalues and eigenvectors that the user intends to store and use
        while computing the expectation value of the observables.
        '''
        return self.config_info['OUTPUT']['lanczos_digits']
    
    def get_magn_output_mode(self) -> str :
        '''
        Returns the chosen definition of the magnetization modulus operator. It also 
        affects how the spin-spin correlation values will be determined.
        '''
        return self.config_info['OUTPUT']['magn_output_mode']
    
    def get_show_plot(self) -> bool :
        '''
        Returns True if the plot (spin-spin correlation vs distance) is requested by the user, False otherwise.
        '''
        return self.config_info['OUTPUT']['show_plot']
    
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
    
    def get_J_couplings(self) -> list :
        '''
        Returns a list of the interaction matrices stored in the Js_info after the reading procedure.
        Each element in the list contains all the interaction matrices of the corresponding NN shell.
        '''
        Js = self.Js_info
        matrices = []
        for section in Js.keys() :
                matrices.append([])
                for mat in Js[section]['J_matrices'].values() :
                    matrices[-1].append(mat)
        return matrices

    def get_T_vectors(self) -> list :
        '''
        Returns a list of the T vectors connecting NN atoms. They are organized in groups for each NN shell
        and  their order cannot be changed since they are in a 1-1 mapping with the interaction matrices 
        given by the get_J_couplings() method. 
        '''
        Js = self.Js_info
        vectors = []
        for section in Js.keys() :
                vectors.append([])
                for vec in Js[section]['T_vectors'].values() :
                    vectors[-1].append(vec)
        return vectors

    def print_summary(self) -> None :
        '''
        Prints a summary of all the quantities that have been read during the construction 
        of the current InputReader instance in order to help the user to have a better 
        understanding of the settings and/or the values in use.
        '''
        config = self.config_info
        struct = self.struct_info
        Js = self.Js_info

        print(f'\nInput parameters from {self.config_file}:')
        for section in config.keys() :
            print(f'\n[{section}]\n')
            for key, value in config[section].items() :
                print(f'{key} = {value}')
        
        print(f'\nStructural information from {self.get_struct_file_name()}:')
        for key, value in struct.items() :
            print(f'\n[{key}]')
            for i in range(len(value)) :
                print(f'{i}: {value[i]}')
        
        J_couplings_file = self.get_J_couplings_file()
        print(f'\nMagnetic interactions from {J_couplings_file}')
        for section in Js.keys() :
            print(f'\n[{section}]')
            for key, value in Js[section].items() :
                if type(value)==float :
                    print(f'{key}: {value}')
                elif type(value)==dict :
                    for k, v in value.items() :
                        aux_key = 'T_vector'*(key=='T_vectors')+'J_matrix'*(key=='J_matrices')
                        print(f'{aux_key} {k}: \n{v}')