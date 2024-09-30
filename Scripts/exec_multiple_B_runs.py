import os
import numpy as np
import shutil
import ast

# Path to main.py
main_path = r"full/path/to/pylasm/main.py"

# Define a set of equally-spaced scaling factors for B_field
B_scale_min = -1.0
B_scale_max = 1.0
B_scale_num = 5
B_scalings = np.linspace(B_scale_min,B_scale_max,B_scale_num)

def prepare_config_files(B_scalings: 'np.ndarray') -> None :
    '''
    Creates as many subdirectories as requested by the user (i.e. =B_scalings.shape[0])) and
    prepares the input files for the associated calculations. The configuration files are indeed 
    modified in order to set up a distinct single-run pylasm execution per subdirectory.
    
    Args:
        B_scalings (np.ndarray): Set of the scaling factors for the specified B_field.
    '''
    # Loop over all the given scaling factors
    for i in range(B_scalings.shape[0]) :
        
        # Create the current subdirectory if needed
        src_dir = '.'
        dest_dir = f'./run_{i}'
        if not os.path.exists(dest_dir) :
            os.mkdir(dest_dir)
        
        # Copy all the files from the working directory to the current subdirectory
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir,filename)
            dest_file = os.path.join(dest_dir,filename)
            if os.path.isfile(src_file) :
                shutil.copy(src_file,dest_file)
        
        # Read the new config file and modify a specific line
        new_config_file = f'run_{i}/init_config.ini'
        with open(new_config_file,'r') as file:
            lines = file.readlines()

        # Loop over the lines
        for n in range(len(lines)) :
            if 'B_field' in lines[n] :
                
                # Extract the old B_field from the current line
                lines[n] = lines[n].replace('B_field','')
                lines[n] = lines[n].replace('=','')
                lines[n] = lines[n].replace('\n','')
                n_empty_spaces = lines[n].count(' ')
                for m in range(n_empty_spaces) :
                    lines[n] = lines[n].replace(' ','')
                old_B_field = np.array(ast.literal_eval(lines[n]))
                
                # Generate the new B_field and update the line
                new_B_field = B_scalings[i]*old_B_field
                lines[n] = f'B_field = {new_B_field.tolist()}\n'
                break
            
            # Raise a ValueError exception if no B_field value is explicitly specified
            if n==len(lines)-1 :
                raise ValueError('No B_field value was found within the given configuration file.')

        # Write the modified lines back to the file
        with open(new_config_file,'w') as file:
            file.writelines(lines)

# Prepare the subdirectories
prepare_config_files(B_scalings)

# Execute main.py within each subdirectory if it actually exists
if os.path.exists(main_path):
    for i in range(B_scalings.shape[0]) :
        os.chdir(f'./run_{i}')
        os.system(f'python \"{main_path}\"')
        os.chdir(f'..')
else:
    raise FileNotFoundError(f"The script at {main_path} does not exist.")