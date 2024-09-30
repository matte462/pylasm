import os
import shutil

# Path to main.py
main_path = r"full/path/to/pylasm/main.py"

# Set of different J_couplings_file (to be prepared independently)
J_files_num = 5 
J_couplings_files = [f'J_couplings_{i}.dat' for i in range(J_files_num)]
# The names can be modified, this is just a personal choice

def prepare_config_files(J_couplings_files: list) -> None :
    '''
    Creates as many subdirectories as requested by the user (i.e. =len(J_couplings_files)) and
    prepares the input files for the associated calculations. The configuration files are indeed 
    modified in order to set up a distinct single-run pylasm execution per subdirectory.
    
    Args:
        J_couplings_files (list): Set of the user-defined J_couplings filenames.
    '''    
    # Loop over the specified J_couplings filenames
    for i in range(len(J_couplings_files)) :
        
        # Check whether the current J_couplings_file exists or not
        if not os.path.exists(J_couplings_files[i]) :
            raise FileNotFoundError(f"The J_couplings file at {J_couplings_files[i]} does not exist.")
        
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
            if 'J_couplings_file' in lines[n] :
                
                # Update the line
                lines[n] = f"J_couplings_file = \'{J_couplings_files[i]}\'\n"
                break

        # Write the modified lines back to the file
        with open(new_config_file,'w') as file:
            file.writelines(lines)

# Prepare the subdirectories
prepare_config_files(J_couplings_files)

# Execute main.py within each subdirectory if it actually exists
if os.path.exists(main_path):
    for i in range(len(J_couplings_files)) :
        os.chdir(f'./run_{i}')
        os.system(f'python \"{main_path}\"')
        os.chdir(f'..')
else:
    raise FileNotFoundError(f"The script at {main_path} does not exist.")