import os

# Path to the script you want to run
script_path = r"path/to/PyLasm/main.py"

# Execute the script if it actually exists
if os.path.exists(script_path):
    os.system(f'python \"{script_path}\"')
else:
    raise FileNotFoundError(f"The script at {script_path} does not exist.")