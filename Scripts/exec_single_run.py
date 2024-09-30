import os

# Path to main.py
main_path = r"full/path/to/PyLasm/main.py"

# Execute main.py if it actually exists
if os.path.exists(main_path):
    os.system(f'python \"{main_path}\"')
else:
    raise FileNotFoundError(f"The script at {main_path} does not exist.")