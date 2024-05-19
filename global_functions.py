import numpy as np
from spin_system import SpinSystem

# Some useful global functions
def print_logo() -> None :
    '''
    Prints the software's logo, to use at the beginning.
    '''
    print("###############################################################")
    print(r" \    __________  ___",end=" ")
    print("\033[91m___       __        _________   ___\033[00m",end=" ")
    print("   / ")
    print(r"  \   |*|  || \*\/*/ ",end=" ")
    print("\033[91m|*|      /**"+r"\ "+"     /**___)|**"+r"\ "+"/**|\033[00m",end=" ")
    print("  /  ")
    print(r"  /   |*|__||  \**/  ",end=" ")
    print("\033[91m|*|__   /____"+r"\ "+"   |__**|  |*|*V*|*|\033[00m",end=" ")
    print(r"  \  ")
    print(r" /    |_|      |__|  ",end=" ")
    print("\033[91m|____| /_/  "+r"\_"+r"\ "+"(_____/   |_|   |_|\033[00m",end=" ")
    print(r"   \ ")
    print("###############################################################")

def clean_line(raw_line: str) -> list :
    '''
    Removes all the empty spaces and the newline character, useful when reading text files.

    Args:
        raw_line (str): String containing the info to be extracted.
    '''
    line_content = raw_line.split(sep=' ')

    # Last element may include the newline character
    # So replace it with an empty character to be removed later
    line_content[-1] = line_content[-1].replace('\n','')
    empty_counts = line_content.count('')
    for c in range(empty_counts) : line_content.remove('')

    return line_content

def is_spin_acceptable(spin_trial: float) -> bool :
    '''
    Returns True if spin_trial is a positive integer or a positive half-integer, False otherwise.
    '''
    double_trial = 2*spin_trial
    double_trial_int = np.abs(np.floor(double_trial))
    if double_trial-double_trial_int==0.0 and double_trial!=0.0 :
        return True
    else :
        return False


def solve_by_lanczos(system: SpinSystem,hamiltonian: 'np.ndarray',lanczos_mode: str,lanczos_par: int) :
    lanczos_mapping = {
        'one_shot' : one_shot_lanczos_solver,
        'scf' : scf_lanczos_solver
    }
    label = 'One-shot'*(lanczos_mode=='one_shot')+'SCF'*(lanczos_mode=='scf')
    print(f'\n{label} Lanczos Algorithm for Exact Diagonalization')
    results = lanczos_mapping[lanczos_mode](system,hamiltonian,lanczos_par)
    return results

def one_shot_lanczos_solver(system: SpinSystem,hamiltonian: 'np.ndarray',n_iterations: int) -> float :
    pass

def scf_lanczos_solver(system: SpinSystem,hamiltonian: 'np.ndarray',energy_digits: int) -> float :
    pass

def print_conclusions(results) -> None :
    print(f'The predicted ground-state energy is {results}')