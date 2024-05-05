import numpy as np
from spin_system import SpinSystem

# Some useful global functions

def print_logo():
    print('##############################################################')
    print(' \    __________  ______       __        _________   ___    / ')
    print('  \   |*|  || \*\/*/ |*|      /**\      /**___)|**\ /**|   /  ')
    print('  /   |*|__||  \**/  |*|__   /____\    |__**|  |*|*V*|*|   \  ')
    print(' /    |_|      |__|  |____| /_/  \_\ (_____/   |_|   |_|    \ ')
    print('##############################################################')

def direct2cartesian(atoms_positions,lattice_vectors) :
    new_atoms_positions = []
    a = np.array(lattice_vectors[0])
    b = np.array(lattice_vectors[1]) 
    c = np.array(lattice_vectors[2])
    for i in range(len(atoms_positions)) :
        ith_proj_a = np.dot(a,atoms_positions[i][0]) 
        ith_proj_b = np.dot(b,atoms_positions[i][1])
        ith_proj_c = np.dot(c,atoms_positions[i][2])
        ith_new_position = ith_proj_a+ith_proj_b+ith_proj_c
        new_atoms_positions.append(ith_new_position)
    return new_atoms_positions

def clean_line(raw_line) :
    line = raw_line.split(sep=' ')
    line[-1] = line[-1].replace('\n','')
    empty_counts = line.count('')
    for c in range(empty_counts) :
        line.remove('')
    return line

def solve_by_lanczos(system: SpinSystem,hamiltonian: 'np.ndarray',lanczos_mode: str,lanczos_par: int) :
    lanczos_mapping = {
        'one_shot' : one_shot_lanczos_solver,
        'scf' : scf_lanczos_solver
    }
    label = 'One-shot'*(lanczos_mode=='one_shot')+'SCF'*(lanczos_mode=='scf')
    print(f'\n{label} Lanczos Algorithm for Exact Diagonalization')
    results = lanczos_mapping[lanczos_mode](system,hamiltonian,lanczos_par)
    print('something',results)
    return results

def one_shot_lanczos_solver(system: SpinSystem,hamiltonian: 'np.ndarray',n_iterations: int) -> float :
    pass

def scf_lanczos_solver(system: SpinSystem,hamiltonian: 'np.ndarray',energy_digits: int) -> float :
    pass

def print_conclusions(results) -> None :
    print(f'The predicted ground-state energy is {results}')