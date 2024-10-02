**<h1 style="text-align: center; font-size: 3em;">FM Ising Chain</h1>**

**<h2 style="font-size: 2em;">Table of Contents</h2>**
- [Single-run calculation](#single-run-calculation)
- [Dependence on the external magnetic field](#dependence-on-the-external-magnetic-field)

## Single-run calculation

First of all, install PyLasm as specified in README.md and prepare the following input files into a working directory.

- **init_config.ini** = Minimal configuration file (default settings are enough, apart from these 4 variables);

~~~
[STRUCTURE]

struct_file_name = 'POSCAR_Ce_6x1x1.vasp'
mag_ion = 'Ce'

[HAMILTONIAN]

J_couplings_file = 'J_couplings.dat'

[OUTPUT]

n_excited = 1
~~~

- **J_couplings.dat** = File storing the 1°NN intersite exchange tensors of the FM Ising model;

~~~
 -----INTERACTION: Ce0_0 - Ce0_0

Shell=0

Coor_num = 2

Dipole-Dipole interactions

 R = [1.889725988 0. 0.     ]
 T = [1.889725988 0. 0.      ]

              y         z         x     
    y       0.000     0.000     0.000   
    z       0.000     -1.000     0.000  
    x       0.000     0.000     0.000   

 R = [-1.889725988 0. 0.     ]
 T = [-1.889725988 0. 0.      ]

              y         z         x     
    y       0.000     0.000     0.000   
    z       0.000     -1.000     0.000  
    x       0.000     0.000     0.000
~~~

- **POSCAR_Ce_6x1x1.vasp** = Structural properties of the chain in the POSCAR format (ions are equally-spaced by 1Å).

~~~
Keep 1 Angstroms of distance between 1° NN sites
1.0
        6.0000000000         0.0000000000         0.0000000000
        0.0000000000        20.0000000000         0.0000000000
        0.0000000000         0.0000000000        20.0000000000
   Ce
    6
Cartesian
     0.000000000         0.000000000         0.000000000
     1.000000000         0.000000000         0.000000000
     2.000000000         0.000000000         0.000000000
     3.000000000         0.000000000         0.000000000
     4.000000000         0.000000000         0.000000000
     5.000000000         0.000000000         0.000000000
~~~

Now, copy and paste **exec_single_run.ipynb** from Scripts within the PyLasm installation folder to the chosen working directory. As indicated in exec_single_run.ipynb, substitute the main_path value properly.

Then, choose the execution method that suits you. Please find here a list of valid options: 
- run the code blocks manually within a source code editor (such as Visual Studio Code) if compatible with Jupyter Notebooks;
- run the code blocks through the Jupyter Notebook Dashboard by the command

```jupyter notebook exec_single_run.ipynb```

- run all the code blocks at once by the command

```jupyter nbconvert --to notebook --execute exec_single_run.ipynb```

In conclusion, find the numerical results within the **SPIN_OUT.json** output file: the converged Lanczos energies for the ground state and/or first excited one, the spin-spin correlation values for all the available spin pairs and the magnetization modulus.

~~~
{
    "Lanczos Energies": [
        -1.5,
        -1.5,
        -0.5,
        (...)
        -0.5
    ],
    "Spin-Spin Corr. Matrices": {
        "X components": {
            "Spins 0-0": 0.25000000000000006,
            "Spins 0-1": 6.199857644706591e-18,
            (...)
            "Spins 5-4": -1.3352246609037516e-17,
            "Spins 5-5": 0.25000000000000006
        },
        "Y components": {
            "Spins 0-0": 0.25000000000000006,
            "Spins 0-1": -6.1998576447065895e-18,
            (...)
            "Spins 5-4": 1.3352246609037516e-17,
            "Spins 5-5": 0.25000000000000006
        },
        "Z components": {
            "Spins 0-0": 0.25000000000000006,
            "Spins 0-1": 0.25000000000000006,
            (...)
            "Spins 5-4": 0.25000000000000006,
            "Spins 5-5": 0.25000000000000006
        },
        "Total": {
            "Spins 0-0": 0.75,
            "Spins 0-1": 0.25000000000000006,
            (...)
            "Spins 5-4": 0.25000000000000006,
            "Spins 5-5": 0.75
        }
    },
    "Magnetization Modulus": 0.5
}
~~~

Also, compare the so obtained **SPIN_CORRS.png** with the following figure

![alt text](https://github.com/matte462/pylasm/Examples/Figures/FM_Ising_Chain_Single-run_SPIN_CORRS.png)

## Dependence on the external magnetic field

Just take exactly the same magnetic interactions and structural properties (i.e. J_couplings.dat and POSCAR_Ce_6x1x1.vasp), but modify the configuration file (i.e. **init_config.ini**) by adding a new line for the external magnetic field as follows

~~~
[STRUCTURE]

struct_file_name = 'POSCAR_Ce_6x1x1.vasp'
mag_ion = 'Ce'

[HAMILTONIAN]

J_couplings_file = 'J_couplings.dat'
B_field = [1.0,0.0,0.0]
~~~

Next, copy and paste **exec_multiple_B_runs.ipynb** from Scripts within the PyLasm installation folder to the chosen working directory. As indicated in exec_multiple_B_runs.ipynb, assign the values that suits you to the variables within the second code block.
Here are the values to reproduce the outcomes given in this example.

~~~
B_scalings = np.linspace(-1.0,1.0,11)
spin_pairs = [(0,1),(0,2),(0,3)]
SSC_type = 'Z components'
~~~

Once all the Pylasm calculations within the sub-directories run_* are finished, find the output plots under the name of **M_vs_Bfield.png** and **SSC_vs_Bfield.png**, which respectively show how the magnetization and the chosen spin-spin correlation function is affected by the variation of the external magnetic field.

![alt text](https://github.com/matte462/pylasm/Examples/Figures/FM_Ising_Chain_Multiple-B-runs_M_vs_Bfield.png)
![alt text](https://github.com/matte462/pylasm/Examples/Figures/FM_Ising_Chain_Multiple-B-runs_SSC_vs_Bfield.png)

Finally, if the just performed PyLasm single-runs are no longer useful to the user, reduce the memory space occupied by the subdirectories by removing all the files they contain apart from SPIN_OUT.json and SPIN_REPORT.txt. This can be easily achieved by running the python script **clean_directory.py** (find it in the Scripts of the PyLasm installation folder) within the working directory.