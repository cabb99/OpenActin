#!/home/cab22/miniconda3/bin/python
#SBATCH --account=commons
#SBATCH --output ./Simulations_nots/Box/slurm-%A_%a.out
#SBATCH --export=All
#SBATCH --partition=commons
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --array=0-23
#SBATCH --mem=16G

import sys # Import the sys module for interacting with the Python interpreter
# sys is a module that gives access to variables and functions
import openactin # imports custom module called openactin
import pandas as pd # imports pandas library, excel for python
import numpy as np # imports the numpy library, matlab for python
import scipy.spatial.transform as strans  # Import the transform module from scipy.spatial for spatial transformations
import scipy.spatial.distance as sdist  # Import the distance module from scipy.spatial for distance calculations
import itertools # Import the itertools module for efficient looping and combination generation

if __name__ == '__main__': # makes sure that the following code is executed only if the script is run directly, not importad as a 
    # module
    ###################################
    # Setting Conditions for simulation#
    ###################################
    """ The objective of this experiment is to simulate a big system containing multiple filaments and abps 
    and observe their behavior"""
    parameters = {"epsilon": [100, 75, 50, 25],
                  "aligned": [False],
                  "actinLen": [100],
                  # "layers": [3],
                  # "repetition":range(3),
                  "disorder": [0],
                  "box_size": [10000],
                  "n_actins": [0],
                  "n_abps": [1],
                  "temperature": [300],
                  "system2D": [False],
                  "frequency": [1000],
                  "run_time": [20],
                  #"run_steps":[10000000],
                  "abp": ['FAS', 'CAM', 'CBP', 'AAC', 'AAC2', 'CAM2'],
                  "simulation_platform": ["OpenCL"]}
    test_parameters = {"simulation_platform": "CUDA",
                       "frequency": 1,
                       "run_time": 0.01,
                       "abp":'CBP',
                       "epsilon":0,
                       #"abp": 'CAM',
                       #"CaMKII_Force": 'multigaussian',
                       }
    job_id = 0
    if len(sys.argv) > 1:
        try:
            job_id = int(sys.argv[1])
        except TypeError:
            pass
    sjob = openactin.SlurmJobArray("Simulations/Box/Box", parameters, test_parameters, job_id)
    sjob.print_parameters()
    sjob.print_slurm_variables()
    sjob.write_csv()

    print("name :", sjob.name)

    ##############
    # Parameters #
    ##############
    aligned = sjob["aligned"]
    system2D = sjob["system2D"]
    actinLen = sjob["actinLen"]
    Sname = sjob.name
    simulation_platform = sjob["simulation_platform"]
    if sjob['abp'] in ['CAM','CAM2']:
        camkii_force='multigaussian'
    else:
        camkii_force = 'abp'

    ###################
    # Build the model #
    ###################
    # Set the points in the actin network
    import random

    full_model = []
    #Add actins
    for i in range(sjob["n_actins"]):
        full_model += [openactin.create_actin(length=sjob["actinLen"],
                       translation=np.random.random(3)*sjob["box_size"],
                       rotation=strans.Rotation.random().as_matrix(),
                       abp=None)]
    #Add ABPs
    for i in range(sjob["n_abps"]):
        full_model += [openactin.create_abp(translation=np.random.random(3)*sjob["box_size"],
                       rotation=strans.Rotation.random().as_matrix(),
                       abp=sjob['abp'])]

    print('Concatenate chains')
    full_model = openactin.Scene.concatenate(full_model)

    full_model = openactin.Scene(full_model.sort_values(['chainID', 'resSeq', 'name']))
    full_model.loc[:, 'chain_resid'] = full_model[['chainID', 'resSeq']].astype(str).T.apply(
        lambda x: '_'.join([str(a) for a in x]))
    full_model_actin = openactin.Scene(full_model[full_model['resName'].isin(['ACT', 'ACD'])])
    full_model_abps = openactin.Scene(full_model[~full_model['resName'].isin(['ACT', 'ACD'])])
    #full_model_abps['chainID'] = 'A'
    resids = full_model_abps['chain_resid'].unique()
    resids_rename = full_model_abps['chain_resid'].replace({a: b for a, b in zip(resids, range(len(resids)))})
    full_model_abps['resSeq'] = resids_rename
    full_model = openactin.Scene.concatenate([full_model_actin, full_model_abps])
    full_model.write_cif(f'{Sname}.cif', verbose=True)

    ##############
    # Simulation #
    ##############
    import sys

    sys.path.insert(0, '.')
    import openmm
    import openmm.app
    import simtk.unit as u
    import time
    from sys import stdout

    time.ctime()

    # Create system
    platform = openmm.Platform.getPlatformByName(simulation_platform)
    s = openactin.openactin.from_topology(f'{Sname}.cif', periodic_box=sjob['box_size'])

    #Add extra bonds for CBP
    extra_bonds=[]
    name_list = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12']
    for _, c in s.atom_list[(s.atom_list['residue_name'] == 'CBP')].groupby('chain_index'):
        cm = c[c['atom_name'].isin(name_list)]
        cb = c[c['atom_name'] == 'Cb']
        assert len(cm.index) == len(cb.index)
        assert len(cm.index) == 12
        for i0, i1 in zip(cm.index,cb.index):
            extra_bonds += [['CBP', i0, i1, 0, 10, 0, 31.77, 0, '1Cb-2Cb']]

    extra_bonds=pd.DataFrame(extra_bonds,columns=s.bonds.columns,index=range(s.bonds.index.max() + 1,s.bonds.index.max() + 1+ len(extra_bonds)))
    s.bonds = pd.concat([s.bonds, extra_bonds], ignore_index=True)

    #Check that the bonds correspond with the correct molecule
    s.bonds = s.bonds[(s.bonds['molecule'] == s.atom_list['residue_name'].loc[s.bonds['i']].values) |
                       s.bonds['molecule'].isin(['Actin-ADP', 'ABP', 'CaMKII'])]

    print(s.system.getDefaultPeriodicBoxVectors())
    s.setForces(AlignmentConstraint=aligned, PlaneConstraint=system2D, CaMKII_Force=camkii_force)
    top = openmm.app.PDBxFile(f'{Sname}.cif')
    coord = openmm.app.PDBxFile(f'{Sname}.cif')

    # Set up simulation
    temperature = sjob["temperature"] * u.kelvin
    integrator = openmm.LangevinIntegrator(temperature, .0001 / u.picosecond, 1 * u.picoseconds)
    simulation = openmm.app.Simulation(top.topology, s.system, integrator, platform)
    simulation.context.setPositions(coord.positions)

    # Modify parameters
    simulation.context.setParameter("g_eps", sjob["epsilon"])

    frequency = sjob["frequency"]
    # Add reporters
    simulation.reporters.append(openmm.app.DCDReporter(f'{Sname}.dcd', frequency), )
    simulation.reporters.append(
        openmm.app.StateDataReporter(stdout, frequency, step=True, time=True, potentialEnergy=True,totalEnergy=True, temperature=True,
                                     separator='\t', ))
    simulation.reporters.append(
        openmm.app.StateDataReporter(f'{Sname}.log', frequency, step=True, time=True, totalEnergy=True,
                                     kineticEnergy=True, potentialEnergy=True, temperature=True))

    #Change simulation parameters
    #simulation.context.setParameter("w1", 5)
    simulation.context.setParameter("g_eps", sjob['epsilon'])

    # Print initial energy
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    print(f'Initial energy: {energy} KJ/mol')
    energies = {}
    for force in s.system.getForces():
        group = force.getForceGroup()
        state = simulation.context.getState(getEnergy=True, groups=2 ** group)
        energies[force] = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    print(energies)

    # Run
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature * u.kelvin)
    time0 = time.ctime()
    time_0 = time.time()
    #simulation.step(sjob['run_steps'])

    # Turn off nematic parameter
    # simulation.context.setParameter('kp_alignment',0)
    simulation.runForClockTime(sjob["run_time"])

    # Save checkpoint
    chk = f'{Sname}.chk'
    simulation.saveCheckpoint(chk)
