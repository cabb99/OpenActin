#!/home/cab22/miniconda3/bin/python
#SBATCH --account=commons
#SBATCH --output ./Simulations/Box/%A_%a.out
#SBATCH --export=All
#SBATCH --partition=commons
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --array=0-288
#SBATCH --mem=16G

import sys
sys.path.insert(0, '.')
import coarseactin
import pandas as pd
import numpy as np
import scipy.spatial.transform as strans

if __name__ == '__main__':
    ###################################
    # Setting Conditions for simulation#
    ###################################

    """ The objective of this simulation is to simulate a range of epsilon and well sizes to determine the binding rate,
    the unbinding rate, the binding constant, and the possible conformations
    """
    parameters = {"epsilon": [100, 150, 200],  # Epsilons from 20 to 500. 20 is too low and 500 is too high
                  "w1": [1.0, 2.0, 3.0],  # Wells from 0.1 to 10. 0.1 seems to narrow and 10 too broad
                  "w2_ratio": [0.1],
                  # It seems to work well with 0.1. #TODO make another experiment to check correct w2
                  "aligned": [False],  # Constant may be different depending on the alignment/ size of filaments.
                  # Aligned=True supposes that the filaments are long and/or had time to align.
                  #"actinLen": [75],  # Filaments with multiple binding sites
                  "monomers":[225],
                  "disorder": [0],  # Not important
                  "box_size": [5000],  # Small box to make long simulations
                  "n_actins": [3],  # 3*75=225 ~ 3uM actin monomers
                  "n_abps": [128],  # More abps to maximize kinetic statistics
                  # 90 monomers ~1.2uM, 15 monomers ~.2uM
                  "temperature": [300],
                  "system2D": [False],
                  "frequency": [100_000],
                  "run_time": [20],
                  "run_steps": [200_000_000],
                  "abp": ['FAS'],  # , 'AAC', 'CAM', 'CBP'],  # 'CBP', 'AAC2', 'CAM2'],
                  "simulation_platform": ["OpenCL"],
                  "repetition": range(8)}
    test_parameters = {"simulation_platform": "CPU",
                       "epsilon": 60,
                       "w1": 3.0,
                       "frequency": 100000,
                       "run_time": 12,
                       "abp": 'FAS',
                       "aligned": True,
                       "n_abps": 64,
                       }
    job_id = None
    if len(sys.argv) > 1:
        try:
            job_id = sys.argv[1]
        except TypeError:
            pass
    # sjob = coarseactin.SlurmJobArray("Simulations_nots/Box/Box", parameters, test_parameters, job_id)
    sjob = coarseactin.SlurmJobArray("Simulations/Box/Epsilon_BoxFAS", parameters, test_parameters, job_id)
    sjob.print_parameters()
    sjob.print_slurm_variables()
    sjob.write_csv()

    print("name :", sjob.name)

    ##############
    # Parameters #
    ##############
    aligned = sjob["aligned"]
    system2D = sjob["system2D"]
    actinLen = int(sjob["monomers"]/sjob["n_actins"])
    Sname = sjob.name
    simulation_platform = sjob["simulation_platform"]
    if sjob['abp'] in ['CAM', 'CAM2']:
        camkii_force = 'multigaussian'
    else:
        camkii_force = 'abp'

    ###################
    # Build the model #
    ###################
    # Set the points in the actin network
    import random

    full_model = []
    # Add unbound actins
    for i in range(sjob["n_actins"] // 2):
        full_model += [coarseactin.create_actin(length=actinLen,
                                                translation=np.random.random(3) * sjob["box_size"],
                                                rotation=strans.Rotation.random().as_matrix(),
                                                abp=None)]
    # Add unbound ABPs
    for i in range(sjob["n_abps"] // 2):
        full_model += [coarseactin.create_abp(translation=np.random.random(3) * sjob["box_size"],
                                              rotation=strans.Rotation.random().as_matrix(),
                                              abp=sjob['abp'])]
    unbound_actins = coarseactin.Scene.concatenate(full_model)

    full_model = []
    # Add bound abps
    for i in range(sjob["n_actins"] - sjob["n_actins"] // 2):
        full_model += [coarseactin.create_actin(length=actinLen,
                                                translation=np.random.random(3) * sjob["box_size"],
                                                rotation=strans.Rotation.random().as_matrix(),
                                                abp=sjob['abp'])]
    bound_actins = coarseactin.Scene.concatenate(full_model)
    selected_bound_abps = np.random.choice(bound_actins[bound_actins['resName'] == sjob['abp']]['chain_index'].unique(),
                                           sjob["n_abps"] // 2, replace=False)
    bound_actins = bound_actins[
        (bound_actins['resName'] != sjob['abp']) | bound_actins['chain_index'].isin(selected_bound_abps)]

    print('Concatenate chains')
    full_model = coarseactin.Scene.concatenate([unbound_actins, bound_actins])

    full_model = coarseactin.Scene(full_model.sort_values(['chainID', 'resSeq', 'name']))
    full_model.loc[:, 'chain_resid'] = full_model[['chainID', 'resSeq']].astype(str).T.apply(
        lambda x: '_'.join([str(a) for a in x]))
    full_model_actin = coarseactin.Scene(full_model[full_model['resName'].isin(['ACT', 'ACD'])])
    full_model_abps = coarseactin.Scene(full_model[~full_model['resName'].isin(['ACT', 'ACD'])])
    # full_model_abps['chainID'] = 'A'
    resids = full_model_abps['chain_resid'].unique()
    resids_rename = full_model_abps['chain_resid'].replace({a: b for a, b in zip(resids, range(len(resids)))})
    full_model_abps['resSeq'] = resids_rename
    full_model = coarseactin.Scene.concatenate([full_model_actin, full_model_abps])
    full_model.write_cif(f'{Sname}.cif', verbose=True)

    ##############
    # Simulation #
    ##############
    try:
        import openmm
        import openmm.app
        from simtk import unit as u
    except ModuleNotFoundError:
        import openmm
        import simtk.openmm.app
        import simtk.unit as u
    import time
    from sys import stdout

    time.ctime()

    # Create system
    platform = openmm.Platform.getPlatformByName(simulation_platform)
    s = coarseactin.CoarseActin.from_topology(f'{Sname}.cif', periodic_box=sjob['box_size'])

    # Add extra bonds for CBP
    extra_bonds = []
    name_list = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12']
    for _, c in s.atom_list[(s.atom_list['residue_name'] == 'CBP')].groupby('chain_index'):
        cm = c[c['atom_name'].isin(name_list)]
        cb = c[c['atom_name'] == 'Cb']
        assert len(cm.index) == len(cb.index)
        assert len(cm.index) == 12
        for i0, i1 in zip(cm.index, cb.index):
            extra_bonds += [['CBP', i0, i1, 0, 10, 0, 31.77, 0, '1Cb-2Cb']]

    extra_bonds = pd.DataFrame(extra_bonds, columns=s.bonds.columns,
                               index=range(s.bonds.index.max() + 1, s.bonds.index.max() + 1 + len(extra_bonds)))
    s.bonds = s.bonds.append(extra_bonds)

    # Check that the bonds correspond with the correct molecule
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
        openmm.app.StateDataReporter(stdout, frequency, step=True, time=True, potentialEnergy=True, totalEnergy=True,
                                     temperature=True,
                                     separator='\t', ))
    simulation.reporters.append(
        openmm.app.StateDataReporter(f'{Sname}.log', frequency, step=True, time=True, totalEnergy=True,
                                     kineticEnergy=True, potentialEnergy=True, temperature=True))

    # Change simulation parameters
    simulation.context.setParameter("w1", sjob['w1'])
    simulation.context.setParameter("w2", sjob['w2_ratio'] * sjob['w1'])
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

    # Minimize
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature * u.kelvin)
    time0 = time.ctime()
    time_0 = time.time()
    simulation.step(sjob['run_steps'])

    # Run
    # simulation.context.setParameter('kp_alignment',0)
    #simulation.runForClockTime(sjob["run_time"])

    # Save checkpoint
    chk = f'{Sname}.chk'
    simulation.saveCheckpoint(chk)
