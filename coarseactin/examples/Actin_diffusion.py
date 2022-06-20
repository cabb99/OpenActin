#!/home/cab22/miniconda3/bin/python
#SBATCH --account=commons
#SBATCH --output ./Simulations/Diffusion/%A_%a.out
#SBATCH --export=All
#SBATCH --partition=commons
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --array=0-1440
#SBATCH --mem=16G

import sys

sys.path.insert(0, '.')
import coarseactin
import pandas as pd
import numpy as np
import scipy.spatial.transform as strans
import os

if __name__ == '__main__':
    ###################################
    # Setting Conditions for simulation#
    ###################################

    """ The objective of this simulation is to determine the diffusion constant of actin filaments
    """
    parameters = {"repetition": range(5),
                  "aligned": [False, True],  # Constant may be different depending on the alignment/ size of filaments.
                  "friction": [.000_1, .001, .000_01],
                  "system2D": [False],
                  "actinLen": [1,35,70,140,280,350,420,490],  # Filaments with multiple binding sites
                  "box_size": [None],  # Small box to make long simulations
                  "n_actins": [1],  # 3*75=225 ~ 3uM actin monomers
                  "temperature": [300],
                  "frequency": [100_000, 10_000, 1_000, 100, 10, 1],
                  "simulation_platform": ["OpenCL"],
                  }
    test_parameters = {"simulation_platform": "CPU",
                       "actinLen": 350,
                       }
    job_id = None
    if len(sys.argv) > 1:
        try:
            job_id = sys.argv[1]
        except TypeError:
            pass
    # sjob = coarseactin.SlurmJobArray("Simulations_nots/Box/Box", parameters, test_parameters, job_id)
    sjob = coarseactin.SlurmJobArray("Simulations/Diffusion/SingleActin", parameters, test_parameters, job_id)
    os.makedirs('/'.join(sjob.name.split('/')[:-1]), exist_ok=True)
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
    run_steps = sjob['frequency'] * 2000

    ###################
    # Build the model #
    ###################
    # Set the points in the actin network
    import random

    full_model = []
    # Add unbound actins
    for i in range(sjob["n_actins"]):
        full_model += [coarseactin.create_actin(length=sjob["actinLen"],
                                                translation=[0, 0, 0],
                                                rotation=strans.Rotation.random().as_matrix(),
                                                abp=None)]

    full_model = coarseactin.Scene.concatenate(full_model)
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
        import simtk.openmm as openmm
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
    s.setForces(AlignmentConstraint=aligned, PlaneConstraint=system2D)
    top = openmm.app.PDBxFile(f'{Sname}.cif')
    coord = openmm.app.PDBxFile(f'{Sname}.cif')

    # Set up simulation
    temperature = sjob["temperature"] * u.kelvin
    integrator = openmm.LangevinIntegrator(temperature, sjob['friction'] / u.picosecond, 1 * u.picoseconds)
    simulation = openmm.app.Simulation(top.topology, s.system, integrator, platform)
    simulation.context.setPositions(coord.positions)

    frequency = sjob["frequency"]

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

    #Equilibrate
    simulation.step(2_000_000)

    #Run
    simulation.reporters.append(
        openmm.app.StateDataReporter(stdout, frequency, step=True, time=True, potentialEnergy=True, totalEnergy=True,
                                     temperature=True,
                                     separator='\t', ))
    simulation.reporters.append(
        openmm.app.StateDataReporter(f'{Sname}.log', frequency, step=True, time=True, totalEnergy=True,
                                     kineticEnergy=True, potentialEnergy=True, temperature=True))
    simulation.reporters.append(openmm.app.DCDReporter(f'{Sname}.dcd', frequency), )
    simulation.step(run_steps)

    # Run
    # simulation.context.setParameter('kp_alignment',0)
    # simulation.runForClockTime(sjob["run_time"])

    # Save checkpoint
    chk = f'{Sname}.chk'
    simulation.saveCheckpoint(chk)
