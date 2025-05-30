#!/home/cab22/miniconda3/bin/python

#SBATCH --account=commons
#SBATCH --output ./Simulations_nots/SingleABP/slurm-%A_%a.out
#SBATCH --export=All
#SBATCH --partition=commons
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --array=0-15
#SBATCH --mem=16G

import sys
import pandas as pd
import openactin
import numpy as np

from scipy.spatial import cKDTree
import itertools


def create_actin(length=100,
                 twist=2.89942054, shift=-28.21600347,
                 rotation=np.array([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]]),
                 translation=np.array([5000, 5000, 5000]),
                 abp=None):
    q = np.array([[np.cos(twist), -np.sin(twist), 0, 0],
                  [np.sin(twist), np.cos(twist), 0, 0],
                  [0, 0, 1, shift],
                  [0, 0, 0, 1]])
    rot = q[:3, :3].T
    trans = q[:3, 3]

    bound_actin_template = pd.read_csv("openactin/data/CaMKII_bound_with_actin.csv", index_col=0)

    if abp is None:
        bound_actin_template = bound_actin_template[bound_actin_template['resName'].isin(['ACT'])]
    else:
        bound_actin_template = bound_actin_template[bound_actin_template['resName'].isin(['ACT', abp])]

    # Create the helix
    point = bound_actin_template[['x', 'y', 'z']]
    points = []
    for i in range(length):
        points += [point]
        point = np.dot(point, rot) + trans
    points = np.concatenate(points)

    # Create the model
    model = pd.DataFrame(points, columns=['x', 'y', 'z'])
    model["resSeq"] = [(j + i if name == 'ACT' else j) for i in range(length) for j, name in
                       zip(bound_actin_template["resSeq"], bound_actin_template["resName"])]
    model['chainID'] = [(0 if j == 'ACT' else i + 1) for i in range(length) for j in bound_actin_template["resName"]]
    model["name"] = [j for i in range(length) for j in bound_actin_template["name"]]
    model["type"] = [j for i in range(length) for j in bound_actin_template["type"]]
    model["resName"] = [j for i in range(length) for j in bound_actin_template["resName"]]
    model["element"] = [j for i in range(length) for j in bound_actin_template["element"]]

    # Remove two binding points
    model = model[~(((model['resSeq'] >= length) | (model['resSeq'] <= 1)) & model['name'].isin(
        ['A5', 'A6', 'A7', 'Aa', 'Ab', 'Ac'])) &
                  ~(((model['chainID'] >= length) | (model['chainID'] == 1)) & ~model['resName'].isin(['ACT']))]

    resmax = model[model['resName'].isin(['ACT'])]['resSeq'].max()
    resmin = model[model['resName'].isin(['ACT'])]['resSeq'].min()
    model.loc[model[(model['resSeq'] == resmax) & model['resName'].isin(['ACT'])].index, 'resName'] = 'ACD'
    model.loc[model[(model['resSeq'] == resmin) & model['resName'].isin(['ACT'])].index, 'resName'] = 'ACD'

    # Center the model
    model[['x', 'y', 'z']] -= model[['x', 'y', 'z']].mean()

    # Move the model
    model[['x', 'y', 'z']] = np.dot(model[['x', 'y', 'z']], rotation) + translation

    return model


if __name__ == '__main__':
    ###################################
    # Setting Conditions for simulation#
    ###################################

    parameters = {"epsilon": [100],
                  "aligned": [False],
                  "actinLen": [500],
                  "bundleWidth": [1000],
                  # "repetition":range(3),
                  "disorder": [0, 0.5],
                  "temperature": [300],
                  "system2D": [False],
                  "frequency": [1000],
                  # "run_time": [20],
                  # "runSteps":[10000000],
                  "abp": ['FAS', 'CAM', 'CBP', 'AAC', 'AAC2', 'CAM2'],
                  "simulation_platform": ["OpenCL"]}
    test_parameters = {"simulation_platform": "CUDA",
                       "run_time": 8,
                       "abp":'CAM',
                       "disorder": 0.2,
                       }
    job_id = 0
    if len(sys.argv) > 1:
        try:
            job_id = int(sys.argv[1])
        except TypeError:
            pass
    sjob = openactin.SlurmJobArray("Simulations/Rebundling/Rebundling", parameters, test_parameters, job_id)
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
    abp = sjob["abp"]
    Sname = sjob.name
    simulation_platform = sjob["simulation_platform"]
    if sjob['abp'] in ['CAM', 'CAM2']:
        camkii_force = 'multigaussian'
    else:
        camkii_force = 'abp'


    ###################
    # Build the model #
    ###################
    # Calculate distance between actin filaments:
    bound_actin_template = pd.read_csv("openactin/data/CaMKII_bound_with_actin.csv", index_col=0)
    c = bound_actin_template[bound_actin_template['resName'] == abp][['x', 'y', 'z']]
    c.mean(axis=0)
    v = c.mean(axis=0)
    d = 2 * (v['x'] ** 2 + v['y'] ** 2) ** .5
    layers = int(sjob["bundleWidth"]*np.sqrt(3)/2/d + 1)
    print('n_layers', layers)
    print(f'{abp} size', d)
    colliding_distance = d/4

    # Set the points in the actin network
    import random

    full_model = []

    if layers == 1:
        hg = openactin.HexGrid(2)
        coords = hg.coords()[:2]
    else:
        hg = openactin.HexGrid(layers)
        coords = hg.coords()

    # Make a bundle of 100nm
    coords = hg.coords()
    if layers > 2:
        coords = coords[(d * (coords ** 2).sum(axis=1) ** .5) < sjob["bundleWidth"] / 2]

    print('Number of actin filaments:', len(coords))

    for c in coords:
        height = (random.random() - 0.5) * sjob["actinLen"] * 28.21600347 * sjob["disorder"]
        t = np.random.random() * np.pi * 2
        rotation = np.array([[np.cos(t), -np.sin(t), 0.],
                             [np.sin(t), np.cos(t), 0.],
                             [0., 0., 1.]])
        print(c[0], c[1], height)
        full_model += [create_actin(length=sjob["actinLen"],
                                    translation=np.array([5000 + d * c[0], 5000 + d * c[1], 5000 + height]),
                                    rotation=rotation, abp=abp)]

    print('Concatenate chains')
    full_model = openactin.Scene.concatenate(full_model)

    # Remove the CaMKII that are not overlapping
    print('Removing Single ABPs')
    mean_abp = full_model[full_model['resName'] == abp].groupby('chain_index')[['x', 'y', 'z']].mean()

    pairs = cKDTree(mean_abp).query_pairs(colliding_distance)
    close_abps = list(set([mean_abp.index[a] for a, b in pairs]+[mean_abp.index[b] for a, b in pairs]))
    close_abps.sort()
    mean_abp_paired = mean_abp.loc[close_abps]

    print('Number of pairs:',len(mean_abp_paired))
    print('Removing Colliding ABPs')
    pairs_colliding = cKDTree(mean_abp_paired).query_pairs(colliding_distance*1.5)
    print('Number of collisions:', len(pairs_colliding))
    print(pairs_colliding)
    while len(pairs_colliding) > 0:
        close_abps = list(set([mean_abp_paired.index[random.choice([a, b])] for a, b in pairs_colliding]))
        close_abps.sort()
        sel = mean_abp_paired.index.difference(close_abps)
        mean_abp_paired = mean_abp_paired.loc[sel]
        print('Number of pairs:', len(mean_abp_paired))
        pairs_colliding = cKDTree(mean_abp_paired).query_pairs(colliding_distance*1.5)
        print('Number of collisions:', len(pairs_colliding))
        print(pairs_colliding)



    full_model = full_model[full_model['chain_index'].isin(mean_abp_paired.index) | full_model['resName'].isin(['ACT', 'ACD'])]

    print('Rejoining model')
    # Split and rejoin the model so that ABPs are on a different chain
    full_model = openactin.Scene(full_model.sort_values(['chainID', 'resSeq', 'name']))
    full_model_actin = openactin.Scene(full_model[full_model['resName'].isin(['ACT', 'ACD'])])
    full_model_abps = openactin.Scene(full_model[~full_model['resName'].isin(['ACT', 'ACD'])])
    # full_model_abps['chainID'] = 'A'
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
    s = openactin.openactin.from_topology(f'{Sname}.cif', )

    # Add extra bonds for alfa actinin
    # extra_bonds=[]
    # for _, c in s.atom_list[(s.atom_list['residue_name'] == 'AAC') & (s.atom_list['atom_name'] == 'Cb')].groupby(
    #        'chain_index'):
    #    assert len(c.index)==2,'multiple Cbs in actinin'
    #    i0,i1=c.index
    #    extra_bonds+=[['AAC', i0, i1, 0, 10, 0, 350.0, 0, '1Cb-2Cb']]

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
    # simulation.context.setParameter("w1", 5)
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
    print('Epsilon_values: np.logspace(2,1,1000) each 1000 steps (1 frames')
    epsilons = np.logspace(2, 1, 200)
    print(epsilons)
    simulation.context.setParameter("w1", 5)
    for eps in epsilons:
        simulation.context.setParameter("g_eps", eps)
        simulation.step(1000)

    simulation.step(200000)

    epsilons = np.logspace(1, 2, 200)
    for eps in epsilons:
        simulation.context.setParameter("g_eps", eps)
        simulation.step(1000)

    simulation.step(1000000)

    # Turn off nematic parameter
    # simulation.context.setParameter('kp_alignment',0)
    # simulation.runForClockTime(sjob["run_time"])

    # Save checkpoint
    chk = f'{Sname}.chk'
    simulation.saveCheckpoint(chk)
