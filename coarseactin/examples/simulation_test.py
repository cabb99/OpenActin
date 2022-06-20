#!/home/cab22/miniconda3/bin/python

#SBATCH --account=commons
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
import coarseactin
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

    bound_actin_template = pd.read_csv("coarseactin/data/CaMKII_bound_with_actin.csv", index_col=0)

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
    model = pandas.DataFrame(points, columns=['x', 'y', 'z'])
    model["resSeq"] = [j + i for i in range(length) for j in bound_actin_template["resSeq"]]
    model["name"] = [j for i in range(length) for j in bound_actin_template["name"]]
    model["type"] = [j for i in range(length) for j in bound_actin_template["type"]]
    model["resName"] = [j for i in range(length) for j in bound_actin_template["resName"]]
    model["element"] = [j for i in range(length) for j in bound_actin_template["element"]]

    # Remove two binding points
    model = model[~(((model['resSeq'] > length-1) | (model['resSeq'] == 1)) & ~model['name'].isin(['A1', 'A2', 'A3', 'A4']))]

    model.loc[model[model['resSeq'] == model['resSeq'].max()].index, 'resName'] = 'ACD'
    model.loc[model[model['resSeq'] == model['resSeq'].min()].index, 'resName'] = 'ACD'

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
                  # "layers": [3],
                  # "repetition":range(3),
                  "disorder": [0],
                  "temperature": [300],
                  "system2D": [False],
                  "frequency": [1000],
                  "run_time": [20],
                  "CaMKII_Force": ['multigaussian', 'doublegaussian', 'singlegaussian'],
                  "simulation_platform": ["OpenCL"]}
    test_parameters = {"simulation_platform": "CUDA",
                       "frequency": 1000,
                       "run_time": 8,
                       "CaMKII_Force": 'multigaussian'
                       }
    job_id = 0
    if len(sys.argv) > 1:
        try:
            job_id = int(sys.argv[1])
        except TypeError:
            pass
    sjob = coarseactin.SlurmJobArray("Unbundling_rebundling_w1_5", parameters, test_parameters, job_id)
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

    ###################
    # Build the model #
    ###################
    # Set the points in the actin network
    import random

    full_model = []

#    if sjob["layers"] == 1:
#        hg = coarseactin.HexGrid(2)
#        coords = hg.coords()[:2]
#        d = 59.499 * 2
#    else:
#        hg = coarseactin.HexGrid(sjob["layers"])
#        coords = hg.coords()
#        d = 59.499 * 2

    #Make a bundle of 100nm
    hg = coarseactin.HexGrid(20)
    coords = hg.coords()
    d = 59.499 * 2
    coords = coords[(d * (coords ** 2).sum(axis=1) ** .5) < 500]

    for c in coords:
        height = (random.random() - 0.5) * sjob["actinLen"] * 28.21600347 * sjob["disorder"]
        t = np.random.random()*np.pi*2
        rotation = np.array([[np.cos(t), -np.sin(t), 0.],
                             [np.sin(t), np.cos(t), 0.],
                             [0., 0., 1.]])
        print(c[0], c[1], height)
        full_model += [create_actin(length=sjob["actinLen"],
                                    translation=np.array([5000 + d * c[0], 5000 + d * c[1], 5000 + height]),
                                    rotation=rotation,abp='CAM')]

    print('Concatenate chains')
    full_model = coarseactin.Scene.concatenate(full_model)

    # Remove the CaMKII that are not overlapping
    print('Removing Single CaMKII')
    sel = full_model[full_model['name'] == 'Cc']
    print('Calculating distance')
    pairs = cKDTree(sel[['x', 'y', 'z']]).query_pairs(35)
    #d = sdist.pdist(sel[['x', 'y', 'z']])
    print('Making selections')
    #d = pandas.Series(d, itertools.combinations(sel.index, 2))
    close_abps=list(set([sel.index[a] for a, b in pairs]))
    sel2 = sel.loc[close_abps]
    print(len(sel2))
    full_model.loc[:, 'chain_resid'] = full_model[['chainID', 'resSeq', ]].apply(lambda x: ''.join([str(a) for a in x]),
                                                                                 axis=1)
    print(len(full_model[full_model['resName'].isin(['ACT', 'ACD'])]))
    print(len(full_model[full_model['chain_resid'].isin(
        sel2[['chainID', 'resSeq', ]].apply(lambda x: ''.join([str(a) for a in x]), axis=1))]))

    full_model = full_model[full_model['resName'].isin(['ACT', 'ACD']) |
                            full_model['chain_resid'].isin(
                                sel2[['chainID', 'resSeq', ]].apply(lambda x: ''.join([str(a) for a in x]), axis=1))]
    print(len(full_model))

    full_model = coarseactin.Scene(full_model.sort_values(['chainID', 'resSeq', 'name']))

    # Remove the CaMKII that are colliding
    print('Removing Collisions')
    sel = full_model[full_model['name'] == 'Cc']
    #d = sdist.pdist(sel[['x', 'y', 'z']])
    #d = pandas.Series(d, itertools.combinations(sel.index, 2))
    pairs = cKDTree(sel[['x', 'y', 'z']]).query_pairs(35)
    sel2 = sel.loc[list(set([sel.index[b] for a, b in pairs]))]

    full_model.loc[:, 'chain_resid'] = full_model[['chainID', 'resSeq', ]].apply(lambda x: ''.join([str(a) for a in x]),
                                                                                 axis=1)
    full_model = full_model[full_model['resName'].isin(['ACT', 'ACD']) | ~full_model['chain_resid'].isin(
        sel2[['chainID', 'resSeq', ]].apply(lambda x: ''.join([str(a) for a in x]), axis=1))]

    full_model['mass'] = 1
    full_model['molecule'] = 1
    full_model['q'] = 0

    full_model = coarseactin.Scene(full_model.sort_values(['chainID', 'resSeq', 'name']))
    full_model_actin = coarseactin.Scene(full_model[full_model['resName'].isin(['ACT','ACD'])])
    full_model_abps = coarseactin.Scene(full_model[~full_model['resName'].isin(['ACT', 'ACD'])])
    full_model_abps['chainID'] = 'A'
    resids = full_model_abps['chain_resid'].unique()
    resids_rename = full_model_abps['chain_resid'].replace({a: b for a, b in zip(resids, range(len(resids)))})
    full_model_abps['resSeq'] = full_model_abps['chain_resid'].replace(resids_rename)
    full_model = coarseactin.Scene.concatenate([full_model_actin, full_model_abps])


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
    s = coarseactin.CoarseActin.from_topology(f'{Sname}.cif', )
    print(s.system.getDefaultPeriodicBoxVectors())
    s.setForces(AlignmentConstraint=aligned, PlaneConstraint=system2D, CaMKII_Force=sjob['CaMKII_Force'])
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
        openmm.app.StateDataReporter(stdout, frequency, step=True, time=True, potentialEnergy=True, temperature=True,
                                     separator='\t', ))
    simulation.reporters.append(
        openmm.app.StateDataReporter(f'{Sname}.log', frequency, step=True, time=True, totalEnergy=True,
                                     kineticEnergy=True, potentialEnergy=True, temperature=True))

    # Print initial energy
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    print(f'Initial energy: {energy} KJ/mol')

    # Run
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature * u.kelvin)
    time0 = time.ctime()
    time_0 = time.time()
    print('Epsilon_values: np.logspace(2,1,1000) each 1000 steps (1 frames')
    epsilons = np.logspace(2 ,1 ,200)
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
    #simulation.runForClockTime(sjob["run_time"])

    # Save checkpoint
    chk = f'{Sname}.chk'
    simulation.saveCheckpoint(chk)
