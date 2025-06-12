#!/usr/bin/env python3
#SBATCH --account=commons
#SBATCH --export=All
#SBATCH --partition=commons
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-user=za32@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-8 
#SBATCH --mem=32G

import sys
import openactin
import pandas as pd
import numpy as np
import scipy.spatial.transform as strans

if __name__ == '__main__':
    ###################################
    # Benchmarking bundle parameters  #
    ###################################
    # Define the parameter sweep for benchmarking bundles of actin with different ABPs (CAM = CaMKII, FAS = fascin),
    # bundle sizes (layers), filament lengths, and other simulation conditions.
    parameters = {
        "abp": ["CAM", "FAS"],                  # Sweep both CaMKII and fascin
        "epsilon_FAS": [100],                   # Fascin epsilon value
        "epsilon_CAM": [100],                   # CaMKII epsilon value
        "actinLen": [100, 250, 500],            # Test different filament lengths
        "layers": [2, 3, 4],                    # Test different bundle sizes (layers)
        "repetition": range(3),                 # Repeat each condition 3 times for statistics
        "disorder": [0],                        # Amount of disorder in filament placement
        "bundleWidth": [1000],                  # nm, used for filtering grid for large bundles
        "temperature": [300],                   # Simulation temperature (K)
        "frequency": [10000],                   # Output frequency for reporters
        "run_time": [20],                       # Simulation run time (arbitrary units)
        "simulation_platform": ["OpenCL"]      # Platform for OpenMM (GPU or CPU)
    }
    # Test parameters for quick runs (CPU, short time, low frequency)
    test_parameters = {
        "simulation_platform": "CPU",
        "frequency": 1,
        "run_time": 0.01,
    }

    # Create a SLURM job array object for this parameter sweep
    sjob = openactin.SlurmJobArray("Simulations_scratch/Bundles/Benchmark", parameters, test_parameters)
    sjob.print_parameters()      # Print the parameters for this job
    sjob.print_slurm_variables() # Print SLURM variables for this job
    sjob.write_csv()             # Write the parameters to a CSV file for record-keeping
    print("name :", sjob.name)   # Print the unique name for this simulation

    # Extract parameters for this job
    actinLen = sjob["actinLen"]
    layers = sjob["layers"]
    abp = sjob["abp"]
    Sname = sjob.name
    simulation_platform = sjob["simulation_platform"]

    # Calculate the distance between actin filaments using the selected ABP as reference
    # This sets the bundle geometry so that crosslinkers (CAM or FAS) fit between filaments
    bound_actin_template = pd.read_csv("openactin/data/CaMKII_bound_with_actin.csv", index_col=0)
    c = bound_actin_template[bound_actin_template['resName'] == abp][['x', 'y', 'z']]
    v = c.mean(axis=0)
    d = 2 * (v['x'] ** 2 + v['y'] ** 2) ** .5  # Effective crosslinker size

    print(f"Grid spacing d: {d}")
    # Generate a hexagonal grid for the bundle cross-section
    # The number of layers determines the bundle size (1 = 2 filaments, 2 = 7, 3 = 19, ...)
    if layers == 1:
        hg = openactin.HexGrid(2)
        coords = hg.coords()[:2]
    else:
        hg = openactin.HexGrid(layers)
        coords = hg.coords()
        # For large bundles, filter out filaments outside the desired bundle width
        if layers > 2:
            coords = coords[(d * (coords ** 2).sum(axis=1) ** .5) < sjob["bundleWidth"] / 2]

    print('Number of actin filaments:', len(coords))

    # Build the model: create actin filaments with the selected ABP (CAM or FAS) at each grid position
    full_model = []
    for c in coords:
        # Optionally add some vertical disorder to the filament position
        height = (np.random.random() - 0.5) * actinLen * 28.21600347 * sjob["disorder"]
        # Randomize the in-plane rotation of each filament
        t = np.random.random() * np.pi * 2
        rotation = np.array([[np.cos(t), -np.sin(t), 0.],
                             [np.sin(t), np.cos(t), 0.],
                             [0., 0., 1.]])
        # Create the actin filament decorated with the selected ABP
        full_model += [openactin.create_actin(
            length=actinLen,
            translation=np.array([5000 + d * c[0], 5000 + d * c[1], 5000 + height]),
            rotation=rotation,
            abp=abp
        )]

    print('Concatenate chains')
    # Combine all filaments into a single Scene object
    full_model = openactin.Scene.concatenate(full_model)

    # --- Remove isolated and colliding ABPs (crosslinkers) ---
    from scipy.spatial import cKDTree
    import random
    # Set colliding distance based on ABP type
    if abp == 'FAS':
        colliding_distance = 40  # or d/4 for FAS
    else:
        colliding_distance = d/4  # for CAM and others
    print(f"Colliding distance: {colliding_distance}")
    abp_count = full_model[full_model['resName'] == abp]['chain_index'].nunique()
    print(f"Number of ABP chains: {abp_count}")
    # Calculate mean position of each ABP chain
    mean_abp = full_model[full_model['resName'] == abp].groupby('chain_index')[['x', 'y', 'z']].mean()
    # Find pairs of ABPs that are close enough to be considered paired
    pairs = cKDTree(mean_abp).query_pairs(colliding_distance)
    close_abps = list(set([mean_abp.index[a] for a, b in pairs] + [mean_abp.index[b] for a, b in pairs]))
    close_abps.sort()
    mean_abp_paired = mean_abp.loc[close_abps]
    print('Number of pairs:', len(mean_abp_paired))
    if len(mean_abp_paired) == 0:
        print("No ABP pairs found. ABP mean positions:")
        print(mean_abp)
        # Save the current model for debugging
        full_model.write_cif(f'{Sname}_failed.cif', verbose=True)
        print(f"No ABP pairs found, saved failed model as {Sname}_failed.cif. Exiting.")
        sys.exit(1)
    # Remove colliding ABPs (those that are too close to each other)
    pairs_colliding = cKDTree(mean_abp_paired).query_pairs(colliding_distance)
    print('Number of collisions:', len(pairs_colliding))
    if len(pairs_colliding) == 0:
        print("No ABP collisions found. ABP paired mean positions:")
        print(mean_abp_paired)
    while len(pairs_colliding) > 0:
        # Randomly remove one from each colliding pair
        close_abps = list(set([mean_abp_paired.index[random.choice([a, b])] for a, b in pairs_colliding]))
        close_abps.sort()
        sel = mean_abp_paired.index.difference(close_abps)
        mean_abp_paired = mean_abp_paired.loc[sel]
        print('Number of pairs:', len(mean_abp_paired))
        pairs_colliding = cKDTree(mean_abp_paired).query_pairs(colliding_distance)
        print('Number of collisions:', len(pairs_colliding))
    # Only keep actin and the selected, non-colliding ABPs in the final model
    full_model = full_model[full_model['chain_index'].isin(mean_abp_paired.index) | full_model['resName'].isin(['ACT', 'ACD'])]
    print('Rejoining model')
    # Split and rejoin the model so that ABPs are on a different chain
    full_model = openactin.Scene(full_model.sort_values(['chainID', 'resSeq', 'name']))
    full_model_actin = openactin.Scene(full_model[full_model['resName'].isin(['ACT', 'ACD'])])
    full_model_abps = openactin.Scene(full_model[~full_model['resName'].isin(['ACT', 'ACD'])])
    full_model = openactin.Scene.concatenate([full_model_actin, full_model_abps])
    # Write the model to a CIF file for OpenMM
    full_model.write_cif(f'{Sname}.cif', verbose=True)

    ##############
    # Simulation #
    ##############
    # Import OpenMM and related modules for running the simulation
    import openmm
    import openmm.app
    import simtk.unit as u
    import time
    from sys import stdout
    import numpy as np

    # Select the simulation platform (CPU, OpenCL, CUDA, etc.)
    platform = openmm.Platform.getPlatformByName(simulation_platform)
    # Load the model and topology from the CIF file
    s = openactin.openactin.from_topology(f'{Sname}.cif')

    # --- Add extra bonds for CBP crosslinkers ---
    extra_bonds = []
    name_list = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12']
    if hasattr(s, 'atom_list') and hasattr(s, 'bonds'):
        for _, c in s.atom_list[(s.atom_list['residue_name'] == 'CBP')].groupby('chain_index'):
            cm = c[c['atom_name'].isin(name_list)]
            cb = c[c['atom_name'] == 'Cb']
            if len(cm.index) == len(cb.index) and len(cm.index) == 12:
                for i0, i1 in zip(cm.index, cb.index):
                    extra_bonds += [['CBP', 'harmonic', i0, i1, 0, 10, 0, 50.00, 0, np.nan, np.nan, '1Cb-2Cb']]
        if extra_bonds:
            extra_bonds = pd.DataFrame(extra_bonds, columns=s.bonds.columns, index=range(s.bonds.index.max() + 1, s.bonds.index.max() + 1 + len(extra_bonds)))
            s.bonds = pd.concat([s.bonds, extra_bonds], ignore_index=True)
        # Check that the bonds correspond with the correct molecule
        s.bonds = s.bonds[(s.bonds['molecule'] == s.atom_list['residue_name'].loc[s.bonds['i']].values) |
                          s.bonds['molecule'].isin(['Actin-ADP', 'ABP', 'CaMKII'])]

    print(s.system.getDefaultPeriodicBoxVectors())

    
    # --- Add all forces  ---
    system2D = False
    if sjob["abp"]!='FAS':
        s.setForces(AlignmentConstraint=False, PlaneConstraint=system2D, forces=['multigaussian','abp'])
    else: 
        s.setForces(AlignmentConstraint=False, PlaneConstraint=system2D, forces=['abp'])

    # --- Add electrostatics force (Debye-Hückel) ---
    electrostatics = openmm.CustomNonbondedForce("epsilon_electrostatics*q1*q2/r*exp(-kappa_electrostatics*r)")
    electrostatics.setNonbondedMethod(electrostatics.CutoffNonPeriodic)
    electrostatics.addPerParticleParameter("q")
    # Use a reasonable default for epsilon and kappa, or add to parameters if needed
    electrostatics.addGlobalParameter("epsilon_electrostatics", 1.736385125)  # (nm/charge^2)
    electrostatics.addGlobalParameter("kappa_electrostatics", 1)  # (nm^-1)
    electrostatics.setCutoffDistance(40*openmm.unit.nanometers)
    electrostatics.setUseLongRangeCorrection(True)
    # Assign charges to particles based on residue and atom name (as in the box script)
    for _, a in s.atom_list.iterrows():
        if a.residue_name in ['ACT','ACD'] and a.name in ['A1','A2','A3','A4']:
            q = -2.5
        elif a.residue_name in ['FAS']:
            q = 3.3/6
        elif a.residue_name in ['CAM']:
            q = -3.5
        else:
            q = 0
        electrostatics.addParticle([q])
    # Exclude close bonded pairs from electrostatics
    if hasattr(s, 'bonds') and hasattr(s.bonds, 'values'):
        electrostatics.createExclusionsFromBonds(s.bonds[['i', 'j']].values.tolist(), 3)
    s.system.addForce(electrostatics)

    # Set up the integrator and simulation object
    temperature = sjob["temperature"] * u.kelvin
    integrator = openmm.LangevinIntegrator(temperature, .0001 / u.picosecond, 1 * u.picoseconds)
    simulation = openmm.app.Simulation(s.top.topology, s.system, integrator, platform)
    simulation.context.setPositions(s.top.positions)

    # Add reporters to save trajectory and log simulation data
    frequency = sjob["frequency"]
    simulation.reporters.append(openmm.app.DCDReporter(f'{Sname}.dcd', frequency))
    simulation.reporters.append(
        openmm.app.StateDataReporter(stdout, frequency, step=True, time=True, potentialEnergy=True, totalEnergy=True, temperature=True, separator='\t'))
    simulation.reporters.append(
        openmm.app.StateDataReporter(f'{Sname}.log', frequency, step=True, time=True, totalEnergy=True, kineticEnergy=True, potentialEnergy=True, temperature=True))

    # Print the initial potential energy for diagnostics
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    print(f'Initial energy: {energy} KJ/mol')

    # Minimize energy, set velocities, and run the simulation for the specified time
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.runForClockTime(sjob["run_time"])

    # Save a checkpoint file for restarting or analysis
    chk = f'{Sname}.chk'
    simulation.saveCheckpoint(chk)
