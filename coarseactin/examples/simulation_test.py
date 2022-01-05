import sys
import coarseactin
import pandas
import numpy as np
import scipy.spatial.distance as sdist
import itertools


def create_actin(length=100,
                    twist=2.89942054, shift=-28.21600347,
                    rotation=np.array([[1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., 1.]]),
                    translation=np.array([5000, 5000, 5000])):
    q = np.array([[np.cos(twist), -np.sin(twist), 0, 0],
                  [np.sin(twist), np.cos(twist), 0, 0],
                  [0, 0, 1, shift],
                  [0, 0, 0, 1]])
    rot = q[:3, :3].T
    trans = q[:3, 3]

    # Create the points
    point = bound_actin_template[['x', 'y', 'z']]
    points = []
    for i in range(length):
        points += [point]
        point = np.dot(point, rot) + trans
    points = np.concatenate(points)

    # Create the model
    model = pandas.DataFrame(points, columns=['x', 'y', 'z'])
    model["resid"] = [j + i for i in range(length) for j in bound_actin_template["resid"]]
    model["name"] = [j for i in range(length) for j in bound_actin_template["name"]]
    model["type"] = [j for i in range(length) for j in bound_actin_template["type"]]
    model["resname"] = [j for i in range(length) for j in bound_actin_template["resname"]]

    # Remove two binding points
    model = model[~((model['resid'] > length - 1) & (model['name'].isin(
        ['A5', 'A6', 'A7'] + ['Cc'] + [f'C{i + 1:02}' for i in range(12)] + [f'Cx{i + 1}' for i in range(3)])))]

    # Remove all CaMKII except resid 50
    # model=model[~((model['resid']!=50) & (model['resname'].isin(['CAM'])))]

    model.loc[model[model['resid'] == model['resid'].max()].index, 'resname'] = 'ACD'
    model.loc[model[model['resid'] == model['resid'].min()].index, 'resname'] = 'ACD'
    #for chain_name in string.ascii_uppercase + string.ascii_lowercase:
        # print(chain_name)
    #    if chain_name in full_model['chainID'].values:
    #        model.loc[model['resname'].isin(['ACT', 'ACD']), 'chainID'] = chain_name
    #        continue
    #    model.loc[model['resname'].isin(['ACT', 'ACD']), 'chainID'] = chain_name
    #    break

    #for chain_name in string.ascii_uppercase + string.ascii_lowercase:
        # print(chain_name,'A' in model['chainID'])
    #    if chain_name in full_model['chainID'].values or chain_name in model['chainID'].values:
    #        model.loc[model['resname'].isin(['CAM']), 'chainID'] = chain_name
    #        continue
    #    model.loc[model['resname'].isin(['CAM']), 'chainID'] = chain_name
    #    break

    # model["name"] = [j for i in range(1000) for j in ['A1', 'A2', 'A3', 'A4']]

    # Center the model
    model[['x', 'y', 'z']] -= model[['x', 'y', 'z']].mean()

    # Move the model
    model[['x', 'y', 'z']] = np.dot(model[['x', 'y', 'z']], rotation) + translation

    #full_model = pandas.concat([full_model, model])
    #full_model.index = range(len(full_model))
    return model

if __name__ == '__main__':
    ###################################
    # Setting Conditions for simulation#
    ###################################

    parameters = {"epsilon": [100],
                  "aligned": [False],
                  "actinLen": [500],
                  "layers": [1],
                  #            "repetition":range(3),
                  "disorder": [.5, .75],
                  "temperature": [300],
                  "system2D": [False],
                  "frequency": [1000],
                  "run_time": [20],
                  "CaMKII_Force": ['multigaussian', 'doublegaussian', 'singlegaussian'],
                  "simulation_platform": ["OpenCL"]}
    test_parameters = {"simulation_platform": "CUDA",
                       "frequency": 1000,
                       "run_time": 0.1,
                       "CaMKII_Force": 'multigaussian'
                       }
    job_id = 0
    if len(sys.argv) > 1:
        try:
            job_id = int(sys.argv[1])
        except TypeError:
            pass
    sjob = coarseactin.SlurmJobArray("ActinBundle", parameters, test_parameters, job_id)
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
    import string
    import random

    bound_actin_template = pandas.read_csv("coarseactin/data/CaMKII_bound_with_actin.csv", index_col=0)

    full_model=[]
    #full_model = pandas.DataFrame(columns=['chainID'])

    if sjob["layers"] == 1:
        hg = coarseactin.HexGrid(2)
        coords = hg.coords()[:2]
        d = 59.499 * 2
    else:
        hg = coarseactin.HexGrid(sjob["layers"])
        coords = hg.coords()
        d = 59.499 * 2

    for c in coords:
        height = (random.random() - 0.5) * 39 * 28.21600347 * sjob["disorder"]
        print(c[0], c[1], height)
        full_model += [create_actin(length=sjob["actinLen"],
                                     translation=np.array([5000 + d * c[0], 5000 + d * c[1], 5000 + height]))]


    print('Converting')
    chain_names = chain_names = [a for a in (string.ascii_uppercase + string.ascii_lowercase)] +\
                                [a+b for a,b in itertools.product(string.ascii_uppercase + string.ascii_lowercase,repeat=2)]
    chain_names = chain_names[:len(full_model)]
    chainID = [c for a, b in zip(chain_names, full_model) for c in [a]*len(b)]
    model = coarseactin.Scene(pandas.concat(full_model))
    model['chainID'] = chainID
    full_model=model
    print('Writing cif')
    full_model.write_cif('full_model_initial.cif')

    # Remove the CaMKII that are not overlapping
    print('Removing Single CaMKII')
    sel = full_model[full_model['name'] == 'Cc']
    i = sel.index
    d = sdist.pdist(sel[['x', 'y', 'z']])
    d = pandas.Series(d, itertools.combinations(i, 2))
    sel2 = sel.loc[[a for a, b in d[d < 35].index]]
    print(len(sel2))
    full_model.loc[:, 'chain_resid'] = full_model[['chainID', 'resid', ]].apply(lambda x: ''.join([str(a) for a in x]),
                                                                                axis=1)
    print(len(full_model[full_model['resname'].isin(['ACT', 'ACD'])]))
    print(len(full_model[full_model['chain_resid'].isin(
        sel2[['chainID', 'resid', ]].apply(lambda x: ''.join([str(a) for a in x]), axis=1))]))

    full_model = full_model[full_model['resname'].isin(['ACT', 'ACD']) |
                            full_model['chain_resid'].isin(
                                sel2[['chainID', 'resid', ]].apply(lambda x: ''.join([str(a) for a in x]), axis=1))]
    print(len(full_model))
    full_model = coarseactin.Scene(full_model.sort_values(['chainID', 'resid', 'name']))
    full_model.write_cif('full_model_initial.cif')

    #Remove the CaMKII that are colliding
    print('Removing Collisions')
    sel=full_model[full_model['name']=='Cc']
    i=sel.index
    d=sdist.pdist(sel[['x','y','z']])
    d=pandas.Series(d,itertools.combinations(i,2))
    sel2=sel.loc[[b for a,b in d[d<35].index]]
    #print(len(sel2))
    full_model.loc[:,'chain_resid']=full_model[['chainID','resid',]].apply(lambda x:''.join([str(a) for a in x]),axis=1)
    #print(len(full_model[full_model['resname'].isin(['ACT','ACD'])]))
    #print(len(full_model[full_model['chain_resid'].isin(sel2[['chainID','resid',]].apply(lambda x:''.join([str(a) for a in x]),axis=1))]))

    full_model=full_model[~full_model['chain_resid'].isin(sel2[['chainID','resid',]].apply(lambda x:''.join([str(a) for a in x]),axis=1))]

    full_model['mass'] = 1
    full_model['molecule'] = 1
    full_model['q'] = 0
    #ss = SystemData(full_model.sort_values(['chainID', 'resid', 'name']))
    #ss.write_data()
    #ss.write_pdb(f'{Sname}.pdb')
    #ss.write_gro(f'{Sname}.gro')
    #ss.print_coeff()

    full_model = coarseactin.Scene(full_model.sort_values(['chainID', 'resid', 'name']))

    print('Writing cif')
    full_model.write_cif('full_model.cif')
    print('Writing gro')
    full_model.write_gro('full_model.gro')