import pandas as pd
from ..Scene import *
import string
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
#Parse coordinate file
coordinates = pd.read_csv(f'{__location__}/../data/coordinates.csv', index_col='id')
actin_template = Scene(coordinates)

#Actin monomer variants
actin_monomer = actin_template[actin_template['resname'].isin(['ACT'])]


def generate_actin(length=100,
                    twist=2.89942054, shift=-28.21600347,
                    rotation=np.array([[1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., 1.]]),
                    translation=np.array([5000, 5000, 5000])):
    quaternion = np.array([[np.cos(twist), -np.sin(twist), 0, 0],
                           [np.sin(twist), np.cos(twist), 0, 0],
                           [0, 0, 1, shift],
                           [0, 0, 0, 1]])
    rot = quaternion[:3, :3].T
    trans = quaternion[:3, 3]

    # Create the points
    point = actin_monomer[['x', 'y', 'z']]
    points = []
    for i in range(length):
        points += [point]
        point = np.dot(point, rot) + trans
    points = np.concatenate(points)

    # Create the model
    model = pandas.DataFrame(points, columns=['x', 'y', 'z'])
    model["resid"] = [j + i for i in range(length) for j in actin_monomer["resid"]]
    model["name"] = [j for i in range(length) for j in actin_monomer["name"]]
    model["type"] = [j for i in range(length) for j in actin_monomer["type"]]
    model["resname"] = [j for i in range(length) for j in actin_monomer["resname"]]

    # Remove two binding points
    model = model[~((model['resid'] > length - 1) & (model['name'].isin(
        ['A5', 'A6', 'A7'] + ['Cc'] + [f'C{i + 1:02}' for i in range(12)] + [f'Cx{i + 1}' for i in range(3)])))]

    # Remove all CaMKII except resid 50
    # model=model[~((model['resid']!=50) & (model['resname'].isin(['CAM'])))]

    model.loc[model[model['resid'] == model['resid'].max()].index, 'resname'] = 'ACD'
    model.loc[model[model['resid'] == model['resid'].min()].index, 'resname'] = 'ACD'
    for chain_name in string.ascii_uppercase + string.ascii_lowercase:
        # print(chain_name)
        if chain_name in full_model['chainID'].values:
            model.loc[model['resname'].isin(['ACT', 'ACD']), 'chainID'] = chain_name
            continue
        model.loc[model['resname'].isin(['ACT', 'ACD']), 'chainID'] = chain_name
        break

    for chain_name in string.ascii_uppercase + string.ascii_lowercase:
        # print(chain_name,'A' in model['chainID'])
        if chain_name in full_model['chainID'].values or chain_name in model['chainID'].values:
            model.loc[model['resname'].isin(['CAM']), 'chainID'] = chain_name
            continue
        model.loc[model['resname'].isin(['CAM']), 'chainID'] = chain_name
        break

    # model["name"] = [j for i in range(1000) for j in ['A1', 'A2', 'A3', 'A4']]

    # Center the model
    model[['x', 'y', 'z']] -= model[['x', 'y', 'z']].mean()

    # Move the model
    model[['x', 'y', 'z']] = np.dot(model[['x', 'y', 'z']], rotation) + translation

    return model
    #full_model = pandas.concat([full_model, model])
    #full_model.index = range(len(full_model))
