import glob
import os
import mdtraj
import typing
import pandas as pd
import numpy as np
import warnings
import prody
import itertools
import coarseactin
import matplotlib.pyplot as plt
import subprocess

def parse_simulations(files: typing.Union[str, list] = None,
                      extensions: tuple = ('param', 'cif', 'pdb', 'dcd', 'log'),
                      ) -> typing.Union[pd.DataFrame, None]:
    # List simulation files
    if isinstance(files, str):
        param_files = glob.glob(f'{files}')
    elif isinstance(files, (list, tuple)):
        param_files = [f1 for f in files for f1 in glob.glob(f)]
    else:
        warnings.warn(f"{files} was not parsed correctly. Invalid type: f{type(files)}")
        return None
    data = []

    #Sort simulation files
    param_files.sort()

    if len(param_files) == 0:
        warnings.warn("No files found")
        return None

    for param_file in param_files:
        # Add parameters
        with open(param_file) as param:
            values = {}
            for line in param:
                if len(line) == 0:
                    continue
                line = line.strip().split(',')
                key = line[0]
                value = line[1]
                if key == '':
                    continue
                values.update({key: value})
                root = '.'.join(param_file.split('.')[:-1])
                values.update({'root': root})
        # Add files
        for extension in extensions:
            file_name = f'{root}.{extension}'
            if os.path.exists(file_name):
                values.update({extension: file_name})
        data += [values]

    simulations = pd.DataFrame(data)
    if len(data) == 0:
        warnings.warn("No data found")
        return None
    #simulations[['epsilon', 'w1', 'w2_ratio']] = simulations[['epsilon', 'w1', 'w2_ratio']].astype(float)

    #simulations = simulations.sort_values(['epsilon', 'w1', 'w2_ratio'])
    simulations = simulations.reindex()
    simulations = simulations.reset_index(drop=True)
    # Get number of simulation frames and throw away simulations with no frames
    dcd_lens = []
    for i, simulation in simulations.iterrows():
        if ~pd.isnull(simulation['cif']):
            pdbx = mdtraj.formats.pdbx.load_pdbx(simulation['cif'])
        else:
            warnings.warn(f"No cif file found for file {simulation['root']}")
            continue
        try:
            dcd = mdtraj.load_dcd(simulation['dcd'], top=pdbx)
            dcd_len = len(dcd)
        except OSError:
            dcd_len = 0
        except TypeError:
            dcd_len = 0
        dcd_lens += [dcd_len]
    simulations['frames'] = dcd_lens
    simulations['time (us)'] = simulations['frames'] * simulations['frequency'].astype(int) / 10 ** 6
    simulations = simulations[simulations['frames'] > 10].copy()
    simulations = simulations.sort_values('time (us)', ascending=False)
    simulations = simulations.reset_index(drop=True)

    #Convert values to float, integer or bool if possible
    for column in simulations.columns:
        try:
            simulations[column] = simulations[column].astype(int)
        except ValueError:
            try:
                simulations[column] = simulations[column].astype(float)
            except ValueError:
                values = simulations[column].unique()
                if len(values) == 2 and 'True' in values and 'False' in values:
                    simulations[column] = simulations[column].map({'True': True, 'False': False})
                    simulations[column] = simulations[column].astype(bool)
    return simulations

class Simulation:
    def __init__(self, simulation, box_size=500):

        self.simulation=simulation
        self.root = simulation['root']
        try:
            self.cif = prody.parseMMCIF(simulation['cif'])
        except KeyError:
            pass
            # continue
        self.pdbx = mdtraj.formats.pdbx.load_pdbx(simulation['cif'])
        self.dcd = mdtraj.load_dcd(simulation['dcd'], top=self.pdbx)
        self.dcd.unitcell_vectors = np.array([[[box_size, 0, 0], [0, box_size, 0], [0, 0, box_size]]] * len(self.dcd))
        self.scene = coarseactin.Scene.from_cif(simulation['cif'])
        self.binding_site_indices=None
        self.binding_site_distances_abp=None
        self.binding_site_rmsd_abp = None
        self.binding_site_distances_cam = None
        self.binding_site_rmsd_cam = None

    def compute_binding_site_indices(self, force=False):
        if self.binding_site_indices and not force:
            return self.binding_site_indices
        self.binding_site_indices = {}
        for name in ['Aa', 'Ab', 'Ac', 'Ca', 'Cb', 'Cd',
                     'A5' , 'A6','A7', 'Cc'] + [f'C{i:02d}' for i in range(12)]:
            selection = self.cif.select(f'name {name}')
            self.binding_site_indices.update({name: selection.getIndices()})
        return self.binding_site_indices


    def compute_binding_site_distances_abp(self,stride=5, force=False):
        if self.binding_site_distances_abp and not force:
            return self.binding_site_distances_abp
        s = self.compute_binding_site_indices(force=force)
        d1 = mdtraj.compute_distances(self.dcd[::stride], np.array([a for a in itertools.product(s['Aa'], s['Ca'])]),
                                      periodic=True)
        d2 = mdtraj.compute_distances(self.dcd[::stride], np.array([a for a in itertools.product(s['Ab'], s['Cb'])]),
                                      periodic=True)
        d3 = mdtraj.compute_distances(self.dcd[::stride], np.array([a for a in itertools.product(s['Ac'], s['Cd'])]),
                                      periodic=True)

        dd = (d1 ** 2 + d2 ** 2 + d3 ** 2) / 3
        self.binding_site_distances_abp = dd
        return self.binding_site_distances_abp

    def compute_binding_site_rmsd_abp(self, stride=5, force=False):
        self.compute_binding_site_indices(force=force)
        if self.binding_site_rmsd_abp and not force:
            return self.binding_site_distances_abp
        s = self.compute_binding_site_indices(force=force)
        actin_com = (self.dcd[::stride].xyz[:, s['Aa']] + self.dcd[::stride].xyz[:, s['Ab']] + self.dcd[::stride].xyz[:,
                                                                                               s['Ac']]) / 3
        abp_com = (self.dcd[::stride].xyz[:, s['Ca']] + self.dcd[::stride].xyz[:, s['Cb']] + self.dcd[::stride].xyz[:,
                                                                                             s['Cd']]) / 3
        translation = np.expand_dims(abp_com, axis=1) - np.expand_dims(actin_com, axis=2)
        rmsd_abp = ((((np.expand_dims(self.dcd[::stride].xyz[:, s['Ca']], axis=1) - translation - np.expand_dims(
            self.dcd[::stride].xyz[:, s['Aa']], axis=2)) ** 2).sum(axis=3) +
                     ((np.expand_dims(self.dcd[::stride].xyz[:, s['Cb']], axis=1) - translation - np.expand_dims(
                         self.dcd[::stride].xyz[:, s['Ab']], axis=2)) ** 2).sum(axis=3) +
                     ((np.expand_dims(self.dcd[::stride].xyz[:, s['Cd']], axis=1) - translation - np.expand_dims(
                         self.dcd[::stride].xyz[:, s['Ac']], axis=2)) ** 2).sum(axis=3)) / 3) ** .5
        self.binding_site_rmsd_abp = rmsd_abp
        return self.binding_site_rmsd_abp

    def compute_binding_site_distances_cam(self, stride=5, force=False):
        # TODO
        self.compute_binding_site_indices(force=force)
        if self.binding_site_distances_cam and not force:
            return self.binding_site_distances_cam
        s = self.compute_binding_site_indices(force=force)
        d1 = mdtraj.compute_distances(self.dcd[::stride], np.array([a for a in itertools.product(s['A5'], s['Cc'])]),
                                      periodic=True)
        d2 = mdtraj.compute_distances(self.dcd[::stride], np.array([a for a in itertools.product(s['A6'], s['C01'])]),
                                      periodic=True)
        d3 = mdtraj.compute_distances(self.dcd[::stride], np.array([a for a in itertools.product(s['A6'], s['C02'])]),
                                      periodic=True)
        d4 = mdtraj.compute_distances(self.dcd[::stride], np.array([a for a in itertools.product(s['A7'], s['C01'])]),
                                      periodic=True)
        d5 = mdtraj.compute_distances(self.dcd[::stride], np.array([a for a in itertools.product(s['A7'], s['C02'])]),
                                      periodic=True)
        d2=np.min([d2,d4],axis=0)
        d3=np.min([d3,d5],axis=0)

        dd = (d1 ** 2 + d2 ** 2 + d3 ** 2) / 3
        self.binding_site_distances_cam = dd
        return self.binding_site_distances_cam

    def compute_binding_site_rmsd_cam(self, stride=5, force=False):
        # TODO
        self.compute_binding_site_indices(force=force)
        if self.binding_site_rmsd_cam and not force:
            return self.binding_site_distances_cam
        s = self.compute_binding_site_indices(force=force)
        actin_com = (self.dcd[::stride].xyz[:, s['A5']] + self.dcd[::stride].xyz[:, s['Ab']] + self.dcd[::stride].xyz[:,
                                                                                               s['Ac']]) / 3
        abp_com = (self.dcd[::stride].xyz[:, s['Ca']] + self.dcd[::stride].xyz[:, s['Cb']] + self.dcd[::stride].xyz[:,
                                                                                             s['Cd']]) / 3
        translation = np.expand_dims(abp_com, axis=1) - np.expand_dims(actin_com, axis=2)
        rmsd_cam = ((((np.expand_dims(self.dcd[::stride].xyz[:, s['Ca']], axis=1) - translation - np.expand_dims(
            self.dcd[::stride].xyz[:, s['Aa']], axis=2)) ** 2).sum(axis=3) +
                     ((np.expand_dims(self.dcd[::stride].xyz[:, s['Cb']], axis=1) - translation - np.expand_dims(
                         self.dcd[::stride].xyz[:, s['Ab']], axis=2)) ** 2).sum(axis=3) +
                     ((np.expand_dims(self.dcd[::stride].xyz[:, s['Cd']], axis=1) - translation - np.expand_dims(
                         self.dcd[::stride].xyz[:, s['Ac']], axis=2)) ** 2).sum(axis=3)) / 3) ** .5
        self.binding_site_rmsd_cam = rmsd_cam
        return self.binding_site_rmsd_cam

    def compute_binding_site_energies(self, stride=5):
        dd = self.binding_site_distances_abp
        g1 = -1 * (np.exp(-dd / 1.0) + np.exp(-dd / 0.01 * 1.0)) / 2 * (d1 < 12.0)
        g2 = g1.reshape(len(g1), len(s['Aa']), len(s['Ca']))
        g2.shape

        g3_total_bind = (g2 < -0.2).sum(axis=1).T > 0
        g3_partial_bind = (g2 < 0).sum(axis=1).T > 0
        g3_states = [g3_total_bind[:, 0]]
        for t in range(1, len(g3_total_bind.T)):
            g3_state = g3_total_bind[:, t] | (g3_states[-1] & g3_partial_bind[:, t])
            g3_states += [g3_state]
        g3 = np.array(g3_states).T
        Actin_binding = g3 * 1

        g3_total_bind = (g2 < -0.2).sum(axis=2).T > 0
        g3_partial_bind = (g2 < 0).sum(axis=2).T > 0
        g3_states = [g3_total_bind[:, 0]]
        for t in range(1, len(g3_total_bind.T)):
            g3_state = g3_total_bind[:, t] | (g3_states[-1] & g3_partial_bind[:, t])
            g3_states += [g3_state]
        g3 = np.array(g3_states).T
        ABP_binding = g3 * 1

        Actin_binding[0].sum()

        Actin_binding_pair = g2.argmin(axis=1).T
        ABP_binding_pair = g2.argmin(axis=2).T
        Actin_binding_pair[Actin_binding == 0] = -99999
        ABP_binding_pair[ABP_binding == 0] = -99999

    def plot_binding_site_free_energy(self):
        plt.figure(figsize=(10, 5))
        try:
            fig, ax = pyemma.plots.plot_free_energy(np.log10((dd ** .5).ravel()), rmsd_abp.ravel(), cmap='viridis')
        except ValueError:
            pass
            # continue
        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(0, 2)
        ax.set_xticklabels([rf'$10^{{{i}}}$' for i in range(-3, 5)])
        ax.set_xlabel('ABP-Binding site distance (nm)')
        ax.set_ylabel('ABP-Binding site RMSD (nm)')
        plt.savefig(f'{self.root}_Freeenergy.png')

    def vmd(self, script='../Box_vis_500.vmd'):
        if script is None:
            return subprocess.Popen(['vmd', self.simulation['cif'], self.simulation['dcd']],
                                    stdin=subprocess.PIPE)
        else:
            return subprocess.Popen(['vmd', '-e', script,
                                     self.simulation['cif'], self.simulation['dcd']],
                                    stdin=subprocess.PIPE)

    def compute_bond_energy(self):
        raise NotImplementedError

    def compute_angle_energy(self):
        raise NotImplementedError

    def compute_dihedral_energy(self):
        raise NotImplementedError

    def color_binding_energy(self):
        raise NotImplementedError

    def color_backbone_energy(self):
        raise NotImplementedError

    def compute_actin_vectors(self):
        raise NotImplementedError

    def compute_msd(self, nparts=1, fft=True):
        import MDAnalysis as mda
        import MDAnalysis.analysis.msd
        import openmm
        timestep_us = int(self.simulation['frequency']) / 10 ** 6
        distance_um = 1 / 10 * 1 / 1000

        cif_openmm = openmm.app.pdbxfile.PDBxFile(self.simulation['cif'], )
        u = mda.Universe(cif_openmm.getTopology(), self.simulation['dcd'], topology_format='OPENMMTOPOLOGY')
        msd_analysis = mda.analysis.msd.EinsteinMSD(u, select='all', fft=fft)
        n = self.simulation.frames
        results = []
        fraction = int(np.floor(n / nparts))
        for i in range(nparts):
            start = fraction * i
            stop = fraction * (i + 1)
            msd_analysis.run(start=start, stop=stop)
            #print(start,stop)
            msd_results = msd_analysis.results.timeseries
            nframes = msd_analysis.n_frames
            lagtimes = np.arange(nframes) * timestep_us  # make the lag-time axis
            results += [msd_results * distance_um ** 2]
        return np.array([lagtimes] + results).T

    @staticmethod
    def general_diffusion(t, D, alpha):
        return 6 * D / 1000 * t ** alpha

    def compute_general_diffusion(self, nparts=5, min_frames=10, std_error_cutoff=0.05, plot=False):
        import scipy.optimize

        test = self.compute_msd(nparts=nparts, fft=True,)


        std_error = (test[min_frames:, 1:].std(axis=1) / test[min_frames:, 1:].mean(axis=1))
        for max_i, e in enumerate(std_error):
            if e > std_error_cutoff:
                break
        max_i += min_frames

        x_0 = test[:max_i, 0]
        x = np.concatenate([x_0] * (test.shape[1] - 1))
        y_0 = test[:max_i, 1:]
        y = y_0.T.ravel()
        #Initial guess
        D_0 = np.mean(y_0[-1]/x_0[-1]/6*1E3)
        alpha_0 = 1
        #Optimization
        parameters, covariance = scipy.optimize.curve_fit(self.general_diffusion, x, y, p0=[D_0, alpha_0])
        #Plot
        if plot:
            for y_i in y_0.T:
                plt.scatter(x_0, y_i, s=0.1)
            plt.plot(x_0, self.general_diffusion(x_0, *parameters))
            plt.xlabel(r'$\mu$s')
            plt.ylabel(r'$\mu$m')
        return parameters

    def compute_diffussion(self, plot=False):
        """
        Calculates the diffusion constant in nm2/s
        """
        import openmm
        import MDAnalysis as mda
        import MDAnalysis.analysis.msd
        import sklearn.linear_model
        cif_openmm = openmm.app.pdbxfile.PDBxFile(self.simulation['cif'], )
        u = mda.Universe(cif_openmm.getTopology(), self.simulation['dcd'], topology_format='OPENMMTOPOLOGY')
        timestep_us = int(self.simulation['frequency']) / 10 ** 6
        distance_um = 1 / 10 * 1 / 1000
        msd_analysis = mda.analysis.msd.EinsteinMSD(u, select='all', fft=True)
        data = []
        nparts = 5
        n = self.simulation.frames
        for i in range(nparts):
            start = n // nparts * i
            stop = n // nparts * (i + 1) if nparts - i > 1 else n
            msd_analysis.run(start=start, stop=stop)
            #print(start, stop)
            msd_results = msd_analysis.results.timeseries
            nframes = msd_analysis.n_frames
            lagtimes = np.arange(nframes) * timestep_us  # make the lag-time axis
            if plot:
                plt.plot(lagtimes, msd_results ** .5 * distance_um, "-", c="black", label=r'3D random walk')
            data += [np.array([lagtimes, msd_results * distance_um ** 2])]
        lr = sklearn.linear_model.LinearRegression()
        data = np.concatenate(data, axis=1).T  # [:20]
        for i in range(1, len(lagtimes)):
            sel = data[data[:, 0] == lagtimes[i], 1]
            if np.std(sel) / np.mean(sel) > 0.05:
                break
        max_i = i
        sel = data[data[:, 0] <= lagtimes[max_i]]
        lr.fit(sel[:, 0:-1], sel[:, -1:])
        diffusion_constant = lr.coef_[0][0] * 1000
        # print(max_i, diffusion_constant, 'nm2/s', 'um2/us')
        x = np.array([lagtimes]).T
        if plot:
            plt.plot(x, lr.predict(x) ** .5)
            plt.xlabel(r'$\mu$s')
            plt.ylabel(r'$\mu$m')
        return diffusion_constant/6

    def compute_persistence_length(self):
        raise NotImplementedError
