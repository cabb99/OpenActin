import os
import warnings
import itertools
import pandas
import time


class SlurmJobArray():
    """ Selects a single condition from an array of parameters using the SLURM_ARRAY_TASK_ID environment variable.
        The parameters need to be supplied as a dictionary. if the task is not in a slurm environment,
        the test parameters will supersede the parameters, and the job_id would be taken as 0.  Example:
        parameters={"epsilon":[100],
                    "aligned":[True,False],
                    "actinLen":[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300],
                    "repetition":range(5),
                    "temperature":[300],
                    "system2D":[False],
                    "simulation_platform":["OpenCL"]}
        test_parameters={"simulation_platform":"CPU"}
        sjob=SlurmJobArray("ActinSimv6", parameters, test_parameters)
        :var test_run: Boolean: This simulation is a test
        :var job_id: SLURM_ARRAY_TASK_ID
        :var all_parameters: Parameters used to initialize the job
        :var parameters: Parameters for this particular job
        :var name: The name (and relative path) of the output
    """

    def __init__(self, name, parameters, test_parameters={}, test_id=0):

        self.all_parameters = parameters
        self.test_parameters = test_parameters

        # Parse the slurm variables
        self.slurm_variables = {}
        for key in os.environ:
            if len(key.split("_")) > 1 and key.split("_")[0] == 'SLURM':
                self.slurm_variables.update({key: os.environ[key]})

        # Check if there is a job id
        self.test_run = False
        try:
            self.job_id = int(self.slurm_variables["SLURM_ARRAY_TASK_ID"])
        except KeyError:
            self.test_run = True
            warnings.warn("Test Run: SLURM_ARRAY_TASK_ID not in environment variables")
            self.job_id = test_id

        keys = parameters.keys()
        self.all_conditions = list(itertools.product(*[parameters[k] for k in keys]))
        self.parameter = dict(zip(keys, self.all_conditions[self.job_id]))

        # The name only includes enough information to differentiate the simulations.
        self.name = f"{name}_{self.job_id:03d}_" + '_'.join(
            [f"{a[0]}_{self[a]}" for a in self.parameter if len(self.all_parameters[a]) > 1])

    def __getitem__(self, name):
        if self.test_run:
            try:
                return self.test_parameters[name]
            except KeyError:
                return self.parameter[name]
        else:
            return self.parameter[name]

    def __getattr__(self, name: str):
        """ The keys of the parameters can be called as attributes
        """
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        elif name in self.parameter:
            return self[name]
        else:
            return object.__getattribute__(self, name)

    def __repr__(self):
        return str(self.parameter)

    def keys(self):
        return str(self.parameters.keys())

    def print_parameters(self):
        print(f"Number of conditions: {len(self.all_conditions)}")
        print("Running Conditions")
        for k in self.parameter.keys():
            print(f"{k} :", f"{self[k]}")
        print()

    def print_slurm_variables(self):
        print("Slurm Variables")
        for key in self.slurm_variables:
            print(key, ":", self.slurm_variables[key])
        print()

    def write_csv(self, out=""):
        s = pandas.concat([pandas.Series(self.parameter), pandas.Series(self.slurm_variables)])
        s['test_run'] = self.test_run
        s['date'] = time.strftime("%Y_%m_%d")
        s['name'] = self.name
        s['job_id'] = self.job_id

        if out == '':
            s.to_csv(self.name + '.param')
        else:
            s.to_csv(out)
