import copy
import logging

from gemseo.core.discipline import MDODiscipline
import numpy as np

class Calibration(MDODiscipline):
    def __init__(self, data_dict, output_name):
        super().__init__() # init the MDODiscipline parent class
        self.logger = logging.getLogger(__name__)
        """ Example of data format:
        data = {
            "fake_p1": {
                    "offset": 60, # beginning year-1900: 1950 -> 50
                    "data": np.full((40,), 2e9) # this is an example ofc
                }, 
        }
        """

        self.data_names = list(data_dict.keys())
        self.data_dict = data_dict
        self.output_name = output_name
        self.n = len(self.data_names)

        self.input_grammar.update_from_names(self.data_names)
        self.output_grammar.update_from_names([output_name])

    def _run(self):
        norms = []
        for data_name in self.data_names:
            start = self.data_dict[data_name]["offset"]
            end = start + len(self.data_dict[data_name]["data"])
            diff = self.data[data_name]["data"] - self.local_data[data_name][start:end]

            norms.append(np.linalg.norm(diff))
            self.logger.debug("A[i]=%s, diff=%s", norms[-1], diff)
        obj = np.linalg.norm(norms)
        self.local_data[self.output_name] = np.array([obj])

