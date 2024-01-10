import copy
import logging

from gemseo.core.discipline import MDODiscipline
import numpy as np

class Calibration(MDODiscipline):
    def __init__(self, data, output_name):
        super().__init__() # init the MDODiscipline parent class
        self.logger = logging.getLogger(__name__)

        self.data_names = list(data.keys())
        self.data = copy.deepcopy(data)
        self.output_name = output_name
        self.n = len(self.data_names)
        self.d = len(self.data[self.data_names[0]])

        self.input_grammar.update_from_names(self.data_names)
        self.output_grammar.update_from_names([output_name])

    def _run(self):
        A = np.empty((self.n, self.d))
        for i in range(self.n):
            data_name = self.data_names[i]
            A[i] = self.data[data_name] - self.local_data[data_name]
            self.logger.debug("A[i]: %s, self.data[data_name] %s", A[i], self.data[data_name])
        self.local_data[self.output_name] = np.array([np.linalg.norm(A)])

