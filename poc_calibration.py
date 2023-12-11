import copy
import logging

from gemseo import create_design_space, create_scenario, configure_logger
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo_calibration import calibrator
import numpy as np

#### Calibration section:
class Calibration(MDODiscipline):
    def __init__(self, data, output_name):
        super().__init__() # init the MDODiscipline parent class
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
            logger.debug("A[i]: %s, self.data[data_name] %s", A[i], self.data[data_name])
        self.local_data[self.output_name] = np.array([np.linalg.norm(A)])


class FakeSimulation(MDODiscipline):
    def __init__(self):
        super().__init__() # init the MDODiscipline parent class
        # self.input_grammar.update_from_names(["in11", "in12", "in13", "in21", "in22", "in23"])
        self.input_grammar.update_from_names(["in11", "in21"])
        self.output_grammar.update_from_names(["out1", "out2"])
        self.n = 2
        self.d = 2
        
    def _run(self):
        in11 = self.local_data["in11"][0]
        # in12 = self.local_data["in12"][0]
        # in13 = self.local_data["in13"][0]
        in21 = self.local_data["in21"][0]
        # in22 = self.local_data["in22"][0]
        # in23 = self.local_data["in23"][0]
        self.local_data["out1"] = np.array([in11*x**2 for x in range(self.d)])
        self.local_data["out2"] = np.array([in21*x**2 for x in range(self.d)])
        logger.debug("local data out1 : %s", self.local_data["out1"])
        logger.debug("local data out2 : %s", self.local_data["out2"])


# Doesn't work:
# fake_sim = AnalyticDiscipline({"out1": "in11*x*x", "out2": "in21*x*x"}, name="fake_sim")
# fake_sim.execute({"x": np.array([1.0])})
# fake_sim.execute({"x": np.array([2.0])})
# fake_sim.set_cache_policy(fake_sim.CacheType.MEMORY_FULL)

fake_sim = FakeSimulation()


def main():
    data = {
        "out1": np.array([ 5*x**2 for x in range(2)]),
        "out2": np.array([-5*x**2 for x in range(2)])
    }
    logger.debug("real data: %s", data)
    disc = [fake_sim, Calibration(data, "obj")]

    design_space = create_design_space()
    for key in ["in11", "in12", "in13", "in21", "in22", "in23"]:
        design_space.add_variable(key, 1, l_b=-10, u_b=10, value=1)

    scenario = create_scenario(disc, "MDF", "obj", design_space)
    scenario.set_differentiation_method("finite_differences", 0.01)

    params = {"max_iter":10, "algo":"SLSQP"}

    scenario.execute(input_data=params)

logger = configure_logger(level=logging.DEBUG)
if __name__ == "__main__":
    main()
