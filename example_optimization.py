# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from pyworld3 import World3

from math import exp

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import generate_n2_plot
from numpy import array
from numpy import onesfrom 
from pyworld3.utils import plot_world_variables

f_obj = None
design_space = None
scenario = None
disciplines = []

def run_sim(X):
    """
    X is expected as a dict 
    """
    world3 = World3()
    world3.init_world3_constants(**X)
    world3.init_world3_variables()
    world3.set_world3_table_functions()
    world3.set_world3_delay_functions()
    world3.run_world3(fast=False)
    return world3

def obj1(w3):
    return w3.ppolx

def f_world3():
    pass

def main():
    disc_world3 = create_discipline("AutoPyDiscipline", py_func=f_world3)
    disciplines = [disc_world3]

    design_space = create_design_space()
    design_space.add_variable("x_local", l_b=0.0, u_b=10.0, value=ones(1))


    scenario = create_scenario(
        disciplines,
        formulation="MDF",
        inner_mda_name="MDAGaussSeidel",
        objective_name="obj",
        design_space=design_space,
    )

    scenario.add_constraint("c_1", "ineq")
    scenario.add_constraint("c_2", "ineq")

    scenario.execute(input_data={"max_iter": 10, "algo": "SLSQP"})



if '__main__' == __name__:
    main()
