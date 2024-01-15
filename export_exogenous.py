# -*- coding: utf-8 -*-
from pyworld3 import World3
from pyworld3.utils import plot_world_variables

def get_exogenous_data(to_export):
    world3 = World3()
    world3.init_world3_constants()
    world3.init_world3_variables()
    world3.set_world3_table_functions()
    world3.set_world3_delay_functions()
    world3.run_world3(fast=False)

    data = {}
    for d in to_export:
        data[d] = getattr(world3, d)

    return data
