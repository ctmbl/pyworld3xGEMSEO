# -*- coding: utf-8 -*-
from pyworld3 import World3
from pyworld3.utils import plot_world_variables

world3 = World3()
world3.init_world3_constants()
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()
world3.run_world3(fast=False)

to_export = ["io", "p1"]
data = ""
for d in to_export:
    data_as_str = str(getattr(world3, d)).replace(" ", ", ")
    data += f"{d} = {data_as_str} \n\n"

with open("exogenous_data.py", "w") as f:
    f.write(data)
