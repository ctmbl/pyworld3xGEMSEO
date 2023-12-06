"""
Copyright or © or Copr. Emile DOSSO, Clément MABILEAU (12 November 2023) 

emile.dosso@student.isae-supaero.fr;clement.mabileau@student.isae-supaero.fr

This software is a computer program whose purpose is to define and calibrate the 
disciplines of the model World3 in an MDO context.

This software is governed by the CeCILL  license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

from gemseo.core.discipline import MDODiscipline
from gemseo import create_design_space, create_scenario, configure_logger
from numpy import array, linalg

from pyworld3 import World3, Population, Pollution, Agriculture, Capital, Resource, hello_world3

"""
TODO: 
separate calibration in (at least) two parts/disciplines:
    - a "simulation" part that runs pyworld3 model and output every data possible
    - a "calibration" part that computes from the 1st the objective fucntion (with respect to calibration data) 
      for this second one a GEMSEO plugin already exists to provide the appropriate Discipline

separate the World3_D discipline into 5 sectors
each sectors will take data as input and output some other data, this will be coupling between disciplines
this will allow us to calibrate each sectors independently of the other
"""

world3_output = ["ppolx", # Pollution sector 
                 "pop",   # Population sector
                 "nr", "nrfr", # Resource sector
                 "al", "pal", "f", "fpc", # Agriculture sector
                 "ic", "io" # Capital sector
                ]
world3_input = ["zpgt"]

class Simulation(MDODiscipline):

    def __init__(self,residual_form=False):
        super().__init__()
        self.input_grammar.update_from_names(world3_input)
        self.output_grammar.update_from_names(world3_output)

    def _run(self):
        params = {}
        for input in world3_input:
            params[input] = next(self.get_inputs_by_name([input]))

        _world3 = World3()
        _world3.init_world3_constants(**params)
        _world3.init_world3_variables()
        _world3.set_world3_table_functions()
        _world3.set_world3_delay_functions()
        _world3.run_world3(fast=True)

        for out in world3_output:
            self.local_data[out] = getattr(_world3, out)


class Calibration(MDODiscipline):

    def __init__(self, residual_form = False):
        super().__init__()
        self.input_grammar.update_from_names(world3_output)
        self.output_grammar.update_from_names(['obj'])

    def _run(self):
        ppolx = self.get_inputs_by_name(["ppolx"])

        
        obj = linalg.norm(ppolx)
        self.local_data['obj'] = array([obj])


class Population_D(MDODiscipline,Population):

    def __init__(self,residual_form=False):
        super(Population,self).__init__()
        self.input_grammar.update_from_names() #à compléter
        self.output_grammar.update_from_names() #à compléter

    def _run(self):
        #à compléter
        pass



class Capital_D(MDODiscipline,Capital):

    def __init__(self,residual_form=False):
        super(Capital,self).__init__()
        self.input_grammar.update_from_names() #à compléter
        self.output_grammar.update_from_names() #à compléter
    def _run(self):
        #à compléter
        pass



class Agriculture_D(MDODiscipline,Agriculture):

    def __init__(self,residual_form=False):
        super(Agriculture,self).__init__()
        self.input_grammar.update_from_names() #à compléter
        self.output_grammar.update_from_names() #à compléter
    def _run(self):
        #à compléter
        pass



class Pollution_D(MDODiscipline,Pollution):

    def __init__(self,residual_form=False):
        super(Pollution,self).__init__()
        self.input_grammar.update_from_names() #à compléter
        self.output_grammar.update_from_names() #à compléter
    def _run(self):
        #à compléter
        pass


class Resource_D(MDODiscipline):

    def __init__(self,residual_form=False):
        super().__init__()
        self.inputs_list = ["nri", "nruf1", "nruf2"]
        self.input_grammar.update_from_names(self.inputs_list) #à compléter
        self.output_grammar.update_from_names() #à compléter

    def _run(self):
        inputs = self.get_inputs_by_name(self.inputs_list)
        nri = inputs[0]
        nruf1 = inputs[1]
        nruf2 = inputs[2]
        print(inputs)

        _res = Resource()
        _res.init_resource_constants(*inputs)
        _res.init_resource_variables()
        _res.set_resource_table_functions(json_file= None) # edit the none to a path to a json file describing table function
        _res.set_resource_delay_functions(method="euler")
        _res.loop0_resource()

        for k_ in range(1, self.n):
            _res.loopk_resource(k_-1, k_, k_-1, k_)

configure_logger()

#disc = [Resource_D(), Capital_D(), Pollution_D(), Population_D(), Agriculture_D(), World3_D()]
disc = [Simulation(), Calibration()]

design_space = create_design_space()
design_space.add_variable("zpgt", 1, l_b=1900, u_b=4000, value=array([2050]))

scenario = create_scenario(disc, "MDF", "obj", design_space)
scenario.set_differentiation_method("finite_differences", 20)

params = {"max_iter":10, "algo":"SLSQP"}

scenario.execute(input_data=params)
