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
from numpy import ones, linalg

import logging

from pyworld3 import World3


def f_obj(model: World3):
    return linalg.norm(model.ppol)

class World3_D(MDODiscipline):

    def __init__(self,residual_form=False):
        super().__init__()
        self.input_var_names = ["zpgt"]
        self.input_grammar.update_from_names(self.input_var_names)
        self.output_grammar.update_from_names(['obj'])

    def _run(self):
        input_vars_generator = self.get_inputs_by_name(self.input_var_names)
        input_var_values = next(input_vars_generator)
        logger.info("generator: %s, values: %s", input_vars_generator, input_var_values)
        
        _world3 = World3()
        _world3.init_world3_constants(zpgt=input_var_values[0])
        _world3.init_world3_variables()
        _world3.set_world3_table_functions()
        _world3.set_world3_delay_functions()
        _world3.run_world3(fast=True)

        self.local_data['obj'] = [f_obj(_world3)]


logger = configure_logger(level=logging.INFO)
disc = [World3_D()]

design_space = create_design_space()
design_space.add_variable("zpgt", 1, l_b=1900, u_b=4000, value=ones(1)*2000)

scenario = create_scenario(disc, "MDF", "obj", design_space)
scenario.set_differentiation_method("finite_differences", 1e-4)

params = {"max_iter":1000, "algo":"SLSQP"}

scenario.execute(input_data=params)
