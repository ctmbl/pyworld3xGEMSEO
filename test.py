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
from gemseo import create_design_space, create_scenario
from numpy import ones

from pyworld3 import World3, Population, Pollution, Agriculture, Capital, Resource, hello_world3


def f_obj(model: World3):
    return max(model.ppol)

class World3_D(MDODiscipline):

    def __init__(self,residual_form=False):
        super(World3, self).__init__()
        self.input_grammar.update_from_names([])
        self.output_grammar.update_from_names(['obj'])

    def _run(self):
        _world3 = World3()
        _world3.init_world3_constants(**self.local_data)
        _world3.init_world3_variables()
        _world3.set_world3_table_functions()
        _world3.set_world3_delay_functions()
        _world3.run_world3(fast=True)

        self.local_data['obj'] = f_obj(_world3)


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


class Resource_D(MDODiscipline,Resource):

    def __init__(self,residual_form=False):
        super(Resource,self).__init__()
        self.input_grammar.update_from_names() #à compléter
        self.output_grammar.update_from_names() #à compléter

    def _run(self):
        #à compléter
        pass



#disc = [Resource_D(), Capital_D(), Pollution_D(), Population_D(), Agriculture_D(), World3_D()]
disc = [World3_D()]

design_space = create_design_space()
design_space.add_variable("zpgt", 1, l_b=1900, u_b=4000, value=ones(1))

scenario = create_scenario(disc, "MDF", "obj", design_space)
scenario.set_differentiation_method("finite_differences", 1e-4)

params = {"max_iter":1000}

scenario.execute(input_data=params)
