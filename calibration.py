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

import copy

from gemseo.core.discipline import MDODiscipline
from gemseo import create_design_space, create_scenario, configure_logger
import numpy as np

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

        
        obj = np.linalg.norm(ppolx)
        self.local_data['obj'] = np.array([obj])


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


        for k_ in range(1, self.n):
            _pol.loopk_pollution(k_-1, k_, k_-1, k_)


"""
Description de l'idée:

input_var_dict est le dictionnaire des variables d'optimisation possible, pour l'instant aucun tri
n'a été fait et il faudra enlever les variables "non calibrable" du style des initialisations de stocks
la clé du dict est le nom de la var d'optim (var de design du design space de GEMSEO) et elle est 
associé à un dict stockant: l_b la lower bound, u_p la upper bound, value, la valeur initiale
pour faciliter la génération du design space cf plus bas le code:

design_space = create_design_space()
for key, values in Resource_D.input_var_dict:
    design_space.add_variable(key, 1, **values)


exogeneous_var_dict représente le dictionnaire pour définir les vairables éxogènes sachant qu'on ne veut pas utiliser les méthodes de calcul
données avec l'option Alone = True, car sinon on utiliserait les courbes par défaut de pyworld3 
Elles ont été choisies directement à partir des variables utilisées dans loop0_exogenous dans Resource de pyworld3
On a besoin des infos données par les modéliseurs pour définir les valeurs, et on les récupère avec ces lignes de code :
for exo_var in self.exogeneous_var_names:
            assert Resource_D.exogeneous_var_dict[exo_var].shape[0] >= self.n
            setattr(self, exo_var, Resource_D.exogeneous_var_dict[exo_var])


"""
class Resource_D(MDODiscipline, Resource):
    input_var_dict = {
        "nri":   {"l_b":la lower bound, "u_b":la upper bound, "value":np.array([la val initiale])}
        "nruf1": {"l_b":la lower bound, "u_b":la upper bound, "value":np.array([la val initiale])}
        "nruf2": {"l_b":la lower bound, "u_b":la upper bound, "value":np.array([la val initiale])}
    }
    exogeneous_var_dict = {
        "pop": np.full(), 
        "pop1": ,
        "ic": , 
        "icir": , 
        "icdr": , 
        "io": , 
        "iopc":
    }
    output_var_names = ["nr", "nrfr", "nruf", "nrur", "pcrum", "fcaor", "fcaor1", "fcaor2"]

    def __init__(self,residual_form=False, year_min=1900, year_max=2100, dt=1, pyear=1975, verbose=False):
        super(self, MDODiscipline).__init__() # init the MDODiscipline parent class
        super(self, Resource).__init__(year_min=year_min, year_max=year_max, dt=dt, pyear=pyear, verbose=verbose) # init the parent Resource class

        self.input_var_names = list(Resource_D.input_var_dict.keys())
        self.exogeneous_var_names = list(Resource_D.exogeneous_var_dict.keys())
        self.output_var_names = Resource_D.output_var_names

        self.input_grammar.update_from_names(self.input_var_names)
        self.output_grammar.update_from_names(self.output_var_names)
        
    def _run(self):
        input_vars_generator = self.get_inputs_by_name(self.input_var_names)
        input_var_values = next(input_vars_generator)
        assert len(input_var_values) == len(self.input_var_names)

        input_var_dict = {self.input_var_names[i]: input_var_values[i] for i in range(len(input_var_values))}
        logger.critical("generator: %s, values: %s", input_vars_generator, input_var_values)

        self.init_resource_constants(**input_var_dict)
        self.init_resource_variables()
        self.set_resource_table_functions(json_file= None) # edit the none to a path to a json file describing table function
        self.set_resource_delay_functions(method="euler")

        #_res.init_exogenous_inputs() # useless c'est set par la boucle suivante:
        for exo_var in self.exogeneous_var_names:
            assert Resource_D.exogeneous_var_dict[exo_var].shape[0] >= self.n
            setattr(self, exo_var, Resource_D.exogeneous_var_dict[exo_var])

        #_res.loop0_resource(alone=True) # --> on veut pas faire ca justement car on utiliserait les fausses courbes de pop1 par defaut de pyworld3
        self.loop0_resource(alone=False)


        for k_ in range(1, self.n):
            self.loopk_resource(k_-1, k_, k_-1, k_, alone=False)


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
            logger.debug("A[i]: %s | data[%s] %s", A[i], data_name, self.data[data_name])
        self.local_data[self.output_name] = np.array([np.linalg.norm(A)])

logger = configure_logger()

#disc = [Resource_D(), Capital_D(), Pollution_D(), Population_D(), Agriculture_D(), World3_D()]
disc = [Resource_D(year_min=1900, year_max=2100, dt=1), Calibration()]

design_space = create_design_space()
for key, values in Resource_D.input_var_dict:
    design_space.add_variable(key, 1, **values)

scenario = create_scenario(disc, "MDF", "obj", design_space)
scenario.set_differentiation_method("finite_differences", 20)

params = {"max_iter":10, "algo":"SLSQP"}

scenario.execute(input_data=params)
