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


class Population_D(MDODiscipline):
        
    input_var_dict = {
        "p1i":   {"l_b":, "u_b":, "value":}
        "p2i": {"l_b":, "u_b":, "value":}
        "p3i": {"l_b":, "u_b":, "value":}
        "p4i": {"l_b":, "u_b":, "value":}
        "dcfsn":{"l_b":, "u_b":, "value":} #choisi pour l'exemple de calibration
        "fcest": {"l_b":, "u_b":, "value":}
        "hsid": {"l_b":, "u_b":, "value":}
        "ieat": {"l_b":, "u_b":, "value":}
        "len": {"l_b":, "u_b":, "value":} #choisi pour l'exemple de calibration
        "lpd": {"l_b":, "u_b":, "value":}
        "mtfn": {"l_b":, "u_b":, "value":}
        "pet": {"l_b":, "u_b":, "value":}
        "rlt": {"l_b":, "u_b":, "value":}
        "sad": {"l_b":, "u_b":, "value":}
        "zpgt": {"l_b":, "u_b":, "value":}

        }
    exogeneous_var_dict = {
        # industrial output
        "io" : np.full(),
        "io1" : np.full(),
        "io11" : np.full(),
        "io12" : np.full(),
        "io2" : np.full(),
        "iopc" : np.full(),
        # index of persistent pollution
        "ppolx" : np.full(),
        # service output
        "so" : np.full(),
        "so1" : np.full(),
        "so11" : np.full(),
        "so12" : np.full(),
        "so2" : np.full(),
        "sopc" : np.full(),
        # food
        "f" : np.full(),
        "f1" : np.full(),
        "f11" : np.full(),
        "f12" : np.full(),
        "f2" : np.full(),
        "fpc" : np.full()
    }
    output_var_names : ["pop","p1","p2","p3","p4","d1","d2","d3","d4","mat1","mat2","mat3","d","cdr",
                        "fpu","le","lmc","lmf","lmhs","lmhs1","lmhs2","lmp","m1","m2","m3","m4","ehspc",
                        "hsapc","b","cbr","cmi","cmple","tf","dtf","dcfs","fce","fie","fm","frsn","mtf"
                        ,"nfc","ple","sfsn","aiopc","diopc","fcapc","fcfpc","fsafc"]

    def __init__(self, residual_form=False, year_min=1900, year_max=2100, dt=1, pyear=1975, verbose=False):
        super(self, MDODiscipline).__init__() # init the MDODiscipline parent class
        self.population_init_params = {"year_min":year_min, "year_max":year_max, "dt":dt, "pyear":pyear, "verbose":verbose}

        self.input_var_names = list(Population_D.input_var_dict.keys())
        self.exogeneous_var_names = list(Population_D.exogeneous_var_dict.keys())
        self.output_var_names = Population_D.output_var_names

        self.input_grammar.update_from_names(self.input_var_names)
        self.output_grammar.update_from_names(self.output_var_names)
          
    def _run(self):
            # shouldn't we use self.local_data[] instead?
        input_vars_generator = self.get_inputs_by_name(self.input_var_names) # wtf?
        input_var_values = next(input_vars_generator)
        logger.debug("Values: %s", input_var_values)
        assert len(input_var_values) == len(self.input_var_names)

        _pop = Population(**self.population_init_params)

        input_var_dict = {_pop.input_var_names[i]: input_var_values[i] for i in range(len(input_var_values))}
        logger.critical("generator: %s, values: %s", input_vars_generator, input_var_values)

        _pop.init_population_constants(**input_var_dict)
        _pop.init_population_variables()
        _pop.set_population_table_functions(json_file= None) # edit the none to a path to a json file describing table function
        _pop.set_population_delay_functions(method="euler")

            #_pop.init_exogenous_inputs() # useless c'est set par la boucle suivante:
        for exo_var in _pop.exogeneous_var_names:
            assert Population_D.exogeneous_var_dict[exo_var].shape[0] >= _pop.n
            setattr(_pop, exo_var, Population_D.exogeneous_var_dict[exo_var])

                #_pop.loop0_population(alone=True) # --> on veut pas faire ca justement car on utiliserait les fausses courbes de pop1 par defaut de pyworld3
            _pop.loop0_population(alone=False)

        for k_ in range(1, _pop.n):
            _pop.loopk_population(k_-1, k_, k_-1, k_, alone=False)

        for output in population_D.output_var_names:
            self.local_data[output] = getattr(_pop, output).copy()



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
class Resource_D(MDODiscipline):
    input_var_dict = {
        "nri":   {"l_b":la lower bound, "u_b":la upper bound, "value":np.array([la val initiale])}
        "nruf1": {"l_b":la lower bound, "u_b":la upper bound, "value":np.array([la val initiale])}
        "nruf2": {"l_b":la lower bound, "u_b":la upper bound, "value":np.array([la val initiale])}
    }
    exogeneous_var_dict = {
        "pop": np.full(), 
        "pop1": np.full(),
        "ic": np.full(), 
        "icir": np.full(), 
        "icdr": np.full(), 
        "io": np.full(), 
        "iopc": np.full()
    }
    output_var_names = ["nr", "nrfr", "nruf", "nrur", "pcrum", "fcaor", "fcaor1", "fcaor2"]

    def __init__(self, residual_form=False, year_min=1900, year_max=2100, dt=1, pyear=1975, verbose=False):
        super(self, MDODiscipline).__init__() # init the MDODiscipline parent class
        self.resource_init_params = {"year_min":year_min, "year_max":year_max, "dt":dt, "pyear":pyear, "verbose":verbose}

        self.input_var_names = list(Resource_D.input_var_dict.keys())
        self.exogeneous_var_names = list(Resource_D.exogeneous_var_dict.keys())
        self.output_var_names = Resource_D.output_var_names

        self.input_grammar.update_from_names(self.input_var_names)
        self.output_grammar.update_from_names(self.output_var_names)
        
    def _run(self):
        # shouldn't we use self.local_data[] instead?
        input_vars_generator = self.get_inputs_by_name(self.input_var_names) # wtf?
        input_var_values = next(input_vars_generator)
        logger.debug("Values: %s", input_var_values)
        assert len(input_var_values) == len(self.input_var_names)

        _res = Resource(**self.resource_init_params)

        input_var_dict = {_res.input_var_names[i]: input_var_values[i] for i in range(len(input_var_values))}
        logger.critical("generator: %s, values: %s", input_vars_generator, input_var_values)

        _res.init_resource_constants(**input_var_dict)
        _res.init_resource_variables()
        _res.set_resource_table_functions(json_file= None) # edit the none to a path to a json file describing table function
        _res.set_resource_delay_functions(method="euler")

        #_res.init_exogenous_inputs() # useless c'est set par la boucle suivante:
        for exo_var in _res.exogeneous_var_names:
            assert Resource_D.exogeneous_var_dict[exo_var].shape[0] >= _res.n
            setattr(_res, exo_var, Resource_D.exogeneous_var_dict[exo_var])

        #_res.loop0_resource(alone=True) # --> on veut pas faire ca justement car on utiliserait les fausses courbes de pop1 par defaut de pyworld3
        _res.loop0_resource(alone=False)


        for k_ in range(1, _res.n):
            _res.loopk_resource(k_-1, k_, k_-1, k_, alone=False)

        for output in Resource_D.output_var_names:
            self.local_data[output] = getattr(_res, output).copy() # equ to: self.local_data["nr"] = _res.nr
            # equivalent to: (but simpler)
            # self.local_data["nr"] = _res.nr.copy()
            # self.local_data["nrfr"] = _res.nrfr.copy()
            # self.local_data["nruf"] = _res.nruf.copy()
            # etc...


#### Calibration section:


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
