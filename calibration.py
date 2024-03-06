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

import logging

from gemseo.core.discipline import MDODiscipline
from gemseo import create_design_space, create_scenario, configure_logger
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd

from pyworld3 import World3, Population, Pollution, Agriculture, Capital, Resource, hello_world3
from calibration_d import Calibration
from export_exogenous import get_exogenous_data


class Population_D(MDODiscipline):
        
    all_input_parameters_names = [
        "p1i",
        "p2i",
        "p3i",
        "p4i",
        "dcfsn",
        "fcest",
        "hsid",
        "ieat",
        "len",
        "lpd",
        "mtfn",
        "pet",
        "rlt", 
        "sad",
        "zpgt",
    ]
    exogenous_data_names = [
        # industrial output
        "io",
        "iopc",
        # index of persistent pollution
        "ppolx",
        # service output
        "so",
        "sopc",
        # food
        "f",
        "fpc"
    ]
    output_data_names = ["pop","p1","p2","p3","p4","d1","d2","d3","d4","mat1","mat2","mat3","d","cdr",
                        "fpu","le","lmc","lmf","lmhs","lmhs1","lmhs2","lmp","m1","m2","m3","m4","ehspc",
                        "hsapc","b","cbr","cmi","cmple","tf","dtf","dcfs","fce","fie","fm","frsn","mtf"
                        ,"nfc","ple","sfsn","aiopc","diopc","fcapc","fcfpc","fsafc"]

    def __init__(self, exogenous_data_dict, input_variables_names, input_parameters_fixed_dict, residual_form=False, year_min=1900, year_max=2100, dt=1, verbose=False):
        super().__init__() # init the MDODiscipline parent class
        self.logger = logging.getLogger("population")
        self.population_init_parameters = {"year_min":year_min, "year_max":year_max, "dt":dt, "verbose":verbose}

        self.input_variables_names = input_variables_names
        self.input_parameters_fixed_dict = input_parameters_fixed_dict
        assert list(exogenous_data_dict.keys()) == Population_D.exogenous_data_names
        self.exogenous_data_dict = exogenous_data_dict
        self.output_data_names = Population_D.output_data_names

        self.input_grammar.update_from_names(self.input_variables_names)
        self.output_grammar.update_from_names(self.output_data_names)
          
    def _run(self):
        ### Instantiate the Population class
        _pop = Population(**self.population_init_parameters)
        # TODO: réfléchir à faire ca mieux:
        _pop.sfpc = 230 # nécessaire pour faire tourner le secteur seul, paramètre scalaire de agriculture

        ### Initialize the exogenous parameters (to the Population sector)

        #_pop.init_exogenous_inputs() # useless c'est set par la boucle suivante:
        for exo_data in Population_D.exogenous_data_names:
            assert self.exogenous_data_dict[exo_data].shape[0] >= _pop.n
            setattr(_pop, exo_data, self.exogenous_data_dict[exo_data])
            self.logger.debug("Set the exogenous data '%s'", exo_data)

        ### Get the model parameters from GEMSEO --> this is what we're calibrating
        input_dict = {key: value[0] for key, value in self.local_data.items() if key in self.input_variables_names}
        input_dict.update(self.input_parameters_fixed_dict)
        self.logger.debug("input_dict=%s", input_dict)

        _pop.init_population_constants(**input_dict)
        _pop.init_population_variables()
        _pop.set_population_table_functions(json_file= None) # edit the none to a path to a json file describing table function
        _pop.set_population_delay_functions(method="euler")

        #_pop.loop0_population(alone=True) # --> on veut pas faire ca justement car on utiliserait les fausses courbes de pop1 par defaut de pyworld3
        _pop.loop0_population(alone=False)


        ### Run the model
        for k_ in range(1, _pop.n):
            _pop.loopk_population(k_-1, k_, k_-1, k_, alone=False)


        ### Export output data (to be used by the Calibration discipline)
        for output in Population_D.output_data_names:
            self.local_data[output] = getattr(_pop, output).copy()



class Animation(MDODiscipline):
    def __init__(self, data, data_names):
        super().__init__() # init the MDODiscipline parent class
        self.logger = logging.getLogger(__name__)

        self.data_names = data_names
        self.n_data = len(data_names)
        self.data = data
        self.input_grammar.update_from_names(data_names)

        ncols = 4
        nrows = int(np.ceil(self.n_data // ncols))
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols)
        self.axes = self.axes.flatten()

        self.artists = []

    def _run(self):
        all_data = [self.local_data[data].copy() for data in self.data_names]
        self.artists.append(all_data)

    def show(self):
        # Heavily inspired from:
        # https://matplotlib.org/stable/gallery/widgets/slider_demo.html

        n_step = len(self.artists)

        # Make room for the slider
        self.fig.subplots_adjust(left=0.1, bottom=0.20)

        # Slider
        ax_slider = self.fig.add_axes([0.20, 0.1, 0.65, 0.03])
        step_slider = Slider(
            ax=ax_slider,
            label="Optimisation",
            valmin=1,
            valmax=n_step,
            valinit=1,
            orientation="horizontal"
        )

        # Plot simulated data
        lines = []
        for i in range(self.n_data):
                lines.extend(self.axes[i].plot(self.artists[0][i], color="blue"))

        # Update the plot on slider changes
        def update(val):
            for i in range(self.n_data):
                lines[i].set_ydata(self.artists[int(step_slider.val)-1][i])
            self.fig.canvas.draw_idle()
        step_slider.on_changed(update)

        # Plot the real data

        for i in range(self.n_data):
            data_name = self.data_names[i]

            start = self.data[data_name]["offset"]
            end = self.data[data_name]["offset"] + len(self.data[data_name]["data"])

            self.axes[i].plot(range(start, end), self.data[data_name]["data"], color="red", label="Real data")
            self.axes[i].set_xlabel("Years from 1900")
            self.axes[i].set_ylabel(data_name)
        plt.show()



# TODO: fix ca:
"""
OBSOLETE JE CROIS:

Description de l'idée:

input_dict est le dictionnaire des variables d'optimisation possible, pour l'instant aucun tri
n'a été fait et il faudra enlever les variables "non calibrable" du style des initialisations de stocks
la clé du dict est le nom de la var d'optim (var de design du design space de GEMSEO) et elle est 
associé à un dict stockant: l_b la lower bound, u_p la upper bound, value, la valeur initiale
pour faciliter la génération du design space cf plus bas le code:

design_space = create_design_space()
for key, values in Resource_D.input_dict:
    design_space.add_variable(key, 1, **values)


exogeneous_var_dict représente le dictionnaire pour définir les vairables éxogènes sachant qu'on ne veut pas utiliser les méthodes de calcul
données avec l'option Alone = True, car sinon on utiliserait les courbes par défaut de pyworld3 
Elles ont été choisies directement à partir des variables utilisées dans loop0_exogenous dans Resource de pyworld3
On a besoin des infos données par les modéliseurs pour définir les valeurs, et on les récupère avec ces lignes de code :
for exo_var in self.exogeneous_var_names:
            assert Resource_D.exogeneous_var_dict[exo_var].shape[0] >= self.n
            setattr(self, exo_var, Resource_D.exogeneous_var_dict[exo_var])
"""


def dict_real_data_to_match(filename):
    data_frame = pd.read_csv(filename, sep=" ")
    d = data_frame.to_dict()
    d = {key: np.array(list(val.values())) for key, val in d.items()}
    return d


def main():
    logger = configure_logger(level=logging.INFO)
    """ Example of data format:
    data = {
        "fake_p1": {
                "offset": 60, # beginning year-1900: 1950 -> 50
                "data": np.full((40,), 2e9) # this is an example ofc
            }, 
    }
    """
    d = dict_real_data_to_match("real_data_to_match.txt")
    data = {key: {"offset": 60, "data": val} for key, val in d.items()}

    
    vars = {
    "len" : {
        "l_b": 20,
        "u_b": 80,#40
        "value": 28 
    },
    "dcfsn": { #très intéressant de la "cross-valider" par la suite car on suppose qu'elle aura tendance à beaucoup changer d'une génénration à l'autre
        "l_b": 0,
        "u_b": 8,
        "value": 4 
    },
    "hsid" : {
        "l_b": 0.1,
        "u_b": 40,
        "value": 20 
    },
    "ieat" : {
        "l_b": 0.1,
        "u_b": 6,
        "value": 3 
    },
    "lpd" : {
        "l_b": 1,
        "u_b": 80,
        "value": 20 
    },
    "mtfn" : {
        "l_b": 2,
        "u_b": 16,
        "value": 12 
    },
    "rlt" : {
        "l_b": 10,
        "u_b": 40,
        "value": 30 
    },
    "sad" : {
    
        "l_b": 0.01,
        "u_b": 80,
        "value": 20 
    }
            
}
    

    disc = [
        Population_D(
            exogenous_data_dict = get_exogenous_data(Population_D.exogenous_data_names),
            input_variables_names = list(vars.keys()),
             input_parameters_fixed_dict = {
                "p1i" : 65e7,
                "p2i" : 70e7,
                "p3i" :19e7,
                "p4i" : 6e7,
                 
                }, # TODO: to be filled
            year_min=1900,
            year_max=2021,
            dt=1
        ),
        Calibration(data, "obj"),
        Animation(data, ["p1", "p2", "p3", "p4", "d1", "d2", "d3", "d4"]),
    ]

    design_space = create_design_space()
    for key, values in vars.items():
        design_space.add_variable(key, 1, **values)

    scenario = create_scenario(disc, "MDF", "obj", design_space)
    # TODO: mind the chosen step
    scenario.set_differentiation_method("finite_differences", 0.001)

    # TODO: mind the max_iter
    params = {"max_iter":500, "algo":"NLOPT_BOBYQA","algo_options" : {"xtol_rel" : 1e-4 , "ftol_rel" : 1e-6 }} 


    scenario.execute(input_data=params)
    scenario.post_process("OptHistoryView", save=True, show=False)
    print(design_space.get_current_value())

    disc[2].show()

if __name__ == "__main__":
    main()
