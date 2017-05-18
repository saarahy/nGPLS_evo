import numpy as np
import gp_conf as neat_gp
from deap import creator
from speciation import calc_intracluster
from conf_primitives import conf_sets
import evoworker_gp as evoworker
import jsonrpclib
import yaml
import os

config = yaml.load(open("conf/conf.yaml"))
num_var = config["num_var"]
pset = conf_sets(num_var)
toolbox = evoworker.getToolBox(config, pset)
server = jsonrpclib.Server(config["server"])


def counter():
    r = server.get_CounterSpecie()
    rs_species = []
    for i in range(1, int(r)+1):
        rs_species.append(int(server.getSpecieInfo(i)['specie']))
    pop = []
    for sp in rs_species:
        evospace_sample = server.getSample_specie(sp)
        for cs in evospace_sample['sample']:
            i = creator.Individual(neat_gp.PrimitiveTree.from_string(cs['chromosome'], pset))
            if isinstance(cs['params'], list):
                i.params_set(np.asarray(cs['params']))
            elif isinstance(cs['params'], unicode):
                i.params_set(np.asarray([float(elem) for elem in cs['params'].strip('[]').split(',')]))
            i.specie(int(cs['specie']))
            i.fitness.values = cs['fitness']['DefaultContext']
            pop.append(i)
    return pop

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

population = counter()
num_p = config["n_problem"]
problem = config["problem"]
n_corr = config["n_corr"]

d = './Global/%s/pop_%d_%d.txt' % (problem, num_p, n_corr)
ensure_dir(d)
g_file = open(d, 'w')

for ind in population:
    g_file.write('\n%s;%s;%s' % (ind.get_specie(), ind.fitness.values[0], ind))