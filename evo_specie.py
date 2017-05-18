import sched, time
import numpy as np
import gp_conf as neat_gp
from deap import creator
from speciation import calc_intracluster
from conf_primitives import conf_sets
import evoworker_gp as evoworker
import jsonrpclib
import yaml

s = sched.scheduler(time.time, time.sleep)
config = yaml.load(open("conf/conf.yaml"))
num_var = config["num_var"]
pset = conf_sets(num_var)
toolbox = evoworker.getToolBox(config, pset)
server = jsonrpclib.Server(config["server"])

def counter():
    r = server.get_CounterSpecie()
    rs_flag = []
    rs_species = []
    for i in range(1, int(r)+1):
        rs_flag.append(eval(server.getSpecieInfo(i)['flag_speciation']))
        rs_species.append(int(server.getSpecieInfo(i)['specie']))
    if all(item is True for item in rs_flag):
        print 'ReSpeciacion'
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
                pop.append(i)
        server.initialize()
        neat_alg = config["neat_alg"]
        if neat_alg:
            a, b, init_pop = evoworker.speciation_init(config, server, pop)
            list_spe = calc_intracluster(pop)
            for elem in list_spe:
                specielist = {'id': None, 'specie': str(elem[0]), 'intra_distance': str(elem[1]),
                              'flag_speciation': 'False'}
                server.putSpecie(specielist)
            server.putZample(init_pop)

        print 'ReSpeciacion- Done'
    s.enter(60, 1, counter, ())

s.enter(60, 1, counter, ())
s.run()

