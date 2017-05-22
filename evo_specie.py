import numpy as np
import gp_conf as neat_gp
from deap import creator
from speciation import calc_intracluster
from conf_primitives import conf_sets
import evoworker_gp
import jsonrpclib
import yaml


def check_(server, r, free_species):
    free_species = []
    for j in range(1, int(r) + 1):
        free_species.append(eval(server.getSpecieFree(j)))
    if  free_species.count(True) >= int(.9 * len(free_species)):
        server.setFreePopulation('False')
    if all(item2 is True for item2 in free_species):
        return False
    else:
        return True


def counter():
    config = yaml.load(open("conf/conf.yaml"))
    num_var = config["num_var"]
    pset = conf_sets(num_var)

    toolbox = evoworker_gp.getToolBox(config, pset)
    server = jsonrpclib.Server(config["server"])

    r = server.get_CounterSpecie()
    free_species = []  # List of free species
    flag_check = True
    rs_flag = []  # List of flags about re - speciation
    rs_species = []  # List of species
    for i in range(1, int(r)+1):
        rs_flag.append(eval(server.getSpecieInfo(i)['flag_speciation']))
        rs_species.append(int(server.getSpecieInfo(i)['specie']))
    if all(item is True for item in rs_flag):
        while flag_check:
            flag_check = check_(server, r, free_species)
        if all(item2 is True for item2 in free_species):
            server.setFreePopulation('False')
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
                a, b, init_pop = evoworker_gp.speciation_init(config, server, pop)
                list_spe = calc_intracluster(pop)
                for elem in list_spe:
                    specielist = {'id': None, 'specie': str(elem[0]), 'intra_distance': str(elem[1]),
                                  'flag_speciation': 'False'}
                    server.putSpecie(specielist)
                server.putZample(init_pop)
            server.setFreePopulation('True')
            print 'ReSpeciacion- Done'

counter()