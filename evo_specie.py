import numpy as np
import gp_conf as neat_gp
from deap import creator
from speciation import calc_intracluster
from neatGPLS import ensure_dir
from conf_primitives import conf_sets
import jsonrpclib
import yaml
import neatGPLS
import datetime
import time
import contSpecie


def check_(server, r, free_species):
    free_species = []
    for j in range(1, int(r) + 1):
        free_species.append(eval(server.getSpecieFree(j)))
    if all(item2 is True for item2 in free_species):
        return False
    else:
        return True


def scheck_(server, r, p_flag, porcentage):
    flag_ = []
    for j in range(1, int(r) + 1):
        flag_.append(eval(server.getSpecieInfo(j)['flag_speciation']))
    if p_flag:
        p_ = porcentage/100.0
        if  flag_.count(True) >= int(p_ * len(flag_)):
            server.setFreePopulation('False')
            return True
        else:
            return False
    else:
        if any(flag_):
            server.setFreePopulation('False')
            return True
        else:
            return False


def speciation_init(config, server, pop):
    neat_h = 0.15
    num_Specie, specie_list = neatGPLS.evo_species(pop, neat_h)
    sample = [{"chromosome": str(ind), "id": None, "fitness": {"DefaultContext": 0.0}, "params": None,  "specie":ind.get_specie()} for ind in pop]
    evospace_sample = {'sample_id': 'None', 'sample_specie': None, 'sample': sample}
    return num_Specie, specie_list, evospace_sample


def counter(toolbox, pset):

    config = yaml.load(open("conf/conf.yaml"))
    num_var = config["num_var"]
    #pset = conf_sets(num_var)

    # toolbox = evoworker_gp.getToolBox(config, pset)
    server = jsonrpclib.Server(config["server"])
    free_pop = eval(server.getFreePopulation())
    if free_pop and contSpecie.cont_specie == 0:
        contSpecie.cont_specie = contSpecie.cont_specie+1
        r = server.get_CounterSpecie()
        free_species = []  # List of free species
        flag_check = True
        rs_flag = []  # List of flags about re - speciation
        rs_species = []  # List of species
        for i in range(1, int(r)+1):
            rs_flag.append(eval(server.getSpecieInfo(i)['flag_speciation']))
            rs_species.append(int(server.getSpecieInfo(i)['specie']))
        if scheck_(server, r, config["porcentage_flag"], config["porcentage"]):  # if all species have the flag speciation in true
            d = './ReSpeciacion/%s/rspecie_%d.txt' % (config["problem"], config["n_problem"])
            ensure_dir(d)
            best = open(d, 'a')
            while flag_check:
                flag_check = check_(server, r, free_species)
            if not flag_check:
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
                server.flushPopulation()
                server.initialize()
                neat_alg = config["neat_alg"]
                if neat_alg:
                    a, b, init_pop = speciation_init(config, server, pop)
                    list_spe = calc_intracluster(pop)
                    for elem in list_spe:
                        specielist = {'id': None, 'specie': str(elem[0]), 'intra_distance': str(elem[1]),
                                      'flag_speciation': 'False', 'sp_event': 'True'}
                        server.putSpecie(specielist)
                    server.putZample(init_pop)
                server.setFreePopulation('True')
                print 'ReSpeciacion- Done'
                contSpecie.cont_specie = 0
                best.write('\n%s;%s' % (str(datetime.datetime.now()), len(pop)))
    else:
        while free_pop == False:
            free_pop = eval(server.getFreePopulation())
            time.sleep(5)
