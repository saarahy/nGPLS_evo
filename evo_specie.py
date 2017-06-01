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


global freepop_
freepop_ = True

def counter(toolbox, pset):

    config = yaml.load(open("conf/conf.yaml"))
    num_var = config["num_var"]
    #pset = conf_sets(num_var)

    # toolbox = evoworker_gp.getToolBox(config, pset)
    server = jsonrpclib.Server(config["server"])
    free_pop = eval(server.getFreePopulation())
    freepop_ = free_pop
    print free_pop
    print contSpecie.cont_specie
    if free_pop and contSpecie.cont_specie == 0:
        contSpecie.cont_specie = contSpecie.cont_specie+1
        r = server.get_CounterSpecie()
        if scheck_(server, r, config["porcentage_flag"], config["porcentage"]):
            freepop_ = False
            print "speciation-required"
            free_species = []  # List of free species
            flag_check = True
            rs_species = []  # List of species
            print contSpecie.freepop_
              # if a specie have the t-flag speciation
            for i in range(1, int(r) + 1):
                rs_species.append(int(server.getSpecieInfo(i)['specie']))
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
                freepop_ = True
                print contSpecie.freepop_
                print 'ReSpeciacion- Done'
                best.write('\n%s;%s' % (str(datetime.datetime.now()), len(pop)))
        contSpecie.cont_specie = 0
    else:
        while free_pop is False:
            print "still waiting", free_pop, freepop_
            try:
                time.sleep(1)
                free_pop = eval(server.getFreePopulation())
            except TypeError:
                time.sleep(5)
                free_pop = eval(server.getFreePopulation())
