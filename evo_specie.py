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
    for j in range(1, int(r)):
        free_species.append(eval(server.getSpecieFree(j)))
    if all(item2 is True for item2 in free_species):
        return False
    else:
        return True


def scheck_(server, r, p_flag, porcentage):
    flag_ = []
    for j in range(1, int(r)):
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


def speciation_init(config, server, pop, neat_h):
    num_Specie, specie_list = neatGPLS.evo_species(pop, neat_h)
    sample = [{"chromosome": str(ind), "id": None, "fitness": {"DefaultContext": 0.0}, "params": None,  "specie":ind.get_specie()} for ind in pop]
    evospace_sample = {'sample_id': 'None', 'sample_specie': None, 'sample': sample}
    return num_Specie, specie_list, evospace_sample


def counter(toolbox, pset):
    print 'Counter'
    re_sp = 0
    config = yaml.load(open("conf/conf.yaml"))
    server = jsonrpclib.Server(config["server"])
    free_pop = eval(server.getFreePopulation())
    free_file = eval(server.getFreeFile())
    print free_file, free_pop
    if free_pop and free_file:
        server.setFreeFile('False')
        print 'Free and Free'
        r = server.get_CounterSpecie()
        print r
        if scheck_(server, r, config["porcentage_flag"], config["porcentage"]):
            print "speciation-required"
            free_species = []  # List of free species
            flag_check = True
            rs_species = []  # List of species
            for i in range(1, int(r)):
                rs_species.append(int(server.getSpecieInfo(i)['specie']))

            # Opening files to save data.
            d = './ReSpeciacion/%s/rspecie_%d.txt' % (config["problem"], config["n_problem"])
            ensure_dir(d)
            best = open(d, 'a')

            d = './ReSpeciacion/%s/nspecie_%d.csv' % (config["problem"], config["n_problem"])
            ensure_dir(d)
            n_specie = open(d, 'a')

            d = './General/%s/datapop_%d_%d.txt' % (config["problem"], config["n_problem"], config["set_specie"])
            neatGPLS.ensure_dir(d)
            datapop_ = open(d, 'a')


            while flag_check:
                flag_check = check_(server, r, free_species)
                print ("waiting  - this worker will make speciation")
            if not flag_check:
                print 'ReSpeciacion'
                sp_init = datetime.datetime.now()
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
                print 'Flush population'
                server.flushPopulation()
                for ind in pop:
                    datapop_.write('\n%s;%s;%s;%s' % (str(sp_init), ind, len(pop), ind.get_specie()))
                server.initialize()
                print 'Initialize population'
                neat_alg = config["neat_alg"]
                if neat_alg:
                    a, b, init_pop = speciation_init(config, server, pop, config["neat_h"])
                    list_spe = calc_intracluster(pop)
                    for elem in list_spe:
                        specielist = {'id': None, 'specie': str(elem[0]), 'intra_distance': str(elem[1]),
                                      'flag_speciation': 'False', 'sp_event': 'True'}
                        server.putSpecie(specielist)
                        n_specie.write('\n%s,%s' % (str(elem[0]), str(elem[1])))
                    server.putZample(init_pop)
                server.setFreePopulation('True')

                num_specie = server.get_CounterSpecie()
                print "numero de especies creadas: ", num_specie


                print 'ReSpeciacion- Done'
                re_sp = 1
                best.write('\n%s;%s;%s;%s' % (str(datetime.datetime.now()), str(sp_init), len(pop), num_specie))
        server.setFreePopulation('True')
        server.setFreeFile('True')
        contSpecie.cont_specie = 0
        return re_sp
    else:
        print 'No Free'
        if free_pop and free_file is False:
            while free_file is False:
                print "waiting"
                time.sleep(5)
                try:
                    free_file = eval(server.getFreeFile())
                except TypeError:
                    free_file = False
        re_sp = 0
        free_pop = eval(server.getFreePopulation())
        while free_pop is False:
            re_sp = 1
            print "still waiting", free_pop
            try:
                time.sleep(1)
                free_pop = eval(server.getFreePopulation())
            except TypeError:
                time.sleep(5)
                free_pop = eval(server.getFreePopulation())
        return re_sp