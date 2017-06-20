#Archivos importados para el algoritmo
import operator
import csv
import funcEval
import numpy as np
import neatGPLS
#import neatGPLS_evospace
import init_conf
import os.path
import datetime
import copy
from deap import base
from deap import creator
from deap import tools
from deap import gp
from speciation import calc_intracluster, intracluster
import gp_conf as neat_gp
from my_operators import safe_div, mylog, mypower2, mypower3, mysqrt, myexp
from conf_primitives import conf_sets
import evo_specie

#Imports de evospace
import random, time
import xmlrpclib
import jsonrpclib


def getToolBox(config, pset):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("FitnessTest", base.Fitness, weights=(-1.0,))
    creator.create("Individual", neat_gp.PrimitiveTree, fitness=creator.FitnessMin, fitness_test=creator.FitnessTest)

    toolbox = base.Toolbox()
    neat_cx = config["neat_cx"]
    # Attribute generator
    if neat_cx:
        toolbox.register("expr", neat_gp.genFull, pset=pset, min_=0, max_=3)
    else:
        toolbox.register("expr", neat_gp.genHalfAndHalf, pset=pset, min_=0, max_=7)
    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", init_conf.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    # Operator registering
    toolbox.register("select", tools.selTournament, tournsize=config["tournament_size"])
    toolbox.register("mate", neat_gp.cxSubtree)
    if neat_cx:
        toolbox.register("expr_mut", neat_gp.genFull, min_=0, max_=3)
    else:
        toolbox.register("expr_mut", neat_gp.genHalfAndHalf, min_=0, max_=7)
    toolbox.register("mutate", neat_gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox


def initialize(config):
    pset = conf_sets(config["num_var"])
    pop = getToolBox(config, pset).population(n=config["population_size"])
    server = jsonrpclib.Server(config["server"])
    server.initialize()
    server.setFreePopulation('True')
    neat_alg = config["neat_alg"]
    if neat_alg:
        a, b, init_pop = speciation_init(config, server, pop)
        list_spe = calc_intracluster(pop)
        for elem in list_spe:
            specielist = {'id': None, 'specie': str(elem[0]), 'intra_distance': str(elem[1]), 'flag_speciation': 'False', 'sp_event': 'True'}
            server.putSpecie(specielist)
    else:
        sample = [{"chromosome": str(ind), "id": None, "fitness": {"DefaultContext": 0.0}, "params": None, "specie": 1} for ind in pop]
        init_pop = {'sample_id': 'None', 'sample_specie': None, 'sample': sample}
        a = 1
        b = 1
    server.putZample(init_pop)
    return a, b


def speciation_init(config, server, pop):
    neat_h = 0.15
    num_Specie, specie_list = neatGPLS.evo_species(pop, neat_h)
    sample = [{"chromosome": str(ind), "id": None, "fitness": {"DefaultContext": 0.0}, "params": None,  "specie":ind.get_specie()} for ind in pop]
    evospace_sample = {'sample_id': 'None', 'sample_specie': None, 'sample': sample}
    return num_Specie, specie_list, evospace_sample


def speciation(config, pop_evo, pset):
    server = jsonrpclib.Server(config["SERVER"])
    #numsampl=server.getSampleNumber()
    evospace_sample = pop_evo
    pop = [creator.Individual(neat_gp.PrimitiveTree.from_string(cs['chromosome'], pset)) for cs in
           evospace_sample['sample']]
    neat_h=0.15
    num_Specie, specie_list = neatGPLS.evo_species(pop, neat_h)
    sample = [{"chromosome": str(ind), "id": None, "fitness": {"DefaultContext": 0.0}, "params": None,  "specie":ind.get_specie()} for ind in pop]
    evospace_sample = {'sample_id': 'None', 'sample': sample}
    server.putZample(evospace_sample)
    return num_Specie, specie_list


def get_Speciedata(config):
    server = jsonrpclib.Server(config["server"])#evospace.Population("pop")
    # evospace_sample = server.getPopulation()#server.getPopulation()
    # if evospace_sample['sample'][00]['specie'] == None:
    #     num_Specie, specie_list = speciation(config, evospace_sample)
    # else:
    a=server.getSpecie()
    specie_list=map(int, a)
    num_Specie=max(specie_list)
    return num_Specie, specie_list


def evalSymbReg(individual, points, toolbox, config):
    points.flags['WRITEABLE'] = False
    func = toolbox.compile(expr=individual)
    vector = copy.deepcopy(points[config["num_var"]])
    data_x = copy.deepcopy(np.asarray(points)[:config["num_var"]])
    vector_x = func(*data_x)
    with np.errstate(divide='ignore', invalid='ignore'):
        if isinstance(vector_x, np.ndarray):
            for e in range(len(vector_x)):
                if np.isnan(vector_x[e]) or np.isinf(vector_x[e]):
                    vector_x[e] = 0.
    result = np.sum((vector_x - vector)**2)
    return np.sqrt(result/len(points[0])),


def data_(n_corr, p, problem, name_database,toolbox, config):
    p = 1000
    n_corr = 1
    n_archivot='./data_corridas/%s/test_%d_%d.txt'%(problem,p,n_corr)
    n_archivo='./data_corridas/%s/train_%d_%d.txt'%(problem,p,n_corr)
    if not (os.path.exists(n_archivo) or os.path.exists(n_archivot)):
        direccion = "./data_corridas/%s/%s" % (problem, name_database)
        with open(direccion) as spambase:
            spamReader = csv.reader(spambase,  delimiter=' ', skipinitialspace=True)
            num_c = sum(1 for line in open(direccion))
            num_r = len(next(csv.reader(open(direccion), delimiter=' ', skipinitialspace=True)))
            Matrix = np.empty((num_r, num_c,))
            for row, c in zip(spamReader, range(num_c)):
                for r in range(num_r):
                    try:
                        Matrix[r, c] = row[r]
                    except ValueError:
                        print 'Line {r} is corrupt', r
                        break
        if not os.path.exists(n_archivo):
            long_train=int(len(Matrix.T)*.7)
            data_train1 = random.sample(Matrix.T, long_train)
            np.savetxt(n_archivo, data_train1, delimiter=",", fmt="%s")
        if not os.path.exists(n_archivot):
            long_test=int(len(Matrix.T)*.3)
            data_test1 = random.sample(Matrix.T, long_test)
            np.savetxt(n_archivot, data_test1, delimiter=",", fmt="%s")
    with open(n_archivo) as spambase:
        spamReader = csv.reader(spambase,  delimiter=',', skipinitialspace=True)
        num_c = sum(1 for line in open(n_archivo))
        num_r = len(next(csv.reader(open(n_archivo), delimiter=',', skipinitialspace=True)))
        Matrix = np.empty((num_r, num_c,))
        for row, c in zip(spamReader, range(num_c)):
            for r in range(num_r):
                try:
                    Matrix[r, c] = row[r]
                except ValueError:
                    print 'Line {r} is corrupt' , r
                    break
        data_train=Matrix[:]
    with open(n_archivot) as spambase:
        spamReader = csv.reader(spambase,  delimiter=',', skipinitialspace=True)
        num_c = sum(1 for line in open(n_archivot))
        num_r = len(next(csv.reader(open(n_archivot), delimiter=',', skipinitialspace=True)))
        Matrix = np.empty((num_r, num_c,))
        for row, c in zip(spamReader, range(num_c)):
            for r in range(num_r):
                try:
                    Matrix[r, c] = row[r]
                except ValueError:
                    print 'Line {r} is corrupt' , r
                    break
        data_test=Matrix[:]
    #return data_train,data_test
    toolbox.register("evaluate", evalSymbReg, points=data_train, toolbox=toolbox, config = config)
    toolbox.register("evaluate_test", evalSymbReg, points=data_test, toolbox=toolbox, config = config)


def evolve(sample_num, config, toolbox, pset):

    start = time.time()
    problem       = config["problem"]
    direccion     = "./data_corridas/%s/train_%d_%d.txt"
    n_corr        = sample_num # config["n_corr"]
    n_prob        = config["n_problem"]

    name_database = config["db_name"]
    print config["set_specie"]
    server = jsonrpclib.Server(config["server"])
    # evospace_sample = server.getSample(config["population_size"])

    if eval(server.getFreePopulation()):
        if eval(server.getSpecieFree(config["set_specie"])):
            evospace_sample = server.getSample_specie(config["set_specie"])
            data_specie = {'id': config["set_specie"], 'b_key': 'False'}
            server.setSpecieFree(data_specie)
        else:
            print 'choose another specie'
            return []
    else:
        print 'Re-speciation process'
        while eval(server.getFreePopulation()) is False:
            time.sleep(5)
            print 'waiting process'
        if eval(server.getSpecieFree(config["set_specie"])):
            evospace_sample = server.getSample_specie(config["set_specie"])
            data_specie = {'id': config["set_specie"], 'b_key': 'False'}
            server.setSpecieFree(data_specie)
        else:
            print 'choose another specie'
            return []

    pop = []
    for cs in evospace_sample['sample']:
        i = creator.Individual(neat_gp.PrimitiveTree.from_string(cs['chromosome'], pset))
        if isinstance(cs['params'], list):
            i.params_set(np.asarray(cs['params']))
        elif isinstance(cs['params'], unicode):
            i.params_set(np.asarray([float(elem) for elem in cs['params'].strip('[]').split(',')]))
        i.specie(int(cs['specie']))
        pop.append(i)

    cxpb                = config["cxpb"]
    mutpb               = config["mutpb"]
    ngen                = config["worker_generations"]

    params              = config["params"]
    neat_cx             = config["neat_cx"]
    neat_alg            = config["neat_alg"]
    neat_pelit          = config["neat_pelit"]
    neat_h              = config["neat_h"]

    funcEval.LS_flag    = config["ls_flag"]
    LS_select           = config["ls_select"]
    funcEval.cont_evalp = 0
    num_salto           = config["num_salto"]
    cont_evalf          = config["cont_evalf"]

    SaveMatrix          = config["save_matrix"]
    GenMatrix           = config["gen_matrix"]
    version             = 3
    testing             = True

    data_(n_corr, n_prob, problem, name_database, toolbox, config)

    begin =time.time()
    print "inicio del proceso"

    pop, log, funcEval_ = neatGPLS.neat_GP_LS(pop, toolbox, cxpb, mutpb, ngen, neat_alg, neat_cx, neat_h, neat_pelit,
                                       funcEval.LS_flag, LS_select, cont_evalf, num_salto, SaveMatrix, GenMatrix, pset,
                                       n_corr, n_prob, params, direccion, problem, testing, version=version, benchmark_flag=False, beta= 0.5,
                                       random_speciation=True, set_specie=config["set_specie"], stats=None, halloffame=None, verbose=True)

    putback = time.time()

    d_intraspecie = intracluster(pop)
    d_intracluster = server.getIntraSpecie(config["set_specie"])
    resp_flag = 0
    id_ = "specie:%s" % config["set_specie"]
    print 'calculation intra-cluster'
    if d_intraspecie > (1.5 * float(d_intracluster)):
        specielist = {'id': id_, 'specie': config["set_specie"], 'intra_distance': str(d_intracluster),
                      'flag_speciation': 'True', 'sp_event': 'False'}
        server.putSpecie(specielist)
        resp_flag = 1


    #

    sample = [{"specie": str(config["set_specie"]), "chromosome":str(ind), "id":None,
               "fitness":{"DefaultContext":[ind.fitness.values[0].item() if isinstance(ind.fitness.values[0], np.float64) else ind.fitness.values[0]]},
               "params":str([x for x in ind.get_params()]) if funcEval.LS_flag else None} for ind in pop]

    evospace_sample = {'sample_id': 'None', 'sample_specie': str(config["set_specie"]), 'sample': sample}

    server.putZample(evospace_sample)
    data_specie = {'id': config["set_specie"], 'b_key': 'True'}
    server.setSpecieFree(data_specie)
    print 'Going to counter'
    re_sp = evo_specie.counter(toolbox, pset)
    print 'Ending to counter'
    d = './Data/%s/dintra_%d_%s.txt' % (problem, n_prob, config["set_specie"])
    neatGPLS.ensure_dir(d)
    dintr = open(d, 'a')
    dintr.write('\n%s;%s;%s;%s' % (n_corr, d_intraspecie, resp_flag, re_sp))

    best_ind = tools.selBest(pop, 1)[0]
    best = [len(best_ind), sample_num, round(time.time() - start, 2),
                                         round(begin - start, 2), round(putback - begin, 2),
                                         round(time.time() - putback, 2), best_ind], len(pop), resp_flag, re_sp, funcEval_
    return best


def work(params):
    worker_id = params[0][0]
    config = params[0][1]
    server = jsonrpclib.Server(config["server"])
    num_var = config["num_var"]
    results = []
    pset = conf_sets(num_var)
    toolbox = getToolBox(config, pset)

    d = './ReSpeciation/%s/time_%d_%d.txt' % (config["problem"], config["n_problem"],config["set_specie"])
    neatGPLS.ensure_dir(d)
    time_r = open(d, 'a')

    for sample_num in range(1, config["max_samples"]+1):
        print  'sample: ', sample_num

        d_intracluster = server.getIntraSpecie(config["set_specie"])
        time_r.write('\n%s;%s;%s;%s;%s;%s;%s;%s' % (config["set_specie"], sample_num, str(datetime.datetime.now()), d_intracluster, 'NA', 'NA','NA', funcEval.cont_evalp))

        gen_data, len_pop, flag_, resp_, fEval_ = evolve(sample_num, config, toolbox, pset)

        d_intracluster = server.getIntraSpecie(config["set_specie"])
        time_r.write('\n%s;%s;%s;%s;%s;%s;%s:%s' % (config["set_specie"], sample_num, str(datetime.datetime.now()), d_intracluster, len_pop, flag_, resp_, fEval_))

        if gen_data == []:
            print 'No-Evolution'
            results = []
        else:
            results.append([worker_id] + gen_data)
    return results

