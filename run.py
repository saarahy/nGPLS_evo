
import evoworker_gp as evoworker
from neatGPLS import ensure_dir
import time
import yaml
from shutil import copyfile

config = yaml.load(open("conf/conf.yaml"))

start = time.time()

start = time.time()
params = [(i, config) for i in range(1)]


num_p = config["n_problem"]
problem = config["problem"]
specie = 7

d = './Timing/%s/time_%d.txt' % (problem, num_p)
ensure_dir(d)
best = open(d, 'a')

c_f = './conf_record/%s/' % (problem)
ensure_dir(c_f)
copyfile('./conf/conf.yaml', ('./conf_record/%s/config_%s_%d_%d.yaml'% (problem, problem, num_p, specie)))

 # int(random.choice(b))
config["set_specie"] = specie
# config["neat_alg"] = False
# for ci in range(1, 2):
# config["n_corr"] = ci
ci = config["n_corr"]
with open("conf/conf.yaml", "w") as f:
    yaml.dump(config, f)

params = [(i, config) for i in range(1)]
result = evoworker.work(params)
if result:
    best.write('\n%s;%s;%s'% (ci, result[0][2], result[0][3]))
    print 'finished'

# config["n_corr"] = ci + 1
# with open("conf/conf.yaml", "w") as f:
#     yaml.dump(config, f)