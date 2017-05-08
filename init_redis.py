import evoworker_gp as evoworker
import yaml

config = yaml.load(open("conf/conf.yaml"))
a, b = evoworker.initialize(config)
