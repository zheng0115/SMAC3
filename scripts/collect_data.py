import glob
import os
import json

import pandas

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.util import fix_types
from ConfigSpace.io.pcs_new import write
from smac.scenario.scenario import Scenario

scens = ["clasp_ricochet", "cplex_corlat", "cplex_rcw2", "probSAT_3SAT1k", "sparrowtoriss_bmc08"]

ACLIB_ROOT = "/home/fr/fr_fr/fr_tl1023/aclib/aclib2-master/"
val_path = "{}/*/run-*/validate-time-train"

cost_fn = "validationObjectiveMatrix*.csv"
config_fn = "validationCallStrings*.csv"


seen_configs = 0
config_ids = {}

def _add(config):
    global seen_configs
    if config_ids.get(config) is not None:
        return config_ids.get(config)
    else:
        seen_configs += 1
        config_ids[config] = seen_configs
        return seen_configs

for scen in scens:
    print(os.path.join(ACLIB_ROOT,"scenarios","*",scen))
    scenario = Scenario(glob.glob(os.path.join(ACLIB_ROOT,"scenarios","*",scen,"scenario.txt")) [0],{"output_dir": ""})
    
    data = pandas.DataFrame()
    config_ids = {}
    seen_configs = 0 
    for d in glob.glob(val_path.format(scen)):

        try:        
            cost_scen_fn = glob.glob(os.path.join(d,cost_fn))[0]
            config_scen_fn = glob.glob(os.path.join(d,config_fn))[0]
        except IndexError:
            continue
        
        cost_scen = pandas.read_csv(cost_scen_fn,index_col=0)
        config_scen = pandas.read_csv(config_scen_fn)
        
        #read and convert configurations
        configs = config_scen["Configuration "].values
        config_ids_list = [] 
        
        for config in configs:
            config = config.split(" ")
            config = dict((name[1:], value.strip("'")) for name, value in zip(config[::2], config[1::2]))
            config = fix_types(configuration=config, configuration_space=scenario.cs)
            config = Configuration(configuration_space=scenario.cs, values=config)
            id_ = _add(config)
            config_ids_list.append(id_)
            
        # read and convert performance data
        del cost_scen["Seed"]
        new_header = config_ids_list
        cost_scen.columns = new_header
        
        data = pandas.concat([data, cost_scen], axis=1)
            
    data = data.loc[:,~data.columns.duplicated()]

    try:
        os.makedirs("all_val/%s/" %(scen))
    except OSError:
        pass
    data.to_csv("all_val/%s/cost.csv" %(scen))
    
    config_dict = dict([(id_, config.get_dictionary()) for config, id_ in config_ids.items()])
    
    with open("all_val/%s/configs.json" %(scen), "w") as fp:
        json.dump(config_dict, fp)
        
    with open("all_val/%s/pcs.txt" %(scen), "w") as fp:
        fp.write(write(scenario.cs))
        