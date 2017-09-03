'''
Created on Sep 1, 2017

@author: mrudolph
'''


import os
from ConfigSpace import Configuration, ConfigurationSpace
from smac.scenario.scenario import Scenario
import glob
from smac.configspace import pcs_new
from smac.utils.io.traj_logging import TrajLogger
import random

def merge_configurations_of_local_optimizations(global_scenario:Scenario, number_of_merged_configurations : int):
    regex_pcs_files = os.path.join(os.getcwd(), os.path.dirname(global_scenario.output_dir),
                                   "preAnalysis*/parameters.pcs") 
    pcs_files = glob.glob(regex_pcs_files)
    cspaces = []

    for pcs_file in pcs_files:
        with open(pcs_file) as fp:
            pcs_str = fp.readlines()
            cspaces.append(pcs_new.read(pcs_str))
    
    regex_traj_files = os.path.join(os.getcwd(), os.path.dirname(global_scenario.output_dir),
                                   "preAnalysis*/output_run1/traj_aclib2.json") 
           
    traj_files = glob.glob(regex_traj_files)
    incumbents = {}
    for index, traj_fn in enumerate(traj_files):
        incumbents[index] = []
        trajectory = TrajLogger.read_traj_aclib_format(
                    fn=traj_fn, cs=cspaces[index])
        for traj_entry in trajectory:
            incumbents[index].append(traj_entry["incumbent"])
        
        if len(incumbents[index]) == 0:
            incumbents[index].append(cspaces[index].get_default_configuration())
        
        
    merged_configs = []
    
    for i in range(0, number_of_merged_configurations):
        merged_config = {}
        for optimization_id in incumbents.keys():
            merged_config.update(random.choice(incumbents[optimization_id]).get_dictionary())
        merged_configs.append(Configuration(configuration_space=global_scenario.cs, values=merged_config))
    
    return merged_configs
#     runhistories = []
#     for index, json_file in enumerate(json_files):
#         rh = RunHistory(aggregate_func=average_cost)
#         rh.load_json(json_file, cspaces[index])
#         runhistories.append(rh)

    

#     merged_configs = []
#     for config_id in range(0, len(runhistories[0].get_all_configs())):
#             merged_config = {}
#             
#             for rh in runhistories:
#                 merged_config.update(rh.get_all_configs()[config_id].get_dictionary())
#             merged_configs.append(Configuration(configuration_space=global_scenario.cs, values=merged_config))
# 
#     return merged_configs
