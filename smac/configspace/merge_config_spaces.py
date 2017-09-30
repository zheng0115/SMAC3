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
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.tae.execute_ta_run import StatusType
from bokeh.sampledata import population

def merge_configurations_of_local_optimizations(global_scenario:Scenario, number_of_merged_configurations : int):
    regex_pcs_files = os.path.join(os.getcwd(), os.path.dirname(global_scenario.output_dir),
                                   "preAnalysis*/parameters.pcs") 
    pcs_files = glob.glob(regex_pcs_files)
    cspaces = []

    for pcs_file in pcs_files:
        with open(pcs_file) as fp:
            pcs_str = fp.readlines()
            cspaces.append(pcs_new.read(pcs_str))
    
    
    rh = RunHistory(aggregate_func=average_cost)
    
    regex_rh_files = os.path.join(os.getcwd(), os.path.dirname(global_scenario.output_dir),
                                   "preAnalysis*/output_run*/runhistory.json") 
    
    rh_files = glob.glob(regex_rh_files)
    successful_configs = {}
    margin_configs={}
    for index, rh_file in enumerate(rh_files):
        successful_configs[index] = []
        margin_configs[index] = []
        rh = RunHistory(aggregate_func=average_cost)
        rh.load_json(rh_file, cspaces[index])
        successful_run_dict = {run: rh.data[run] for run in rh.data.keys()
                      if rh.data[run].status == StatusType.SUCCESS}
        for key in successful_run_dict.keys():
            successful_config = rh.ids_config[key.config_id]
            successful_configs[index].append(successful_config)
            margin_configs[index].append(rh.get_cost(successful_config))
            
        if len(successful_configs[index]) == 0:
            successful_config = cspaces[index].get_default_configuration()
            successful_configs[index].append(successful_config)
            margin_configs[index].append(rh.get_cost(successful_config))
        
    merged_configs = []
    
    for i in range(0, number_of_merged_configurations):
        merged_config = {}
        for optimization_id in successful_configs.keys():
            merged_config.update(random.choices(population=successful_configs[optimization_id], 
                                                weights=margin_configs[optimization_id], k=1)[0].get_dictionary())
        merged_configs.append(Configuration(configuration_space=global_scenario.cs, values=merged_config))
    
    return merged_configs
