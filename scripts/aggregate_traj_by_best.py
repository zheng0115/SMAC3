import os
import typing
import glob
import json
import logging
from sqlalchemy.orm.session import ACTIVE
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlotTraj")

#######################################
# Boiler plate to use SMAC from src code and not installed version
import sys
import inspect
cmd_folder = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
########################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from ConfigSpace.io.pcs_new import read

from smac.utils.io.traj_logging import TrajLogger
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.configspace import convert_configurations_to_array, Configuration

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"

def read_val_data(val_data_dn:str, cs):
    '''
        read (training) validation data from val_data_dn 
    
        Arguments
        ---------
        val_data_dn: str
            directory name/path of with validation data (training)
        cs: ConfigurationSpace
            
        Returns
        -------
        cost_data: pandas.Dataframe
            index instance names; columns config id
        config_dict: dict()
            maps from config id 
    '''
    
    cost_data = pd.read_csv(os.path.join(val_data_dn, "cost.csv"), index_col=0, header=0)
    
    with open(os.path.join(val_data_dn, "configs.json")) as fp:
        config_dict = json.load(fp)
    
    #===========================================================================
    # with open(os.path.join(val_data_dn, "pcs.txt")) as fp:
    #     pcs_string = fp.readlines()
    #     cs = read(pcs_string)
    #===========================================================================
    
    config_dict = {Configuration(configuration_space=cs, values=config): id_ for id_, config in config_dict.items()}
        
    return cost_data, config_dict

def setup_SMAC_from_file(smac_out_dns: str):
    '''
        read all files from disk
        and initialize SMAC data structures with it

        Arguments
        ---------
        smac_out_dns: typing.List[str]
            output directory names of a SMAC runs

        Returns
        -------
        trajs: typing.List
            list of trajectories
        cs: ConfigurationSpace
    '''

    cwd = os.getcwd()
    smac_out_path, smac_out_dn = os.path.split(smac_out_dns[0])
   # os.chdir(smac_out_path)

    # use first run as reference
    scenario_fn = os.path.join(smac_out_dn, "scenario.txt")
    scenario = Scenario(scenario_fn, {"output_dir": ""})
    
   # os.chdir(cwd)

    trajs = []
    for dn in smac_out_dns:
        traj = TrajLogger.read_traj_aclib_format(fn=os.path.join(dn, "traj_aclib2.json"),
                                             cs=scenario.cs)
        trajs.append(traj)

    return trajs, scenario.cs


def race_traj(trajs,
              cost_data, config_dict):
    '''
        iterates over all trajectories simultaneously and 
        races these based on the available cost_data
        
        Arguments
        ---------
        trajs: typing.List
            list of trajectories
        cost_data: pandas.Dataframe
            index instance names; columns config id
        config_dict: dict()
            maps from config id 
    '''

    # consider only incumbents at every 
    # 2**-1, 2**0, 2**1, ... time stamp
    wc_time = 0.6591796875

    final_traj = []
    
    while True:
        print(wc_time)
        
        incs = []
        last = []
        for traj in trajs:
            last_entry = traj[0]
            for entry in traj:
                if entry["wallclock_time"] > wc_time:
                    incs.append(last_entry)
                    break
                else:
                    last_entry = entry
            if traj[-1]["wallclock_time"] == last_entry["wallclock_time"]:
                last.append(True)
                incs.append(traj[-1])
            else:
                last.append(False)

        # get relevant cost data
        cost_data_incs = []
        incs_avail = []
        for inc in incs:
            for config_prime, id_prime in config_dict.items():
                if config_prime == inc["incumbent"]:
                     id_ = id_prime
                     break

            if id_ is None:
                print(inc["incumbent"])
                #print(config_dict)
                print("WARN: Configuration not found -- skipped")
                continue
            incs_avail.append(inc)
            cost_data_incs.append(cost_data[id_].values)
            
        cost_data_incs = np.array(cost_data_incs).T
            
        if not incs_avail:
            print("WARN: Haven't found any data for configurations to be raced")
            wc_time *= 2
            
            if np.all(last):
                break
            
            continue
            
        # race
        avg_cost = np.mean(cost_data_incs)
        best_idx = np.argmin(avg_cost)
        inc = incs_avail[best_idx]
            
        final_traj.append(inc)
        
        wc_time *= 2
        
        if np.all(last):
            break
            
    return final_traj

def perm_test(x, y, alpha:float=0.05, reps:int=10000):
    '''
        one-sided paired permutation test
        Alternative Hypothese: x is sig better than y
        
        Arguments
        ---------
        x: np.ndarray
        y: np.ndarray
        alpha: float
        reps: int
            number of samples/repetitions
            
        Returns
        -------
        True: reject null hypothesis
        False: don't reject
        
    '''
    if np.all(x == y):
        return False
    
    ground_truth = np.sum(x-y)
    permutations = [np.sum((x-y) * np.random.choice([1,-1], size=x.shape[0])) for _ in range(reps)]
    p = percentileofscore(a=permutations, score=ground_truth) / 100
    
    #print(p)
    return p < alpha
        
        
def save_traj(traj, save_dn:str):
    '''
        save trajectory to disk
        using the TrajLogger
        
        Arguments
        ---------
        traj: typing.List
            trajectory 
        save_dn: str
            directory name to save the trajectory into it
    '''
    
    tj = TrajLogger(output_dir=save_dn, stats=None)
    
    for id_, entry in enumerate(traj):
        tj._add_in_old_format(train_perf=entry["cost"], 
                              incumbent_id=id_+1, 
                              incumbent=entry["incumbent"], 
                              ta_time_used=entry["cpu_time"], 
                              wallclock_time=entry["wallclock_time"])
        tj._add_in_aclib_format(train_perf=entry["cost"], 
                              incumbent_id=id_+1, 
                              incumbent=entry["incumbent"], 
                              ta_time_used=entry["cpu_time"], 
                              wallclock_time=entry["wallclock_time"],
                              evaluations=entry["evaluations"])
    

if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--smac_output_dir", required=True, nargs="*",
                        help="Output directories of SMAC")
    parser.add_argument("--val_data", required=True,
                        help="Directory with all validation information")
    parser.add_argument("--save_dir", required=True,
                        help="Output directoriy to save aggregated trajectory")
    args_ = parser.parse_args()
    
    out_dirs = []
    for dn in args_.smac_output_dir:
        out_dirs.extend(glob.glob(dn))

    trajs, cs = setup_SMAC_from_file(smac_out_dns=out_dirs)
    
    cost_data, config_dict = read_val_data(args_.val_data, cs)

    aggr_traj = race_traj(trajs=trajs,
                          cost_data=cost_data, config_dict=config_dict)
    
    save_traj(traj=aggr_traj, save_dn=args_.save_dir)
