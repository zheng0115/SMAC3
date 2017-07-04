import os
import typing
import glob
import json
import logging
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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from smac.utils.io.traj_logging import TrajLogger
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.configspace import convert_configurations_to_array

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"

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
        smac: SMAC()
            SMAC Facade object
        trajs: typing.List
            list of trajectories
    '''

    # use first run as reference
    scenario_fn = os.path.join(smac_out_dns[0], "scenario.txt")
    scenario = Scenario(scenario_fn, {"output_dir": ""})
    smac = SMAC(scenario=scenario)

    rh = smac.solver.runhistory
    rh.load_json(os.path.join(smac_out_dns[0], "runhistory.json"), cs=scenario.cs)
    
    for dn in smac_out_dns[1:]:
        try:
            rh.update_from_json(fn=os.path.join(dn, "runhistory.json"), cs=scenario.cs)
        except json.decoder.JSONDecodeError:
            print("Failed to read %s" %(os.path.join(dn, "runhistory.json")))

    trajs = []
    for dn in smac_out_dns:
        traj = TrajLogger.read_traj_aclib_format(fn=os.path.join(dn, "traj_aclib2.json"),
                                             cs=scenario.cs)
        trajs.append(traj)

    return smac, trajs


def predict_perf_of_traj(trajs, smac: SMAC):
    '''
        iterates over all trajectories simultaneously 
        and predicts the performance of all current incumbents
        and chooses the best of them

        Arguments
        ---------
        smac: SMAC()
            SMAC Facade object
        traj: typing.List
            list of trajectories

    '''

    wc_time = 0
    # skip dummy entries
    incs = [traj[1] for traj in trajs]
    idxs = np.array([1]*len(trajs))
    
    final_traj = []
    
    while True:
        wc_time = max(inc["wallclock_time"] for inc in incs)
        print(wc_time)
        X, Y = smac.solver.rh2EPM.transform(smac.solver.runhistory, wc_time)
        logger.info("Fit EPM on %d observations." % (X.shape[0]))
        smac.solver.model.train(X, Y)
        
        config_array = convert_configurations_to_array([inc["incumbent"] for inc in incs])
        m, v = smac.solver.model.predict_marginalized_over_instances(
                X=config_array)
        
        if smac.solver.scenario.run_obj == "runtime":
            p = 10**m[:, 0]
        else:
            p = m[:, 0]

        best_idx = np.argmin(p)
        incs[best_idx]["cost"] = p[best_idx]
        final_traj.append(incs[best_idx])
        print(p[best_idx])
        print(incs[best_idx]["incumbent"])
        
        min_wc_time = np.inf
        min_idx = None
        for traj_idx, idx in enumerate(idxs):
            try:
                wc = trajs[traj_idx][idx+1]["wallclock_time"]
                if wc < min_wc_time:
                    min_idx = traj_idx
                    min_wc_time = wc
            except IndexError:
                pass
            
        if min_idx is not None:
            idxs[min_idx] += 1
            incs[min_idx] = trajs[min_idx][idxs[min_idx]]
        else:
            break
            
    return final_traj
        
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
    parser.add_argument("--save_dir", required=True,
                        help="Output directoriy to save aggregated trajectory")
    args_ = parser.parse_args()
    
    out_dirs = []
    for dn in args_.smac_output_dir:
        out_dirs.extend(glob.glob(dn))

    smac, trajs = setup_SMAC_from_file(smac_out_dns=out_dirs)

    aggr_traj = predict_perf_of_traj(trajs=trajs, smac=smac)
    
    save_traj(traj=aggr_traj, save_dn=args_.save_dir)
