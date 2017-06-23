import os
import typing
import glob
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
        rh.update_from_json(fn=os.path.join(dn, "runhistory.json"), cs=scenario.cs)

    logger.info("Fit EPM on %d observations." % (len(rh.data)))
    X, Y = smac.solver.rh2EPM.transform(rh)
    smac.solver.model.train(X, Y)

    trajs = []
    for dn in smac_out_dns:
        traj = TrajLogger.read_traj_aclib_format(fn=os.path.join(dn, "traj_aclib2.json"),
                                             cs=scenario.cs)
        trajs.append(traj)

    return smac, trajs


def predict_perf_of_traj(trajs, smac: SMAC):
    '''
        predict the performance of all entries in the trajectory
        marginalized across all instances
        
        Overwrites all cost estimates in the traj entries

        Arguments
        ---------
        smac: SMAC()
            SMAC Facade object
        traj: typing.List
            list of trajectories

    '''

    for traj in trajs:
        logger.info("Predict performance of %d entries in trajectory." %
                    (len(traj)))
        for entry in traj:
            config = entry["incumbent"]
            wc_time = entry["wallclock_time"]
            config_array = convert_configurations_to_array([config])
            m, v = smac.solver.model.predict_marginalized_over_instances(
                X=config_array)
    
            if smac.solver.scenario.run_obj == "runtime":
                p = 10**m[0, 0]
            else:
                p = m[0, 0]
                
            entry["cost"] = p
            
def aggregate_trajs(trajs:typing.List):
    '''
        at each point in time,
        the current best inc is used 
        given the cost value estimates
        
        Arguments
        ---------
        trajs:typing.List
            list of trajectory
            
        Returns
        -------
        new_traj:
            aggregated trajectory
    '''
    
    # put all entries in one list
    traj_list = []
    for traj in trajs:
        traj_list.extend(traj)
    
    # sort them by wc time stamp
    traj_list = sorted(traj_list, key=lambda x: x["wallclock_time"])
    
    # store first inc
    inc_entry = traj_list[0]
    # new traj
    new_traj = [inc_entry]
    
    for entry in traj_list:
        if entry["cost"] < inc_entry["cost"]:
            new_traj.append(entry)
            inc_entry = entry
            
    return new_traj
            
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

    predict_perf_of_traj(trajs=trajs, smac=smac)
    
    aggr_traj = aggregate_trajs(trajs=trajs)
    
    save_traj(traj=aggr_traj, save_dn=args_.save_dir)
