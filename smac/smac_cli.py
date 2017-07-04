import os
import sys
import logging
import typing
from joblib import Parallel, delayed

import numpy as np

from smac.utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.facade.roar_facade import ROAR
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.utils.merge_foreign_data import merge_foreign_data_from_file
from smac.utils.io.traj_logging import TrajLogger
from smac.tae.execute_ta_run import TAEAbortException, FirstRunCrashedException
from smac.configspace import Configuration

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


def start(scens:typing.List[Scenario], 
          mode:str, 
          rh:RunHistory, 
          initial_configs:typing.List[Configuration], 
          seed:int,
          diversify:bool=False, 
          rr_portfolio:bool=False):
    '''
        decoupled method to be used in joblib
        
        Arguments
        ---------
        scens: typing.List[Scenario]
            list of scenario objects -- scens[seed] is used
        mode: str
            AC mode -- will be overwritten if rr_portfolio is set to True
        rh: RunHistory
            runhistory object
        initial_configs: typing.List[Configuration]
            list of initial configurations
        seed: int
            random seed -- special semantic of seed==1 -> only run starting from Default if diversify==True
        diversify: bool
            if set to True, all but seed==1 will start with a random configuration as initial inc
        rr_portfolio: bool
            if set to True, use a round robin portfolio of different AC modes
    '''

    portfolio_seq = ["SMAC", "ROAR", "EPILS"]

    if (not rr_portfolio and seed > 0 and diversify) or\
       (rr_portfolio and seed > len(portfolio_seq) and  diversify):
        scens[seed].initial_incumbent = "RANDOM"
       
    if (not rr_portfolio and mode == "SMAC") or (rr_portfolio and portfolio_seq[seed % 3] == "SMAC"):
        print("SMAC")
        optimizer = SMAC(
            scenario=scens[seed],
            rng=np.random.RandomState(seed),
            runhistory=rh,
            initial_configurations=initial_configs)
    elif (not rr_portfolio and mode == "ROAR") or (rr_portfolio and portfolio_seq[seed % 3] == "ROAR"):
        print("ROAR")
        optimizer = ROAR(
            scenario=scens[seed],
            rng=np.random.RandomState(seed),
            runhistory=rh,
            initial_configurations=initial_configs)
    elif (not rr_portfolio and mode == "EPILS") or (rr_portfolio and portfolio_seq[seed % 3] == "EPILS"):
        print("EPILS")
        optimizer = EPILS(
            scenario=scens[seed],
            rng=np.random.RandomState(seed),
            runhistory=rh,
            initial_configurations=initial_configs)
    try:
        optimizer.optimize()
        
    except (TAEAbortException, FirstRunCrashedException) as err:
        self.logger.error(err)


class SMACCLI(object):

    '''
    main class of SMAC
    '''

    def __init__(self):
        '''
            constructor
        '''
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

    def main_cli(self):
        '''
            main function of SMAC for CLI interface
        '''
        self.logger.info("SMAC call: %s" % (" ".join(sys.argv)))

        cmd_reader = CMDReader()
        args_, misc_args = cmd_reader.read_cmd()

        root_logger = logging.getLogger()
        root_logger.setLevel(args_.verbose_level)
        logger_handler = logging.StreamHandler(
                stream=sys.stdout)
        if root_logger.level >= logging.INFO:
            formatter = logging.Formatter(
                "%(levelname)s:\t%(message)s")
        else:
            formatter = logging.Formatter(
                "%(asctime)s:%(levelname)s:%(name)s:%(message)s",
                "%Y-%m-%d %H:%M:%S")
        logger_handler.setFormatter(formatter)
        root_logger.addHandler(logger_handler)
        # remove default handler
        root_logger.removeHandler(root_logger.handlers[0])
        
        if args_.parallel > 1:
            misc_args = {"shared_model": True}

        scens = [Scenario(args_.scenario_file, misc_args,
                        run_id=seed)
                 for seed in range(args_.seed, args_.seed+args_.parallel)]
        scen = scens[0]

        rh = None
        if args_.warmstart_runhistory:
            aggregate_func = average_cost
            rh = RunHistory(aggregate_func=aggregate_func)

            scen, rh = merge_foreign_data_from_file(
                scenario=scen,
                runhistory=rh,
                in_scenario_fn_list=args_.warmstart_scenario,
                in_runhistory_fn_list=args_.warmstart_runhistory,
                cs=scen.cs,
                aggregate_func=aggregate_func)

        initial_configs = None
        if args_.warmstart_incumbent:
            initial_configs = [scen.cs.get_default_configuration()]
            for traj_fn in args_.warmstart_incumbent:
                trajectory = TrajLogger.read_traj_aclib_format(
                    fn=traj_fn, cs=scen.cs)
                initial_configs.append(trajectory[-1]["incumbent"])
                
                
        Parallel(n_jobs=args_.parallel, backend="multiprocessing")\
                (delayed(start)\
                (scens=scens, mode=args_.mode, rh=rh, initial_configs=initial_configs, seed=seed,
                 diversify=args_.diversify, rr_portfolio=args_.rr_portfolio) 
                  for seed in range(args_.parallel))
