import os
import sys
import logging
import numpy as np

from smac.utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.facade.roar_facade import ROAR
from smac.runhistory.runhistory import RunHistory
from smac.smbo.objective import average_cost
from smac.utils.merge_foreign_data import merge_foreign_data_from_file
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.constants import MAXINT
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rf_with_instances_warmstarted import WarmstartedRandomForestWithInstances
from smac.utils.util_funcs import get_types

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class SMACCLI(object):

    '''
    main class of SMAC
    '''

    def __init__(self):
        '''
            constructor
        '''
        self.logger = logging.getLogger("SMAC")

    def main_cli(self):
        '''
            main function of SMAC for CLI interface
        '''

        cmd_reader = CMDReader()
        args_, misc_args = cmd_reader.read_cmd()

        logging.basicConfig(level=args_.verbose_level)

        root_logger = logging.getLogger()
        root_logger.setLevel(args_.verbose_level)

        scen = Scenario(args_.scenario_file, misc_args)

        initial_configs = None
        if args_.warmstart_incumbent:
            initial_configs = [scen.cs.get_default_configuration()]
            for traj_fn in args_.warmstart_incumbent:
                trajectory = TrajLogger.read_traj_aclib_format(
                    fn=traj_fn, cs=scen.cs)
                initial_configs.append(trajectory[-1]["incumbent"])

        rh = None
        if args_.warmstart_runhistory:
            if args_.warmstart_mode == "FULL":
                aggregate_func = average_cost
                rh = RunHistory(aggregate_func=aggregate_func)

                scen, rh = merge_foreign_data_from_file(
                    scenario=scen,
                    runhistory=rh,
                    in_scenario_fn_list=args_.warmstart_scenario,
                    in_runhistory_fn_list=args_.warmstart_runhistory,
                    cs=scen.cs,
                    aggregate_func=aggregate_func)
            elif args_.warmstart_mode == "WEIGHTED":
                SMAC = self.weighted_warmstart(scenario, args)

        if args_.modus == "SMAC":
            optimizer = SMAC(
                scenario=scen,
                rng=np.random.RandomState(args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs)
        elif args_.modus == "ROAR":
            optimizer = ROAR(
                scenario=scen,
                rng=np.random.RandomState(args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs)
        try:
            optimizer.optimize()

        finally:
            # ensure that the runhistory is always dumped in the end
            if scen.output_dir is not None:
                optimizer.solver.runhistory.save_json(
                    fn=os.path.join(scen.output_dir, "runhistory.json"))
        #smbo.runhistory.load_json(fn="runhistory.json", cs=smbo.config_space)

    def weighted_warmstart(self, scenario: Scenario, args):
        '''
            train models from loaded runhistories 
            and uses them to later weight predictions
        '''

        aggregate_func = average_cost

        # initial EPM
        types = get_types(scenario.cs, scenario.feature_array)
        model = RandomForestWithInstances(types=types,
                                          instance_features=scenario.feature_array,
                                          seed=rng.randint(MAXINT))

        # initial conversion of runhistory into EPM data
        num_params = len(scenario.cs.get_hyperparameters())
        if scenario.run_obj == "runtime":

            # if we log the performance data,
            # the RFRImputator will already get
            # log transform data from the runhistory
            cutoff = np.log10(scenario.cutoff)
            threshold = np.log10(scenario.cutoff *
                                 scenario.par_factor)

            imputor = RFRImputator(rs=rng,
                                   cutoff=cutoff,
                                   threshold=threshold,
                                   model=model,
                                   change_threshold=0.01,
                                   max_iter=2)

            runhistory2epm = RunHistory2EPM4LogCost(
                scenario=scenario, num_params=num_params,
                success_states=[StatusType.SUCCESS, ],
                impute_censored_data=True,
                impute_state=[StatusType.TIMEOUT, ],
                imputor=imputor)

        elif scenario.run_obj == 'quality':
            runhistory2epm = RunHistory2EPM4Cost\
                (scenario=scenario, num_params=num_params,
                 success_states=[StatusType.SUCCESS, ],
                 impute_censored_data=False, impute_state=None)

        else:
            raise ValueError('Unknown run objective: %s. Should be either '
                             'quality or runtime.' % self.scenario.run_obj)

        warmstart_models = []
        for rh_fn in args_.warmstart_runhistory:
            rh = RunHistory(aggregate_func)
            rh.load_json(rh_fn, cs)

            model = RandomForestWithInstances(types=types,
                                              instance_features=scenario.feature_array,
                                              seed=rng.randint(MAXINT))
            
            X,y = runhistory2epm.transform(rh)
            model.train(X,y)
            warmstart_models.append(model)
        
        model = WarmstartedRandomForestWithInstances(types=types,
                                              instance_features=scenario.feature_array,
                                              seed=rng.randint(MAXINT),
                                              warmstart_models=warmstart_models)
        
        #TODO: continue
            
