import logging
import time
import numpy as np

from smac.configspace import get_one_exchange_neighbourhood, \
    convert_configurations_to_array, Configuration

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class LocalSearch(object):

    """Implementation of SMAC's local search

    Attributes
    ----------

    """

    def __init__(self, acquisition_function, config_space,
                 epsilon=0.00001, max_iterations=None, rng=None):
        """Constructor

        Parameters
        ----------
        acquisition_function:  function
            The function which the local search tries to maximize
        config_space:  ConfigSpace
            Parameter configuration space
        epsilon: float
            In order to perform a local move one of the incumbent's neighbors
            needs at least an improvement higher than epsilon
        max_iterations: int
            Maximum number of iterations that the local search will perform
        """
        self.config_space = config_space
        self.acquisition_function = acquisition_function
        self.epsilon = epsilon
        self.max_iterations = max_iterations

        if rng is None:
            self.rng = np.random.RandomState(seed=np.random.randint(10000))
        else:
            self.rng = rng

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

    def maximize(self, start_points, *args):
        """Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters
        ----------
        start_points:  np.array(1, D)
            The points from where the local search starts
        *args:
            Additional parameters that will be passed to the
            acquisition function

        Returns
        -------
        incumbent: np.array(1, D)
            The best found configuration
        acq_val_incumbent: np.array(1,1)
            The acquisition value of the incumbent

        """
        if isinstance(start_points, Configuration):
            start_points = [start_points]
        incumbents = start_points
        # Compute the acquisition value of the incumbent
        incumbent_array = convert_configurations_to_array(incumbents)
        num_incumbents = len(incumbents)
        acq_val_incumbents = self.acquisition_function(incumbent_array, *args)
        if num_incumbents == 1:
            acq_val_incumbents = [acq_val_incumbents]
        active = [True] * num_incumbents

        local_search_steps = [0] * num_incumbents
        neighbors_looked_at = [0] * num_incumbents
        times = []

        while True:

            # local_search_steps += 1
            #if local_search_steps % 1000 == 0:
            #    self.logger.warning("Local search took already %d iterations."
            #                        "Is it maybe stuck in a infinite loop?",
            #                        local_search_steps)

            changed_inc = np.zeros((num_incumbents), dtype=np.bool)
            neighborhood_iterators = []
            for i, inc in enumerate(incumbents):
                if active[i]:
                    neighborhood_iterators.append(get_one_exchange_neighbourhood(
                        inc, seed=self.rng.seed()))
                    local_search_steps[i] += 1
                else:
                    neighborhood_iterators.append(None)

            while np.any(active) and np.any(~changed_inc[active]):

                # gather all neighbors
                neighbors = []
                for i, neighborhood_iterator in enumerate(neighborhood_iterators):
                    if active[i] and not changed_inc[i]:
                        try:
                            neighbors.append(next(neighborhood_iterator))
                            neighbors_looked_at[i] += 1
                        except StopIteration:
                            active[i] = False

                if len(neighbors) == 0:
                    continue

                neighbor_array_ = convert_configurations_to_array(neighbors)
                start_time = time.time()
                acq_val = self.acquisition_function(neighbor_array_, *args)
                end_time = time.time()
                times.append(end_time - start_time)
                if num_incumbents == 1:
                    acq_val = [acq_val]

                acq_index = 0
                for i in range(num_incumbents):
                    if not active[i] or changed_inc[i]:
                        continue
                    if acq_val[acq_index] > acq_val_incumbents[i] + self.epsilon:
                        self.logger.debug("Switch to one of the neighbors")
                        incumbents[i] = neighbors[acq_index]
                        acq_val_incumbents[i] = acq_val[acq_index]
                        changed_inc[i] = True
                    acq_index += 1

            if (not np.any(changed_inc)) or (self.max_iterations != None
                                             and local_search_steps == self. max_iterations):
                self.logger.debug("Local search took %s steps and looked at %s configurations. "
                                  "Call to acquisition function took %f seconds on average.",
                                  local_search_steps, neighbors_looked_at, np.mean(times))
                break

        for inc in incumbents:
            inc.origin = 'Local search'

        return [(i, a) for i, a in zip(acq_val_incumbents, incumbents)]
