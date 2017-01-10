import numpy as np
import scipy as sp
import logging

import pyrfr.regression

from smac.epm.rf_with_instances import RandomForestWithInstances


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class WarmstartedRandomForestWithInstances(RandomForestWithInstances):

    '''
    Interface to the random forest that takes instance features
    into account.

    Parameters
    ----------
    types: np.ndarray (D)
        Specifies the number of categorical values of an input dimension. Where
        the i-th entry corresponds to the i-th input dimension. Let say we have
        2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass np.array([2, 0]). Note that we count starting from 0.
    instance_features: np.ndarray (I, K)
        Contains the K dimensional instance features
        of the I different instances
    num_trees: int
        The number of trees in the random forest.
    do_bootstrapping: bool
        Turns on / off bootstrapping in the random forest.
    ratio_features: float
        The ratio of features that are considered for splitting.
    min_samples_split: int
        The minimum number of data points to perform a split.
    min_samples_leaf: int
        The minimum number of data points in a leaf.
    max_depth: int

    eps_purity: float

    max_num_nodes: int

    seed: int
        The seed that is passed to the random_forest_run library
        
    warmstart_models: list
        list of already trained models
    '''

    def __init__(self, types,
                 instance_features=None,
                 num_trees=10,
                 do_bootstrapping=True,
                 n_points_per_tree=0,
                 ratio_features=5. / 6.,
                 min_samples_split=3,
                 min_samples_leaf=3,
                 max_depth=20,
                 eps_purity=1e-8,
                 max_num_nodes=1000,
                 seed=42,
                 warmstart_models=None):

        RandomForestWithInstances.__init__(self, types=types, 
                                           instance_features=instance_features, 
                                           num_trees=num_trees, 
                                           do_bootstrapping=do_bootstrapping, 
                                           n_points_per_tree=n_points_per_tree, 
                                           ratio_features=ratio_features, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf, 
                                           max_depth=max_depth, 
                                           eps_purity=eps_purity, 
                                           max_num_nodes=max_num_nodes, 
                                           seed=seed)
        
        self.warmstart_models = warmstart_models
        self.weights = []

        self.logger = logging.getLogger("WarmstartedRandomForestWithInstances")

    def train(self, X, y, **kwargs):
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """
        
        super(WarmstartedRandomForestWithInstances,self).train(X,y)

        y_pred = super(WarmstartedRandomForestWithInstances,self).predict(X)[0]
        tau = self._do_test(y=y, y_pred=y_pred)
        
        self.weights = [tau]
        for model in self.warmstart_models:
            y_pred = model.predict(X)[0]
            tau = self._do_test(y=y, y_pred=y_pred)
            self.weights.append(tau)

        self.logger.debug("Model weights: %s" %(str(self.weights)))
            
        sum_weights = np.sum(self.weights)
        if sum_weights == 0:
            self.weights = np.zeros((len(self.weights))) + 1/len(self.weights)
        else:
            self.weights = np.array(self.weights) / sum_weights
        
        self.logger.debug("Normalized Model weights: %s" %(str(self.weights)))
            
        return self
    
    def _do_test(self, y, y_pred):
        tau, _p_value = sp.stats.kendalltau(x=y, y=y_pred)
        if tau < 0: # anti correlated -> weight of 0
            tau = 0
        elif np.isnan(tau): #if X has only one sample
                tau = 0
        return tau
    

    def predict(self, X):
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance
        features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        
        weights = self.weights
        base_pred = super(WarmstartedRandomForestWithInstances,self).predict(X)
        means = [base_pred[0]]
        vars = [base_pred[1]]
        
        for model in self.warmstart_models:
            warm_y = model.predict(X)
            means.append(warm_y[0])
            vars.append(warm_y[1])

        mean = np.average(means, weights=weights, axis=0)
        var_of_means = np.average((means - mean)**2, weights=weights, axis=0)
        var = np.average(vars, weights=weights, axis=0) + var_of_means
        
        return mean, var  
        
