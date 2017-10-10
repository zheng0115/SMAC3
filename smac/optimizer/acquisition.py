# encoding=utf8
import abc
import logging
from scipy.stats import norm
import typing
import numpy as np

from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestClassifierWithInstances
from smac.configspace import Configuration
import json
import requests

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class AbstractAcquisitionFunction(object, metaclass=abc.ABCMeta):
    """Abstract base class for acquisition function

    Attributes
    ----------
    model
    logger
    """

    def __str__(self):
        return type(self).__name__ + " (" + self.long_name + ")"

    def __init__(self, model: AbstractEPM, **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            Models the objective function.
        """
        self.model = model
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

    def update(self, **kwargs):
        """Update the acquisition functions values.

        This method will be called if the model is updated. E.g.
        entropy search uses it to update its approximation of P(x=x_min),
        EI uses it to update the current fmin.

        The default implementation takes all keyword arguments and sets the
        respective attributes for the acquisition function object.

        Parameters
        ----------
        kwargs
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, X: np.ndarray):
        """Computes the acquisition value for a given X

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            acquisition values for X
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(np.float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X: np.ndarray):
        """Computes the acquisition value for a given point X. This function has
        to be overwritten in a derived class.

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Acquisition function values wrt X
        """
        raise NotImplementedError()

class EI(AbstractAcquisitionFunction):

    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi\right] \} ]`,
    with :math:`f(X^+)` as the incumbent.
    """

    def __init__(self,
                 model: AbstractEPM,
                 par: float=0.0,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super(EI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        z = (self.eta - m - self.par) / s

        f = (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            self.logger.warn("Predicted std is 0.0 for at least one sample.")
            f[s == 0.0] = 0.0

        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")

        return f
    

class EI_CONSTRAINT_VARIANT2(EI):
    def __init__(self, model: AbstractEPM, constraint_model:  RandomForestClassifierWithInstances, 
                 par: float=0.0, **kwargs):
        super().__init__(model=model, par=par,**kwargs)
        self.constraint_model = constraint_model

    def compute_success_probabilities(self, X: np.ndarray):
        success_probabilties = self.constraint_model.predict_marginalized_over_instances(X)[:,0]
        return success_probabilties.reshape((-1,1))
    
    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EIPS value.
 
        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
 
        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement per Second of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        ei_values = super()._compute(X=X, **kwargs)

        success_probabilities = self.compute_success_probabilities(X)

        product_ei_values_constraint_probs = success_probabilities * ei_values

        return product_ei_values_constraint_probs
 
    
class EI_CONSTRAINT_VARIANT3(EI):
    def __init__(self, model: AbstractEPM, instance_id,
                 step_size_of_sigmoid=0.0001, par: float=0.0, **kwargs):
        super().__init__(model=model, par=par,**kwargs)
        self.step_size_of_sigmoid = step_size_of_sigmoid
        self.instance_id = instance_id
        self.ignore_ei_term = False
        self.session = requests.Session()
        logging.getLogger("urllib3").setLevel(logging.WARNING)
   
       
    def update_success_probobilities(self, configs=typing.List[Configuration]):
        self.success_probabilities = np.zeros(shape=(len(configs), 1))
        
        for config_id, config in enumerate(configs):
            parameters = []
            for p in config:
                if not config.get(p) is None:
                    parameters.extend(["-" + str(p), str(config[p])])
            
            headers = {'content-type': 'application/json', 'Connection': 'Keep-Alive'}
            url = 'http://localhost:8180/approximateSuccessProbility/'
            #+ "," + str(sys.argv[1])
            parameters.append(self.instance_id)
            data = json.dumps(parameters)
            r_simulate = self.session.post(url, data=data, headers=headers)
            self.success_probabilities[config_id,0] = r_simulate.json()

    def compute_success_probabilities(self, X: np.ndarray):
        return self.success_probabilities
    
    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EIPS value.
 
        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
 
        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement per Second of X
        """
        if self.ignore_ei_term:
            return self.success_probabilities
        else:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]
            ei_values = super()._compute(X=X, **kwargs)
            return self.success_probabilities * ei_values


class EIPS(EI):
    def __init__(self,
                 model: AbstractEPM,
                 par: float=0.0,
                 **kwargs):
        r"""Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) := \frac{\mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi\right] \} ]} {np.log10(r(x))}`,
        with :math:`f(X^+)` as the incumbent and :math:`r(x)` as runtime.

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X) returning a tuples of
                   predicted cost and running time
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(EIPS, self).__init__(model, par=par)
        self.long_name = 'Expected Improvement per Second'

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EIPS value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement per Second of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        assert m.shape[1] == 2
        assert v.shape[1] == 2
        m_cost = m[:, 0]
        v_cost = v[:, 0]
        # The model already predicts log(runtime)
        m_runtime = m[:, 1]
        s = np.sqrt(v_cost)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        z = (self.eta - m_cost - self.par) / s
        f = (self.eta - m_cost - self.par) * norm.cdf(z) + s * norm.pdf(z)
        f = f / m_runtime
        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            self.logger.warn("Predicted std is 0.0 for at least one sample.")
            f[s == 0.0] = 0.0

        if (f < 0).any():
            raise ValueError("Expected Improvement per Second is smaller than "
                             "0 for at least one sample.")


        return f.reshape((-1, 1))


class LogEI(AbstractAcquisitionFunction):

    def __init__(self,
                 model: AbstractEPM,
                 par: float=0.0,
                 **kwargs):
        r"""Computes for a given x the logarithm expected improvement as
        acquisition value.

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LogEI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)

        f_min = self.eta - self.par
        v = (np.log(f_min) - m) / std
        log_ei = (f_min * norm.cdf(v)) - \
            (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

        if np.any(std == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            self.logger.warn("Predicted std is 0.0 for at least one sample.")
            log_ei[std == 0.0] = 0.0

        if (log_ei < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one sample.")

        return log_ei.reshape((-1, 1))
