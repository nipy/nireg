# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module presents an interface to use the glm implemented in
nipy.algorithms.statistics.models.regression
It contains the GLM and contrast classes that are meant to be the main objects
of fMRI data analyses.

It is important to note that the GLM is meant as a one-session
General Linear Model. But inference can be performed on multiple sessions
by computing fixed effects on contrasts

>>> from nipy.modalities.fmri.glm import GeneralLinearModel
>>> import numpy as np
>>> n, p, q = 100, 80, 10
>>> X, Y = np.random.randn(p, q), np.random.randn(p, n)
>>> cval = np.hstack((1, np.zeros(9)))
>>> mulm = GeneralLinearModel(X)
>>> mulm.fit(Y)
>>> z_vals = mulm.contrast(cval).z_score() # z-transformed statistics
>>> # example of fixed effects statistics across two contrasts
>>> cval_ = cval.copy()
>>> np.random.shuffle(cval_)
>>> z_ffx = (mulm.contrast(cval) + mulm.contrast(cval_)).z_score()
"""

import numpy as np
import scipy.stats as sps
from nipy.algorithms.statistics.models.regression import OLSModel, ARModel
from nipy.labs.utils import mahalanobis
from nipy.labs.utils.zscore import zscore

DEF_TINY = 1e-50
DEF_DOFMAX = 1e10


class GeneralLinearModel(object):
    """ This class handles the so-called on General Linear Model

    Most of what it does in the fit() and contrast() methods
    fit() performs the standard two-step ('ols' then 'ar1') GLM fitting
    contrast() returns a contrast instance, yileding statistics and p-values.
    The link between fit() and constrast is done vis the two class members:
    glm_results: dictionary of nipy.algorithms.statistics.models.\
                 regression.RegressionResults instances,
                 describing results of a GLM fit
    labels: array of shape(n_voxels),
            labels that associate each voxel with a results key
    """

    def __init__(self, X):
        """
        Parameters
        ----------
        X: array of shape(n_time_points, n_regressors),
           the design matrix
        """
        self.X = X
        self.labels_ = None
        self.results_ = None

    def fit(self, Y, model='ar1', steps=100):
        """GLM fitting of a dataset using 'ols' regression or the two-pass

        Parameters
        ----------
        Y: array of shape(n_time_points, n_samples), the fMRI data
        model: string, to be chosen in ['ar1', 'ols'], optional,
               the temporal variance model. Defaults to 'ar1'
        steps: int, optional,
               Maximum number of discrete steps for the AR(1) coef histogram
        """
        if model not in ['ar1', 'ols']:
            raise ValueError('Unknown model')

        if Y.ndim == 1:
            Y = Y[:, np.newaxis]

        if Y.shape[0] != self.X.shape[0]:
            raise ValueError('Response and predictors are inconsistent')

        # fit the OLS model
        ols_result = OLSModel(self.X).fit(Y)

        # compute and discretize the AR1 coefs
        ar1 = ((ols_result.resid[1:] * ols_result.resid[:-1]).sum(0) /
               (ols_result.resid ** 2).sum(0))
        ar1 = (ar1 * steps).astype(np.int) * 1. / steps

        # Fit the AR model acccording to current AR(1) estimates
        if model == 'ar1':
            self.results_ = {}
            self.labels_ = ar1
            # fit the model
            for val in np.unique(self.labels_):
                m = ARModel(self.X, val)
                self.results_[val] = m.fit(Y[:, self.labels_ == val])
        else:
            self.labels_ = np.zeros(Y.shape[1])
            self.results_ = {0.0: ols_result}

    def contrast(self, con_val, contrast_type=None):
        """ Specify and estimate a linear contrast

        Parameters
        ----------
        con_val: numpy.ndarray of shape (p) or (q, p),
                 where q = number of contrast vectors
                 and p = number of regressors
        contrast_type: string, optional, either 't' or 'F',
                       type of the contrast

        Returns
        -------
        con: Contrast instance
        """
        if self.labels_ == None or self.results_ == None:
            raise ValueError('The model has not been estimated yet')
        con_val = np.asarray(con_val)
        if con_val.ndim == 1:
            dim = 1
        else:
            dim = con_val.shape[0]
        if contrast_type is None:
            if dim == 1:
                contrast_type = 't'
            else:
                contrast_type = 'F'
        if contrast_type not in ['t', 'F']:
            raise ValueError('Unknown contrast type: %s' % contrast_type)

        effect_ = np.zeros((dim, self.labels_.size), dtype=np.float)
        var_ = np.zeros((dim, dim, self.labels_.size), dtype=np.float)
        if contrast_type == 't':
            for l in self.results_.keys():
                resl = self.results_[l].Tcontrast(con_val)
                effect_[:, self.labels_ == l] = resl.effect.T
                var_[:, :, self.labels_ == l] = (resl.sd ** 2).T
        else:
            for l in self.results_.keys():
                resl = self.results_[l].Fcontrast(con_val)
                effect_[:, self.labels_ == l] = resl.effect
                var_[:, :, self.labels_ == l] = resl.covariance
        dof_ = self.results_[l].df_resid
        return Contrast(effect=effect_, variance=var_, dof=dof_,
                        contrast_type=contrast_type)


class Contrast(object):
    """ The contrast class handles the estimation of statistical contrasts
    After application of the GLM.
    The important feature is that it supports addition,
    thus opening the possibility of fixed-effects models.

    The current implementation is meant to be simple,
    and could be enhanced in the future on the computational side
    (high-dimensional F constrasts may lead to memory breakage)
    """

    def __init__(self, effect, variance, dof=DEF_DOFMAX, contrast_type='t',
                 tiny=DEF_TINY, dofmax=DEF_DOFMAX):
        """
        Parameters
        ==========
        effect: array of shape (contrast_dim, n_voxels)
                the effects related to the contrast
        variance: array of shape (contrast_dim, contrast_dim, n_voxels)
                  the associated variance estimate
        dof: scalar, the degrees of freedom
        contrast_type: string to be chosen among 't' and 'F'
        """
        if variance.ndim != 3:
            raise ValueError('Variance array should have 3 dimensions')
        if effect.ndim != 2:
            raise ValueError('Variance array should have 2 dimensions')
        if variance.shape[0] != variance.shape[1]:
            raise ValueError('Inconsistent shape for the variance estimate')
        if ((variance.shape[1] != effect.shape[0]) or
            (variance.shape[2] != effect.shape[1])):
            raise ValueError('Effect and variance have inconsistent shape')
        self.effect = effect
        self.variance = variance
        self.dof = float(dof)
        self.dim = effect.shape[0]
        if self.dim > 1 and contrast_type is 't':
            print 'Automatically converted multi-dimensional t to F contrast'
            contrast_type = 'F'
        self.contrast_type = contrast_type
        self.stat_ = None
        self.p_value_ = None
        self.baseline = 0
        self.tiny = tiny
        self.dofmax = dofmax

    def stat(self, baseline=0.0):
        """ Return the decision statistic associated with the test of the
        null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ==========
        baseline: float, optional,
                  Baseline value for the test statistic
        """
        self.baseline = baseline

        # Case: one-dimensional contrast ==> t or t**2
        if self.dim == 1:
            # avoids division by zero
            stat = (self.effect - baseline) / np.sqrt(
                np.maximum(self.variance, self.tiny))
            if self.contrast_type == 'F':
                stat = stat ** 2
        # Case: F contrast
        elif self.contrast_type == 'F':
            # F = |t|^2/q ,  |t|^2 = e^t inv(v) e
            if self.effect.ndim == 1:
                self.effect = self.effect[np.newaxis]
            if self.variance.ndim == 1:
                self.variance = self.variance[np.newaxis, np.newaxis]
            stat = (mahalanobis(self.effect - baseline, self.variance)
                    / self.dim)
        # Case: tmin (conjunctions)
        elif self.contrast_type == 'tmin':
            vdiag = self.variance.reshape([self.dim ** 2] + list(
                    self.variance.shape[2:]))[:: self.dim + 1]
            stat = (self.effect - baseline) / np.sqrt(
                np.maximum(vdiag, self.tiny))
            stat = stat.min(0)

        # Unknwon stat
        else:
            raise ValueError('Unknown statistic type')
        self.stat_ = stat
        return stat.ravel()

    def p_value(self, baseline=0.0):
        """Return a parametric estimate of the p-value associated
        with the null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ==========
        baseline: float, optional,
        Baseline value for the test statistic
        """
        if self.stat_ == None or not self.baseline == baseline:
            self.stat_ = self.stat(baseline)
        # Valid conjunction as in Nichols et al, Neuroimage 25, 2005.
        if self.contrast_type in ['t', 'tmin']:
            p = sps.t.sf(self.stat_, np.minimum(self.dof, self.dofmax))
        elif self.contrast_type == 'F':
            p = sps.f.sf(self.stat_, self.dim, np.minimum(
                    self.dof, self.dofmax))
        else:
            raise ValueError('Unknown statistic type')
        self.p_value_ = p
        return p

    def z_score(self, baseline=0.0):
        """Return a parametric estimation of the z-score associated
        with the null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ==========
        baseline: float, optional,
                  Baseline value for the test statistic
        """
        if self.p_value_ == None or not self.baseline == baseline:
            self.p_value_ = self.p_value(baseline)

        # Avoid inf values kindly supplied by scipy.
        return zscore(self.p_value_)

    def __add__(self, other):
        """Addition of selfwith others, Yields an new Contrast instance
        This should be used only on indepndent contrasts"""
        if self.contrast_type != other.contrast_type:
            raise ValueError(
                'The two contrasts do not have consistant type dimensions')
        if self.dim != other.dim:
            raise ValueError(
                'The two contrasts do not have compatible dimensions')
        effect_ = self.effect + other.effect
        variance_ = self.variance + other.variance
        dof_ = self.dof + other.dof
        return Contrast(effect=effect_, variance=variance_, dof=dof_,
                        contrast_type=self.contrast_type)

    def __rmul__(self, scalar):
        """Multiplication of the contrast by a scalar"""
        scalar = float(scalar)
        effect_ = self.effect * scalar
        variance_ = self.variance * scalar ** 2
        dof_ = self.dof
        return Contrast(effect=effect_, variance=variance_, dof=dof_,
                        contrast_type=self.contrast_type)

    __mul__ = __rmul__

    def __div__(self, scalar):
        return self.__rmul__(1 / float(scalar))
