from ._register import _L1_moments

import numpy as np
from scipy.ndimage import gaussian_filter

TINY = float(np.finfo(np.double).tiny)
SIGMA_FACTOR = 0.05
RENORMALIZATIONS = {'default': 0, 'ml': 1, 'nml': 2}
OVERLAP_MIN = 0.01

# A lambda function to force positive values
nonzero = lambda x: np.maximum(x, TINY)


def correlation2loglikelihood(rho2, npts):
    """
    Re-normalize correlation.

    Convert a squared normalized correlation to a proper
    log-likelihood associated with a registration problem. The result
    is a function of both the input correlation and the number of
    points in the image overlap.

    See: Roche, medical image registration through statistical
    inference, 2001.

    Parameters
    ----------
    rho2: float
      Squared correlation measure

    npts: int
      Number of points involved in computing `rho2`

    Returns
    -------
    ll: float
      Log-likelihood re-normalized `rho2`
    """
    return -.5 * npts * np.log(nonzero(1 - rho2))


def dist2loss(q, qI=None, qJ=None):
    """
    Convert a joint distribution model q(i,j) into a pointwise loss:

    L(i,j) = - log q(i,j)/(q(i)q(j))

    where q(i) = sum_j q(i,j) and q(j) = sum_i q(i,j)

    See: Roche, medical image registration through statistical
    inference, 2001.
    """
    qT = q.T
    if qI is None:
        qI = q.sum(0)
    if qJ is None:
        qJ = q.sum(1)
    q /= nonzero(qI)
    qT /= nonzero(qJ)
    return -np.log(nonzero(q))


class SimilarityMeasure(object):
    """
    Template class
    """
    def __init__(self, shape, total_npoints, renormalize=None, dist=None):
        self.shape = shape
        self.J, self.I = np.indices(shape)
        self.renormalize = RENORMALIZATIONS[renormalize]
        if dist is None:
            self.dist = None
        else:
            self.dist = dist.copy()
        self.total_npoints = nonzero(float(total_npoints))
        self.penalty = .5 * self.degrees_of_freedom() / self.total_npoints

    def loss(self, H):
        return np.zeros(H.shape)

    def npoints(self, H):
        return H.sum()

    def overlap_penalty(self, npts):
        overlap = npts / self.total_npoints
        return self.penalty * np.log(max(OVERLAP_MIN, overlap))

    def degrees_of_freedom(self):
        return 0

    def __call__(self, H):
        total_loss = np.sum(H * self.loss(H))
        if self.renormalize == 0:
            total_loss /= nonzero(self.npoints(H))
        elif self.renormalize > 0:
            total_loss /= self.total_npoints
            if self.renormalize == 2:
                total_loss += self.overlap_penalty(self.npoints(H))
        return -total_loss


class SupervisedLikelihoodRatio(SimilarityMeasure):
    """
    Assume a joint intensity distribution model is given by self.dist
    """
    def loss(self, H):
        if not hasattr(self, 'L'):
            if self.dist is None:
                raise ValueError('SupervisedLikelihoodRatio: dist attribute cannot be None')
            if not self.dist.shape == H.shape:
                raise ValueError('SupervisedLikelihoodRatio: wrong shape for dist attribute')
            self.L = dist2loss(self.dist)
        return self.L


class MutualInformation(SimilarityMeasure):
    """
    Use the normalized joint histogram as a distribution model
    """
    def loss(self, H):
        return dist2loss(H / nonzero(self.npoints(H)))

    def degrees_of_freedom(self):
        return np.prod(self.shape) - np.sum(self.shape)



class ParzenMutualInformation(MutualInformation):
    """
    Use Parzen windowing to estimate the distribution model
    """
    def loss(self, H):
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_FACTOR * np.array(H.shape)
        npts = nonzero(self.npoints(H))
        Hs = H / npts
        gaussian_filter(Hs, sigma=self.sigma, mode='constant', output=Hs)
        return dist2loss(Hs)


class DiscreteParzenMutualInformation(MutualInformation):
    """
    Use Parzen windowing in the discrete case to estimate the
    distribution model
    """
    def loss(self, H):
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_FACTOR * np.array(H.shape)
        Hs = gaussian_filter(H, sigma=self.sigma, mode='constant')
        Hs /= nonzero(Hs.sum())
        return dist2loss(Hs)


class NormalizedMutualInformation(SimilarityMeasure):
    """
    NMI = [H(I) + H(H)] / H(I,J)

    Note the previous implementation returned the entropy correlation
    coefficient:

    ECC = 2*(1 - H(I,J) / [H(I) + H(J)])
        
    which is equivalent to NMI (in the sense that it is an increasing
    function of NMI) but is not the NMI measure as defined by
    Studholme et al, Pattern Recognition, 1998.
    """
    def __call__(self, H):
        H = H / nonzero(self.npoints(H))
        hI = H.sum(0)
        hJ = H.sum(1)
        entIJ = -np.sum(H * np.log(nonzero(H)))
        entI = -np.sum(hI * np.log(nonzero(hI)))
        entJ = -np.sum(hJ * np.log(nonzero(hJ)))
        #return 2 * (1 - entIJ / nonzero(entI + entJ))
        return (entI + entJ) / nonzero(entIJ)


class CorrelationCoefficient(SimilarityMeasure):
    """
    Use a bivariate Gaussian as a distribution model
    """
    def loss(self, H):
        rho2 = self(H)
        I = (self.I - self.mI) / np.sqrt(nonzero(self.vI))
        J = (self.J - self.mJ) / np.sqrt(nonzero(self.vJ))
        L = rho2 * I ** 2 + rho2 * J ** 2 - 2 * self.rho * I * J
        tmp = nonzero(1. - rho2)
        L *= .5 / tmp
        L += .5 * np.log(tmp)
        return L

    def degrees_of_freedom(self):
        return 1

    def __call__(self, H):
        npts = nonzero(self.npoints(H))
        mI = np.sum(H * self.I) / npts
        mJ = np.sum(H * self.J) / npts
        vI = np.sum(H * (self.I) ** 2) / npts - mI ** 2
        vJ = np.sum(H * (self.J) ** 2) / npts - mJ ** 2
        cIJ = np.sum(H * self.J * self.I) / npts - mI * mJ
        rho2 = (cIJ / nonzero(np.sqrt(vI * vJ))) ** 2
        if self.renormalize:
            rho2 = correlation2loglikelihood(rho2, npts) / self.total_npoints
            if self.renormalize == 2:
                rho2 -= self.overlap_penalty(npts)
        return rho2


class CorrelationRatio(SimilarityMeasure):
    """
    Use a nonlinear regression model with Gaussian errors as a
    distribution model
    """
    def __call__(self, H):
        npts_J = np.sum(H, 1)
        tmp = nonzero(npts_J)
        mI_J = np.sum(H * self.I, 1) / tmp
        vI_J = np.sum(H * (self.I) ** 2, 1) / tmp - mI_J ** 2
        npts = np.sum(npts_J)
        tmp = nonzero(npts)
        hI = np.sum(H, 0)
        hJ = np.sum(H, 1)
        mI = np.sum(hI * self.I[0, :]) / tmp
        vI = np.sum(hI * self.I[0, :] ** 2) / tmp - mI ** 2
        mean_vI_J = np.sum(hJ * vI_J) / tmp
        eta2 = 1. - mean_vI_J / nonzero(vI)
        if self.renormalize:
            eta2 = correlation2loglikelihood(eta2, npts) / self.total_npoints
            if self.renormalize == 2:
                eta2 -= self.overlap_penalty(npts)
        return eta2

    def degrees_of_freedom(self):
        return self.shape[0] - 1


class CorrelationRatioL1(CorrelationRatio):
    """
    Use a nonlinear regression model with Laplace distributed errors
    as a distribution model
    """
    def __call__(self, H):
        moments = np.array([_L1_moments(H[j, :]) for j in range(H.shape[0])])
        npts_J, mI_J, sI_J = moments[:, 0], moments[:, 1], moments[:, 2]
        hI = np.sum(H, 0)
        hJ = np.sum(H, 1)
        npts, mI, sI = _L1_moments(hI)
        mean_sI_J = np.sum(hJ * sI_J) / nonzero(npts)
        tmp = mean_sI_J / nonzero(sI)
        if self.renormalize == 0:
            eta = 1. - tmp
        elif self.renormalize > 0:
            eta = -(npts / self.total_npoints) * np.log(nonzero(tmp))
            if self.renormalize == 2:
                eta -= self.overlap_penalty(npts)
        return eta


similarity_measures = {
    'slr': SupervisedLikelihoodRatio,
    'mi': MutualInformation,
    'nmi': NormalizedMutualInformation,
    'pmi': ParzenMutualInformation,
    'dpmi': DiscreteParzenMutualInformation,
    'cc': CorrelationCoefficient,
    'cr': CorrelationRatio,
    'crl1': CorrelationRatioL1}
