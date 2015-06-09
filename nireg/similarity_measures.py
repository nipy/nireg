from ._register import _L1_moments

import numpy as np
from scipy.ndimage import gaussian_filter

TINY = float(np.finfo(np.double).tiny)
SIGMA_FACTOR = 0.05
RENORMALIZATIONS = {'default': 0, 'ml': 1, 'nml': 2}
OVERLAP_MIN = 0.01

# A lambda function to force positive values
nonzero = lambda x: np.maximum(x, TINY)


def correlation2loglikelihood(rho2, npts, total_npts):
    """Re-normalize correlation.

    Convert a squared normalized correlation to a composite
    log-likelihood function of the registration transformation
    parameters. The result is a function of both the input correlation
    and the fraction of points in the image overlap.

    See: Roche, medical image registration through statistical
    inference, 2001.

    Parameters
    ----------
    rho2: float
      Squared correlation measure

    npts: int
      Number of source image voxels that transform within the domain
      of the reference image.

    total_npts: int
      Total number of source image voxels involved in computing the
      correlation, including voxels transforming outside the target
      image domain.

    Returns
    -------
    ll: float
      Logarithm of composite likelihood registration function.
    """
    tmp = float(npts) / total_npts
    return -.5 * tmp * np.log(nonzero(1 - rho2))


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

    def loss(self, H):
        return np.zeros(H.shape)

    def npoints(self, H):
        return H.sum()

    def overlap_penalty(self, npts):
        overlap = npts / self.total_npoints
        return self.penalty * np.log(max(OVERLAP_MIN, overlap))

    def __call__(self, H):
        total_loss = np.sum(H * self.loss(H))
        if self.renormalize == 0:
            total_loss /= nonzero(self.npoints(H))
        else:
            total_loss /= self.total_npoints
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

    def __call__(self, H):
        npts = nonzero(self.npoints(H))
        mI = np.sum(H * self.I) / npts
        mJ = np.sum(H * self.J) / npts
        vI = np.sum(H * (self.I) ** 2) / npts - mI ** 2
        vJ = np.sum(H * (self.J) ** 2) / npts - mJ ** 2
        cIJ = np.sum(H * self.J * self.I) / npts - mI * mJ
        rho2 = (cIJ / nonzero(np.sqrt(vI * vJ))) ** 2
        if self.renormalize:
            rho2 = correlation2loglikelihood(rho2, npts, self.total_npoints)
        return rho2


def correlation_ratio(H, Y):
    """Use a nonlinear regression model with Gaussian errors as a
    distribution model.

    Assume the input joint histogram has shape (dimX, dimY) where X is
    the predictor and Y is the response variable.

    Input array Y must be of same shape as H.
    """
    npts_X = np.sum(H, 1)
    tmp = nonzero(npts_X)
    mY_X = np.sum(H * Y, 1) / tmp
    vY_X = np.sum(H * (Y ** 2), 1) / tmp - mY_X ** 2
    npts = np.sum(npts_X)
    tmp = nonzero(npts)
    hY = np.sum(H, 0)
    hX = np.sum(H, 1)
    mY = np.sum(hY * Y[0, :]) / tmp
    vY = np.sum(hY * (Y[0, :] ** 2)) / tmp - mY ** 2
    mean_vY_X = np.sum(hX * vY_X) / tmp
    eta2 = 1. - mean_vY_X / nonzero(vY)
    return eta2, npts


class CorrelationRatio(SimilarityMeasure):
    def __call__(self, H):
        eta2, npts = correlation_ratio(H, self.I)
        if self.renormalize:
            eta2 = correlation2loglikelihood(eta2, npts, self.total_npoints)
        return eta2


class ReverseCorrelationRatio(SimilarityMeasure):
    def __call__(self, H):
        eta2, npts = correlation_ratio(H.T, self.J.T)
        if self.renormalize:
            eta2 = correlation2loglikelihood(eta2, npts, self.total_npoints)
        return eta2


def correlation_ratio_L1(H):
    """
    Use a nonlinear regression model with Laplace distributed errors
    as a distribution model.

    Assume the input joint histogram has shape (dimX, dimY) where X is
    the predictor and Y is the response variable.
    """
    moments = np.array([_L1_moments(H[x, :]) for x in range(H.shape[0])])
    npts_X, mY_X, sY_X = moments[:, 0], moments[:, 1], moments[:, 2]
    hY = np.sum(H, 0)
    hX = np.sum(H, 1)
    npts, mY, sY = _L1_moments(hY)
    mean_sY_X = np.sum(hX * sY_X) / nonzero(npts)
    tmp = mean_sY_X / nonzero(sY)
    return 1 - tmp, npts


class CorrelationRatioL1(CorrelationRatio):
    """
    Use a nonlinear regression model with Laplace distributed errors
    as a distribution model
    """
    def __call__(self, H):
        eta, npts = correlation_ratio_L1(H)
        if self.renormalize:
            eta = -(npts / self.total_npoints) * np.log(nonzero(1 - eta))
        return eta


class ReverseCorrelationRatioL1(CorrelationRatio):
    """
    Use a nonlinear regression model with Laplace distributed errors
    as a distribution model
    """
    def __call__(self, H):
        eta, npts = correlation_ratio_L1(H.T)
        if self.renormalize:
            eta = -(npts / self.total_npoints) * np.log(nonzero(1 - eta))
        return eta


similarity_measures = {
    'slr': SupervisedLikelihoodRatio,
    'mi': MutualInformation,
    'nmi': NormalizedMutualInformation,
    'pmi': ParzenMutualInformation,
    'dpmi': DiscreteParzenMutualInformation,
    'cc': CorrelationCoefficient,
    'cr': CorrelationRatio,
    'crl1': CorrelationRatioL1,
    'rcr': ReverseCorrelationRatio,
    'rcrl1': ReverseCorrelationRatioL1}
