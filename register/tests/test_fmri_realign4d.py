from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import warnings

from nose.tools import assert_equal
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal,
                           assert_raises)
import numpy as np
from nibabel import (load, Nifti1Image, io_orientation)

from ..testing import funcfile
from ..groupwise_registration import (Image4d,
                                      resample4d,
                                      SpaceTimeRealign,
                                      SpaceRealign,
                                      Realign4d,
                                      Realign4dAlgorithm,
                                      make_grid)
from ..slicetiming.timefuncs import st_43210, st_02413, st_42031
from ..affine import Rigid

im = load(funcfile)


def test_scanner_time():
    im4d = Image4d(im.get_data(), im.get_affine(), tr=3.,
                   slice_times=(0, 1, 2))
    assert_equal(im4d.scanner_time(0, 0), 0.)
    assert_equal(im4d.scanner_time(0, im4d.tr), 1.)


def test_slice_info():
    im4d = Image4d(im.get_data(), im.get_affine(), tr=3.,
                   slice_times=(0, 1, 2), slice_info=(2, -1))
    assert_equal(im4d.slice_axis, 2)
    assert_equal(im4d.slice_direction, -1)


def test_slice_timing():
    affine = np.eye(4)
    affine[0:3, 0:3] = im.get_affine()[0:3, 0:3]
    im4d = Image4d(im.get_data(), affine, tr=2., slice_times=0.0)
    x = resample4d(im4d, [Rigid() for i in range(im.shape[3])])
    assert_array_almost_equal(im4d.get_data(), x)


def test_realign4d_no_time_interp():
    runs = [im, im]
    R = SpaceRealign(runs)
    assert R.slice_times == 0


def test_realign4d_ascending():
    runs = [im, im]
    R = SpaceTimeRealign(runs, tr=3, slice_times='ascending', slice_info=2)
    assert_array_equal(R.slice_times, (0, 1, 2))
    assert R.tr == 3


def test_realign4d_descending():
    runs = [im, im]
    R = SpaceTimeRealign(runs, tr=3, slice_times='descending', slice_info=2)
    assert_array_equal(R.slice_times, (2, 1, 0))
    assert R.tr == 3


def test_realign4d_ascending_interleaved():
    runs = [im, im]
    R = SpaceTimeRealign(runs, tr=3, slice_times='asc_alt_2', slice_info=2)
    assert_array_equal(R.slice_times, (0, 2, 1))
    assert R.tr == 3


def test_realign4d_descending_interleaved():
    runs = [im, im]
    R = SpaceTimeRealign(runs, tr=3, slice_times='desc_alt_2', slice_info=2)
    assert_array_equal(R.slice_times, (1, 2, 0))
    assert R.tr == 3


def test_realign4d():
    """This tests whether realign4d yields the same results depending on
    whether the slice order is input explicitely or as
    slice_times='ascending'.

    Due to the very small size of the image used for testing (only 3
    slices), optimization is numerically unstable. It seems to make
    the default optimizer, namely scipy.fmin.fmin_ncg, adopt a random
    behavior. To work around the resulting inconsistency in results,
    we use a custom steepest gradient descent as the optimizer,
    although it's generally not recommended in practice.
    """
    runs = [im, im]
    orient = io_orientation(im.get_affine())
    slice_axis = int(np.where(orient[:, 0] == 2)[0])
    R1 = SpaceTimeRealign(runs, tr=2., slice_times='ascending',
                          slice_info=slice_axis)
    R1.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    nslices = im.shape[slice_axis]
    slice_times = (2. / float(nslices)) * np.arange(nslices)
    R2 = SpaceTimeRealign(runs, tr=2., slice_times=slice_times,
                          slice_info=slice_axis)
    R2.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    for r in range(2):
        for i in range(im.shape[3]):
            assert_array_almost_equal(R1._transforms[r][i].translation,
                                      R2._transforms[r][i].translation)
            assert_array_almost_equal(R1._transforms[r][i].rotation,
                                      R2._transforms[r][i].rotation)
    for i in range(im.shape[3]):
            assert_array_almost_equal(R1._mean_transforms[r].translation,
                                      R2._mean_transforms[r].translation)
            assert_array_almost_equal(R1._mean_transforms[r].rotation,
                                      R2._mean_transforms[r].rotation)


def test_realign4d_runs_with_different_affines():
    aff = im.get_affine()
    aff2 = aff.copy()
    aff2[0:3, 3] += 5
    im2 = Nifti1Image(im.get_data(), aff2)
    runs = [im, im2]
    R = SpaceTimeRealign(runs, tr=2., slice_times='ascending', slice_info=2)
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    cor_im, cor_im2 = R.resample()
    assert_array_equal(cor_im2.get_affine(), aff)


def test_realign4d_params():
    # Some tests for input parameters to realign4d
    R = Realign4d(im, 3, [0, 1, 2], None) # No slice_info - OK
    assert_equal(R.tr, 3)
    # TR cannot be None for set slice times
    assert_raises(ValueError, Realign4d, im, None, [0, 1, 2], None)
    # TR can be None if slice times are None
    R = Realign4d(im, None, None)
    assert_equal(R.tr, 1)


def test_spacetimerealign_params():
    runs = [im, im]
    for slice_times in ('descending', '43210', st_43210, [2, 1, 0]):
        R = SpaceTimeRealign(runs, tr=3, slice_times=slice_times, slice_info=2)
        assert_array_equal(R.slice_times, (2, 1, 0))
        assert_equal(R.tr, 3)
    for slice_times in ('asc_alt_2', '02413', st_02413, [0, 2, 1]):
        R = SpaceTimeRealign(runs, tr=3, slice_times=slice_times, slice_info=2)
        assert_array_equal(R.slice_times, (0, 2, 1))
        assert_equal(R.tr, 3)
    for slice_times in ('desc_alt_2', '42031', st_42031, [1, 2, 0]):
        R = SpaceTimeRealign(runs, tr=3, slice_times=slice_times, slice_info=2)
        assert_array_equal(R.slice_times, (1, 2, 0))
        assert_equal(R.tr, 3)
    # Check changing axis
    R = SpaceTimeRealign(runs, tr=21, slice_times='ascending', slice_info=1)
    assert_array_equal(R.slice_times, np.arange(21))
    # Check slice_times and slice_info and TR required
    R = SpaceTimeRealign(runs, 3, 'ascending', 2) # OK
    assert_raises(ValueError, SpaceTimeRealign, runs, 3, None, 2)
    assert_raises(ValueError, SpaceTimeRealign, runs, 3, 'ascending', None)
    assert_raises(ValueError, SpaceTimeRealign, runs, None, [0, 1, 2], 2)
    # Test when TR and nslices are not the same
    R1 = SpaceTimeRealign(runs, tr=2., slice_times='ascending', slice_info=2)
    assert_array_equal(R1.slice_times, np.arange(3) / 3. * 2.)
    # Smoke test run
    R1.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')


def reduced_dim(dim, subsampling, border):
    return max(1, int(np.ceil((dim - 2 * border) / float(subsampling))))


def test_lowlevel_params():
    runs = [im, im]
    R = SpaceTimeRealign(runs, tr=21, slice_times='ascending', slice_info=1)
    borders=(3,2,1)
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest', borders=borders)
    # Test tighter borders for motion estimation
    r = Realign4dAlgorithm(R._runs[0], borders=borders)
    nvoxels = np.prod(np.array([reduced_dim(im.shape[i], 1, borders[i]) for i in range(3)]))
    assert_array_equal(r.xyz.shape, (nvoxels, 3))
    # Test wrong argument types raise errors
    assert_raises(ValueError, Realign4dAlgorithm, R._runs[0], subsampling=(3,3,3,1))
    assert_raises(ValueError, Realign4dAlgorithm, R._runs[0], refscan='first')
    assert_raises(ValueError, Realign4dAlgorithm, R._runs[0], borders=(1,1,1,0))
    assert_raises(ValueError, Realign4dAlgorithm, R._runs[0], xtol=None)
    assert_raises(ValueError, Realign4dAlgorithm, R._runs[0], ftol='dunno')
    assert_raises(ValueError, Realign4dAlgorithm, R._runs[0], gtol=(.1,.1,.1))
    assert_raises(ValueError, Realign4dAlgorithm, R._runs[0], stepsize=None)
    assert_raises(ValueError, Realign4dAlgorithm, R._runs[0], maxiter=None)
    assert_raises(ValueError, Realign4dAlgorithm, R._runs[0], maxfun='none')


def _test_make_grid(dims, subsampling, borders, expected_nvoxels):
    x = make_grid(dims, subsampling, borders)
    assert_equal(x.shape[0], expected_nvoxels)


def test_make_grid_funfile():
    dims = im.shape[0:3]
    borders = (3,2,1)
    nvoxels = np.prod(np.array([reduced_dim(dims[i], 1, borders[i]) for i in range(3)]))
    _test_make_grid(dims, (1,1,1), borders, nvoxels)           


def test_make_grid_default():
    dims = np.random.randint(100, size=3) + 1
    _test_make_grid(dims, (1,1,1), (0,0,0), np.prod(dims))           


def test_make_grid_random_subsampling():
    dims = np.random.randint(100, size=3) + 1
    subsampling = np.random.randint(5, size=3) + 1
    nvoxels = np.prod(np.array([reduced_dim(dims[i], subsampling[i], 0) for i in range(3)]))
    _test_make_grid(dims, subsampling, (0,0,0), nvoxels)           


def test_make_grid_random_borders():
    dims = np.random.randint(100, size=3) + 1
    borders = np.minimum((dims - 1) / 2, np.random.randint(10, size=3))
    nvoxels = np.prod(np.array([reduced_dim(dims[i], 1, borders[i]) for i in range(3)]))
    _test_make_grid(dims, (1,1,1), borders, nvoxels)           


def test_make_grid_full_monthy():
    dims = np.random.randint(100, size=3) + 1
    subsampling = np.random.randint(5, size=3) + 1
    borders = np.minimum((dims - 1) / 2, np.random.randint(10, size=3))
    nvoxels = np.prod(np.array([reduced_dim(dims[i], subsampling[i], borders[i]) for i in range(3)]))
    _test_make_grid(dims, subsampling, borders, nvoxels)           


def test_spacerealign():
    # Check space-only realigner
    runs = [im, im]
    R = SpaceRealign(runs)
    assert_equal(R.tr, 1)
    assert_equal(R.slice_times, 0.)
    # Smoke test run
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')


def test_single_image():
    # Check we can use a single image as argument
    R = SpaceTimeRealign(im, tr=3, slice_times='ascending', slice_info=2)
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    R = SpaceRealign(im)
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
    R = Realign4d(im, 3, [0, 1, 2], (2, 1))
    R.estimate(refscan=None, loops=1, between_loops=1, optimizer='steepest')
