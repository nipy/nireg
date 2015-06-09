# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The testing directory contains a small set of imaging files to be
used for doctests only.
"""
import os

#__all__ = ['funcfile', 'anatfile']

# Discover directory path
filepath = os.path.abspath(__file__)
basedir = os.path.dirname(filepath)
funcfile = os.path.join(basedir, 'functional.nii.gz')
anatfile = os.path.join(basedir, 'anatomical.nii.gz')

