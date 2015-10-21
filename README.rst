.. -*- rest -*-
.. vim:syntax=rst

=====
NIREG
=====

Image registration package for Python.


Website
=======

Current information can always be found at the `NIPY project website
<http://nipy.org>`_.

Mailing Lists
=============

For questions on how to use nipy or on making code contributions, please see
the ``neuroimaging`` mailing list:

    https://mail.python.org/mailman/listinfo/neuroimaging

Please report bugs at github issues:

    https://github.com/nipy/nireg/issues

You can see the list of current proposed changes at:

    https://github.com/nipy/nireg/pulls

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github;
* Documentation_ for all releases and current development tree;
* Download the `current development version`_ as a tar/zip file;
* Downloads of all `available releases`_.

.. _main repository: http://github.com/nipy/nireg
.. _Documentation: http://nipy.org/nipy
.. _current development version: https://github.com/nipy/nireg/archive/master.zip
.. _available releases: http://pypi.python.org/pypi/nireg

Tests
=====

To run nipy's tests, you will need to install the nose_ Python testing
package.  Then::

    python -c "import nireg; nireg.test()"


Dependencies
============


To run NIREG, you will need:
=======
* python_ >= 2.5 (tested with 2.5, 2.6, 2.7, 3.2, 3.3)
* numpy_ >= 1.2
* scipy_ >= 0.7.0
* nibabel_ >= 1.2

You will probably also like to have:

* ipython_ for interactive work
* matplotlib_ for 2D plotting
* mayavi_ for 3D plotting

.. _python: http://python.org
.. _numpy: http://numpy.scipy.org
.. _scipy: http://www.scipy.org
.. _nibabel: http://nipy.org/nibabel
.. _ipython: http://ipython.org
.. _matplotlib: http://matplotlib.org
.. _mayavi: http://code.enthought.com/projects/mayavi/
.. _nose: http://nose.readthedocs.org/en/latest

License
=======

We use the 3-clause BSD license; the full license is in the file ``LICENSE``
in the nipy distribution.
