User API
========

.. toctree::

.. note::
    The common classes ``Oxides``, ``Ions`` and ``APFU`` could be imported
    directly from main namespace, e.g.:

        >>> from epmatools import Oxides

    The modules ``minerals`` and ``plotting`` are available in main namespace as
    ``mindb`` and ``minplot`` for convenience.

    For interactive use you can import all above using:

        >>> from epmatools import *

EPMAtools provides following modules:

epmatools.datatables
--------------------

.. automodule:: epmatools.datatables
    :members:
    :inherited-members:

epmatools.minerals
------------------

.. automodule:: epmatools.minerals
    :members:
    :inherited-members:

epmatools.plotting
------------------

.. automodule:: epmatools.plotting
    :members:

To work with EDS maps you can use ``maps`` module. The class ``MapStore`` could be used
for H5 storage of your maps, allowing quick access of yout data. Individual dataset are
manipulated using ``Mapset`` class:

epmatools.maps
--------------

.. automodule:: epmatools.maps
    :members:
    :inherited-members:
