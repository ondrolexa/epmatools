User API
========

.. toctree::

.. note::
    The common classes ``Oxides``, ``Ions`` and ``APFU`` could be imported
    directly from main namespace, e.g.:

        >>> from empatools import Oxides

    The modules ``minerals`` and ``plotting`` are available in main namespace as
    ``mindb`` and ``minplot`` for convenience.

    For interactive use you can import all above using:

        >>> from empatools import *

EMPAtools provides following modules:

empatools.datatables
--------------------

.. automodule:: empatools.datatables
    :members:
    :inherited-members:

empatools.minerals
------------------

.. automodule:: empatools.minerals
    :members:
    :inherited-members:

empatools.plotting
------------------

.. automodule:: empatools.plotting
    :members:

To work with EDS maps you can use ``maps`` module. The class ``MapStore`` could be used
for H5 storage of your maps, allowing quick access of yout data. Individual dataset are
manipulated using ``Mapset`` class:

empatools.maps
--------------

.. automodule:: empatools.maps
    :members:
    :inherited-members:
