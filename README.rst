How to run
-------------

* Make an output folder:
``mkdir /data/experiments/experiment_xyz``

* Put a ``config.ini`` in that folder. Set all the options that differ from
the values given in ``default.ini`` (the rest will be taken from default.ini automatically).
For example in ``config.ini`` could contain the following:

.. code:: ini

    [model]
    inn_coupling_blocks = [0, 3, 4, 6, 3]


* Then, start the training as follows, by giving the direcory:

.. code:: sh

    export DATASET_DIR=datasets/maps
    python main.py /data/experiments/experiment_xyz

