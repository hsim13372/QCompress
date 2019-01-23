
.. _intro:

Installing QCompress
====================

There are a few options for installing QCompress:

1. To install QCompress using ``pip``, execute:

.. code-block:: bash

	pip install qcompress


2. To install QCompress using ``conda``, execute:

.. code-block:: bash

	conda install -c rigetti -c hsim13372 qcompress


3. To instead install QCompress from source, clone this repository, ``cd`` into it, and run:

.. code-block:: bash

	git clone https://github.com/hsim13372/QCompress
	cd QCompress
	python -m pip install -e .


Try executing ``import qcompress`` to test the installation in your terminal.

Note that the pyQuil version used requires Python 3.6 or later.



Installing QCompress on QMI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For installing QCompress on a user's Quantum Machine Image (QMI), we recommend the following steps:

1. Connect to your QMI with SSH.

2. Launch a Python virtual environment:

.. code-block:: bash

	source ~/pyquil/venv/bin/activate


3. To install QCompress, clone then install from github or install using ``pip``:

.. code-block:: bash

	git clone https://github.com/hsim13372/QCompress
	cd QCompress
	pip install -e .


or


.. code-block:: bash

	pip install qcompress


4. To execute the Jupyter notebook demos or run QCompress on Jupyter notebooks in general, execute:

.. code-block:: bash

	tmux new -s <ENTER-SESSION-NAME>
	source ~/pyquil/venv/bin/activate

	pip install jupyter
	cd <ENTER-DIRECTORY-FOR-NOTEBOOK>
	jupyter notebook

5. (Optional) To run your quantum autoencoder instance on the QPU, book reservations in the compute schedule via ``qcs reserve``.


**NOTE**: We assume the user has already set up his/her QMI. If the user is new to QCS, please refer to `Rigetti QCS docs <https://www.rigetti.com/qcs>`__ to get started!
