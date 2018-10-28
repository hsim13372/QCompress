
===========================
Installing QCompress on QMI
===========================

For installing QCompress on a user's Quantum Machine Image (QMI), we recommend the following steps:

1. Connect to your QMI with SSH.

2. To install QCompress, clone then install from github or install using ``pip``:

.. code-block:: bash

	git clone https://github.com/hsim13372/QCompress
	cd QCompress
	pip install -e .


or


.. code-block:: bash

	pip install qcompress


3. To execute the Jupyter notebook demos or run QCompress on Jupyter notebooks in general, execute:

.. code-block:: bash

	tmux new -s <ENTER-SESSION-NAME>
	source pyquil/venv/bin/activate

	pip install jupyter
	cd <ENTER-DIRECTORY-FOR-NOTEBOOK>
	jupyter notebook

4. (Optional) To run your quantum autoencoder instance on the QPU, book reservations in the compute schedule via ``qcs reserve``.


**NOTE**: We assume the user has already set up his/her QMI. If the user is new to QCS, please refer to `Rigetti QCS docs <URLHERE>`__ to get started!