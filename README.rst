
=========
QCompress
=========


Description
===========

QCompress is a Python framework for the quantum autoencoder (QAE) algorithm. Using the code, the user can execute instances of the algorithm on either a quantum simulator or a quantum processor provided by Rigetti Computing's `Quantum Cloud Services <https://www.rigetti.com/qcs>`__. For a more in-depth description of QCompress (including the naming convention for the types of qubits involved in the QAE circuit), click `here <https://github.com/hsim13372/QCompress/blob/master/examples/intro.rst>`__. 

For more information about the algorithm, see `Romero et al <https://arxiv.org/abs/1612.02806>`__. Note that we deviate from the training technique used in the original paper and instead introduce two alternative autoencoder training schemes that require lower-depth circuits (see `Sim et al <https://arxiv.org/abs/1810.10576>`__).

Features
--------

This code is based on an older `version <https://github.com/hsim13372/QCompress-1>`__ written during Rigetti Computing's hackathon in April 2018. Since then, we've updated and enhanced the code, supporting the following features:

* Executability on Rigetti's quantum processor(s)
* Several training schemes for the autoencoder
* Use of the ``RESET`` operation for the encoding qubits (lowers qubit requirement)
* User-definable training circuit and/or classical optimization routine


Installation
============

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

Note that the pyQuil version used requires Python 3.6 or later. For installation on a user QMI, please click `here <https://github.com/hsim13372/QCompress/blob/master/qmi_instructions.rst>`__.


Examples
========

We provide several Jupyter notebooks to demonstrate the utility of QCompress. We recommend going through the notebooks in the order shown in the table (top-down).

.. csv-table::
   :header: Notebook, Feature(s)

   `qae_h2_demo.ipynb <https://github.com/hsim13372/QCompress/blob/master/examples/qae_h2_demo.ipynb>`__, Simulates the compression of the ground states of the hydrogen molecule. Uses OpenFermion and grove to generate data. Demonstrates the "halfway" training scheme.
   `qae_two_qubit_demo.ipynb <https://github.com/hsim13372/QCompress/blob/master/examples/qae_two_qubit_demo.ipynb>`__, Simulates the compression of a two-qubit data set. Outlines how to run an instance on an actual device. Demonstrates the "full with reset" training scheme.
   `run_landscape_scan.ipynb <https://github.com/hsim13372/QCompress/blob/master/examples/run_landscape_scan.ipynb>`__, Shows user how to run landscape scans for small (few-parameter) instances. Demonstrates setup of the "full with no reset" training scheme.


Disclaimer
==========

We note that there is a lot of room for improvement and fixes. Please feel free to submit issues and/or pull requests!


How to cite
===========

When using QCompress for research projects, please cite:

	Sukin Sim, Yudong Cao, Jonathan Romero, Peter D. Johnson and Alán Aspuru-Guzik.
	*A framework for algorithm deployment on cloud-based quantum computers*.
	`arXiv:1810.10576 <https://arxiv.org/abs/1810.10576>`__. 2018.


Authors
=======

`Sukin (Hannah) Sim <https://github.com/hsim13372>`__ (Harvard), `Zapata Computing, Inc. <https://zapatacomputing.com/>`__
