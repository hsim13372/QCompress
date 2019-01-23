
Welcome to the Docs for QCompress!
==================================

QCompress is a Python framework for the quantum autoencoder (QAE) algorithm. Using the code, the user can execute instances of the algorithm on either a quantum simulator or a quantum processor provided by Rigetti Computing's `Quantum Cloud Services <https://www.rigetti.com/qcs>`__. For a more in-depth description of QCompress (including the naming convention for the types of qubits involved in the QAE circuit), please go to section :ref:`qae_description`.

For more information about the algorithm, see `Romero et al <https://arxiv.org/abs/1612.02806>`__. Note that we deviate from the training technique used in the original paper and instead introduce two alternative autoencoder training schemes that require lower-depth circuits (see `Sim et al <https://arxiv.org/abs/1810.10576>`__).

Features
--------

This code is based on an older `version <https://github.com/hsim13372/QCompress-1>`__ written during Rigetti Computing's hackathon in April 2018. Since then, we've updated and enhanced the code, supporting the following features:

* Executability on Rigetti's quantum processor(s)
* Several training schemes for the autoencoder
* Use of the ``RESET`` operation for the encoding qubits (lowers qubit requirement)
* User-definable training circuit and/or classical optimization routine


Contents
--------

.. toctree::
   :maxdepth: 3

   intro
   qae_description


API Reference
-------------

.. toctree::
   :maxdepth: 1

   source/qcompress


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`