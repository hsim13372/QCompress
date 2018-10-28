
##############################################################################
# Copyright 2018 Sukin Sim and Zapata Computing, Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

"""Implementation of the quantum autoencoder (QAE). See arXiv:1612.02806."""

from pyquil.api import get_qc, pyquil_protect
from pyquil.api._errors import ApiError
from pyquil.api._qpu import QPU
from pyquil.api._qvm import QVM
from pyquil.gates import MEASURE
from pyquil.quil import Program
import numpy
import scipy.optimize

from qcompress.utils import is_parametrized_circuit


class QAutoencoderError(Exception):
    """quantum_autoencoder-related error."""
    pass

class quantum_autoencoder:
    """The Quantum Autoencoder (QAE) algorithm

    quantum_autoencoder is an object that encapsulates the QAE algorithm which, similar to VQE, involves
    functional minimization. The main objective of QAE is to compress (then recover) quantum data,
    or a set of quantum states using a fewer number of qubits than what was originally required to encode the data.

    Usage:

        1) Prepare data (state preparation circuits) and the training circuit
        2) Initialize: `inst = quantum_autoencoder(...)`
        3) Set up Forest connection: `inst.setup_forest_cxn([same arguments as get_qc])`
        4) Divide data set into training and test sets: `inst.train_test_split(...)`
        5) Set initial guess for parameters then train: `inst.train(initial_guess)`
        6) Predict against test set: `inst.predict()`

    :param q_in: (dict) Dictionary of abstract-physical qubit mappings for input qubits,
                        i.e. qubits to encode the input data
    :param q_latent: (dict) Dictionary of abstract-physical qubit mappings for latent space qubits,
                    i.e. qubits to "hold" the compressed information
    :param state_prep_circuits: (list) List of pyQuil Programs to prepare input data
    :param training_circuit: (Program) Parametrized circuit for training
    :param state_prep_circuits_dag: (list, Optional) List of daggered state preparation circuits
                                        assuming adjusted qubit indices.
    :param training_circuit_dag: (Program, Optional) Daggered training circuit assuming adjusted
                                    qubit indices.
    :param q_refresh: (dict, Optional) Dictionary of abstract-physical qubit mappings for refresh qubits,
                    i.e. auxiliary qubits added to help recover the original data, Default=None
    :param trash_training: (bool, Optional) Boolean for full-cost function or halfway
                            (trash state) cost function. Default is full-cost function, i.e. Default=False
    :param reset: (bool, Optional) Boolean for resetting input qubits for full-cost function training.
                    Default=True
    :param parametric_compilation: (bool, Optional) Boolean for enabling parametric compilation. Default=False
    :param param_name: (str, Optional) Name of parameter if enabling parametric compilation
                        i.e. name of MemoryReference. Default='theta'
    :param minimizer: Function that minimizes objective f(obj, param). For
                    example the function scipy.optimize.minimize() needs
                    at least two parameters, the objective and an initial
                    point for the optimization. The args for minimizer
                    are the cost function (provided by this class),
                    initial parameters (passed to train() method, and
                    jacobian (defaulted to None). kwargs can be passed
                    in below. Default is scipy's COBYLA optimizer.
    :param minimizer_args: (list, Optional) Arguments for minimizer function. Default=[]
    :param minimizer_kwargs: (dict, Optional) Arguments for keyword args. Default={}
    :param opt_result_parse: (callable, Optional) Function for parsing custom optimizer output.
                                Syntax: xopt, fopt = opt_result_parse(OptResult). Default=None
    :param n_shots: (int, Optional) Number of runs for a QAE circuit. Default=5000
    :param verbose: (bool, Optional) If True, prints mean loss at every n steps defined by
                        print_interval. Default=True
    :param print_interval: (int, Optional) Printing frequency. Default=10
    """
    def __init__(self, q_in, q_latent, state_prep_circuits, training_circuit, state_prep_circuits_dag=None,
                 training_circuit_dag=None, q_refresh={}, trash_training=False, reset=True, parametric_compilation=False,
                 param_name='theta', minimizer=None, minimizer_args=[], minimizer_kwargs={}, opt_result_parse=None,
                 n_shots=1000, verbose=True, print_interval=10):
        """
        Initializes QAE instance.
        
        :ivar n_in: (int) Number of input qubits
        :ivar n_latent: (int) Number of latent space qubits
        :ivar n_refresh: (int) Number of refresh qubits
        :ivar data_size: (int) Size of input data
        :ivar train_indices: (list) List of indices poinint from state prep circuits
                                to define training set, initialized to []
        :ivar test_indices: (list) List of indices pointing from state prep circuits
                                to define test set, initialized to []
        :ivar forest_cxn: (ForestConnection) Connection to QVM or QPU, initialized to None
        :ivar optimized_params: (list) Vector of optimized parameters, initialized to None
        :ivar n_iter: (int) Number of loss function evaluations
        :ivar train_history: (list) List of mean losses over training iterations
        :ivar test_history: (list) List of mean loss(es) for predicting/testing iteration(s)
        """
        # Autoencoder property setting
        self.q_in = q_in
        self.q_latent = q_latent

        self.n_in = len(self.q_in.keys())
        self.n_latent = len(self.q_latent.keys())

        self.state_prep_circuits = state_prep_circuits
        self.training_circuit = training_circuit

        self.trash_training = trash_training
        self.reset = reset

        # Determine refresh qubits for full training techniques
        if not self.trash_training:
            # Reset
            if self.reset:
                self.q_refresh = dict(set(self.q_in.items()) - set(self.q_latent.items()))
                
            # No reset
            else:
                if q_refresh == {}:
                    raise ValueError("The full, no reset training requires q_refresh to be non-empty.")
                self.q_refresh = q_refresh
            
            self.n_refresh = len(self.q_refresh.keys())

            self.state_prep_circuits_dag = state_prep_circuits_dag
            self.training_circuit_dag = training_circuit_dag
            if self.state_prep_circuits_dag is None or self.training_circuit_dag is None:
                raise ValueError("Full training requires daggered circuits for state preparation and training.")
        else:
            self.reset = False

        # Data setting
        self.data_size = len(self.state_prep_circuits)
        self.train_indices = []
        self.test_indices = []

        # Circuit exexcution setting
        self.n_shots = n_shots
        self.forest_cxn = None
        self.cxn_type = None

        # Parametric compilation setting
        self.parametric_compilation = parametric_compilation
        if self.parametric_compilation:
            try:
                self.parametric_compilation = is_parametrized_circuit(self.training_circuit,
                                                                      param_name)
            except AttributeError:
                raise QAutoencoderError('Training circuit cannot be parametrically compiled.')

        if self.parametric_compilation:
            self.param_name = param_name
            self.compiled_qae_circuits = []

        # Optimizer setting
        self.minimizer = minimizer
        self.minimizer_args = minimizer_args
        self.minimizer_kwargs = minimizer_kwargs
        self.opt_result_parse= opt_result_parse

        # Output setting
        self.n_iter = 0
        self.optimized_params = None
        self.train_history = []
        self.test_history = []
        self.verbose = verbose
        self.print_interval = print_interval

    def __str__(self):
        qae_str  = 'QCompress Setting\n'
        qae_str += '=================\n'
        qae_str += 'QAE type: {0}-{1}-{0}\n'.format(self.n_in, self.n_latent)
        qae_str += 'Data size: {0}\n'.format(self.data_size)
        qae_str += 'Training set size: {0}\n'.format(len(self.train_indices))
        if self.trash_training:
            qae_str += 'Training mode: halfway cost function\n'
        else: 
            qae_str += 'Training mode: full cost function\n'
            qae_str += '  Reset qubits: {0}\n'.format(self.reset)
        qae_str += 'Parametric compilation: {0}\n'.format(self.parametric_compilation)
        qae_str += 'Forest connection: {0}\n'.format(self.forest_cxn)
        qae_str += '  Connection type: {0}'.format(self.cxn_type)
        return qae_str

    def setup_forest_cxn(self, *args, **kwargs):
        """
        Sets up Forest connection to simulator or quantum device.
        Enter arguments for get_qc.

        :raises: NotImplementedError
        """
        try:
            self.forest_cxn = get_qc(*args, **kwargs)
            
            if isinstance(self.forest_cxn.qam, QVM):
            	self.cxn_type = "QVM"
            elif isinstance(self.forest_cxn.qam, QPU):
            	self.cxn_type = "QPU"
        except: 
            raise NotImplementedError("This qvm/qpu specification is invalid. Please see args for get_qc.")

    def train_test_split(self, train_indices=None, train_ratio=0.25):
        """
        Splits data set into training and test sets. By default, it will randomly
        divide the data set using train_ratio. 
        
        :param list train_indices: (list) list of integer indices pointing to
                                state preparation circuits for the training set
        :param train_ratio: (float) ratio of training set (rest will be testing set). Default=0.25
        :return: None
        :rtype: NoneType
        :raises: QAutoencoderError
        """
        # User-input (check for invalid input)
        if train_indices is not None:
            self.train_indices = train_indices
            if any(train_index >= self.data_size or train_index < 0 for train_index in train_indices):
                raise QAutoencoderError("Invalid training index/indices. They must be >= 0 and < data size")

        # Automatically generated
        else:
            train_set_size = int(train_ratio * self.data_size)
            self.train_indices = numpy.random.randint(low=0, high=self.data_size - 1,
                                                      size=train_set_size)
    
        self.test_indices = (list(set(range(self.data_size)) - set(self.train_indices)))

    def construct_compression_circuit(self, parameters, index):
        """
        Returns a circuit for compressing the input data,
        i.e. state preparation followed by encoding (training) circuit.
        
        :param parameters: (list) Vector of training circuit parameters
        :param index: (int) Index pointing to corresponding state preparation circuit
        :returns: Quantum circuit implementing state preparation followed by training circuit
        :rtype: Program
        """
        compression_circuit = Program()
        compression_circuit += self.state_prep_circuits[index]
        if self.parametric_compilation:
            try:
                compression_circuit += self.training_circuit
            except:
                raise QAutoencoderError("Circuit cannot be constructed.")
        else:
            compression_circuit += self.training_circuit(parameters)
        return compression_circuit

    def construct_recovery_circuit(self, parameters, index):
        """
        Returns a circuit for recovering the input data after compression,
        i.e. daggered state preparation followed by decoding (daggered training) circuit.
        
        :param parameters: (list) Vector of training circuit parameters
        :param index: (int) Index pointing to corresponding state un-preparation circuit
        :returns: Quantum circuit implementing daggered state preparation
                    followed by daggered training circuit
        :rtype: Program
        """
        recovery_circuit = Program()
        
        if self.reset:
            refresh_qubits = list(self.q_refresh.values())
            for refresh_qubit in refresh_qubits:
                recovery_circuit.reset(refresh_qubit)

        if self.trash_training:
            raise QAutoencoderError("Invalid command for halfway training!")
        
        if self.parametric_compilation:
            try:
                recovery_circuit += self.training_circuit_dag
            except:
                raise QAutoencoderError("Circuit cannot be constructed.")
        else:
            recovery_circuit += self.training_circuit_dag(parameters)
        recovery_circuit += self.state_prep_circuits_dag[index]
        return recovery_circuit

    def _determine_qubits_to_measure(self):
        """
        Helper function for determining which physical
        qubits to measure, based on autoencoder setting
        """
        # Option 1: Halfway/trash training
        if self.trash_training:
            self.memory_size = self.n_in - self.n_latent
            self._physical_labels = [v for k, v in self.q_in.items() if k not in self.q_latent]

        # Option 2: Full training
        else:
            self.memory_size = self.n_in
            
            # Option 2A: reset trash qubits
            if self.reset:
                self._physical_labels = list(self.q_in.values())

            # Option 2B: introduce refresh qubits to replace trash qubits
            else:
                self._physical_labels = list(self.q_latent.values()) + list(self.q_refresh.values())

    @pyquil_protect
    def _execute_circuit(self, parameters, qae_circuit_executable, memory_size):
        """
        Executes a QAE circuit (corresponding to a particular data point) and computes
        the individual loss (i.e. negated frequency of measuring all 0's on relevant qubits).

        :param qae_circuit_executable: (Program) QAE circuit corresponding to a data point
        :param memory_size: (int) Number of array elements in the declared memory
        :returns: Loss value for the QAE circuit
        :rtype: float
        :raises: ApiError
        """
        if not self.parametric_compilation:
            try:
                bitstrings = self.forest_cxn.run(qae_circuit_executable)
            except (ApiError, AttributeError) as err:
                raise err
        else:
            try:
                self.forest_cxn.qam.load(qae_circuit_executable)
                for j, param_val in enumerate(parameters):
                    self.forest_cxn.qam.write_memory(region_name=self.param_name, offset=j, value=parameters[j])
                self.forest_cxn.qam.run()
                self.forest_cxn.qam.wait()
                bitstrings = self.forest_cxn.qam.read_from_memory_region(region_name='ro')
            except (ApiError, AttributeError) as err:
                raise err

        if isinstance(bitstrings, numpy.ndarray):
            bitstrings = bitstrings.tolist()
        single_loss = float(bitstrings.count([0] * memory_size)) / float(self.n_shots)
        return single_loss

    def _manage_measurement(self, memory_size, physical_labels):
        """
        Returns a program that manages readout memory and measures relevant
        qubits, depending on the training technique.

        :param memory_size: (int) Number of array elements in the declared memory
        :param physical_labels: (list) List of physical indices of qubits to measure
        :returns: Circuit component for measurements
        :rtype: Program
        """
        meas_circuit = Program()
        ro = meas_circuit.declare('ro', memory_type='BIT', memory_size=memory_size)
        for i in range(memory_size):
            meas_circuit += MEASURE(physical_labels[i], ro[i])
        return meas_circuit

    def _compute_loss(self, parameters, history_list, dataset_type, indices=None):
        """
        Computes mean loss for the given data subset (training or test).
        
        :param parameters: (list) Vector of training circuit parameters
        :param history_list: (list) List to store losses
        :param dataset_type: (bool) Indicator for training (0) or testing (1) set
        :param indices: (list) List of indices pointing to state preparation circuits
                            (for training or testing)
        :returns: Average loss value for a given data set (training or test)
        :rtype: float
        """
        losses = []

        # Compute cost function value for each data point
        for i, index in enumerate(indices):

            if self.n_iter == 0 or not self.parametric_compilation or dataset_type == 1:

                qae_circuit = Program()
                qae_circuit += self.construct_compression_circuit(parameters, index)

                self._determine_qubits_to_measure()

                if not self.trash_training:
                    qae_circuit += self.construct_recovery_circuit(parameters, index)

                # Apply measurement operations
                qae_circuit += self._manage_measurement(self.memory_size, self._physical_labels)
                
                # Compile circuit
                qae_circuit_wrapped = qae_circuit.wrap_in_numshots_loop(self.n_shots)
                native_quil_circuit = self.forest_cxn.compiler.quil_to_native_quil(qae_circuit_wrapped)
                qae_circuit_executable = self.forest_cxn.compiler.native_quil_to_executable(native_quil_circuit)

            # Save parametrized circuits if using parametric compilation feature
            if self.parametric_compilation and dataset_type == 0:
                if self.n_iter == 0:
                    self.compiled_qae_circuits.append(qae_circuit_executable)
                else:
                    qae_circuit_executable = self.compiled_qae_circuits[i]

            # Compute loss for the data point and store
            single_loss = self._execute_circuit(parameters, qae_circuit_executable, self.memory_size)
            losses.append(single_loss)

        mean_loss = -1. * numpy.mean(losses)
        history_list.append(mean_loss)

        if self.verbose:
            if (len(history_list) - 1) % self.print_interval == 0:
                print("Iter {0:4d} Mean Loss: {1:.7f}".format(self.n_iter, mean_loss))
        self.n_iter += 1
        return mean_loss

    def compute_loss_function(self, parameters):
        """
        Computes mean loss for the training set. Users can directly
        call this routine to compute the training loss for e.g. parameter scans,
        evaluating best qubit mapping, etc.

        :param parameters: (list) Vector of parameters
        :returns: Mean loss computed for the training set
        :rtype: float
        """
        mean_loss = self._compute_loss(parameters=parameters,
                                       history_list=self.train_history,
                                       dataset_type=0,
                                       indices=self.train_indices)
        return mean_loss

    def train(self, initial_guess):
        """
        Trains QAE circuit using a classical optimization routine.
        
        :param initial_guess: (list) Vector of parameters as initial guess
        :returns: The average loss for the training set
        :rtype: float
        :raises: QAutoencoderError, ValueError
        """
        if len(self.train_indices) == 0 or len(self.test_indices) == 0:
            raise QAutoencoderError("Please split your data set into training and test sets before training.")

        # Default minimizer
        if self.minimizer is None:
            self.minimizer = scipy.optimize.minimize
            self.minimizer_args = []
            self.minimizer_kwargs = ({'method': 'COBYLA', 
                                'constraints':[{'type': 'ineq', 'fun': lambda x: x},
                                {'type': 'ineq', 'fun': lambda x: 2. * numpy.pi - x}],
                                'options': {'disp': True, 'maxiter': 1000,
                                'tol': 1e-04, 'rhobeg': 0.10}})

        compute_loss = lambda params: self.compute_loss_function(parameters=params)

        args = [compute_loss, initial_guess]
        args.extend(self.minimizer_args)
        sol = self.minimizer(*args, **self.minimizer_kwargs)
        
        # Parse optimal parameter value(s) and loss function
        if hasattr(sol, 'x'):
            self.optimized_params = sol.x
            try:
                avg_loss = sol.fun
            except:
                pass
        elif self.opt_result_parse is not None:
            try:
                self.optimized_params = self.opt_result_parse(sol)[0]
                avg_loss = self.opt_result_parse(sol)[1]
            except:
                raise ValueError('Unsupported opt_result_parse syntax.')
        else:
            raise ValueError('Could not parse optimization output.')

        if self.verbose:
            print("Mean loss for training data: {}".format(avg_loss))
        return avg_loss

    def predict(self):
        """
        Computes mean loss against the test set using trained/optimized parameters.
        
        :returns: The average loss for the test set
        :rtype: float
        :raises: QAutoencoderError
        """
        if self.optimized_params is not None:
            avg_loss = self._compute_loss(parameters=self.optimized_params,
                                          history_list=self.test_history,
                                          dataset_type=1,
                                          indices=self.test_indices)
        else:
            raise QAutoencoderError("Parameters have not yet been optimized. Please train first.")

        if self.verbose:
            print("Mean loss for test data: {}".format(avg_loss))
        return avg_loss
