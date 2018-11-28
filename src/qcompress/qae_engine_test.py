
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

"""Tests for qae_engine.py"""

import numpy
import pytest
import scipy.optimize
from pyquil.gates import *
from pyquil.quil import Program
from qcompress.qae_engine import QAutoencoderError, quantum_autoencoder


@pytest.fixture
def full_no_reset_inst(trash_training=False, reset=False,
                       q_refresh={'q2': 2}, parametric_compilation=False):
    """Returns a valid QAE instance, employing full training without reset option."""
    q_in = {'q0': 0, 'q1': 1}
    q_latent = {'q1': 1}
    n_shots = 10

    sp_circuit = lambda theta, qubit_indices: Program(RY(theta[0], qubit_indices[1]),
                                                      CNOT(qubit_indices[1], qubit_indices[0]))
    sp_circuit_dag = lambda theta, qubit_indices: Program(CNOT(qubit_indices[1], qubit_indices[0]),
                                                          RY(-theta[0], qubit_indices[1]))

    list_SP_circuits, list_SP_circuits_dag = [], []
    angle_list = numpy.linspace(-10, 10, 5) # Generate 5 data pts

    for angle in angle_list:
        state_prep_unitary = sp_circuit([angle], [0, 1])
        state_prep_unitary_dag = sp_circuit_dag([angle], [2, 1])
        list_SP_circuits.append(state_prep_unitary)
        list_SP_circuits_dag.append(state_prep_unitary_dag)

    training_circuit = lambda theta, qubit_indices=[0, 1]: Program(RY(-theta[0]/2, qubit_indices[0]),
                                                            CNOT(qubit_indices[1], qubit_indices[0]))
    training_circuit_dag = lambda theta, qubit_indices=[2, 1]: Program(CNOT(qubit_indices[1], qubit_indices[0]),
                                                                RY(theta[0]/2, qubit_indices[0]))
    return quantum_autoencoder(state_prep_circuits=list_SP_circuits,
                               training_circuit=training_circuit,
                               q_in=q_in, q_latent=q_latent, q_refresh=q_refresh,
                               state_prep_circuits_dag=list_SP_circuits_dag,
                               training_circuit_dag=training_circuit_dag,
                               parametric_compilation=parametric_compilation,
                               trash_training=trash_training, reset=reset, 
                               n_shots=n_shots, verbose=False)

@pytest.fixture
def full_reset_inst():
    """Returns a valid QAE instance, employing full training with reset option."""
    q_in = {'q0': 0, 'q1': 1}
    q_latent = {'q1': 1}
    n_shots = 10
    trash_training = False
    reset = True

    sp_circuit = lambda theta, qubit_indices: Program(RY(theta[0], qubit_indices[1]),
                                                      CNOT(qubit_indices[1], qubit_indices[0]))
    sp_circuit_dag = lambda theta, qubit_indices: Program(CNOT(qubit_indices[1], qubit_indices[0]),
                                                          RY(-theta[0], qubit_indices[1]))

    list_SP_circuits, list_SP_circuits_dag = [], []
    angle_list = numpy.linspace(-10, 10, 5) # Generate 5 data pts

    for angle in angle_list:
        state_prep_unitary = sp_circuit([angle], [0, 1])
        state_prep_unitary_dag = sp_circuit_dag([angle], [0, 1])
        list_SP_circuits.append(state_prep_unitary)
        list_SP_circuits_dag.append(state_prep_unitary_dag)

    training_circuit = lambda theta, qubit_indices=[0, 1]: Program(RY(-theta[0]/2, qubit_indices[0]),
                                                            CNOT(qubit_indices[1], qubit_indices[0]))
    training_circuit_dag = lambda theta, qubit_indices=[0, 1]: Program(CNOT(qubit_indices[1], qubit_indices[0]),
                                                                RY(theta[0]/2, qubit_indices[0]))
    return quantum_autoencoder(state_prep_circuits=list_SP_circuits,
                               training_circuit=training_circuit,
                               q_in=q_in, q_latent=q_latent,
                               state_prep_circuits_dag=list_SP_circuits_dag,
                               training_circuit_dag=training_circuit_dag,
                               trash_training=trash_training, reset=reset, 
                               n_shots=n_shots, verbose=False)

@pytest.fixture
def halfway_inst():
    """Returns a valid QAE instance, employing halfway training."""
    q_in = {'q0': 0, 'q1': 1}
    q_latent = {'q1': 1}
    n_shots = 10
    trash_training = True

    sp_circuit = lambda theta, qubit_indices: Program(RY(theta[0], qubit_indices[1]),
                                                      CNOT(qubit_indices[1], qubit_indices[0]))

    list_SP_circuits = []
    angle_list = numpy.linspace(-10, 10, 5) # Generate 5 data pts

    for angle in angle_list:
        state_prep_unitary = sp_circuit([angle], [0, 1])
        list_SP_circuits.append(state_prep_unitary)

    training_circuit = lambda theta, qubit_indices=[0, 1]: Program(RY(-theta[0]/2, qubit_indices[0]),
                                                            CNOT(qubit_indices[1], qubit_indices[0]))
    return quantum_autoencoder(state_prep_circuits=list_SP_circuits,
                               training_circuit=training_circuit,
                               q_in=q_in, q_latent=q_latent,
                               trash_training=trash_training,
                               n_shots=n_shots, verbose=False)

def test_full_no_reset_attributes(full_no_reset_inst):
    """Test for attributes of full, no reset instance."""
    assert full_no_reset_inst.trash_training == False
    assert full_no_reset_inst.reset == False

    assert full_no_reset_inst.n_in == 2
    assert full_no_reset_inst.n_latent == 1
    assert full_no_reset_inst.n_refresh == 1
    assert full_no_reset_inst.data_size == 5

    test_str = ('QCompress Setting\n'
                '=================\n'
                'QAE type: 2-1-2\n'
                'Data size: 5\n'
                'Training set size: 0\n'
                'Training mode: full cost function\n'
                '  Reset qubits: False\n'
                'Parametric compilation: False\n'
                'Forest connection: None\n'
                '  Connection type: None')
    assert str(full_no_reset_inst) == test_str

def test_faulty_full_no_reset_no_daggered_circuits():
    """Test for trying to instantiate a QAE employing full training, no reset
    without inputing daggered circuits. Should also apply to full, reset case."""
    q_in = {'q0': 0, 'q1': 1}
    q_latent = {'q1': 1}
    q_refresh = {'q2': 2}
    n_shots = 10
    trash_training = False
    reset = False
    parametric_compilation = False

    sp_circuit = lambda theta, qubit_indices: Program(RY(theta[0], qubit_indices[1]),
                                                      CNOT(qubit_indices[1], qubit_indices[0]))
    
    list_SP_circuits = []
    angle_list = numpy.linspace(-10, 10, 5) # Generate 5 data pts

    for angle in angle_list:
        state_prep_unitary = sp_circuit([angle], [0, 1])
        list_SP_circuits.append(state_prep_unitary)

    training_circuit = lambda theta, qubit_indices=[0, 1]: Program(RY(-theta[0]/2, qubit_indices[0]),
                                                            CNOT(qubit_indices[1], qubit_indices[0]))

    with pytest.raises(ValueError):
        faulty = quantum_autoencoder(state_prep_circuits=list_SP_circuits,
                                   training_circuit=training_circuit,
                                   q_in=q_in, q_latent=q_latent, q_refresh=q_refresh,
                                   parametric_compilation=parametric_compilation,
                                   trash_training=trash_training, reset=reset, 
                                   n_shots=n_shots, verbose=False)

def test_full_reset_attributes(full_reset_inst):
    """Test for attributes of full, reset instance."""
    assert full_reset_inst.trash_training == False
    assert full_reset_inst.reset == True

    assert set(full_reset_inst.q_refresh) == set({'q0': 0})
    assert full_reset_inst.n_in == 2
    assert full_reset_inst.n_latent == 1
    assert full_reset_inst.n_refresh == 1
    assert full_reset_inst.data_size == 5

    test_str = ('QCompress Setting\n'
                '=================\n'
                'QAE type: 2-1-2\n'
                'Data size: 5\n'
                'Training set size: 0\n'
                'Training mode: full cost function\n'
                '  Reset qubits: True\n'
                'Parametric compilation: False\n'
                'Forest connection: None\n'
                '  Connection type: None')
    assert str(full_reset_inst) == test_str

def test_halfway_attributes(halfway_inst):
    """Test for attributes of halfway instance."""
    assert halfway_inst.trash_training == True
    assert halfway_inst.reset == False
    assert halfway_inst.n_in == 2
    assert halfway_inst.n_latent == 1
    assert halfway_inst.data_size == 5

    assert not hasattr(halfway_inst, 'n_refresh')

    halfway_inst.trash_training = True
    test_str = ('QCompress Setting\n'
                '=================\n'
                'QAE type: 2-1-2\n'
                'Data size: 5\n'
                'Training set size: 0\n'
                'Training mode: halfway cost function\n'
                'Parametric compilation: False\n'
                'Forest connection: None\n'
                '  Connection type: None')
    assert str(halfway_inst) == test_str

def test_invalid_forest_cxn(full_no_reset_inst):
    """Test for trying to set up invalid Forest connection."""
    with pytest.raises(NotImplementedError):
        full_no_reset_inst.setup_forest_cxn('MADEUP_NONSENSE')

def test_faulty_qae_setup_parametric():
    """Test for trying to set parametric compilation to unfit QAE setup.
    That is, the input circuits do not employ parametric circuits using 
    MemoryReference."""
    q_in = {'q0': 0, 'q1': 1}
    q_latent = {'q1': 1}
    n_shots = 10
    trash_training = True
    reset = True
    parametric_compilation = True # Turn on

    sp_circuit = lambda theta, qubit_indices: Program(RY(theta[0], qubit_indices[1]),
                                                      CNOT(qubit_indices[1], qubit_indices[0]))
    list_SP_circuits = []
    angle_list = numpy.linspace(-10, 10, 5) # Generate 5 data pts

    for angle in angle_list:
        state_prep_unitary = sp_circuit([angle], [0, 1])
        list_SP_circuits.append(state_prep_unitary)

    training_circuit = lambda theta, qubit_indices=[0, 1]: Program(RY(-theta[0]/2, qubit_indices[0]),
                                                            CNOT(qubit_indices[1], qubit_indices[0]))

    with pytest.raises(QAutoencoderError):
        faulty = quantum_autoencoder(state_prep_circuits=list_SP_circuits,
                                   training_circuit=training_circuit,
                                   q_in=q_in, q_latent=q_latent,
                                   parametric_compilation=parametric_compilation,
                                   trash_training=trash_training, reset=reset, 
                                   n_shots=n_shots, verbose=False)
    
def test_faulty_parametric_compilation_attempt(full_no_reset_inst):
    """Test for trying to parametrically compile unsuitable circuits."""
    # User changes the flag in the middle of the computation
    full_no_reset_inst.parametric_compilation = True
    parameters = [0, 0]
    index = 0
    with pytest.raises(QAutoencoderError):
        full_no_reset_inst.construct_compression_circuit(parameters, index)

    with pytest.raises(QAutoencoderError):
        full_no_reset_inst.construct_recovery_circuit(parameters, index)

def test_inputing_invalid_train_indices(full_no_reset_inst):
    """Test for user trying to input invalid training indices when
    splitting up data set."""
    # Only 5 data points so can choose from: [0, 1, 2, 3, 4]
    with pytest.raises(QAutoencoderError):
        full_no_reset_inst.train_test_split(train_indices=[5, 2])
    
    with pytest.raises(QAutoencoderError):
        full_no_reset_inst.train_test_split(train_indices=[2, -1])

    # Valid case
    full_no_reset_inst.train_test_split(train_indices=[0, 2])
    assert set(full_no_reset_inst.test_indices) == set([1, 3, 4])

def test_not_inputing_train_indices(full_no_reset_inst):
    """Test using train ratio to split data set."""
    full_no_reset_inst.train_test_split(train_ratio=0.2)
    assert len(full_no_reset_inst.train_indices) == 1
    assert len(full_no_reset_inst.test_indices) == 4

def test_trying_to_train_without_proper_setup(full_no_reset_inst):
    """Test user trying to compute loss without (1) splitting data set
    or (2) setting up Forest connection."""
    initial_guess = [0, 0]

    # User trying to train without splitting data set
    with pytest.raises(QAutoencoderError):
        train_loss = full_no_reset_inst.train(initial_guess)

    full_no_reset_inst.train_test_split(train_ratio=0.2)

    # User trying to train without setting up Forest cxn
    with pytest.raises(AttributeError):
        train_loss = full_no_reset_inst.train(initial_guess)

    full_no_reset_inst.setup_forest_cxn('9q-square-qvm')
    assert full_no_reset_inst.minimizer == scipy.optimize.minimize

def test_trying_to_predict_without_training(full_no_reset_inst):
    """Test user trying to predict/test without training first."""
    with pytest.raises(QAutoencoderError):
        test_loss = full_no_reset_inst.predict()

def test_cobyla_output(full_no_reset_inst):
    """Test COBYLA optimizer-like output."""
    niter = 1

    full_no_reset_inst.minimizer = scipy.optimize.minimize
    full_no_reset_inst.minimizer_kwargs = ({'method': 'COBYLA', 
                                      'constraints':[{'type': 'ineq', 'fun': lambda x: x},
                                      {'type': 'ineq', 'fun': lambda x: 2. * numpy.pi - x}],
                                      'options': {'disp': False, 'maxiter': niter,
                                      'tol': 1e-04, 'rhobeg': 0.10}})

    full_no_reset_inst.setup_forest_cxn('9q-square-qvm')
    
    test_str = ('QCompress Setting\n'
                '=================\n'
                'QAE type: 2-1-2\n'
                'Data size: 5\n'
                'Training set size: 0\n'
                'Training mode: full cost function\n'
                '  Reset qubits: False\n'
                'Parametric compilation: False\n'
                'Forest connection: 9q-square-qvm\n'
                '  Connection type: QVM')
    assert str(full_no_reset_inst) == test_str

    full_no_reset_inst.train_test_split(train_ratio=0.2)
    initial_guess = [0, 0]

    opt_result_parse = lambda opt_res: (opt_res.x, opt_res.fun)
    full_no_reset_inst.opt_result_parse = opt_result_parse
    train_loss = full_no_reset_inst.train(initial_guess)

    assert full_no_reset_inst.optimized_params is not None
    assert isinstance(train_loss, numpy.float)

    test_loss = full_no_reset_inst.predict()
    assert isinstance(test_loss, numpy.float)

# def test_powell_output(full_no_reset_inst):
#     """Test POWELL optimizer-like output. (Slow)"""
#     niter = 1
#     full_no_reset_inst.minimizer = scipy.optimize.fmin_powell
#     full_no_reset_inst.minimizer_kwargs = ({'xtol':0.0001, 'ftol':0.0001, 'maxiter':0,
#                                       'full_output':1, 'retall': 0})

#     full_no_reset_inst.setup_forest_cxn('9q-square-qvm')
#     full_no_reset_inst.train_test_split(train_ratio=0.2)
#     initial_guess = [0, 0]

#     # Cannot parse optimizer output
#     with pytest.raises(ValueError):
#         train_loss = full_no_reset_inst.train(initial_guess)

#     # Nonsense parsing function #1
#     opt_result_parse = lambda opt_res: 1
#     full_no_reset_inst.opt_result_parse = opt_result_parse
#     with pytest.raises(ValueError):
#         train_loss = full_no_reset_inst.train(initial_guess)

#     # Nonsense parsing function #2 (function value not fetchable)
#     opt_result_parse = lambda opt_res: (opt_res[0], )
#     full_no_reset_inst.opt_result_parse = opt_result_parse
#     with pytest.raises(ValueError):
#         train_loss = full_no_reset_inst.train(initial_guess)

#     # Proper syntax for parsing function
#     opt_result_parse = lambda opt_res: (opt_res[0], opt_res[1])
#     full_no_reset_inst.opt_result_parse = opt_result_parse

#     train_loss = full_no_reset_inst.train(initial_guess)
#     assert full_no_reset_inst.optimized_params is not None
#     assert isinstance(train_loss, numpy.float)

#     assert full_no_reset_inst.memory_size == 2
#     assert set(full_no_reset_inst._physical_labels) == set([2, 1])

#     test_loss = full_no_reset_inst.predict()
#     assert isinstance(test_loss, numpy.float)

def test_bfgs_output(full_no_reset_inst):
    """Test BFGS optimizer-like output."""
    niter = 1
    full_no_reset_inst.minimizer = scipy.optimize.fmin_bfgs
    full_no_reset_inst.minimizer_kwargs = ({'maxiter': niter, 'full_output': 1,
                                           'retall': 0, 'disp': 0})

    full_no_reset_inst.setup_forest_cxn('9q-square-qvm')
    full_no_reset_inst.train_test_split(train_ratio=0.2)
    initial_guess = [0, 0]

    # Cannot parse optimizer output
    with pytest.raises(ValueError):
        train_loss = full_no_reset_inst.train(initial_guess)

    opt_result_parse = lambda opt_res: (opt_res[0], opt_res[1])
    full_no_reset_inst.opt_result_parse = opt_result_parse
    train_loss = full_no_reset_inst.train(initial_guess)
    assert isinstance(train_loss, numpy.float)

def test_full_training_with_reset(full_reset_inst):
    """Test when full training with reset option."""
    assert full_reset_inst.reset == True
    assert full_reset_inst.q_refresh == {'q0': 0}

    full_reset_inst.setup_forest_cxn('9q-square-qvm')
    full_reset_inst.train_test_split(train_ratio=0.2)
    params = [numpy.pi/2., 0.]
    index = 0
    compress_prog = full_reset_inst.construct_compression_circuit(params, index)
    recovery_prog = full_reset_inst.construct_recovery_circuit(params, index)

    assert compress_prog.get_qubits() == {0, 1}
    assert recovery_prog.get_qubits() == {0, 1}

    loss_val = full_reset_inst.compute_loss_function(params)
    assert full_reset_inst.memory_size == 2
    assert set(full_reset_inst._physical_labels) == set([0, 1])

def test_halfway_training(halfway_inst):
    """Test when halfway training."""
    assert halfway_inst.trash_training == True

    halfway_inst.setup_forest_cxn('9q-square-qvm')
    halfway_inst.train_test_split(train_ratio=0.2)
    params = [numpy.pi/2., 0.]
    index = 0
    compress_prog = halfway_inst.construct_compression_circuit(params, index)
    assert compress_prog.get_qubits() == {0, 1}
    
    with pytest.raises(QAutoencoderError):
        recovery_prog = halfway_inst.construct_recovery_circuit(params, index)

    loss_val = halfway_inst.compute_loss_function(params)
    assert halfway_inst.memory_size == 1
    assert set(halfway_inst._physical_labels) == set([0])
