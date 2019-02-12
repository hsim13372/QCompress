
"""Implementation of the quantum autoencoder (QAE). See arXiv:1612.02806."""


import matplotlib.pyplot as plt
import numpy
import scipy.optimize
import cirq


class quantum_autoencoder:
    """Class for the quantum autoencoder (QAE)"""
    def __init__(self, n_qubits_in, n_qubits_latent_space, state_preparation_circuits,
                 state_preparation_circuits_dag, training_circuit, minimizer=None,
                 minimizer_args=[], minimizer_kwargs={}, n_samples=5000, device=None, gate_noise=None,
                 meas_noise=None, qvm_random_seed=None, verbose=True, print_interval=10, display_progress=False):
        """Initializes quantum autoencoder.

        Args:
        =====
        n_qubits_in : int, required
            Number of qubits used to encode input data
        n_qubits_latent_space : int, required
            Number of qubits in latent space
        state_preparation_circuits : list[pyquil.quil.Program], required
            List of quil programs to prepare set of quantum states, i.e. input data
        state_preparation_circuits_dag : list[pyquil.quil.Program], required
            List of quil programs to prepare daggered state preparation circuits using adjusted qubit indices
        training_circuit : Program, required
            Parametrized circuit to train to compress input data
        minimizer : callable, optional (default : None)
            Function that minimizes objective f(obj, param). For example the function scipy.optimize.minimize() needs
            at least two parameters, the objective and an initial point for the optimization. Default minimizer
            is scipy's COBYLA.
        minimizer_args : list, optional (default: [])
            Arguments for minimizer function
        minimizer_kwargs : dict, optional (default: {})
            Arguments for keyword args
        n_samples : int, optional (default: 5000)
            Number of circuit runs for a given circuit
        device : pyquil.device.Device
            Device object with hardware specs + noise model
        gate_noise : list or numpy.ndarray, optional
            Probabilities of gate being applied to every gate after each gate application, [Px, Py, Pz]
        meas_noise : list or numpy.ndarray, optional
            Probabilities of a X, Y, or Z being applied before a measurement, [Px', Py', Pz']
        qvm_random_seed : int, optional
            Random seed for QVM
        verbose : bool, optional (default: True)
            If True, saves loss values for training and test sets
        print_interval : int, optional (default: 10)
            Printing frequency
        display_progress : bool, optional (default: False)
            If True, displays loss value plot during training process.

        Attributes:
        ===========
        n_ancillas : int
            Number of 'extra' or refresh qubits for compression
        n_data_points : int
            Size of input data set
        train_indices : list[int]
            List of indices pointing to training set
        test_indices : list[int]
            List of indices pointing to test set
        connection : pyquil.api.QVMConnection
            Connection for QVM
        optimized_params : list or numpy.ndarray
            Vector of optimized parameters, initially set to None
        train_history : list
            List of loss values during the training process
        test_history : list
            List of loss value(s) for the test set
        """
        # Autoencoder setting
        self.n_qubits_in = n_qubits_in
        self.n_qubits_latent_space = n_qubits_latent_space
        self.n_ancillas = int(self.n_qubits_in - self.n_qubits_latent_space)

        self.state_preparation_circuits = state_preparation_circuits
        self.state_preparation_circuits_dag = state_preparation_circuits_dag
        self.n_data_points = len(self.state_preparation_circuits)
        self.training_circuit = training_circuit

        self.train_indices = []
        self.test_indices = []

        self.minimizer = minimizer
        self.minimizer_args = minimizer_args
        self.minimizer_kwargs = minimizer_kwargs

        # QVM noise setting
        self.n_samples = n_samples
        #self.device = device
        #self.gate_noise = gate_noise
        #self.meas_noise = meas_noise
        #self.qvm_random_seed = qvm_random_seed

        # Data setting
        self.optimized_params = None
        self.verbose = verbose
        self.print_interval = print_interval
        self.train_history = []
        self.test_history = []
        self.display_progress = display_progress

    def train_test_split(self, train_indices=None, train_ratio=0.25):
        """Splits data set into training and testing sets.

        Args:
        =====
        train_indices : list[int], optional
            List of indices pointing to state preparation circuits for training set
        train_ratio : float, optional
            Ratio of training set (rest will be testing set)

        Notes:
        ======
            - You can manually input indices of training set but by
                default, it will randomly split the data set for you, using the ratio.
        """
        if train_indices is not None:
            self.train_indices = train_indices

        else:
            train_set_size = int(train_ratio * self.n_data_points)
            self.train_indices = numpy.random.randint(
                                    low=0, high=self.n_data_points,
                                    size=train_set_size)
        
        self.test_indices = (list(set(range(self.n_data_points)) -
                                      set(self.train_indices)))

    def construct_compression_program(self, parameters, index):
        """Constructs quantum program for compressing quantum states,
        i.e. state preparation followed by encoding circuit.
        
        Args:
        =====
        parameters : list or numpy.ndarray
            Vector of circuit parameters
        index : int
            Index pointing to corresponding state preparation circuit

        Returns:
        ========
        compression_circuit : pyquil.quil.Program
            Quantum circuit implementing state preparation
            followed by parametrized training circuit
        """
        compression_circuit = cirq.Circuit()

        # Apply state preparation circuit
        compression_circuit.append(state_preparation_circuits[index])

        # Apply training circuit
        compression_circuit.append(self.training_circuit(parameters,
                                                     None,
                                                     range(self.n_qubits_in)))
        return compression_circuit

    def compute_loss(self, parameters,symbols history_list, indices=None):
        """Helper routine to compute loss.

        Args:
        =====
        parameters : list or numpy.ndarray
            Vector of circuit parameters
        history_list : list, required
            List to store losses
        indices : list[int]
            List of indices pointing to state preparation circuits (for training or testing set)

        Returns:
        ========
        loss_values : list or numpy.ndarray
            List of average loss values for given set (training or test)
        """
        total_qubits = self.n_qubits_in + (self.n_qubits_in - self.n_qubits_latent_space)
        
        losses = []

        for index in indices:

            # Apply state preparation then training circuit
            qae_circuit = self.construct_compression_program(parameters, index)

            # Apply daggered training circuit (with adjusted indices)
            new_range = range(total_qubits - self.n_qubits_in, total_qubits)
            new_range = new_range[::-1]
            qae_circuit.append(cirq.inverse(self.training_circuit(parameters, None, new_range)))
            
            # Apply daggered state preparation circuit (with adjusted indices)
            qae_circuit.append(self.state_preparation_circuits_dag[index])

            # Measure data qubits
            measure_moment=cirq.Moment([cirq.measure(qubits[i]) for q,i in enumerate(new_range)])
            
            qae_circuit.append(measure_moment)

               
            # Run circuit
            #result = self.connection.run(quil_program=qae_circuit,
                                         #classical_addresses=range(self.n_qubits_in),
                                         #trials=self.n_samples)
            simulator= cirq.Simulator.Simulator()
            resolver=cirq.ParamResolver({symbols[i]):parameters[i] for i in range(len(parameters))})
            result= simulator.run(qae_circuit,resolver, repititions=self.n_samples)

            # Count measurements of all 0's on data qubits
            n = self.n_qubits_in
            result_count = result.count([0] * self.n_qubits_in)
            losses.append(result_count / self.n_samples)
        
        mean_loss = -1. * numpy.mean(losses)

        self._prepare_loss_history(history_list, mean_loss)

        if self.verbose:
            if (len(history_list) - 1) % self.print_interval == 0:
            # losses_str = ["{0:.4f}".format(loss_val) for loss_val in losses]
            # print("Loss values: {}".format(losses_str))
                print("Iter {0:4d} Mean Loss: {1:.7f}".format(len(history_list) - 1, mean_loss))

        return mean_loss

    def _compute_loss_training_set(self, parameters):
        """Helper routine for computing loss for the training set,
        with the option to display training progress.

        Args:
        =====
        parameters : list or numpy.ndarray
            Vector of parameters

        Returns:
        ========
        mean_loss : float
            Mean loss computed
        """
        mean_loss = self.compute_loss(parameters, self.train_history,
                                      self.train_indices)

        if self.display_progress:
            plt.ion()
            plt.close('all')
            plt.plot(self.train_history, 'o-')
            plt.xlabel('Iteration')
            plt.ylabel('Mean Loss')
            plt.title('Training Progress')
            plt.show()
            plt.pause(0.05)
        return mean_loss

    def train(self, initial_guess):
        """Trains QAE circuit using classical optimization routine.

        Args:
        =====
        initial_guess : list or numpy.ndarray
            Vector of parameters as initial guess

        Returns: 
        ========
        avg_loss : float
            Mean loss value
        """
        compute_loss = lambda params: self._compute_loss_training_set(parameters=params)

        # Default minimizer
        if self.minimizer is None:
            self.minimizer = scipy.optimize.minimize
            self.minimizer_args = []
            self.minimizer_kwargs = ({'method': 'COBYLA',
                                'constraints':[{'type': 'ineq', 'fun': lambda x: x},
                                {'type': 'ineq', 'fun': lambda x: 2. * numpy.pi - x}],
                                'options': {'disp': False, 'maxiter': 500,
                                'tol': 1e-04, 'rhobeg': 0.10}})

        args = [compute_loss, initial_guess]
        args.extend(self.minimizer_args)

        sol = self.minimizer(*args, **self.minimizer_kwargs)

        self.optimized_params = sol.x
        avg_loss = sol.fun

        if self.verbose:
            print("Mean loss for training data: {}".format(avg_loss))

        return avg_loss

    def predict(self):
        """Computes loss for test set."""
        avg_loss = self.compute_loss(parameters=self.optimized_params,
                                     history_list=self.test_history,
                                     indices=self.test_indices)
        if self.verbose:
            print("Mean loss for testing data: {}".format(avg_loss))
        return avg_loss

    def _prepare_loss_history(self, history_list, loss_value):
        """Helper routine to populate loss histories.
        
        Args:
        =====
        history_list : list
            List to store loss values
        loss_value : float
            Loss value to add
        """
        if history_list is None:
            history_list = []
        history_list.append(loss_value)
