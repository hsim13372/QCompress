
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

"""Utility functions for the quantum autoencoder."""

import numpy as np
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quilbase import Declare


def is_parametrized_circuit(program, param_name):
    """
    Returns True if the circuit is parametrized.
    
    :param program: (Program) A quantum program
    :param param_name: (str) Name of parameter, e.g. 'theta'
    :returns: True if the Program contains parameters
    """
    for instr in program._instructions:
        if isinstance(instr, Declare):
            if instr.name == param_name:
                return True
    return False

def order_qubit_labels(qubit_label_dict):
    """
    Helper function for returning physical qubit indices
    ordered by abstract qubits (i.e. q0, q1, ...).

    :param qubit_label_dict: (dict) Dictionary of abstract-physical
                                mapping for qubits, e.g. {'q0': 10, 'q1': 2}
    :returns: Ordered array of physical qubit indices
    :rtype: numpy.ndarray
    """
    abstract_labels = []
    physical_labels = []
    for key, val in qubit_label_dict.items():
        abstract_labels.append(int(key[1:]))
        physical_labels.append(val)
    sorted_indices = np.argsort(abstract_labels)
    return np.array(physical_labels)[sorted_indices]

def merge_two_dicts(dict1, dict2):
    """
    Helper function for merging two dictionaries into a
    new dictionary as a shallow copy.

    :param dict1: (dict) First of two dictonaries to merge
    :param dict2: (dict) Second dictionary
    :returns: Merged dictionary
    :rtype: dict
    """
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict
