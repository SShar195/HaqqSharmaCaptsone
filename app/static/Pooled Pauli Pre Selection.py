from qiskit import transpile
from qiskit.quantum_info import random_density_matrix, Operator, Statevector, DensityMatrix, random_unitary, random_statevector, random_clifford, Pauli, SparsePauliOp
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, FiniteDiffEstimatorGradient
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter

import numpy as np
import pandas as pd
import scipy as sp
from math import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from itertools import product

def create_random_pauli_strings(num_qubits, num_strings, seed):
    pauli_dict = {
        "0": "I",
        "1": "X",
        "2": "Y",
        "3": "Z"
    }

    #Randomly sample from 1 to 4 ** num_qubits without replacement (and convert it to base4)
    random.seed(seed)
    random_pauli_strings = random.sample(range(1, 4 ** num_qubits), k=num_strings)
    random_pauli_strings = [np.base_repr(num, base=4).zfill(num_qubits) for num in random_pauli_strings]

    #Replace integer for associated pauli
    for num in pauli_dict:
        for i, pauli_string in enumerate(random_pauli_strings):
            random_pauli_strings[i] = pauli_string.replace(num, pauli_dict[num])
    
    return random_pauli_strings

def random_algorithm(initial_state, problem_hamiltonian, num_iterations, step_size, num_tests, options, num_qubits):
    #Initialize with state and identity matrix
    adaptive_unitary = QuantumCircuit(num_qubits)
    
    adaptive_unitary.initialize(initial_state, range(num_qubits))
    adaptive_unitary.i(range(num_qubits))
    
    #Estimator and Gradient Estimator
    estimator = Estimator(options=options)
    gradient_estimator = ParamShiftEstimatorGradient(estimator)
    
    cost_list = []
    variance_list = []
    gradients_list = []
    for iteration in range(num_iterations):
        #Calculate cost
        cost = None
        variance = None
        try:
            cost_job = estimator.run(adaptive_unitary, problem_hamiltonian).result()
            cost = cost_job.values[0] #Jk
            variance = cost_job.metadata[0].get("variance")
        except:
            break
        
        #Update adaptive unitary
        random_pauli_strings = create_random_pauli_strings(num_qubits, num_strings=num_tests, seed=iteration)
        
        test_circuits = [] 
        for random_pauli_string in random_pauli_strings:
            theta = Parameter(f"theta_{random_pauli_string}") #Parameter for adapative unitary

            test_circuit = adaptive_unitary.copy()
            test_circuit.append(PauliEvolutionGate(Pauli(random_pauli_string),
                                                         time=theta/2,
                                                         synthesis=LieTrotter(reps=1)),
                                                         range(num_qubits))

            test_circuits.append(test_circuit)
        
        #Calculate gradient and objective input (Theta k)
        gradients = None
        try:
            gradient_job = gradient_estimator.run(test_circuits, [problem_hamiltonian] * num_tests, [[0]] * num_tests).result()
            gradients = gradient_job.gradients
        except:
            break
        
        gradients = np.array(gradients).flatten()
        absolute_gradients = np.abs(gradients)
        max_gradient_index = absolute_gradients.argmax()
        
        current_objective_input = -step_size * gradients[max_gradient_index]  
            
        #Assign new parameter
        adaptive_unitary = test_circuits[max_gradient_index]
        adaptive_unitary.assign_parameters([current_objective_input], inplace=True)
        
        #Update cost and jobs list
        cost_list.append(cost)
        variance_list.append(variance)
        gradients_list.append(gradients)

        #DEBUG
        grad = gradients[max_gradient_index]
        print(f"{iteration=:5}, {cost=:20}, {gradients=}, {max_gradient_index=}, {grad=:20}")

        #DEBUG
        # display(adaptive_unitary.draw(output="mpl"))
        # if(iteration >= 3):
        #     break
    
    return adaptive_unitary, cost_list, variance_list, gradients_list

#TESTBENCH
NUM_QUBITS = 2 #Number of qubits
NUM_ITERATIONS = 150 #Number of iterations in the optimization loop
STEP_SIZE = 0.25 #Gamma
NUM_TESTS = 5 #Number of tests

INITIAL_STATE = random_statevector(2 ** NUM_QUBITS, seed=7) #psi 0
PROBLEM_HAMILTONIAN = Operator.from_label("ZZ") #Hp

service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q-asu/main/asu-arenz")
backend = service.get_backend("ibmq_mumbai")