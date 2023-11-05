#Libraries
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

def random_algorithm(initial_state, problem_hamiltonian, num_iterations, step_size, estimator, num_qubits):
    #Initialize with state and identity matrix
    adaptive_unitary = QuantumCircuit(num_qubits)
    
    adaptive_unitary.initialize(initial_state, range(num_qubits))
    adaptive_unitary.i(range(num_qubits))
    
    #Gradient Estimator
    gradient_estimator = ParamShiftEstimatorGradient(estimator)
    
    cost_list = []
    variance_list = []
    gradient_list = []
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
        theta = Parameter("theta") #Paramater for adapative unitary
        
        random_unitary_matrix = random_unitary(2 ** num_qubits, seed=iteration) #Vk
        adaptive_unitary.append(random_unitary_matrix.adjoint(), range(num_qubits))
        adaptive_unitary.rx(theta, 0)
        adaptive_unitary.append(random_unitary_matrix, range(num_qubits))
        
        #Calculate gradient and objective input (Theta k)
        gradient = None
        try:
            gradient_job = gradient_estimator.run(adaptive_unitary, problem_hamiltonian, [[0]]).result()
            gradient = gradient_job.gradients[0][0]
        except:
            break
        current_objective_input = -step_size * gradient
            
        #Assign new parameter
        adaptive_unitary.assign_parameters([current_objective_input], inplace=True)
        
        #Update cost and jobs list
        cost_list.append(cost)
        variance_list.append(variance)
        gradient_list.append(gradient)

        #DEBUG
        print(f"{iteration=:5}, {cost=:20}, {gradient=}")

        #DEBUG
        # display(adaptive_unitary.draw(output="mpl"))
        # if(iteration >= 3):
        #     break
    
    return adaptive_unitary, cost_list, variance_list, gradient_list