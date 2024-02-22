from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector, Pauli, SparsePauliOp
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
import numpy as np
import scipy as sp
from math import pi
from itertools import repeat
from multiprocessing import Pool
from collections import namedtuple
from enum import Enum

KBodyOperator = namedtuple("KBodyOperator", "pauli_string plus_half_pi_operator minus_half_pi_operator kbody_depth")

GradientSamplingType = Enum(value="GradientSamplingType",
							names=[
							("Full Gradient", 1),
							("FullGradient", 1),
							("Cubed Samples", 2),
							("CubedSamples", 2),
							("Squared Samples", 3),
							("SquaredSamples", 3)])

def create_chain_ising_hamiltonian(num_qubits):
	if(num_qubits < 2):
		raise Exception("num_qubits must be greater than or equal to 2")
	
	chain_ising_hamiltonian = Operator.from_label("ZZ" + "I" * (num_qubits - 2))
	for i in range(1, num_qubits - 1):
		chain_ising_hamiltonian += Operator.from_label("I" * i + "ZZ" + "I" * (num_qubits - 2 - i))
	
	return chain_ising_hamiltonian

def create_chain_ising_hamiltonian_sparseop(num_qubits):
	if(num_qubits < 2):
		raise Exception("num_qubits must be greater than or equal to 2")
	
	chain_ising_hamiltonian_terms = ["ZZ" + "I" * (num_qubits - 2)]
	for i in range(1, num_qubits - 1):
		chain_ising_hamiltonian_terms.append("I" * i + "ZZ" + "I" * (num_qubits - 2 - i))
	
	return SparsePauliOp(chain_ising_hamiltonian_terms)

def create_all_pauli_strings(num_qubits):
	pauli_dict = {
			"0": "I",
			"1": "X",
			"2": "Y",
			"3": "Z"
			}
	
	all_pauli_strings = range(1, 4 ** num_qubits)
	all_pauli_strings = [np.base_repr(num, base=4).zfill(num_qubits) for num in all_pauli_strings]
	
	#Replace integer for associated pauli
	for num in pauli_dict:
		for i, pauli_string in enumerate(all_pauli_strings):
			all_pauli_strings[i] = pauli_string.replace(num, pauli_dict[num])
	
	return all_pauli_strings

def create_kbody_operator(num_qubits, backend, pauli_string):
	#Create raw circuit
	theta = Parameter(f"theta")
	
	kbody_circuit = QuantumCircuit(num_qubits)
	kbody_circuit.append(PauliEvolutionGate(
		Pauli(pauli_string),
		time=theta,
		synthesis=LieTrotter(reps=1)),
		range(num_qubits))
	
	#Plus and minus pi/2 to calculate gradient
	plus_half_pi_circuit  = kbody_circuit.assign_parameters([pi/4])
	minus_half_pi_circuit = kbody_circuit.assign_parameters([-pi/4])
	
	plus_half_pi_operator  = Operator(plus_half_pi_circuit)
	minus_half_pi_operator = Operator(minus_half_pi_circuit)
	
	#Determine depth of kbody circuit
	kbody_transpiled_circuit = transpile(kbody_circuit, backend, optimization_level=0, seed_transpiler=0)
	kbody_depth              = kbody_transpiled_circuit.depth()
	
	return KBodyOperator(pauli_string, plus_half_pi_operator, minus_half_pi_operator, kbody_depth)

def create_all_kbody_operators(num_qubits, backend_str):
	#Create IBM Backend for calculating transpiled circuit depth
	service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q-asu/main/asu-arenz")
	backend = service.get_backend(backend_str)
	
	all_pauli_strings = create_all_pauli_strings(num_qubits)
	
	#Multithread each pauli string
	pool                = Pool()
	function_args       = zip(repeat(num_qubits), repeat(backend), all_pauli_strings)
	all_kbody_operators = pool.starmap(create_kbody_operator, function_args)
	pool.close()
	
	return all_kbody_operators

def calculate_fidelity_projector(state, states_to_project):
	projector = sum(Statevector(state_to_project).to_operator() for state_to_project in states_to_project)
	
	return np.linalg.norm(state.evolve(projector))

def commutator(A_matrix, B_matrix):
	return (A_matrix @ B_matrix) - (B_matrix @ A_matrix)

def create_trotter_operator(hamiltonian, time, reps):
	one_rep_trotter_operator = Operator(np.eye(*hamiltonian.dim))
	
	for hamiltonian_term in hamiltonian:
		one_rep_trotter_operator = one_rep_trotter_operator.compose(sp.linalg.expm(-1j * time / reps * hamiltonian_term))
	
	trotter_operator = one_rep_trotter_operator.power(reps)
	
	return trotter_operator

