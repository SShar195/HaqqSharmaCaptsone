from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
import numpy as np

from utils.operator_utils import create_all_pauli_strings, GradientSamplingType

class ModifiedQDriftExperimentalRandomAlgorithm:
	def __init__(self, num_qubits, qdrift_reps, max_iterations, initial_state, problem_hamiltonian, step_size, gradient_sampling_type, estimator):
		self.num_qubits             = num_qubits
		self.qdrift_reps            = qdrift_reps
		self.max_iterations         = max_iterations
		self.initial_state          = initial_state
		self.problem_hamiltonian    = problem_hamiltonian
		self.step_size              = step_size
		self.gradient_sampling_type = gradient_sampling_type
		self.estimator              = estimator
		self.gradient_estimator     = ParamShiftEstimatorGradient(estimator)
		
		#Precalculate all Pauli strings
		self.all_pauli_strings = create_all_pauli_strings(num_qubits)
	
	def run(self):
		#Initialize with state and identity matrix
		adaptive_unitary = QuantumCircuit(self.num_qubits)
		
		adaptive_unitary.initialize(self.initial_state, range(self.num_qubits))
		adaptive_unitary.id(range(self.num_qubits))
		
		#Calculate minimum eigenvalue and associated eigenstates
		eigenvalues, eigenstates = np.linalg.eig(self.problem_hamiltonian)
		min_eigenvalue           = np.min(eigenvalues).real
		min_eigenstates          = eigenstates[eigenvalues == min_eigenvalue]
		
		#Keep running random algorithm for max_iterations
		iteration                = 0
		cost_list                = []
		approximation_ratio_list = []
		variance_list            = []
		gradients_list           = []
		for iteration in range(self.max_iterations):
			#Generate random state with set seed
			random_state_iteration = np.random.RandomState(iteration) #Each iteration should sample the same Paulis
			
			#Sample paulis according to gradient sampling type
			if(self.gradient_sampling_type == GradientSamplingType.FullGradient):
				sampled_indices_gradient = list(range(len(self.all_pauli_strings)))
			elif(self.gradient_sampling_type == GradientSamplingType.CubedSamples):
				sampled_indices_gradient = random_state_iteration.choice(len(self.all_pauli_strings), size=self.num_qubits ** 3)
			elif(self.gradient_sampling_type == GradientSamplingType.SquaredSamples):
				sampled_indices_gradient = random_state_iteration.choice(len(self.all_pauli_strings), size=self.num_qubits ** 2)
			else:
				raise Exception("Unhandled GradientSamplingType")
			
			sampled_gradient_pauli_strings = [self.all_pauli_strings[sampled_index] for sampled_index in sampled_indices_gradient]
			
			#Calculate cost and approximation ratio
			cost                = None #Jk
			approximation_ratio = None
			variance            = None
			try:
				cost_job = self.estimator.run(adaptive_unitary, self.problem_hamiltonian).result()
				cost     = cost_job.values[0] #Jk
				variance = cost_job.metadata[0].get("variance")
				
				approximation_ratio = cost / min_eigenvalue
			except:
				break
			
			#Generate circuits to measure gradients
			gradient_measurement_circuits = []
			for pauli_string in sampled_gradient_pauli_strings:
				theta = Parameter(f"theta_{pauli_string}")
				
				gradient_measurement_circuit = adaptive_unitary.copy()
				gradient_measurement_circuit.append(
					PauliEvolutionGate(Pauli(pauli_string),
					time=theta,
					synthesis=LieTrotter(reps=1)),
					range(self.num_qubits))
				
				gradient_measurement_circuits.append(gradient_measurement_circuit)
			
			#Calculate gradients
			gradients = None #hj
			try:
				gradient_job = self.gradient_estimator.run(gradient_measurement_circuits, [self.problem_hamiltonian] * len(gradient_measurement_circuits), [[0]] * len(gradient_measurement_circuits)).result()
				gradients    = np.array(gradient_job.gradients).flatten() / (2 ** self.num_qubits) #Normalization factor is sqrt(2 ** self.num_qubits), but since computing sum(<gradJ, iBj> iBj), square root is cancelled out
			except:
				break
			
			#Save signs of gradient
			gradient_signs = ["+" if gradient >= 0 else "-" for gradient in gradients]
			
			#Calculate sum of gradient coefficients and normalize for associated probabilities
			absolute_gradients     = np.abs(gradients)
			sum_absolute_gradients = np.sum(absolute_gradients) #lambda
			
			if(sum_absolute_gradients > 0):
				probabilites = absolute_gradients / sum_absolute_gradients #pj
			else:
				probabilites = np.full(len(absolute_gradients), 1 / len(absolute_gradients)) #Give equal probabilities if lambda is 0
			
			#Sample num_gates number of paulis for evolution
			num_gates             = self.qdrift_reps #N
			sampled_indices       = random_state_iteration.choice(len(sampled_gradient_pauli_strings), size=num_gates, p=probabilites)
			sampled_pauli_strings = [sampled_gradient_pauli_strings[sampled_index] for sampled_index in sampled_indices]
			sampled_signs         = [gradient_signs[sampled_index] for sampled_index in sampled_indices]
			
			#Update adaptive unitary
			operator_to_evolve = sum(SparsePauliOp(sampled_sign + sampled_pauli_string) for sampled_sign, sampled_pauli_string in zip(sampled_signs, sampled_pauli_strings)) #Hj
			evolution_time     = (sum_absolute_gradients * -self.step_size) / num_gates
			adaptive_unitary.append(
					PauliEvolutionGate(operator_to_evolve,
					time=evolution_time,
					synthesis=LieTrotter(reps=1)),
					range(self.num_qubits))
			
			#Update cost, approximation_ratio, variance, and gradients list
			cost_list.append(cost)
			approximation_ratio_list.append(approximation_ratio)
			variance_list.append(variance)
			gradients_list.append(gradients)
			
			#DEBUG
			print(f"{iteration=:5}, {cost=:20.17f}, {approximation_ratio=:20.17f}")
		
		return cost_list, approximation_ratio_list, variance_list, gradients_list

