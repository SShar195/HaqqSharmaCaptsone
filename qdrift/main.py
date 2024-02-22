#!/usr/bin/env python3

from qiskit.quantum_info import Statevector
from qiskit.primitives import Estimator as PrimitiveEstimator
from qiskit_ibm_runtime import Estimator as RuntimeEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
import numpy as np
from math import sqrt
import argparse
from pathlib import Path

from qdrift_experimental import ModifiedQDriftExperimentalRandomAlgorithm
from utils.operator_utils import create_chain_ising_hamiltonian_sparseop, GradientSamplingType
from utils.data_utils import graph_simulation_data, export_qdrift_experimental_initial_data, export_qdrift_experimental_simulation_data

if __name__ == "__main__":
	#Parse system arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--gradient_sampling", choices=[gradient_sampling_type.name for gradient_sampling_type in GradientSamplingType], required=True)
	parser.add_argument("--num_qubits", nargs="?", default=2)
	parser.add_argument("--qdrift_reps", nargs="?", default=1)
	parser.add_argument("--max_iterations", required=True)
	parser.add_argument("--optimization_level", nargs="?", default=0)
	parser.add_argument("--ibm_backend_str", nargs="?", default="ibmq_mumbai")
	parser.add_argument("--folder_name", nargs="?")
	parser.add_argument("--dry_run", action="store_true")
	args = parser.parse_args()
	
	#Initialize constants
	GRADIENT_SAMPLING_TYPE = GradientSamplingType[args.gradient_sampling] #Method type to sample from Riemannian gradient
	NUM_QUBITS             = int(args.num_qubits) #Number of qubits
	QDRIFT_REPS            = int(args.qdrift_reps) #Number of qDRIFT repetitions
	MAX_ITERATIONS         = int(args.max_iterations) #Number of iterations to attempt to run for
	OPTIMIZATION_LEVEL     = int(args.optimization_level) #Level of circuit optimization that Qiskit will utilize
	IBM_BACKEND_STR        = args.ibm_backend_str #Specify backend for quantum computation
	DRY_RUN                = args.dry_run #If True, run locally
	
	INITIAL_STATE       = Statevector.from_label("+" * NUM_QUBITS) #psi 0
	PROBLEM_HAMILTONIAN = create_chain_ising_hamiltonian_sparseop(NUM_QUBITS) #Hp
	STEP_SIZE           = 1 / (4 * sqrt(abs(np.linalg.norm(PROBLEM_HAMILTONIAN, ord=2)))) #Gamma
	
	#Run modified qDRIFT experimental random algorithm
	service   = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q-asu/main/asu-arenz")
	backend   = service.get_backend(IBM_BACKEND_STR)
	session   = Session(service=service, backend=backend)
	options   = Options(optimization_level=OPTIMIZATION_LEVEL)
	estimator = PrimitiveEstimator() if DRY_RUN else RuntimeEstimator(options=options, session=session)
	
	modified_qdrift_experimental_random_algorithm = ModifiedQDriftExperimentalRandomAlgorithm(
		num_qubits=NUM_QUBITS,
		qdrift_reps=QDRIFT_REPS,
		max_iterations=MAX_ITERATIONS,
		initial_state=INITIAL_STATE,
		problem_hamiltonian=PROBLEM_HAMILTONIAN,
		step_size=STEP_SIZE,
		gradient_sampling_type=GRADIENT_SAMPLING_TYPE,
		estimator=estimator)
	
	cost_list, \
	approximation_ratio_list, \
	variance_list, \
	gradients_list = modified_qdrift_experimental_random_algorithm.run()
	
	#Close Qiskit IBM Runtime session
	session.close()
	
	#Export graph and simulation data
	folder_name = f"{GRADIENT_SAMPLING_TYPE.name} - {NUM_QUBITS} Qubits" if args.folder_name == None else args.folder_name
	if(DRY_RUN and args.folder_name == None):
		folder_name += " (Dry run)"
	data_path = Path(__file__).parents[1] / "data" / "qdrift_experimental" / folder_name
	data_path.mkdir(parents=True, exist_ok=True)
	
	#Export cost over adaptive steps and approximation ratio over adaptive steps graphs
	graph_simulation_data(data_path, cost_list, GRADIENT_SAMPLING_TYPE.name, "Average Cost",  NUM_QUBITS)
	graph_simulation_data(data_path, approximation_ratio_list, GRADIENT_SAMPLING_TYPE.name, "Average Approximation Ratio", NUM_QUBITS)
	
	#Export qDRIFT experimental initial and simulation data
	export_qdrift_experimental_initial_data(data_path, GRADIENT_SAMPLING_TYPE.name, INITIAL_STATE, PROBLEM_HAMILTONIAN, STEP_SIZE, NUM_QUBITS, QDRIFT_REPS, MAX_ITERATIONS, OPTIMIZATION_LEVEL, IBM_BACKEND_STR, DRY_RUN)
	export_qdrift_experimental_simulation_data(data_path, cost_list, approximation_ratio_list, variance_list, gradients_list, GRADIENT_SAMPLING_TYPE.name, NUM_QUBITS)

