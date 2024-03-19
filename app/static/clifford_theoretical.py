from clifford import random_algorithm
from qiskit.quantum_info import random_statevector, Operator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from qiskit.primitives import Estimator
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

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
    NUM_ITERATIONS             = 150 #Number of iterations in the optimization loop
    INITIAL_STATE       = Statevector.from_label("+" * NUM_QUBITS) #psi 0
    PROBLEM_HAMILTONIAN = create_chain_ising_hamiltonian_sparseop(NUM_QUBITS) #Hp
    STEP_SIZE           = 1 / (4 * sqrt(abs(np.linalg.norm(PROBLEM_HAMILTONIAN, ord=2)))) #Gamma    
    service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q-asu/main/asu-arenz")
    backend = service.get_backend("ibmq_mumbai")
    estimator = Estimator(options=None)
    plot_name = "Haar, Theoretical"
    csv_name = "theoretical_haar.csv"

    service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q-asu/main/asu-arenz")
    backend = service.get_backend("ibmq_mumbai")
    estimator = Estimator(options=None)

    plot_name = "Clifford, Theoretical"
    csv_name = "theoretical_clifford.csv"

    #Run Random Algorithm
    trained_unitary, \
    cost_list, \
    variance_list, \
    gradient_list = random_algorithm(initial_state=INITIAL_STATE,
                                     problem_hamiltonian=PROBLEM_HAMILTONIAN,
                                     num_iterations=NUM_ITERATIONS,
                                     step_size=STEP_SIZE,
                                     estimator=estimator,
                                     num_qubits=NUM_QUBITS)
    
    #Graph cost over time
    plt.plot(cost_list)
    plt.title(f"Cost vs. Iterations ({plot_name})")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
    
    results_df = pd.DataFrame({
    "cost": cost_list,
    "gradient": gradient_list
    })
    results_df.to_csv(csv_name, index=False)
