from haar import random_algorithm
from qiskit.quantum_info import random_statevector, Operator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from qiskit.primitives import Estimator
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    NUM_QUBITS = 2 #Number of qubits
    NUM_ITERATIONS = 150 #Number of iterations in the optimization loop
    STEP_SIZE = 0.25 #Gamma

    INITIAL_STATE = random_statevector(2 ** NUM_QUBITS, seed=7) #psi 0
    PROBLEM_HAMILTONIAN = Operator.from_label("ZZ") #Hp

    service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q-asu/main/asu-arenz")
    backend = service.get_backend("ibmq_mumbai")
    estimator = Estimator(options=None)

    plot_name = "Haar, Theoretical"
    csv_name = "theoretical_haar.csv"

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