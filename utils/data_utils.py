import pandas as pd
import matplotlib.pyplot as plt

def sanitize_to_file_name(string):
	#Would be better if I used "slugify", but for this its ok
	return string.lower().replace(" ", "_")

def graph_simulation_data(data_path, data_list, sampling_type_str, data_str, num_qubits):
	plt.clf()
	plt.rc('font',**{'family':'serif','serif':['Palatino']})
	plt.rc('text', usetex=True)
	
	title_str = f"{data_str} vs. Adaptive Steps ({sampling_type_str}, {num_qubits} Qubits)"
	
	plt.figure(figsize=(15, 9), linewidth=2 * 1.15)
	plt.plot(data_list, linewidth=2 * 1.15)
	plt.title(title_str, fontsize=20 * 1.15, pad=10)
	plt.xlabel("Adaptive Steps", fontsize=17 * 1.15)
	plt.ylabel(data_str, fontsize=17 * 1.15)
	plt.tick_params(labelsize=15 * 1.15)
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	lowercase_data_str          = sanitize_to_file_name(data_str)
	
	plt.savefig(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_{lowercase_data_str}_graph.png")

def export_trotter_initial_data(data_path, sampling_type_str, initial_state, problem_hamiltonian, step_size, num_qubits, approximation_ratio_threshold, trotter_reps, num_samples, ibm_backend_str):
	initial_df = pd.DataFrame({
		"gradient_sampling_type": [sampling_type_str],
		"initial_state": [initial_state],
		"problem_hamiltonian": [problem_hamiltonian],
		"step_size": [step_size],
		"num_qubits": [num_qubits],
		"approximation_ratio_threshold": [approximation_ratio_threshold],
		"trotter_reps": [trotter_reps],
		"num_samples": [num_samples],
		"ibm_backend_str": [ibm_backend_str]
		})
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	
	initial_df.to_csv(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_trotter_initial_data.csv", index=False)

def export_trotter_scaling_data(data_path, num_kbodies, upper_bound_depth, sampling_type_str, num_qubits):
	scaling_df = pd.DataFrame({
		"num_kbodies": [num_kbodies],
		"upper_bound_depth": [upper_bound_depth]
		})
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	
	scaling_df.to_csv(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_trotter_scaling_data.csv", index=False)

def export_trotter_simulation_data(data_path, cost_list, fidelity_list, approximation_ratio_list, sampling_type_str, num_qubits):
	simulation_df = pd.DataFrame({
		"cost": cost_list,
		"fidelity": fidelity_list,
		"approximation_ratio": approximation_ratio_list
		})
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	
	simulation_df.to_csv(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_trotter_sim_data.csv", index=False)

def export_qdrift_initial_data(data_path, sampling_type_str, initial_state, problem_hamiltonian, step_size, num_qubits, approximation_ratio_threshold, qdrift_reps, num_samples, ibm_backend_str):
	initial_df = pd.DataFrame({
		"gradient_sampling_type": [sampling_type_str],
		"initial_state": [initial_state],
		"problem_hamiltonian": [problem_hamiltonian],
		"step_size": [step_size],
		"num_qubits": [num_qubits],
		"approximation_ratio_threshold": [approximation_ratio_threshold],
		"qdrift_reps": [qdrift_reps],
		"num_samples": [num_samples],
		"ibm_backend_str": [ibm_backend_str]
		})
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	
	initial_df.to_csv(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_qdrift_initial_data.csv", index=False)

def export_qdrift_scaling_data(data_path, avg_num_kbodies, avg_upper_bound_depth, sampling_type_str, num_qubits):
	scaling_df = pd.DataFrame({
		"avg_num_kbodies": [avg_num_kbodies],
		"avg_upper_bound_depth": [avg_upper_bound_depth]
		})
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	
	scaling_df.to_csv(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_qdrift_scaling_data.csv", index=False)

def export_qdrift_simulation_data(data_path, avg_cost_list, avg_fidelity_list, avg_approximation_ratio_list, sampling_type_str, num_qubits):
	simulation_df = pd.DataFrame({
		"avg_cost": avg_cost_list,
		"avg_fidelity": avg_fidelity_list,
		"avg_approximation_ratio": avg_approximation_ratio_list
		})
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	
	simulation_df.to_csv(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_qdrift_sim_data.csv", index=False)

def export_kbody_depths_data(data_path, all_kbody_operators, sampling_type_str, num_qubits):
	kbody_depth_df = pd.DataFrame({
		"kbody_depth": [kbody_operator.kbody_depth for kbody_operator in all_kbody_operators]
		})
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	
	kbody_depth_df.index = [kbody_operator.pauli_string for kbody_operator in all_kbody_operators]
	kbody_depth_df.to_csv(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_kbody_depths_data.csv")

def graph_scaling_plot(data_path, trotter_data, qdrift_data, trotter_label, qdrift_label, sampling_type_str, data_type, use_logscale):
	qubit_range = range(2, len(trotter_data) + 2)
	scale_str   = None
	
	plt.rc('font',**{'family':'serif','serif':['Palatino']})
	plt.rc('text', usetex=True)
	
	plt.figure(figsize=(10, 7))
	plt.xticks(qubit_range)
	if(use_logscale):
		plt.yscale("log")
		scale_str = "log"
	else:
		scale_str= "linear"
	plt.plot(qubit_range, trotter_data, label=trotter_label, linewidth=2 * 1.15)
	plt.plot(qubit_range, qdrift_data, label=qdrift_label, linewidth=2 * 1.15)
	plt.legend(fontsize=15 * 1.15)
	plt.title(f"{trotter_label} vs. {qdrift_label} ({data_type}, {sampling_type_str})", fontsize=20 * 1.15, pad=10)
	plt.xlabel("Number of Qubits", fontsize=17 * 1.15)
	plt.ylabel(f"{data_type}", fontsize=17 * 1.15)
	plt.tick_params(labelsize=15 * 1.15)
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	lowercase_data_str          = sanitize_to_file_name(data_type)
	
	plt.savefig(data_path / f"{lowercase_sampling_type_str}_{lowercase_data_str}_{scale_str}_scaling_plot.png", bbox_inches="tight")

def export_qdrift_experimental_initial_data(data_path, sampling_type_str, initial_state, problem_hamiltonian, step_size, num_qubits, qdrift_reps, max_iterations, optimization_level, ibm_backend_str, dry_run):
	initial_df = pd.DataFrame({
		"gradient_sampling_type": [sampling_type_str],
		"initial_state": [initial_state],
		"problem_hamiltonian": [problem_hamiltonian],
		"step_size": [step_size],
		"num_qubits": [num_qubits],
		"qdrift_reps": [qdrift_reps],
		"max_iterations": [max_iterations],
		"optimization_level": [optimization_level],
		"ibm_backend_str": [ibm_backend_str],
		"dry_run": [dry_run]
		})
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	
	initial_df.to_csv(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_qdrift_experimental_initial_data.csv", index=False)

def export_qdrift_experimental_simulation_data(data_path, cost_list, approximation_ratio_list, variance_list, gradients_list, sampling_type_str, num_qubits):
	simulation_df = pd.DataFrame({
		"cost": cost_list,
		"approximation_ratio": approximation_ratio_list,
		"variance": variance_list,
		"gradients": gradients_list
		})
	
	lowercase_sampling_type_str = sanitize_to_file_name(sampling_type_str)
	
	simulation_df.to_csv(data_path / f"{lowercase_sampling_type_str}_{num_qubits}_qubits_qdrift_experimental_sim_data.csv", index=False)

