#!/usr/bin/env python
# coding: utf-8

# # Random Algorithm (Pooled Pauli, 2 Qubits, Theoretical)

# In[1]:


get_ipython().run_line_magic('run', 'Pooled\\ Pauli.ipynb')


# In[2]:


plot_name = "Pooled Pauli, Theoretical"
csv_name = "theoretical_pooled_pauli.csv"


# In[3]:


from qiskit.primitives import Estimator

#Run Random Algorithm
trained_unitary, \
cost_list, \
variance_list, \
gradient_list = random_algorithm(initial_state=INITIAL_STATE,
                                 problem_hamiltonian=PROBLEM_HAMILTONIAN,
                                 num_iterations=NUM_ITERATIONS,
                                 step_size=STEP_SIZE,
                                 options=None,
                                 num_qubits=NUM_QUBITS)


# In[4]:


#Graph cost over time
plt.plot(cost_list)
plt.title(f"Cost vs. Iterations ({plot_name})")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()


# ## Export Data

# In[5]:


results_df = pd.DataFrame({
    "cost": cost_list,
    "gradient": gradient_list
})
results_df.to_csv(csv_name, index=False)

