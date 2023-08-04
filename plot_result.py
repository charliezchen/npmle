import ast

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys

# Read data from file
with open(sys.argv[1], 'r') as f:
    data = f.readlines()

# Parse JSON objects and collect data for N and W1
N_values = []
W1_values = []
Number_of_supp = []
for line in data:
    if 'W1' in line:
        obj = ast.literal_eval(line)
        N_values.append(obj['N'])
        W1_values.append(obj['W1'])
        Number_of_supp.append(obj['number of support'])

# Plot W1 vs N
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(N_values, W1_values)
plt.title('W1 versus N')
plt.xlabel('N')
plt.ylabel('W1')

# Plot number of support over N
plt.subplot(1,2,2)
plt.plot(N_values, Number_of_supp)
plt.title('Number of Support versus N')
plt.xlabel('N')
plt.ylabel('Number of Support')
plt.tight_layout()
plt.show()

# Convert to logarithmic scale
log_N_values = np.log(N_values)
log_W1_values = np.log(W1_values)

# TODO: std for monte carlo
# TODO: Overleaf write up
# Algorithm + Implementation details + Overleaf
# One mixture + resolution(N 1000, 30 000) + finer grid (at least 20)
# convidence interval (W1, # of support)
    # Scale: log scale
    # Different parameters:
        # Stop threshold (experiment, small?), maxiter
    # 
# Two mixture
# 

# Fit a line to the log-log data
slope, intercept, r_value, p_value, std_err = stats.linregress(log_N_values, log_W1_values)
# TODO: std error
print(std_err)

# Plot log-log of W1 vs N
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(log_N_values, log_W1_values)
plt.plot(log_N_values, intercept + slope*log_N_values, 'r', label='fitted line')
plt.title('Log-log plot of W1 versus N')
plt.xlabel('log(N)')
plt.ylabel('log(W1)')

print('The slope of the line (approximate linear coefficient) is:', slope)

plt.show()
