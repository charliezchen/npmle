import ast

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
import os
import re
import pickle

# Read data from file
target_folder = sys.argv[1]
data = []
for file in os.listdir(target_folder):
    path = os.path.join(target_folder, file)
    # N = re.search("(.+).pkl", file).group(1)
    # N = int(N)
    with open(path, 'rb') as f:
        pkl = pickle.load(f)
    data.append(pkl)
    
data.sort(key=lambda x:x['N'])
data_dict = {}
for key in data[0].keys():
    data_dict[key] = np.array([d[key] for d in data])


# Plot W1 vs N
plt.plot(data_dict['N'], data_dict['w1_mean'], 'ro--')
upper = data_dict['w1_mean'] + data_dict['w1_std']
lower = data_dict['w1_mean'] - data_dict['w1_std']

plt.fill_between(data_dict['N'], lower, upper, alpha=0.3)
plt.title('W1 versus N')
plt.xlabel('N')
plt.ylabel('W1')
plt.show()


# Plot W1 vs N
plt.plot(data_dict['N'], data_dict['n_mean'], 'ro--')
upper = data_dict['n_mean'] + data_dict['n_std']
lower = data_dict['n_mean'] - data_dict['n_std']

plt.fill_between(data_dict['N'], lower, upper, alpha=0.3)
plt.title('Number of support versus N')
plt.xlabel('N')
plt.ylabel('Number of support')
plt.show()

# Convert to logarithmic scale
log_N_values = np.log(data_dict['N'])
log_W1_values = np.log(data_dict['w1_mean'])
w1_std = (data_dict['w1_std'])
upper = log_W1_values + np.log(w1_std)
lower = log_W1_values - np.log(w1_std) # TODO: The proper transformation here

# Fit a line to the log-log data
slope, intercept, r_value, p_value, std_err = stats.linregress(log_N_values, log_W1_values)

# Plot log-log of W1 vs N
plt.scatter(log_N_values, log_W1_values)
plt.plot(log_N_values, intercept + slope*log_N_values, 'r--', label='fitted line')
plt.fill_between(log_N_values, lower, upper, alpha=0.3)
plt.title('Log-log plot of W1 versus N')
plt.xlabel('log(N)')
plt.ylabel('log(W1)')

print('The slope of the line (approximate linear coefficient) is:', slope)

plt.show()
