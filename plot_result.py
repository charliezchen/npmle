import ast

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
import os
import re
import pickle
import seaborn as sns

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

def get_mean_and_std(series):
    return series.mean(axis=1), series.std(axis=1)

number_of_simulation = data_dict['w1'].shape[-1]

# W1 vs N
x=data_dict['N'].repeat(number_of_simulation, axis=-1)
y=data_dict['w1'].reshape(-1)
sns.violinplot(x=x,y=y,hue=np.ones_like(x))
plt.xlabel("N")
plt.ylabel("W1")
plt.show()

y=data_dict['n'].reshape(-1)
sns.violinplot(x=x,y=y,hue=np.ones_like(x))
plt.xlabel("N")
plt.ylabel("# supp")
plt.show()

y=np.log(data_dict['w1'].reshape(-1))
sns.violinplot(x=np.log(x),y=y,hue=np.ones_like(x))
plt.xlabel("log N")
plt.ylabel("log W1")
plt.show()


# Plot W1 vs N
w1_mean, w1_std = get_mean_and_std(data_dict['wa'])
plt.plot(data_dict['N'], w1_mean, 'ro--')
upper = w1_mean + w1_std
lower = w1_mean - w1_std

plt.fill_between(data_dict['N'], lower, upper, alpha=0.3)
plt.title('W1 versus N')
plt.xlabel('N')
plt.ylabel('W1')
plt.show()

exit(0)


# Plot W1 vs N
w1_mean, w1_std = get_mean_and_std(data_dict['w1'])
plt.plot(data_dict['N'], w1_mean, 'ro--')
upper = w1_mean + w1_std
lower = w1_mean - w1_std


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
