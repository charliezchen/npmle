import ast

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
import os
import re
import pickle
import seaborn as sns

EPSILON=1e-8

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

# 
for key,value in data_dict.items():
    if key == 'N':
        data_dict[key] = value.repeat(number_of_simulation, axis=-1)
    else:
        data_dict[key] = value.reshape(-1)

data_dict['Nk'] = data_dict['N'] // 1000

def violinplot(data_dict, x_key, y_key, x_name, y_name):
    x, y = data_dict[x_key], data_dict[y_key]
    sns.violinplot(x=x,y=y,hue=["1k simulation" for _ in range(len(x))])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

violinplot(data_dict, "Nk", "w1", "number of thousands of samples", 'W1')
violinplot(data_dict, "Nk", "n", "number of thousands of samples", 'Number of support')

data_dict['log_N'] = np.log(data_dict['N'])
data_dict['log_w1'] = np.log(data_dict['w1']+EPSILON)
data_dict['log_N_str'] = [f"{i:.1f}" for i in data_dict['log_N']]
violinplot(data_dict, 'log_N_str', 'log_w1', 'log N', 'log W1')

# Use sns
def lineplot(data_dict, x_key, y_key, x_name, y_name):
    x, y = data_dict[x_key], data_dict[y_key]
    sns.lineplot(x=x, y=y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
 
    plt.show()

    x = data_dict[x_key].reshape(-1,1000).mean(axis=1)
    mean, std = get_mean_and_std(data_dict[y_key].reshape(-1, number_of_simulation))
    plt.plot(x, mean, 'ro--')    
    lower, upper = mean-std, mean+std
    plt.fill_between(x, lower, upper, alpha=0.3)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
# def lineplot(data_dict, key, key_name):
#     w1_mean, w1_std = get_mean_and_std(data_dict[key])
#     plt.plot(data_dict['N'], w1_mean, 'ro--')
#     upper = w1_mean + w1_std
#     lower = w1_mean - w1_std

#     plt.fill_between(data_dict['N'], lower, upper, alpha=0.3)
#     plt.xlabel("Number of thousands of support")
#     plt.ylabel(key_name)
    plt.show()

lineplot(data_dict, 'Nk', 'w1', "number of thousands of samples", 'W1')
lineplot(data_dict, 'Nk', 'n', "number of thousands of samples", 'number of support')
lineplot(data_dict, 'log_N', 'w1', "log N", 'log W1')
exit(0)

lineplot(data_dict, 'w1', 'W1')
lineplot(data_dict, 'n', 'Number of support')

data_dict['log_w1'] = np.log(data_dict['w1'] + epsilon)
lineplot(data_dict, 'log_w1', 'log W1', 'log N')



# Plot W1 vs N

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
