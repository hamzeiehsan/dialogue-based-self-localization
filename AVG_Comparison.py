import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics


def read_file(address):
    with open(address, encoding='utf-8') as fp:
        res = json.load(fp)
    return res


base_address = 'out/{0}_{1}.csv.json'
letters = ['A', 'C', 'E', 'H', 'K', 'L', 'P', 'T', 'Y']
methods = {'r': 'Random', 'rs': 'Random Supervised', 'e': 'Entropy'}

final_results = {}
final_comparison = {}
for letter in letters:
    final_results[letter] = {}
    for key, method in methods.items():
        final_results[letter][method] = {}
        address = base_address.format(key, letter)
        data = read_file(address)

        initial_count = {}
        for region, vals in data.items():
            for info_count, info_vals in vals.items():
                if info_count not in initial_count.keys():
                    initial_count[info_count] = []
                for pattern, count in info_vals.items():
                    initial_count[info_count].append(count)
        for info_count, count_list in initial_count.items():
            if round(statistics.mean(count_list)) > 0:
                final_results[letter][method][int(info_count)] = round(statistics.mean(count_list))

final_comparison = {}
for l, methods in final_results.items():
    m = {}
    for method, ini in methods.items():
        m[method] = round(statistics.mean(ini.values()))
    final_comparison[l] = m

shift = -0.2
for environment, env_vals in final_results.items():
    for method, met_vals in env_vals.items():
        x = list(met_vals.keys())
        y = list(met_vals.values())
        plt.bar(np.array(x) + shift, y, width=0.2, label=method)
        shift += 0.2
    shift = -0.2
    plt.title(environment)
    plt.legend()
    plt.xlabel("Number of initial information")
    plt.ylabel("Dialogue Length")
    x = list(env_vals['Random'].keys())
    y = list(env_vals['Random'].values())
    plt.xticks(list(range(0, max(x) + 1)), list(range(0, max(x) + 1)))
    plt.yticks(list(range(0, max(y) + 1)), list(range(0, max(y) + 1)))
    plt.savefig('output_figures/' + environment + '.pdf', dpi=300)
    plt.show()

sorted_envs = ['A', 'H', 'T', 'Y', 'E', 'L']

# sorted_envs = ['L', 'K', 'E', 'T', 'P', 'C', 'Y', 'H', 'A']
method_overall_y = {}
for env in sorted_envs:
    env_vals = final_results[env]
    for method, method_vals in env_vals.items():
        if method not in method_overall_y.keys():
            method_overall_y[method] = []
        method_overall_y[method].append(statistics.mean(method_vals.values()))
shift = -0.2

for method, method_y in method_overall_y.items():
    plt.bar(np.arange(len(sorted_envs)) + shift, method_y, width=0.2, label=method)
    shift += 0.2
plt.title('Overall comparison per environment')
plt.xticks(np.arange(len(sorted_envs)), sorted_envs)
plt.xlabel("Environments")
plt.ylabel("AVG Dialogue Length")
plt.legend()
plt.savefig('output_figures/final_comparison.pdf', dpi=300)
plt.show()
