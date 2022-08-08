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
