import csv
import io
import json
import numpy as np
import os
import random
import statistics
from itertools import combinations
from scipy.stats import entropy
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree


def train_target(env):
    """"if ist is the first time, the :A_visibilitymatrix  should be opened. elese: the other one named: updatedvisibilitymatrixforpositioning"""
    with open('Input_data/' + env) as csv_file:
        csv_visibility_matrix = csv.reader(csv_file, delimiter=',')
        visibility_matrix = []
        target_test = []
        for row in csv_visibility_matrix:
            target_test.append(row[0])
            row.pop(0)
            row1 = []
            for r in row:
                row1.append(int(r))
            visibility_matrix.append(row1)
        train = visibility_matrix
        feature_cols = train[0]
        train.pop(0)
        target_test.pop(0)
    target = []
    for t in target_test:
        target.append(int(t))

    return train, target, feature_cols


def dialogue_evaluation():
    for root, dirs, files in os.walk("Input_data"):
        for file in files:
            if file.endswith(".csv"):
                train, target, feature_cols = train_target(file)
                communication_count = {}
                communication_count_random = {}
                communication_count_random_supervised = {}
                for region in target:
                    known_signature = train[target.index(region)]
                    visible_objects = []
                    count_based_on_initial_info = {}
                    count_based_on_initial_info_random = {}
                    count_based_on_initial_info_srandom = {}
                    for v in range(0, len(known_signature)):
                        if known_signature[v] == 1:
                            visible_objects.append(feature_cols[v])

                    for number_of_initial_object in range(0, len(visible_objects) + 1):
                        initial_information_combination = list(combinations(visible_objects, number_of_initial_object))
                        count_based_on_initial_info_combination = {}
                        count_based_on_initial_info_combination_random = {}
                        count_based_on_initial_info_combination_srandom = {}
                        for initial_information in initial_information_combination:
                            count = number_of_qa_entropy(known_signature, list(initial_information), train, target,
                                                         feature_cols)
                            count_random = number_of_qa_random_robust(known_signature,
                                                                      list(initial_information), train, target,
                                                                      feature_cols,
                                                                      len(feature_cols) - len(initial_information) + 1)
                            count_srandom = number_of_qa_random_srobust(known_signature,
                                                                        list(initial_information), train, target,
                                                                        feature_cols,
                                                                        len(feature_cols) - len(
                                                                            initial_information) + 1)

                            count_based_on_initial_info_combination['-'.join([str(doorId) for doorId in
                                                                              list(initial_information)])] = count
                            count_based_on_initial_info_combination_random['-'.join([str(doorId) for doorId in
                                                                                     list(
                                                                                         initial_information)])] = count_random
                            count_based_on_initial_info_combination_srandom['-'.join([str(doorId) for doorId in
                                                                                      list(
                                                                                          initial_information)])] = count_srandom

                        count_based_on_initial_info[number_of_initial_object] = count_based_on_initial_info_combination
                        count_based_on_initial_info_random[
                            number_of_initial_object] = count_based_on_initial_info_combination_random
                        count_based_on_initial_info_srandom[
                            number_of_initial_object] = count_based_on_initial_info_combination_srandom
                    communication_count[region] = count_based_on_initial_info
                    communication_count_random[region] = count_based_on_initial_info_random
                    communication_count_random_supervised[region] = count_based_on_initial_info_srandom

                with open('out/e_' + file + '.json', 'w', encoding='utf-8') as fp:
                    json.dump(communication_count, fp)
                with open('out/r_' + file + '.json', 'w', encoding='utf-8') as fp:
                    json.dump(communication_count_random, fp)
                with open('out/rs_' + file + '.json', 'w', encoding='utf-8') as fp:
                    json.dump(communication_count_random_supervised, fp)
    return communication_count, communication_count_random, communication_count_random_supervised


def find_valid_regions(first_object, train, yn, feature_cols):
    if first_object:
        selected_train = train
        for fo in first_object:
            left_train = []
            for row in range(0, len(selected_train)):
                if selected_train[row][feature_cols.index(fo)] == yn:
                    left_train.append(selected_train[row])
            selected_train = left_train
    else:
        selected_train = train
    return selected_train


def number_of_qa_random_robust(known_signature, initial_information, train, target, feature_cols, n_iteration):
    counts = []
    for i in range(0, n_iteration):
        counts.append(number_of_qa_random(known_signature, initial_information, train, target, feature_cols))
    return statistics.median(counts)


def number_of_qa_random_srobust(known_signature, initial_information, train, target, feature_cols, n_iteration):
    counts = []
    for i in range(0, n_iteration):
        counts.append(number_of_qa_random_supervised(known_signature, initial_information, train, target, feature_cols))
    return statistics.median(counts)


def number_of_qa_random_supervised(known_signature, initial_information, train, target, feature_cols):
    signature = [1 if idx in initial_information else 0 for idx in range(0, len(known_signature))]
    already_asked = []
    count = 0
    initial_information_update = initial_information[:]
    valid_train = train
    not_visible = []
    while signature != known_signature:
        valid_train = find_valid_regions(initial_information_update, valid_train, 1, feature_cols)
        valid_train = find_valid_regions(not_visible, valid_train, 0, feature_cols)
        diff = set()
        for row in valid_train:
            diff.update([i for i, x in enumerate(row) if x == 1 and feature_cols[i] not in initial_information_update
                         and i not in already_asked])

        if signature == known_signature or len(valid_train) == 1:
            break
        random_choice = random.choice(list(diff))
        count += 1
        if known_signature[random_choice] == 1:
            signature[random_choice] = 1
            initial_information_update.append(feature_cols[random_choice])
        else:
            not_visible.append(random_choice)
        already_asked.append(random_choice)
    return count


def number_of_qa_random(known_signature, initial_information, train, target, feature_cols):
    signature = [1 if idx in initial_information else 0 for idx in range(0, len(known_signature))]
    already_asked = []
    count = 0
    initial_information_update = initial_information[:]
    valid_train = train
    not_visible = []
    while signature != known_signature:
        valid_train = find_valid_regions(initial_information_update, valid_train, 1, feature_cols)
        valid_train = find_valid_regions(not_visible, valid_train, 0, feature_cols)
        diff = set()
        for row in valid_train:
            diff.update([i for i, x in enumerate(row) if feature_cols[i] not in initial_information_update
                         and i not in already_asked])

        if signature == known_signature or len(valid_train) == 1:
            break
        random_choice = random.choice(list(diff))
        count += 1
        if known_signature[random_choice] == 1:
            signature[random_choice] = 1
            initial_information_update.append(feature_cols[random_choice])
        else:
            not_visible.append(random_choice)
        already_asked.append(random_choice)
    return count


def number_of_qa_entropy(known_signature, initial_information, train, target, feature_cols):
    signature = [1 if idx in initial_information else 0 for idx in range(0, len(known_signature))]
    already_asked = []
    count = 0
    initial_information_update = initial_information[:]
    valid_train = train
    not_visible = []
    while signature != known_signature:

        valid_train = find_valid_regions(initial_information_update, valid_train, 1, feature_cols)
        valid_train = find_valid_regions(not_visible, valid_train, 0, feature_cols)
        diff = set()
        for row in valid_train:
            diff.update([i for i, x in enumerate(row) if x == 1 and feature_cols[i] not in initial_information_update
                         and i not in already_asked])
        sum_check = [sum(row[i] for row in valid_train) for i in range(len(valid_train[0]))]
        for s in range(0, len(sum_check)):
            if sum_check[s] == len(valid_train) and feature_cols[s] in diff:
                diff.remove(feature_cols[s])
                signature[s] = 1

        if signature == known_signature or len(valid_train) == 1:
            break

        entropy_choice = entropy_calc(valid_train, diff, feature_cols)
        count += 1
        if known_signature[entropy_choice] == 1:
            signature[entropy_choice] = 1
            initial_information_update.append(feature_cols[entropy_choice])
        else:
            not_visible.append(entropy_choice)
        already_asked.append(entropy_choice)
    return count


def entropy_calc(valid_train, diff, feature_cols):
    sum_one = [sum(row[i] for row in valid_train) for i in range(len(valid_train[0]))]
    entropy_feature = {}
    for i in range(0, len(sum_one)):
        if feature_cols[i] in diff:
            entropy_feature[feature_cols[i]] = entropy(
                [(sum_one[i] / len(valid_train)), 1 - (sum_one[i] / len(valid_train))], base=2)
    max_entropy = max(entropy_feature, key=entropy_feature.get)

    return max_entropy


communication_matrix, communication_matrix_random, communication_matrix_random_supervised = dialogue_evaluation()
