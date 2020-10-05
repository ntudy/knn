# -*- coding: utf-8 -*- # @Time    : 4/10/20 21:03
# @Author  : Deng Yue

import csv
from collections import Counter


class DataProcessor(object):

    def __init__(self, file_path):
        self.columns = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income"
        ]
        self.data, self.missing_attr_set = self.read_data(file_path)

    # read file
    def read_data(self, path):
        train_data = [[] for _ in range(len(self.columns))]
        with open(path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            if 'test' in path:
                next(reader)
            for row in reader:
                for i, ele in enumerate(row):
                    train_data[i].append(ele.strip())
        # dict of lists
        train_data_dict = dict()
        missing_value_set = set()
        for index, column in enumerate(self.columns):
            train_data_dict[column] = train_data[index]
            if "?" in train_data[index]:
                missing_value_set.add(column)
        return train_data_dict, missing_value_set

    def preprocess_data(self):

        def scale(x_list):
            x_list = [float(i) for i in x_list]
            min_value = min(x_list)
            max_value = max(x_list)
            diff = max_value - min_value
            return [(i - min_value) / diff for i in x_list]

        train_data_dict = self.data
        missing_value_set = self.missing_attr_set

        # remove unused attributes
        del train_data_dict['fnlwgt']

        # revert income to 0-1 integer
        train_data_dict['income'] = [1 if i == '>50K' else 0 for i in train_data_dict['income']]

        # handle missing value
        for missing_attr in missing_value_set:
            mode = Counter(train_data_dict[missing_attr]).most_common(1)[0][0]
            train_data_dict[missing_attr] = [i if i != '?' else mode for i in train_data_dict[missing_attr]]

        # min-max feature scailing for age, education_num, capital-gain, capital-loss, hours-per-week
        for attr in ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            train_data_dict[attr] = scale(train_data_dict[attr])

        # handle categorical data, replace with prob of being rich among that category
        for attr in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                     'native-country']:
            data = train_data_dict[attr]

            prob_dict = {}
            for i_data, v_data in enumerate(data):
                if v_data not in prob_dict:
                    prob_dict[v_data] = (0, 0)
                total = prob_dict[v_data][0]
                total += 1
                rich = prob_dict[v_data][1]
                if train_data_dict['income'][i_data] == 1:
                    rich += 1
                prob_dict[v_data] = (total, rich)
            for k in prob_dict.keys():
                prob_dict[k] = prob_dict[k][1] / float(prob_dict[k][0])

            data = [prob_dict[i] for i in data]
            train_data_dict[attr] = data

        return train_data_dict


# class KNN(object):
#
#     def __init__(self):
#         self.data =


def main():

    train_data_object = DataProcessor('./data/adult.data')
    train_data = train_data_object.preprocess_data()
    # print(len(list(train_data.values())[0]))

    test_data_object = DataProcessor('./data/adult.test')
    test_data = test_data_object.preprocess_data()
    # print(len(list(test_data.values())[0]))


if __name__ == '__main__':
    main()
