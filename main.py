# -*- coding: utf-8 -*- # @Time    : 4/10/20 21:03
# @Author  : Deng Yue

import csv
from collections import Counter
from math import sqrt
import operator


class DataProcessor(object):

    def __init__(self, file_path, test_mode=False, kernel={}):
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
        self.test_mode = test_mode
        self.kernel = kernel
        self.data, self.missing_attr_set = self.read_data(file_path)

    # read file
    def read_data(self, path):
        data = [[] for _ in range(len(self.columns))]
        with open(path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            if self.test_mode:
                next(reader)
            for row in reader:
                for i, ele in enumerate(row):
                    data[i].append(ele.strip())
        # dict of lists
        data_dict = dict()
        missing_value_set = set()
        for index, column in enumerate(self.columns):
            data_dict[column] = data[index]
            if "?" in data[index]:
                missing_value_set.add(column)
        return data_dict, missing_value_set

    def scale(self, x_list, attr_name):
        x_list = [float(i) for i in x_list]
        if self.test_mode:
            min_value = self.kernel['scale'][attr_name]['min']
            max_value = self.kernel['scale'][attr_name]['max']
        else:
            min_value = min(x_list)
            max_value = max(x_list)
            if 'scale' not in self.kernel:
                self.kernel['scale'] = {}
            self.kernel['scale'][attr_name] = {'min': min_value, 'max': max_value}
        diff = max_value - min_value
        return [(i - min_value) / diff for i in x_list]

    def preprocess_data(self):

        data_dict = self.data
        missing_value_set = self.missing_attr_set

        # remove unused attributes
        del data_dict['fnlwgt']
        self.columns.remove('fnlwgt')

        # revert income to 0-1 integer
        data_dict['income'] = [1 if '>50K' in i else 0 for i in data_dict['income']]

        # handle missing value
        for missing_attr in missing_value_set:
            if self.test_mode:
                mode = self.kernel['missing'][missing_attr]
            else:
                mode = Counter(data_dict[missing_attr]).most_common(1)[0][0]
                if 'missing' not in self.kernel:
                    self.kernel['missing'] = {}
                self.kernel['missing'][missing_attr] = mode
            data_dict[missing_attr] = [i if i != '?' else mode for i in data_dict[missing_attr]]

        # min-max feature scailing for age, education_num, capital-gain, capital-loss, hours-per-week
        for attr in ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            data_dict[attr] = self.scale(data_dict[attr], attr_name=attr)

        # handle categorical data, replace with prob of being rich among that category
        for attr in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                     'native-country']:
            data = data_dict[attr]

            if self.test_mode:
                prob_dict = self.kernel['category'][attr]
            else:
                prob_dict = {}
                for i_data, v_data in enumerate(data):
                    if v_data not in prob_dict:
                        prob_dict[v_data] = (0, 0)
                    total = prob_dict[v_data][0]
                    total += 1
                    rich = prob_dict[v_data][1]
                    if data_dict['income'][i_data] == 1:
                        rich += 1
                    prob_dict[v_data] = (total, rich)
                for k in prob_dict.keys():
                    prob_dict[k] = prob_dict[k][1] / float(prob_dict[k][0])
                if 'category' not in self.kernel:
                    self.kernel['category'] = {}
                self.kernel['category'][attr] = prob_dict

            data = [prob_dict[i] for i in data]
            data_dict[attr] = data

        data_y = data_dict['income']
        length = len(data_y)
        data_x = [[] for i in data_y]
        for i in range(length):
            for index, column in enumerate(self.columns[:-1]):
                data_x[i].append(data_dict[column][i])

        return data_x, data_y


class KNN(object):

    def __init__(self, k, X, Y):
        self.k = k
        self.X = X
        self.Y = Y

    @staticmethod
    def calculate_distance(a, b):
        total = 0
        for i in range(len(a)):
            total += (a[i] - b[i]) ** 2
        return sqrt(total)

    def predict(self, data):
        result_list = []
        for index, value in enumerate(data):
            if index % 100 == 0:
                print(index)
            dist_list = []
            for xi in self.X:
                dist_list.append(self.calculate_distance(value, xi))
            indexed = list(enumerate(dist_list))
            top_k = sorted(indexed, key=operator.itemgetter(1))[-self.k:]
            top_k_index = list(reversed([i for i, v in top_k]))
            prediction = [self.Y[i] for i in top_k_index]
            prediction = Counter(prediction).most_common(1)[0][0]
            result_list.append(prediction)
        return result_list

    @staticmethod
    def evaluate(pred, real):
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(pred)):
            if pred[i] == 1 and real[i] == 1:
                tp += 1
            elif pred[i] == 1 and real[i] == 0:
                fp += 1
            if pred[i] == 0 and real[i] == 0:
                tn += 1
            if pred[i] == 0 and real[i] == 1:
                fn += 1
        print(tp, tn, fp, fn)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp) if tp != 0 else 0.0
        recall = tp / (tp + fn) if tp != 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision != 0 or recall != 0 else 0.0

        return accuracy, precision, recall, f1


def main():
    train_data_object = DataProcessor('./data/adult.data')
    train_x, train_y = train_data_object.preprocess_data()
    # print(len(list(train_data.values())[0]))

    test_data_object = DataProcessor('./data/adult.test', test_mode=True, kernel=train_data_object.kernel)
    test_x, test_y = test_data_object.preprocess_data()
    # print(len(list(test_data.values())[0]))

    print('{} train samples.\n{} test samples.'.format(len(train_x), len(test_y)))
    test_index = None
    clf = KNN(20, train_x[:test_index], train_y[:test_index])
    prediction = clf.predict(test_x[:test_index])
    print(prediction)
    print(test_y[:test_index])
    print(clf.evaluate(prediction, test_y[:test_index]))


if __name__ == '__main__':
    main()
