# -*- coding: utf-8 -*- # @Time    : 4/10/20 21:03
# @Author  : Deng Yue

import csv
from collections import Counter
from math import sqrt, ceil
import operator
import statistics
import time
import os


def chunk(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


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

    def standardize(self, x_list, attr_name):
        x_list = [float(i) for i in x_list]
        if self.test_mode:
            mean = self.kernel['standardize'][attr_name]['mean']
            variance = self.kernel['standardize'][attr_name]['variance']
        else:
            mean = statistics.mean(x_list)
            variance = statistics.variance(x_list)
            if 'standardize' not in self.kernel:
                self.kernel['standardize'] = {}
            self.kernel['standardize'][attr_name] = {'mean': mean, 'variance': variance}
        return [(i - mean) / variance for i in x_list]

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

        # for attr in self.columns[:-1]:
        #     data_dict[attr] = self.standardize(data_dict[attr], attr)

        data_y = data_dict['income']
        length = len(data_y)
        data_x = [[] for i in data_y]
        for i in range(length):
            for index, column in enumerate(self.columns[:-1]):
                data_x[i].append(data_dict[column][i])

        return data_x, data_y

    def k_fold(self, k, data_x, data_y):
        step_size = ceil(len(data_x) / k)
        data_folds = chunk(data_x, step_size)
        label_folds = chunk(data_y, step_size)
        return data_folds, label_folds


class KNN(object):

    def __init__(self, k, X, Y):
        self.k = k
        self.X = X
        self.Y = Y

    @staticmethod
    def calculate_distance(a, b, distance='euclidean'):
        if distance == 'euclidean':
            total = 0
            for i in range(len(a)):
                total += (a[i] - b[i]) ** 2
            return sqrt(total)
        elif distance == "manhattan":
            total = 0
            for i in range(len(a)):
                total += abs(a[i] - b[i])
            return total

    def calculate(self, data, distance='euclidean'):
        result_list = [0] * len(data)
        for index, value in enumerate(data):
            # if index % 100 == 0:
            #     print("Process", os.getpid(), index)
            dist_list = []
            for xi in self.X:
                dist_list.append(self.calculate_distance(value, xi, distance))
            indexed = list(enumerate(dist_list))
            top_k = sorted(indexed, key=operator.itemgetter(1))[:self.k]
            top_k_index = list(reversed([i for i, v in top_k]))
            prediction_list = [self.Y[i] for i in top_k_index]
            prediction = Counter(prediction_list).most_common(1)[0][0]
            result_list[index] = prediction
        return result_list

    def predict(self, data, distance='euclidean', thread=1):

        if thread > 1:
            import multiprocessing
            data = chunk(data, int(len(data) / thread))
            pool = multiprocessing.Pool(thread)
            result = pool.map(self.calculate, data)
            pool.close()
            return result[0]
        else:
            return self.calculate(data, distance)

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
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp) if tp != 0 else 0.0
        recall = tp / (tp + fn) if tp != 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision != 0 or recall != 0 else 0.0

        return accuracy, precision, recall, f1


def main():
    train_data_object = DataProcessor('./data/adult.data')
    train_x, train_y = train_data_object.preprocess_data()

    folds = 4
    train_x_folds, train_y_folds = train_data_object.k_fold(folds, train_x, train_y)
    running_time_list = []
    result_list = []
    model_list = []
    for i in range(folds):
        print('-' * 100)
        print('Working on fold {}'.format(i))
        x_train = [row for index, fold in enumerate(train_x_folds) if index != i for row in fold]
        y_train = [row for index, fold in enumerate(train_y_folds) if index != i for row in fold]
        x_test = [row for row in train_x_folds[i]]
        y_test = [row for row in train_y_folds[i]]

        model = KNN(10, x_train, y_train)
        start = time.time()
        prediction = model.predict(x_test, distance='euclidean', thread=8)
        end = time.time()
        running_time = end - start
        result = model.evaluate(prediction, y_test)
        print("running time: {:.2f}s".format(running_time))
        print("evaluation", result)
        running_time_list.append(running_time)
        result_list.append(result)
        model_list.append(model)

    accuracy_list = [i[0] for i in result_list]
    precision_list = [i[1] for i in result_list]
    recall_list = [i[2] for i in result_list]
    f1_list = [i[3] for i in result_list]
    print('=' * 100)
    print("average running time: {:.2f}s".format(sum(running_time_list) / len(running_time_list)))
    print("average accuracy: {}".format(sum(accuracy_list) / len(accuracy_list)))
    print("average precision: {}".format(sum(precision_list) / len(precision_list)))
    print("average recall: {}".format(sum(recall_list) / len(recall_list)))
    print("average f1: {}".format(sum(f1_list) / len(f1_list)))

    best_model_index = accuracy_list.index(max(accuracy_list))
    best_model = model_list[best_model_index]
    print('best model is from fold {}'.format(best_model_index))

    test_data_object = DataProcessor('./data/adult.test', test_mode=True, kernel=train_data_object.kernel)
    test_x, test_y = test_data_object.preprocess_data()
    start = time.time()
    prediction = best_model.predict(test_x, distance='euclidean', thread=8)
    end = time.time()
    print('test running time: {:.2f}s'.format(end - start))
    print('evaluation result:', model.evaluate(prediction, test_y))

    # test_data_object = DataProcessor('./data/adult.test', test_mode=True, kernel=train_data_object.kernel)
    # test_x, test_y = test_data_object.preprocess_data()
    #
    # print('{} train samples.\n{} test samples.'.format(len(train_x), len(test_y)))
    # train_index = None
    # test_index = None
    # clf = KNN(10, train_x[:train_index], train_y[:train_index])
    # start = time.time()
    # prediction = clf.predict(test_x[:test_index], distance='euclidean', thread=8)
    # end = time.time()
    # print("running time: {:.2f}s".format(end - start))
    # print("prediction", prediction)
    # print("test y", test_y[:test_index])
    # print("evaluation", clf.evaluate(prediction, test_y[:test_index]))


if __name__ == '__main__':
    main()
