import os, json
import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model

from sklearn.ensemble import VotingClassifier as VC
from mlxtend.classifier import StackingClassifier


class CompareMedicine:

    HEALTHCODE = 'healthcode'
    MEDTIMEPOINT = 'medtimepoint'
    ZCR = "ZCR"
    FEATURES = "features"
    AUDIO = "audio"
    ENERGY = "energy"
    ENTROPY = "entropy"
    BEFORE_MED = "Immediately before Parkinson medication"
    AFTER_MED = "Just after Parkinson medication (at your best)"
    data_path = '/Volumes/AwesomeBackup/Parkinsons_mPower_Voice_Features'
    test_path = '/Volumes/AwesomeBackup/test_data'

    def __init__(self):
        self.before = []
        self.after = []

    def divide_sample_by_medtimepoint(self):
        path_to_json = self.data_path
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        for index, js in enumerate(json_files):
            with open(os.path.join(path_to_json, js)) as json_file:
                json_text = json.load(json_file)
                sample = {
                    self.HEALTHCODE: json_text[self.HEALTHCODE],
                    self.MEDTIMEPOINT: json_text[self.MEDTIMEPOINT],
                    self.ZCR: json_text[self.FEATURES][self.AUDIO][self.ZCR],
                    self.ENERGY: json_text[self.FEATURES][self.AUDIO][self.ENERGY],
                    self.ENTROPY: json_text[self.FEATURES][self.AUDIO][self.ENTROPY]
                }
                if sample[self.MEDTIMEPOINT] == self.BEFORE_MED:
                    self.before.append(sample)
                if sample[self.MEDTIMEPOINT] == self.AFTER_MED:
                    self.after.append(sample)
        with open("./before.csv", 'w') as before_file:
            filed_name = [self.HEALTHCODE, self.MEDTIMEPOINT, self.ZCR, self.ENERGY, self.ENTROPY]
            writer = csv.DictWriter(before_file, fieldnames=filed_name)
            writer.writeheader()
            for sample in self.before:
                writer.writerow(sample)
        before_file.close()

        with open("./after.csv", 'w') as after_file:
            filed_name = [self.HEALTHCODE, self.MEDTIMEPOINT, self.ZCR, self.ENERGY, self.ENTROPY]
            writer = csv.DictWriter(after_file, fieldnames=filed_name)
            writer.writeheader()
            for sample in self.after:
                writer.writerow(sample)
        after_file.close()

    def convert_data(self, input_data):
        output_data = []
        for sample in input_data:
            sample = sample[1:-1]
            sample = sample.split(", ")
            sample = [float(_) for _ in sample]
            output_data.append(sample)
        output_data = np.array(output_data)
        return output_data

    def split_data(self):
        def myFloat(mylist):
            return map(float, mylist)

        with open('./before.csv', newline='') as before:
            d = csv.reader(before)
            e = np.array([_ for _ in d])
            e = np.delete(e, (0), axis = 0)
            f = e[:, [2, 3, 4]]
            zcr = f[:, 0]
            energy = f[:,1]
            entropy = f[:,2]
            before_zcr = self.convert_data(zcr)
            before_energy = self.convert_data(energy)
            before_entropy = self.convert_data(entropy)

        with open('./after.csv', newline= '') as after:
            d = csv.reader(after)
            e = np.array([_ for _ in d])
            e = np.delete(e, (0), axis=0)
            f = e[:, [2, 3, 4]]
            zcr = f[:, 0]
            energy = f[:, 1]
            entropy = f[:, 2]
            after_zcr = self.convert_data(zcr)
            after_energy = self.convert_data(energy)
            after_entropy = self.convert_data(entropy)

        np.save('./before_zcr', before_zcr)
        np.save('./before_energy', before_energy)
        np.save('./before_entropy', before_entropy)
        np.save('./after_zcr', after_zcr)
        np.save('./after_energy', after_energy)
        np.save('./after_entropy', after_entropy)

    def truncation(self, filename, output_name):
        threshold = 300
        data = np.load(filename)
        data = np.array([np.array(_) for _ in data])
        new_data = []
        for sample in data:
            if sample.shape[0] > threshold:
                sample = sample[:threshold]
                new_data.append(sample)
            else:
                continue
        np.save(output_name, new_data)
        return new_data

    def normalization(self, filename, output_name):
        data = np.load(filename)
        new_data = preprocessing.normalize(data, norm='l1')
        np.save(output_name, new_data)
        return new_data

    def combinedata(self, filename1, filename2, output_name):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        new_data = np.concatenate((data1, data2), axis=0)
        np.save(output_name, new_data)
        return new_data

    def combinedata2(self, filename1, filename2, output_name):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        new_data = np.concatenate((data1, data2), axis=1)
        np.save(output_name, new_data)
        return new_data

    def separatedata(self, filename, size1, output1, output2):
        data = np.load(filename)
        new_data1 = data[:size1,:]
        new_data2 = data[size1:,:]
        np.save(output1, new_data1)
        np.save(output2, new_data2)
        return new_data1, new_data2


if __name__ == '__main__':
    c = CompareMedicine()
    # c.divide_sample_by_medtimepoint()
    # c.split_data()
    # #
    # c.truncation("./before_zcr.npy", "./before_zcr.npy")
    # c.truncation("./after_zcr.npy", "./after_zcr.npy")
    # c.truncation("./before_energy.npy", "./before_energy.npy")
    # c.truncation("./after_energy.npy", "./after_energy.npy")
    # c.truncation("./after_entropy.npy", "./after_entropy.npy")
    # c.truncation("./before_entropy.npy", "./before_entropy.npy")
    #
    # c.combinedata("./before_energy.npy", "./after_energy.npy", "./ba_energy.npy")
    # c.normalization("./ba_energy.npy", "./ba_energy.npy")
    # c.separatedata("./ba_energy.npy", len("./before_energy.npy"), "./before_energy1.npy", "./after_energy1.npy")
    #
    # c.combinedata("./before_zcr.npy", "./after_zcr.npy", "./ba_zcr.npy")
    # c.normalization("./ba_zcr.npy", "./ba_zcr.npy")
    # c.separatedata("./ba_zcr.npy", len("./before_zcr.npy"), "./before_zcr1.npy", "./after_zcr1.npy")
    #
    # c.combinedata("./before_entropy.npy", "./after_entropy.npy", "./ba_entropy.npy")
    # c.normalization("./ba_entropy.npy","./ba_entropy.npy")
    # c.separatedata("./ba_entropy.npy", len("./before_entropy.npy"), "./before_entropy1.npy", "./after_entropy1.npy")

# # for energy & entropy
#     c.combinedata2("./before_energy1.npy", "./before_entropy1.npy",
#                    "./before_energy_entropy1.npy")
#     b_energy = np.load("./before_energy_entropy1.npy")
#     sample_num_b = b_energy.shape[0]
#     train_num_b = int(0.8*sample_num_b)
#     train_sample_b = b_energy[:train_num_b, :]
#     test_sample_b = b_energy[train_num_b:, :]
#
#     c.combinedata2("./after_energy1.npy", "./after_entropy1.npy", "./after_energy_entropy1.npy")
#     a_energy = np.load("./after_energy_energy1.npy")
#     sample_num_a = a_energy.shape[0]
#     train_num_a = int(0.8 * sample_num_a)
#     train_sample_a = a_energy[:train_num_a, :]
#     test_sample_a = a_energy[train_num_a:, :]
#
#     train_sample_energy = np.append(train_sample_b, train_sample_a, axis= 0)
#     test_sample_energy = np.append(test_sample_b, test_sample_a, axis = 0)
#
#     y_train_energy = np.append(np.zeros(train_num_b), np.ones(train_num_a))
#     y_test_energy = np.append(np.zeros(sample_num_b - train_num_b), np.ones(sample_num_a - train_num_a))
#
#     lr = LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced')
#     lr.fit(train_sample_energy, y_train_energy)
#     print("if combine energy and entropy together, we get score:", lr.score(test_sample_energy, y_test_energy))

    # method 1 just combine them together and them run the model
    # train_sample_together1 = np.concatenate((train_sample_energy, train_sample_entropy), axis=0)
    # y_train_together1 = np.concatenate((y_train_energy, y_train_entropy),axis=0)
    # test_energy_entropy = np.concatenate((test_sample_energy, test_sample_entropy), axis=0)
    # y_test_energy_entropy = np.concatenate((y_test_energy, y_test_entropy), axis=0)
    #
    # lr_energy_entropy = lr1.fit(train_sample_together1, y_train_together1)
    # print(lr1.score(test_energy_entropy, y_test_energy_entropy))

    # # method 2
    # combine_lr1 = lr1.fit(Stacker(lr1).fit_transformation(train_sample_energy, y_train_energy))
    # combine_lr2 = lr1.fit(Stacker(lr1).fit_transformation(train_sample_entropy, y_train_entropy))
    #
    # # method 3
    # lrr = Your_Meta_Classifier()
    # sclf = StackingClassifier(classifiers=[])

    # # Method 4 apply the sklearn.ensemble.VotingClassifier
    # combine_lr = VC(estimators=[("lr1",lr1),('lr2',lr2)])
    # combine_lr.fit(train_sample_together1, y_train_together1)
    # print(combine_lr.score(test_energy_entropy, y_test_energy_entropy))


    # regr = ElasticNet(random_state= 0)
    # regr.fit(train_sample, y_train)
    # print(regr.coef_)
    # print(regr.intercept_)
    # # print(regr.predict([[0, 0]]))