import os, json
import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, MultiTaskLasso, Lasso, LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model, svm

from sklearn.ensemble import VotingClassifier as VC
from mlxtend.classifier import StackingClassifier

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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
    OTHER_MED = "Another time"
    NO_IDEA = "nan"
    NO_MED = 'I don\'t take Parkinson medications'
    data_path = '/Volumes/AwesomeBackup/Parkinsons_mPower_Voice_Features'
    test_path = '/Volumes/AwesomeBackup/test_data'

    def __init__(self):
        self.other = []
        # self.noidea = []

    def divide_sample_by_medtimepoint(self):
        path_to_json = self.data_path
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        # normal_json_files = pd.io.json.json_normalize(json_files)
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

                if sample[self.MEDTIMEPOINT] == self.OTHER_MED:
                    self.other.append(sample)

                # elif sample[self.MEDTIMEPOINT] == self.NO_IDEA:
                #     self.noidea.append(sample)


        with open("./other.csv", 'w') as other_file:
            filed_name = [self.HEALTHCODE, self.MEDTIMEPOINT, self.ZCR, self.ENERGY, self.ENTROPY]
            writer = csv.DictWriter(other_file, fieldnames=filed_name)
            writer.writeheader()
            for sample in self.other:
                writer.writerow(sample)
        other_file.close()

        # with open("./noidea.csv", 'w') as noidea_file:
        #     filed_name = [self.HEALTHCODE, self.MEDTIMEPOINT, self.ZCR, self.ENERGY, self.ENTROPY]
        #     writer = csv.DictWriter(noidea_file, fieldnames=filed_name)
        #     writer.writeheader()
        #     for sample in self.noidea:
        #         writer.writerow(sample)
        # noidea_file.close()

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

        # with open('./noidea.csv', newline='') as noidea:
        #     d = csv.reader(noidea)
        #     e = np.array([_ for _ in d])
        #     e = np.delete(e, (0), axis = 0)
        #     f = e[:, [2, 3, 4]]
        #     zcr = f[:, 0]
        #     energy = f[:,1]
        #     entropy = f[:,2]
        #     noidea_zcr = self.convert_data(zcr)
        #     noidea_energy = self.convert_data(energy)
        #     noidea_entropy = self.convert_data(entropy)

        with open('./other.csv', newline='') as other:
            d = csv.reader(other)
            e = np.array([_ for _ in d])
            e = np.delete(e, (0), axis=0)
            f = e[:, [2, 3, 4]]
            zcr = f[:, 0]
            energy = f[:, 1]
            entropy = f[:, 2]
            other_zcr = self.convert_data(zcr)
            other_energy = self.convert_data(energy)
            other_entropy = self.convert_data(entropy)

        np.save('./other_zcr', other_zcr)
        np.save('./other_energy', other_energy)
        np.save('./other_entropy', other_entropy)

        # np.save('./noidea_zcr', noidea_zcr)
        # np.save('./noidea_energy', noidea_energy)
        # np.save('./noidea_entropy', noidea_entropy)

    def truncation(self, filename, output_name):
        threshold = 350
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

    def combinedata3(self, filename1, filename2, filename3, output_name):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        data3 = np.load(filename3)
        new_data = np.concatenate((data1, data2, data3), axis=0)
        np.save(output_name, new_data)
        return new_data

    def separatedata3(self, filename, size1, size2, output1, output2, output3):
        data = np.load(filename)
        new_data1 = data[:size1, :]
        new_data2 = data[size1:size1+size2, :]
        new_data3 = data[size1+size2:, :]
        np.save(output1, new_data1)
        np.save(output2, new_data2)
        np.save(output3, new_data3)
        return new_data1, new_data2, new_data3

    def combinedata(self, filename1, filename2, output_name):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        new_data = np.concatenate((data1, data2), axis=0)
        np.save(output_name, new_data)
        return new_data

    def separatedata(self, filename, size1, output1, output2):
        data = np.load(filename)
        new_data1 = data[:size1, :]
        new_data2 = data[size1:, :]
        np.save(output1, new_data1)
        np.save(output2, new_data2)
        return new_data1, new_data2

    def plot2D_list(self, data, colors, ax):
        file = np.load(data)
        (n,_) = file.shape
        print(file.shape)
        print(n)
        index = np.arange(350)

        for i in range(n):
            ax.plot(index, file[i],  color=colors)

    def com_norm_sep2(self, filename1, filename2, output1, output2):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        new_data = np.concatenate((data1, data2), axis=0)
        norm_data = preprocessing.normalize(new_data, norm='l1')
        new_data1 = norm_data[:data1.shape[0], :]
        new_data2 = norm_data[data1.shape[0]:, :]
        np.save(output1, new_data1)
        np.save(output2, new_data2)
        return new_data1, new_data2


    def plot2D_list_diff(self, filename1, filename2, colors):
        file1 = np.load(filename1)
        file2 = np.load(filename2)

        (n1,_) = file1.shape
        (n2,_) = file2.shape
        if n1 < n2:
            n = n1
        else:
            n = n2

        index = np.arange(350)

        for i in range(n):
            plt.plot(index, file1[i] - file2[i],  color=colors)

if __name__ == '__main__':
    c = CompareMedicine()
    # c.divide_sample_by_medtimepoint()
    # c.split_data()

    # c.truncation("./other_zcr.npy", "./other_zcr.npy")
    # c.truncation("./other_energy.npy", "./other_energy.npy")
    # c.truncation("./other_entropy.npy", "./other_entropy.npy")
    # # c.truncation("./noidea_zcr.npy", "./noidea_zcr.npy")
    # c.truncation("./noidea_energy.npy", "./noidea_energy.npy")
    # c.truncation("./noidea_entropy.npy", "./noidea_entropy.npy")

    # c.combinedata("./before_energy.npy", "./after_energy.npy",
    #               "./energy.npy")
    # c.normalization("./energy.npy", "./energy.npy")
    # c.separatedata("./energy.npy", len("./before_energy.npy"),
    #                "./before_energy1.npy", "./after_energy1.npy")
    #
    # c.combinedata("./before_zcr.npy", "./after_zcr.npy",
    #               "./zcr.npy")
    # c.normalization("./zcr.npy", "./zcr.npy")
    # c.separatedata("./zcr.npy", len("./before_zcr.npy"),
    #                "./before_zcr1.npy", "./after_zcr1.npy")
    #
    # c.combinedata("./before_entropy.npy", "./after_entropy.npy",
    #               "./entropy.npy")
    # c.normalization("./entropy.npy","./entropy.npy")
    # c.separatedata("./entropy.npy", len("./before_entropy.npy"),
    #                "./before_entropy1.npy", "./after_entropy1.npy")


    # fig, ax = plt.subplots()  # create figure and axes
    # c.plot2D_list("./before_entropy2.npy", 'red', ax)
    # c.plot2D_list("./after_entropy2.npy", 'blue', ax)
    # # c.plot2D_list("./other_entropy2.npy", 'green', ax)
    # # c.plot2D_list("./noidea_entropy2.npy", 'pink', ax)
    # plt.show()

    # c.com_norm_sep2("./before2_spectal_entropy.npy", "./after2_spectal_entropy.npy",
    #                 "./before2_spectal_entropy1.npy", "./after2_spectal_entropy1.npy")
    # c.com_norm_sep2("./before2_rolloff.npy", "./after2_rolloff.npy",
    #                 "./before2_rolloff1.npy", "./after2_rolloff1.npy")
    # c.com_norm_sep2("./before2_entropy.npy", "./after2_entropy.npy",
    #                 "./before2_entropy1.npy", "./after2_entropy1.npy")

    # for entropy
    b_entropy = np.load("./before_single_patient_spectal_entropy_cutted.npy")
    sample_num_b = b_entropy.shape[0]
    train_num_b = int(0.8 * sample_num_b)
    train_sample_b = b_entropy[:train_num_b, :]
    test_sample_b = b_entropy[train_num_b:, :]

    a_entropy = np.load("./after_single_patient_spectal_entropy_cutted.npy")
    sample_num_a = a_entropy.shape[0]
    train_num_a = int(0.8 * sample_num_a)
    train_sample_a = a_entropy[:train_num_a, :]
    test_sample_a = a_entropy[train_num_a:, :]

    train_sample_entropy1 = np.append(train_sample_b, train_sample_a, axis=0)
    test_sample_entropy1 = np.append(test_sample_b, test_sample_a, axis=0)

    y_train_entropy1 = np.append(np.zeros(train_num_b), np.ones(train_num_a))
    y_test_entropy1 = np.append(np.zeros(sample_num_b - train_num_b), np.ones(sample_num_a - train_num_a))

    # for energy
    # b_entropy = np.load("./before2_rolloff1.npy")
    # sample_num_b = b_entropy.shape[0]
    # train_num_b = int(0.8 * sample_num_b)
    # train_sample_b = b_entropy[:train_num_b, :]
    # test_sample_b = b_entropy[train_num_b:, :]
    #
    # a_entropy = np.load("./after2_rolloff1.npy")
    # sample_num_a = a_entropy.shape[0]
    # train_num_a = int(0.8 * sample_num_a)
    # train_sample_a = a_entropy[:train_num_a, :]
    # test_sample_a = a_entropy[train_num_a:, :]
    #
    # train_sample_entropy2 = np.append(train_sample_b, train_sample_a, axis=0)
    # test_sample_entropy2 = np.append(test_sample_b, test_sample_a, axis=0)
    #
    # y_train_entropy2 = np.append(np.zeros(train_num_b), np.ones(train_num_a))
    # y_test_entropy2 = np.append(np.zeros(sample_num_b - train_num_b), np.ones(sample_num_a - train_num_a))
    #
    # # for zcr
    # b_entropy = np.load("./before2_entropy1.npy")
    # sample_num_b = b_entropy.shape[0]
    # train_num_b = int(0.8 * sample_num_b)
    # train_sample_b = b_entropy[:train_num_b, :]
    # test_sample_b = b_entropy[train_num_b:, :]
    #
    # a_entropy = np.load("./after2_entropy1.npy")
    # sample_num_a = a_entropy.shape[0]
    # train_num_a = int(0.8 * sample_num_a)
    # train_sample_a = a_entropy[:train_num_a, :]
    # test_sample_a = a_entropy[train_num_a:, :]
    #
    # train_sample_entropy3 = np.append(train_sample_b, train_sample_a, axis=0)
    # test_sample_entropy3 = np.append(test_sample_b, test_sample_a, axis=0)
    #
    # y_train_entropy3 = np.append(np.zeros(train_num_b), np.ones(train_num_a))
    # y_test_entropy3 = np.append(np.zeros(sample_num_b - train_num_b), np.ones(sample_num_a - train_num_a))
    #

    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced')
    lr.fit(train_sample_entropy1, y_train_entropy1)
    print("for the logistic regression model, the score for entropy is: ",
          lr.score(test_sample_entropy1, y_test_entropy1))

    # lr = LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced')
    # lr.fit(train_sample_entropy2, y_train_entropy2)
    # print("for the logistic regression model, the score for energy is: ",
    #       lr.score(test_sample_entropy2, y_test_entropy2))
    #
    # lr = LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced')
    # lr.fit(train_sample_entropy3, y_train_entropy3)
    # print("for the logistic regression model, the score for zcr is: ",
    #       lr.score(test_sample_entropy3, y_test_entropy3))

    svc = svm.SVC(C=1, kernel='linear')
    score = svc.fit(train_sample_entropy1, y_train_entropy1).score(test_sample_entropy1, y_test_entropy1)
    print("for the svm model, the score for entropy is: ", score)

    # svc = svm.SVC(C=1, kernel='linear')
    # score = svc.fit(train_sample_entropy2, y_train_entropy2).score(test_sample_entropy2, y_test_entropy2)
    # print("for the svm model, the score for energy is: ", score)
    #
    # svc = svm.SVC(C=1, kernel='linear')
    # score = svc.fit(train_sample_entropy3, y_train_entropy3).score(test_sample_entropy3, y_test_entropy3)
    # print("for the svm model, the score for zcr is: ", score)

    # lir = LinearRegression()
    # lir.fit(train_sample_entropy, y_train_entropy)
    # print("for the linear regression model, the score is: ", lir.score(test_sample_entropy, y_test_entropy))



