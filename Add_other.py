import os, json
import csv

import numpy as np
import pandas as pd
import warnings
from pandas.io.json import json_normalize
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import preprocessing, svm
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import scipy.cluster.hierarchy as hc
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model

from sklearn.ensemble import VotingClassifier as VC
from mlxtend.classifier import StackingClassifier

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs



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

    def com_norm_sep3(self, filename1, filename2, filename3, output1, output2, output3):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        data3 = np.load(filename3)
        new_data = np.concatenate((data1, data2, data3), axis=0)
        norm_data = preprocessing.normalize(new_data, norm='l1')
        new_data1 = norm_data[:data1.shape[0], :]
        new_data2 = norm_data[data1.shape[0]:data1.shape[0]+data2.shape[0], :]
        new_data3 = norm_data[data1.shape[0]+data2.shape[0]:, :]
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

    def plot2D_list(self, data, colors, ax):
        file = np.load(data)
        (n,_) = file.shape
        print(file.shape)
        print(n)
        index = np.arange(300)

        for i in range(n):
            ax.plot(index, file[i],  color=colors)


    def plot2D_list_diff(self, filename1, filename2, colors):
        file1 = np.load(filename1)
        file2 = np.load(filename2)

        (n1,_) = file1.shape
        (n2,_) = file2.shape
        if n1 < n2:
            n = n1
        else:
            n = n2

        index = np.arange(300)

        for i in range(n):
            plt.plot(index, file1[i] - file2[i],  color=colors)

    def calculate_mean(self, filename, label, colors, ax):
        file = np.load(filename)
        mean = file.mean(axis = 0)

        index = np.arange(300)

        plt.plot(index, mean, label = label, color=colors)

    def calculate_mean_patient(self, filename, label, colors, ax):
        file = np.load(filename)
        mean = file.mean(axis = 1)

        plt.plot(mean, label = label, color=colors)


if __name__ == '__main__':
    c = CompareMedicine()

    # # draw the dendrogram
    # plt.figure(figsize=(25, 10))
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('sample index')
    # plt.ylabel('distance')
    # entropy = np.load('./entropy.npy')
    # dendrogram(
    #     entropy,
    #     leaf_rotation=90.,  # rotates the x axis labels
    # )
    # plt.show()

    # c.com_norm_sep3("./before2_spectal_entropy.npy", "./after2_spectal_entropy.npy", "./other2_spectal_entropy.npy",
    #                 "./before2_spectal_entropy2.npy", "./after2_spectal_entropy2.npy", "./other2_spectal_entropy2.npy")
    # c.com_norm_sep3("./before2_rolloff.npy", "./after2_rolloff.npy", "./other2_rolloff.npy",
    #                 "./before2_rolloff2.npy", "./after2_rolloff2.npy", "./other2_rolloff2.npy")
    # c.com_norm_sep3("./before2_entropy.npy", "./after2_entropy.npy", "./other2_entropy.npy",
    #                 "./before2_entropy2.npy", "./after2_entropy2.npy", "./other2_entropy2.npy")

    # draw the hiarachy clustering
    # c.combinedata3("./before2_spectal_entropy2.npy", "./after2_spectal_entropy2.npy",
    #                "./other2_spectal_entropy2.npy", "./spectal_entropy2.npy")
    np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
    X = np.load('./spectal_entropy2.npy')
    a = np.load("./before2_spectal_entropy2.npy")
    b = np.load("./after2_spectal_entropy2.npy")
    c = np.load("./other2_spectal_entropy2.npy")
    ashape = a.shape[0]
    bshape = b.shape[0]
    cshape = c.shape[0]

    label = np.append(np.zeros(int(ashape)), np.ones(int(bshape)))
    l = np.ones(int(cshape))
    l = [x * 2 for x in l]
    labels = np.append(label, l)


    Z = linkage(X, 'ward')

    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    c, coph_dists = cophenet(Z, pdist(X))

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('patient samples')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,
        labels=labels,  # font size for the x axis labels
    )
    plt.show()

    # # draw the cluster map with sns
    # sns.set(color_codes=True)
    # entropy = np.load('./entropy.npy')
    # g = sns.clustermap(entropy)

    # c.divide_sample_by_medtimepoint()
    # c.split_data()
    #
    # c.truncation("./other_zcr.npy", "./other_zcr.npy")
    # c.truncation("./other_energy.npy", "./other_energy.npy")
    # c.truncation("./other_entropy.npy", "./other_entropy.npy")

    # c.combinedata3("./before_energy.npy", "./after_energy.npy",
    #               "./other_energy.npy", "./energy.npy")
    # c.normalization("./energy.npy", "./energy.npy")
    # c.separatedata3("./energy.npy", len("./before_energy.npy"),
    #                len("./after_energy.npy"),
    #                "./before_energy2.npy", "./after_energy2.npy",
    #                "./other_energy2.npy")
    #
    # c.combinedata3("./before_zcr.npy", "./after_zcr.npy", "./other_zcr.npy",
    #               "./zcr.npy")
    # c.normalization("./zcr.npy", "./zcr.npy")
    # c.separatedata3("./zcr.npy", len("./before_zcr.npy"), len("./after_zcr.npy"),
    #                "./before_zcr2.npy", "./after_zcr2.npy",
    #                "./other_zcr2.npy")
    #
    # c.combinedata3("./before_entropy.npy", "./after_entropy.npy",
    #               "./other_entropy.npy", "./entropy.npy")
    # c.normalization("./entropy.npy","./entropy.npy")
    # c.separatedata3("./entropy.npy", len("./before_entropy.npy"),
    #                len("./after_entropy.npy"),
    #                "./before_entropy2.npy", "./after_entropy2.npy",
    #                "./other_entropy2.npy")






# # draw the plots to show the differences
#     fig, ax = plt.subplots()  # create figure and axes
#     c.calculate_mean("./before_entropy2.npy", 'before', 'red', ax)
#     c.calculate_mean("./after_entropy2.npy", 'after', 'blue', ax)
#     c.calculate_mean("./other_entropy2.npy", 'other', 'green', ax)
#     # c.plot2D_list("./noidea_entropy2.npy", 'pink', ax)
#     plt.legend()
#     plt.show()

# # draw the plots to show the differences between each patient (failed)
#     fig, ax = plt.subplots()  # create figure and axes
#     c.calculate_mean_patient("./before_entropy.npy", 'before', 'red', ax)
#     c.calculate_mean_patient("./after_entropy.npy", 'after', 'blue', ax)
#     c.calculate_mean_patient("./other_entropy.npy", 'other', 'green', ax)
#     # c.plot2D_list("./noidea_entropy2.npy", 'pink', ax)
#     plt.legend()
#     plt.show()
#     #
    # fig, ax = plt.subplots()  # create figure and axes
    # c.plot2D_list("./before_entropy2.npy", 'red', ax)
    # c.plot2D_list("./after_entropy2.npy", 'blue', ax)
    # plt.show()
    #
    # fig, ax = plt.subplots()  # create figure and axes
    # c.plot2D_list("./before_entropy2.npy", 'red', ax)
    # c.plot2D_list("./other_entropy2.npy", 'green', ax)
    # # c.plot2D_list("./noidea_entropy2.npy", 'C3')
    # plt.show()
    #
    # c.plot2D_list_diff("./before_entropy2.npy", "./after_entropy2.npy", 'red')
    # c.plot2D_list_diff("./before_entropy2.npy", "./other_entropy2.npy", 'blue')
    # c.plot2D_list_diff("./after_entropy2.npy", "./other_entropy2.npy", 'green')
    #
    # # c.plot2D_list("./noidea_entropy2.npy", 'C3')
    # plt.show()

# re-calculate the value
#     b_entropy = np.load("./before_zcr2.npy")
#     sample_num_b = b_entropy.shape[0]
#     train_num_b = int(0.8 * sample_num_b)
#     train_sample_b = b_entropy[:train_num_b, :]
#     test_sample_b = b_entropy[train_num_b:, :]
#
#     a_entropy = np.load("./after_zcr2.npy")
#     sample_num_a = a_entropy.shape[0]
#     train_num_a = int(0.8 * sample_num_a)
#     train_sample_a = a_entropy[:train_num_a, :]
#     test_sample_a = a_entropy[train_num_a:, :]
#
#     train_sample_entropy = np.append(train_sample_b, train_sample_a, axis=0)
#     test_sample_entropy = np.append(test_sample_b, test_sample_a, axis=0)
#
#     y_train_entropy = np.append(np.zeros(train_num_b), np.ones(train_num_a))
#     y_test_entropy = np.append(np.zeros(sample_num_b - train_num_b), np.ones(sample_num_a - train_num_a))
#
#     lr = LogisticRegression(solver= 'newton-cg', multi_class='multinomial', class_weight='balanced')
#     lr.fit(train_sample_entropy, y_train_entropy)
#     print(lr.score(test_sample_entropy, y_test_entropy))

# # combine the other and after together for entropy.
#     b_entropy = np.load("./before_entropy2.npy")
#     sample_num_b = b_entropy.shape[0]
#     train_num_b = int(0.8 * sample_num_b)
#     train_sample_b = b_entropy[:train_num_b, :]
#     test_sample_b = b_entropy[train_num_b:, :]
#
#
#     a_entropy = c.combinedata("./after_entropy2.npy", "./other_entropy2.npy", "./combine_entropy2.npy")
#     sample_num_a = a_entropy.shape[0]
#     train_num_a = int(0.8 * sample_num_a)
#     train_sample_a = a_entropy[:train_num_a, :]
#     test_sample_a = a_entropy[train_num_a:, :]
#
#     train_sample_entropy = np.append(train_sample_b, train_sample_a, axis=0)
#     test_sample_entropy = np.append(test_sample_b, test_sample_a, axis=0)
#
#     y_train_entropy = np.append(np.zeros(train_num_b), np.ones(train_num_a))
#     y_test_entropy = np.append(np.zeros(sample_num_b - train_num_b), np.ones(sample_num_a - train_num_a))
#
#     lr = LogisticRegression(solver= 'newton-cg', multi_class='multinomial', class_weight='balanced')
#     lr.fit(train_sample_entropy, y_train_entropy)
#     print("for the logistic regression model, the score for entropy is: ", lr.score(test_sample_entropy, y_test_entropy))
#
#     svc = svm.SVC(C=1, kernel='linear')
#     score = svc.fit(train_sample_entropy, y_train_entropy).score(test_sample_entropy, y_test_entropy)
#     print("for the svm model, the score for entropy is: ", score)
#
# # combine the other and after together for engery.
#     b_entropy = np.load("./before_energy2.npy")
#     sample_num_b = b_entropy.shape[0]
#     train_num_b = int(0.8 * sample_num_b)
#     train_sample_b = b_entropy[:train_num_b, :]
#     test_sample_b = b_entropy[train_num_b:, :]
#
#
#     a_entropy = c.combinedata("./after_energy2.npy", "./other_energy2.npy", "./combine_energy2.npy")
#     sample_num_a = a_entropy.shape[0]
#     train_num_a = int(0.8 * sample_num_a)
#     train_sample_a = a_entropy[:train_num_a, :]
#     test_sample_a = a_entropy[train_num_a:, :]
#
#     train_sample_entropy = np.append(train_sample_b, train_sample_a, axis=0)
#     test_sample_entropy = np.append(test_sample_b, test_sample_a, axis=0)
#
#     y_train_entropy = np.append(np.zeros(train_num_b), np.ones(train_num_a))
#     y_test_entropy = np.append(np.zeros(sample_num_b - train_num_b), np.ones(sample_num_a - train_num_a))
#
#     lr = LogisticRegression(solver= 'newton-cg', multi_class='multinomial', class_weight='balanced')
#     lr.fit(train_sample_entropy, y_train_entropy)
#     print("for the logistic regression model, the score for energy is: ", lr.score(test_sample_entropy, y_test_entropy))
#
#     svc = svm.SVC(C=1, kernel='linear')
#     score = svc.fit(train_sample_entropy, y_train_entropy).score(test_sample_entropy, y_test_entropy)
#     print("for the svm model, the score for energy is: ", score)
#
# # combine the other and after together for zcr.
#     b_entropy = np.load("./before_zcr2.npy")
#     sample_num_b = b_entropy.shape[0]
#     train_num_b = int(0.8 * sample_num_b)
#     train_sample_b = b_entropy[:train_num_b, :]
#     test_sample_b = b_entropy[train_num_b:, :]
#
#
#     a_entropy = c.combinedata("./after_zcr2.npy", "./other_zcr2.npy", "./combine_zcr2.npy")
#     sample_num_a = a_entropy.shape[0]
#     train_num_a = int(0.8 * sample_num_a)
#     train_sample_a = a_entropy[:train_num_a, :]
#     test_sample_a = a_entropy[train_num_a:, :]
#
#     train_sample_entropy = np.append(train_sample_b, train_sample_a, axis=0)
#     test_sample_entropy = np.append(test_sample_b, test_sample_a, axis=0)
#
#     y_train_entropy = np.append(np.zeros(train_num_b), np.ones(train_num_a))
#     y_test_entropy = np.append(np.zeros(sample_num_b - train_num_b), np.ones(sample_num_a - train_num_a))
#
#     lr = LogisticRegression(solver= 'newton-cg', multi_class='multinomial', class_weight='balanced')
#     lr.fit(train_sample_entropy, y_train_entropy)
#     print("for the logistic regression model, the core for zcr is: ", lr.score(test_sample_entropy, y_test_entropy))
#
#     svc = svm.SVC(C=1, kernel='linear')
#     score = svc.fit(train_sample_entropy, y_train_entropy).score(test_sample_entropy, y_test_entropy)
#     print("for the svm model, the score for zcr is: ", score)


# # combine the other and after together.
#     b_entropy = np.load("./before2_spectal_entropy.npy")
#     sample_num_b = b_entropy.shape[0]
#     train_num_b = int(0.8 * sample_num_b)
#     train_sample_b = b_entropy[:train_num_b, :]
#     test_sample_b = b_entropy[train_num_b:, :]
#
#
#     a_entropy = c.combinedata("./other2_spectal_entropy.npy", "./after2_spectal_entropy.npy", "./combine2_entropy2.npy")
#     sample_num_a = a_entropy.shape[0]
#     train_num_a = int(0.8 * sample_num_a)
#     train_sample_a = a_entropy[:train_num_a, :]
#     test_sample_a = a_entropy[train_num_a:, :]
#
#     train_sample_entropy = np.append(train_sample_b, train_sample_a, axis=0)
#     test_sample_entropy = np.append(test_sample_b, test_sample_a, axis=0)
#
#     y_train_entropy = np.append(np.zeros(train_num_b), np.ones(train_num_a))
#     y_test_entropy = np.append(np.zeros(sample_num_b - train_num_b), np.ones(sample_num_a - train_num_a))
#
#     lr = LogisticRegression(solver= 'newton-cg', multi_class='multinomial', class_weight='balanced')
#     lr.fit(train_sample_entropy, y_train_entropy)
#     print(lr.score(test_sample_entropy, y_test_entropy))


