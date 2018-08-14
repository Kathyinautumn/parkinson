import os, json
import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, MultiTaskLasso, Lasso, LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn import cross_validation
from sklearn import preprocessing, model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model, svm, tree
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


from sklearn.ensemble import VotingClassifier as VC
from mlxtend.classifier import StackingClassifier

import matplotlib.pyplot as plt


class CompareMedicine:

    def normalization(self, filename, output_name):
        data = np.load(filename)
        new_data = preprocessing.normalize(data, norm='l1')
        np.save(output_name, new_data)
        return new_data

    def combinedatc3(self, filename1, filename2, filenamc3, output_name):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        datc3 = np.load(filenamc3)
        new_data = np.concatenate((data1, data2, datc3), axis=0)
        np.save(output_name, new_data)
        return new_data

    def separatedatc3(self, filename, size1, size2, output1, output2, outpuc3):
        data = np.load(filename)
        new_data1 = data[:size1, :]
        new_data2 = data[size1:size1+size2, :]
        new_datc3 = data[size1+size2:, :]
        np.save(output1, new_data1)
        np.save(output2, new_data2)
        np.save(outpuc3, new_datc3)
        return new_data1, new_data2, new_datc3

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

    def com_norm_sep2(self, filename1, filename2, output1, output2):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        new_data = np.concatenate((data1, data2), axis=0)
        norm_data = preprocessing.normalize(new_data, norm='l2')
        new_data1 = norm_data[:data1.shape[0], :]
        new_data2 = norm_data[data1.shape[0]:, :]
        np.save(output1, new_data1)
        np.save(output2, new_data2)
        return new_data1, new_data2

    def truncation(self, filename):
        threshold = 300
        data = np.load(filename)
        data = np.array([np.array(_) for _ in data])
        new_data = []
        for sample in data:
            if sample.shape[0] >= threshold:
                sample = sample[7:threshold]
                new_data.append(sample)
            else:
                continue
        new_data = np.asarray(new_data)
        new_data = new_data.astype(float)
        mean_total = np.mean(new_data, axis=1)
        mean_total = mean_total.reshape(-1, 1)
        return mean_total

    def reject_outliers(self, data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    def cross_validatinon_model_for_sample_number_selection(self, file1, file2, model):
        data1 = self.truncation(file1)
        data2 = self.truncation(file2)

        b = []
        for i in range(100):
            data1 = data1[np.random.choice(data1.shape[0], 11, replace=False), :]
            number1 = data1.shape[0]
            label1 = np.zeros(number1)
            data2 = data2[np.random.choice(data2.shape[0], 11, replace=False), :]
            number2 = data2.shape[0]
            label2 = np.ones(number2)
            data = np.append(data1, data2, axis=0)
            label = np.append(label1, label2, axis=0)
            X_train, X_test, y_train, y_test = train_test_split(data*100, label, test_size=0.1, random_state=42)

            clf = model
            scores = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(5))
            b = sorted(sorted(b) + sorted(scores))
        print("For the %s of %s, the Accuracy: %0.2f" % (model, file1, np.mean(b)))

        # data1 = self.truncation(file1)
        # number1 = data1.shape[0]
        # label1 = np.zeros(number1)
        # data2 = self.truncation(file2)
        # number2 = data2.shape[0]
        # label2 = np.ones(number2)
        # new_data = np.concatenate((data1, data2), axis=0)
        # label = np.append(label1, label2, axis=0)
        # X_train, X_test, y_train, y_test = train_test_split(new_data, label, test_size=0.1, random_state=42)
        #
        # clf = model
        # scores = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(10))
        # # y_train_predict = cross_val_predict(clf, X_train, y_train, cv=StratifiedKFold(5))
        # # clf.fix(X_train, y_train)
        # # y_pred = clf.predict(X_test)
        # print("For the %s of %s, the Accuracy: %0.2f (+/- %0.2f)" % (model, file1, scores.mean(), scores.std() * 2))

    def cross_validatinon_model(self, file1, file2):
        data1 = self.truncation(file1)
        number1 = data1.shape[0]
        label1 = np.zeros(number1)
        data2 = self.truncation(file2)
        number2 = data2.shape[0]
        label2 = np.ones(number2)
        new_data = np.concatenate((data1, data2), axis=0)
        norm_data = preprocessing.StandardScaler().fit_transform(new_data)
        data = norm_data
        label = np.append(label1, label2, axis=0)

        SVM = svm.NuSVC(kernel='sigmoid', gamma=.2)
        LR = LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced')
        RF = RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini")
        Tree = tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
        MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        Gaussian = GaussianNB()
        model = [SVM, LR, RF, Tree, MLP, Gaussian]

        for i in range(6):
            clf = model[i]
            X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=1)
            scores = cross_val_score(clf, X_train, y_train, cv=7, scoring='accuracy')

                # y_train_predict = cross_val_predict(clf, X_train, y_train, cv=StratifiedKFold(5))
                # clf.fix(X_train, y_train)
                # y_pred = clf.predict(X_test)
            print("the Accuracy: %0.2f" % np.mean(scores))
            print('-------------------')
                # print(recall_score(y_train, y_train_predict))
        print("=================")


if __name__ == '__main__':
    c = CompareMedicine()
    #    #
    #
    # c.cross_validatinon_model("./before_single_patient_male_entropy.npy", "./after_single_patient_male_entropy.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_rolloff.npy", "./after_single_patient_male_rolloff.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_spectal_entropy.npy",
    #                           "./after_single_patient_male_spectal_entropy.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    #
    # c.cross_validatinon_model("./before_single_patient_male_flus.npy", "./after_single_patient_male_flus.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_mfcc1.npy", "./after_single_patient_male_mfcc1.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_mfcc2.npy", "./after_single_patient_male_mfcc2.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_mfcc3.npy", "./after_single_patient_male_mfcc3.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_mfcc4.npy", "./after_single_patient_male_mfcc4.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_mfcc5.npy", "./after_single_patient_male_mfcc5.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_mfcc6.npy", "./after_single_patient_male_mfcc6.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_mfcc12.npy", "./after_single_patient_male_mfcc12.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_chroma_vector2.npy",
    #                           "./after_single_patient_male_chroma_vector2.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial',
    #                                              class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_chroma_vector3.npy",
    #                           "./after_single_patient_male_chroma_vector3.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial',
    #                                              class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_chroma_vector4.npy",
    #                           "./after_single_patient_male_chroma_vector4.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial',
    #                                              class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_male_chroma_vector6.npy",
    #                           "./after_single_patient_male_chroma_vector6.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial',
    #                                              class_weight='balanced'))
    #
    # c.cross_validatinon_model("./before_single_patient_female_entropy.npy", "./after_single_patient_female_entropy.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_rolloff.npy", "./after_single_patient_female_rolloff.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # # c.cross_validatinon_model("./before_single_patient_female_spectal_entropy.npy",
    # #                           "./after_single_patient_female_spectal_entropy.npy", RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    #
    # c.cross_validatinon_model("./before_single_patient_female_flus.npy", "./after_single_patient_female_flus.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # # c.cross_validatinon_model("./before_single_patient_female_mfcc1.npy", "./after_single_patient_female_mfcc1.npy",
    # #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_mfcc2.npy", "./after_single_patient_female_mfcc2.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_mfcc3.npy", "./after_single_patient_female_mfcc3.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_mfcc4.npy", "./after_single_patient_female_mfcc4.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_mfcc5.npy", "./after_single_patient_female_mfcc5.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_mfcc6.npy", "./after_single_patient_female_mfcc6.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_mfcc12.npy", "./after_single_patient_female_mfcc12.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))

    # c.cross_validatinon_model("./before_single_patient_female_chroma_vector2.npy",
    #                           "./after_single_patient_female_chroma_vector2.npy", RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_chroma_vector3.npy",
    #                           "./after_single_patient_female_chroma_vector3.npy", RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_chroma_vector4.npy",
    #                           "./after_single_patient_female_chroma_vector4.npy", RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_chroma_vector6.npy",
    #                           "./after_single_patient_female_chroma_vector6.npy", RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))

    # c.cross_validatinon_model("./before_single_patient_male_flus.npy", "./after_single_patient_male_flus.npy",
    #                           svm.SVC(C=1, gamma=1))
    # c.cross_validatinon_model("./before_single_patient_female_flus.npy", "./after_single_patient_female_flus.npy",
    #                           svm.SVC(C=1, gamma=1))
    #
    # c.cross_validatinon_model("./before_single_patient_male_flus.npy", "./after_single_patient_male_flus.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    # c.cross_validatinon_model("./before_single_patient_female_flus.npy", "./after_single_patient_female_flus.npy",
    #                           RandomForestClassifier(n_estimators=100, n_jobs=1, criterion="gini"))
    #
    # c.cross_validatinon_model("./before_single_patient_male_flus.npy", "./after_single_patient_male_flus.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    # c.cross_validatinon_model("./before_single_patient_female_flus.npy", "./after_single_patient_female_flus.npy",
    #                           LogisticRegression(solver='newton-cg', multi_class='multinomial', class_weight='balanced'))
    #
    # c.cross_validatinon_model("./before_single_patient_male_flus.npy", "./after_single_patient_male_flus.npy",
    #                           tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1))
    # c.cross_validatinon_model("./before_single_patient_female_flus.npy", "./after_single_patient_female_flus.npy",
    #                           tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1))
    #
    # c.cross_validatinon_model("./before_single_patient_male_flus.npy", "./after_single_patient_male_flus.npy",
    #                           GaussianNB())
    # c.cross_validatinon_model("./before_single_patient_female_flus.npy", "./after_single_patient_female_flus.npy",
    #                           GaussianNB())
    #
    # c.cross_validatinon_model("./before_single_patient_male_flus.npy", "./after_single_patient_male_flus.npy", MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                                                                                      hidden_layer_sizes=(15,), random_state=1))
    # c.cross_validatinon_model("./before_single_patient_female_flus.npy", "./after_single_patient_female_flus.npy", MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                                                                                      hidden_layer_sizes=(15,), random_state=1))

    c.cross_validatinon_model("./before_single_patient_male_flus.npy", "./after_single_patient_male_flus.npy")
    c.cross_validatinon_model("./before_single_patient_male2_flus.npy", "./after_single_patient_male2_flus.npy")
    c.cross_validatinon_model("./before_single_patient_male3_flus.npy", "./after_single_patient_male3_flus.npy")
    c.cross_validatinon_model("./before_single_patient_female_flus.npy", "./after_single_patient_female_flus.npy")
    c.cross_validatinon_model("./before_single_patient_female2_flus.npy", "./after_single_patient_female2_flus.npy")
    c.cross_validatinon_model("./before_single_patient_female3_flus.npy", "./after_single_patient_female3_flus.npy")