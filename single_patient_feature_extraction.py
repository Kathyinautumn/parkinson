import os
import csv
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing


class CompareMedicine:
    HEALTHCODE = 'healthcode'
    MEDTIMEPOINT = 'medtimepoint'
    ZCR = "ZCR"
    SPECTAL_ENTROPY = "spectral_entropy"
    MFCC = "MFCC"
    ROLLOFF = "spectral_rolloff"
    ENERGY = "energy"
    FLUS = "spectral_flux"
    ENTROPY = "entropy"
    SPREAD = "spectral_spread"
    CENTROID = "spectral_centroid"
    CHROMA_VECTOR = "chroma_vector"
    CHROMA_DEVIATION = "chroma_deviation"
    FEATURES = "features"
    AUDIO = "audio"
    BEFORE_MED = "Immediately before Parkinson medication"
    AFTER_MED = "Just after Parkinson medication (at your best)"
    OTHER_MED = "Another time"
    NO_IDEA = "nan"
    NO_MED = 'I don\'t take Parkinson medications'
    data_path_before0 = '/Volumes/AwesomeBackup/for single patient/patient 3 male/before_med'
    data_path_after0 = '/Volumes/AwesomeBackup/for single patient/patient 3 male/after_med'
    data_path_before1 = '/Volumes/AwesomeBackup/for single patient/patient 5 male/before_med'
    data_path_after1 = '/Volumes/AwesomeBackup/for single patient/patient 5 male/after_med'
    data_path_before2 = '/Volumes/AwesomeBackup/for single patient/patient 4 female/before_med'
    data_path_after2 = '/Volumes/AwesomeBackup/for single patient/patient 4 female/after_med'
    data_path_before3 = '/Volumes/AwesomeBackup/for single patient/patient 6 female/before_med'
    data_path_after3 = '/Volumes/AwesomeBackup/for single patient/patient 6 female/after_med'

    def __init__(self):
        self.before = []
        self.after = []

    def divide_sample_by_medtimepoint(self):

        data_path_before = [self.data_path_before0, self.data_path_before1, self.data_path_before2, self.data_path_before3]
        list_before = ['./before_single_patient_male2', './before_single_patient_male3',
                       './before_single_patient_female2', './before_single_patient_female3']
        data_path_after = [self.data_path_after0, self.data_path_after1, self.data_path_after2, self.data_path_after3]
        list_after = ['./after_single_patient_male2', './after_single_patient_male3',
                       './after_single_patient_female2', './after_single_patient_female3']
        for i in range(4):
            path_to_json_before = data_path_before[i]
            json_files_before = [pos_json for pos_json in os.listdir(path_to_json_before) if pos_json.endswith('.json')]
            # normal_json_files = pd.io.json.json_normalize(json_files)
            for index, js in enumerate(json_files_before):
                with open(os.path.join(path_to_json_before, js)) as json_file:
                    json_text = json.load(json_file)
                    sample = {
                        self.HEALTHCODE: json_text[self.HEALTHCODE],
                        self.MEDTIMEPOINT: json_text[self.MEDTIMEPOINT],
                        self.ZCR: json_text[self.FEATURES][self.AUDIO][self.ZCR],
                        self.SPECTAL_ENTROPY: json_text[self.FEATURES][self.AUDIO][self.SPECTAL_ENTROPY],
                        self.MFCC: json_text[self.FEATURES][self.AUDIO][self.MFCC],
                        self.ROLLOFF: json_text[self.FEATURES][self.AUDIO][self.ROLLOFF],
                        self.ENERGY: json_text[self.FEATURES][self.AUDIO][self.ENERGY],
                        self.FLUS: json_text[self.FEATURES][self.AUDIO][self.FLUS],
                        self.ENTROPY: json_text[self.FEATURES][self.AUDIO][self.ENTROPY],
                        self.SPREAD: json_text[self.FEATURES][self.AUDIO][self.SPREAD],
                        self.CENTROID: json_text[self.FEATURES][self.AUDIO][self.CENTROID],
                        self.CHROMA_VECTOR: json_text[self.FEATURES][self.AUDIO][self.CHROMA_VECTOR],
                        self.CHROMA_DEVIATION: json_text[self.FEATURES][self.AUDIO][self.CHROMA_DEVIATION]
                    }
                    self.before.append(sample)

            with open("%s.csv" % list_before[i], 'w') as before_file:
                filed_name = [self.HEALTHCODE, self.MEDTIMEPOINT, self.ZCR, self.SPECTAL_ENTROPY,
                          self.MFCC, self.ROLLOFF, self.ENERGY, self.FLUS, self.ENTROPY,
                          self.SPREAD, self.CENTROID, self.CHROMA_VECTOR, self.CHROMA_DEVIATION]
                writer = csv.DictWriter(before_file, fieldnames=filed_name)
                writer.writeheader()
                for sample in self.before:
                    writer.writerow(sample)
            before_file.close()

        for i in range(4):
            path_to_json_after = data_path_after[i]
            json_files_after = [pos_json for pos_json in os.listdir(path_to_json_after) if
                                 pos_json.endswith('.json')]
            # normal_json_files = pd.io.json.json_normalize(json_files)
            for index, js in enumerate(json_files_after):
                with open(os.path.join(path_to_json_after, js)) as json_file:
                    json_text = json.load(json_file)
                    sample = {
                        self.HEALTHCODE: json_text[self.HEALTHCODE],
                        self.MEDTIMEPOINT: json_text[self.MEDTIMEPOINT],
                        self.ZCR: json_text[self.FEATURES][self.AUDIO][self.ZCR],
                        self.SPECTAL_ENTROPY: json_text[self.FEATURES][self.AUDIO][self.SPECTAL_ENTROPY],
                        self.MFCC: json_text[self.FEATURES][self.AUDIO][self.MFCC],
                        self.ROLLOFF: json_text[self.FEATURES][self.AUDIO][self.ROLLOFF],
                        self.ENERGY: json_text[self.FEATURES][self.AUDIO][self.ENERGY],
                        self.FLUS: json_text[self.FEATURES][self.AUDIO][self.FLUS],
                        self.ENTROPY: json_text[self.FEATURES][self.AUDIO][self.ENTROPY],
                        self.SPREAD: json_text[self.FEATURES][self.AUDIO][self.SPREAD],
                        self.CENTROID: json_text[self.FEATURES][self.AUDIO][self.CENTROID],
                        self.CHROMA_VECTOR: json_text[self.FEATURES][self.AUDIO][self.CHROMA_VECTOR],
                        self.CHROMA_DEVIATION: json_text[self.FEATURES][self.AUDIO][self.CHROMA_DEVIATION]
                    }
                    self.after.append(sample)

            with open("%s.csv" % list_after[i], 'w') as after_file:
                filed_name = [self.HEALTHCODE, self.MEDTIMEPOINT, self.ZCR, self.SPECTAL_ENTROPY,
                          self.MFCC, self.ROLLOFF, self.ENERGY, self.FLUS, self.ENTROPY,
                          self.SPREAD, self.CENTROID, self.CHROMA_VECTOR, self.CHROMA_DEVIATION]
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
            # sample = [float(_) for _ in sample]
            # map(float, sample)
            output_data.append(sample)
        output_data = np.array(output_data)
        return output_data

    def split_data_before(self):
        before = pd.read_csv('./before_single_patient_male2.csv')
        flus = before[before.columns[7]]
        flus = np.array([_ for _ in flus])
        before_male_flus = self.convert_data(flus)
        np.save('./before_single_patient_male2_flus', before_male_flus)

        before = pd.read_csv('./before_single_patient_male3.csv')
        flus = before[before.columns[7]]
        flus = np.array([_ for _ in flus])
        before_male_flus = self.convert_data(flus)
        np.save('./before_single_patient_male3_flus', before_male_flus)

        before = pd.read_csv('./before_single_patient_female2.csv')
        flus = before[before.columns[7]]
        flus = np.array([_ for _ in flus])
        before_male_flus = self.convert_data(flus)
        np.save('./before_single_patient_female2_flus', before_male_flus)

        before = pd.read_csv('./before_single_patient_female3.csv')
        flus = before[before.columns[7]]
        flus = np.array([_ for _ in flus])
        before_male_flus = self.convert_data(flus)
        np.save('./before_single_patient_female3_flus', before_male_flus)

    def split_data_after(self):
        after = pd.read_csv('./after_single_patient_male2.csv')
        flus = after[after.columns[7]]
        flus = np.array([_ for _ in flus])
        after_male_flus = self.convert_data(flus)
        np.save('./after_single_patient_male2_flus', after_male_flus)

        after = pd.read_csv('./after_single_patient_male3.csv')
        flus = after[after.columns[7]]
        flus = np.array([_ for _ in flus])
        after_male_flus = self.convert_data(flus)
        np.save('./after_single_patient_male3_flus', after_male_flus)

        after = pd.read_csv('./after_single_patient_female2.csv')
        flus = after[after.columns[7]]
        flus = np.array([_ for _ in flus])
        after_male_flus = self.convert_data(flus)
        np.save('./after_single_patient_female2_flus', after_male_flus)

        after = pd.read_csv('./after_single_patient_female3.csv')
        flus = after[after.columns[7]]
        flus = np.array([_ for _ in flus])
        after_male_flus = self.convert_data(flus)
        np.save('./after_single_patient_female3_flus', after_male_flus)

    def truncation(self, filename, output_name):
        threshold = 400
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

    def com_norm_sep2(self, filename1, filename2, output1, output2):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        new_data = np.append(data1, data2, axis=0)
        norm_data = preprocessing.normalize(new_data, norm='l1')
        new_data1 = norm_data[:data1.shape[0], :]
        new_data2 = norm_data[data1.shape[0]:, :]
        np.save(output1, new_data1)
        np.save(output2, new_data2)
        return new_data1, new_data2

    def com_norm_sep3(self, filename1, filename2, filename3, output1, output2, output3):
        data1 = np.load(filename1)
        data2 = np.load(filename2)
        data3 = np.load(filename3)
        new_data_begin = np.append(data1, data2, axis=0)
        new_data = np.append(new_data_begin, data3, axis=0)
        norm_data = preprocessing.normalize(new_data, norm='l1')
        new_data1 = norm_data[:data1.shape[0], :]
        new_data2 = norm_data[data1.shape[0]:data1.shape[0]+data2.shape[0], :]
        new_data3 = norm_data[data1.shape[0]+data2.shape[0]:, :]
        np.save(output1, new_data1)
        np.save(output2, new_data2)
        np.save(output3, new_data3)
        return new_data1, new_data2, new_data3


if __name__ == '__main__':

    c = CompareMedicine()
    c.divide_sample_by_medtimepoint()
    c.split_data_before()
    c.split_data_after()

    # c.truncation('./before_male_spectal_flus.npy', './before_cutted_spectal_flus.npy')
    # c.truncation('./before_male_rolloff.npy', './before_cutted_rolloff.npy')
    # c.truncation('./before_male_flus.npy', './before_cutted_flus.npy')
    # c.truncation('./before_male_flus.npy', './before_male_flus.npy')
    # c.truncation('./other2_flus.npy', './other2_flus.npy')


    #
    # c.truncation('./after_male_spectal_flus.npy', './after_cutted_spectal_flus.npy')
    # c.truncation('./after_male_rolloff.npy', './after_cutted_rolloff.npy')
    # c.truncation('./after_male_flus.npy', './after_cutted_flus.npy')
    # #
    # c.truncation('./other2_spectal_flus.npy', './other2_spectal_flus.npy')
    # c.truncation('./other2_rolloff.npy', './other2_rolloff.npy')
    # c.truncation('./other2_flus.npy', './other2_flus.npy')
    #
    # c.com_norm_sep2("./before_male_spectal_flus.npy", "./after_male_spectal_flus.npy",
    #                 "./before_male_spectal_flus1.npy", "./after_male_spectal_flus1.npy")
    # c.com_norm_sep2("./before_male_rolloff.npy", "./after_male_rolloff.npy",
    #                 "./before_male_rolloff1.npy", "./after_male_rolloff1.npy")
    # c.com_norm_sep2("./before_male_flus.npy", "./after_male_flus.npy",
    #                 "./before_male_flus1.npy", "./after_male_flus1.npy")
    #
    # c.com_norm_sep3('./before_male_flus.npy', './after_male_flus.npy', './other2_flus.npy',
    #                 './before_male_flus2.npy', './after_male_flus2.npy', './other2_flus2.npy')
    # c.com_norm_sep3('./before_male_rolloff.npy', './after_male_rolloff.npy', './other2_rolloff.npy',
    #                 './before_male_rolloff2.npy', './after_male_rolloff2.npy', './other2_rolloff2.npy')
    # c.com_norm_sep3('./before_male_spectal_flus.npy', './after_male_spectal_flus.npy', './other2_spectal_flus.npy',
    #                 './before_male_spectal_flus2.npy', './after_male_spectal_flus2.npy', './other2_spectal_flus2.npy')
    #
    #
    #





