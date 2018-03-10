import os
import csv
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
    data_path = '/Volumes/AwesomeBackup/Parkinsons_mPower_Voice_Features'
    test_path = '/Volumes/AwesomeBackup/test_data'

    def __init__(self):
        self.before = []
        self.after = []
        self.other = []
        # self.noidea = []

    def divide_sample_by_medtimepoint(self):
        '''
        MFCC = "MFCC"
        ROLLOFF = "spectral_rolloff"
        ENERGY = "energy"
        FLUS = "spectral_flux"
        ENTROPY = "entropy"
        SPREAD = "spectral_spread"
        CENTROID = "spectral_centroid"
        CHROMA_VECTOR = "chroma_vector"
        CHROMA_DEVIATION = "chroma_deviation"
        :return:
        '''
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
                    self.SPECTAL_ENTROPY: json_text[self.FEATURES][self.AUDIO][self.SPECTAL_ENTROPY],
                    self.MFCC: json_text[self.FEATURES][self.AUDIO][self.MFCC],
                    self.ROLLOFF: json_text[self.FEATURES][self.AUDIO][self.ROLLOFF],
                    self.ENERGY: json_text[self.FEATURES][self.AUDIO][self.ENERGY],
                    self.FLUS: json_text[self.FEATURES][self.AUDIO][self.FLUS],
                    self.ENTROPY: json_text[self.FEATURES][self.AUDIO][self.ENTROPY],
                    self.SPREAD: json_text[self.FEATURES][self.AUDIO][self.SPREAD],
                    self.CENTROID : json_text[self.FEATURES][self.AUDIO][self.CENTROID],
                    self.CHROMA_VECTOR: json_text[self.FEATURES][self.AUDIO][self.CHROMA_VECTOR],
                    self.CHROMA_DEVIATION: json_text[self.FEATURES][self.AUDIO][self.CHROMA_DEVIATION]
                }

                if sample[self.MEDTIMEPOINT] == self.BEFORE_MED:
                    self.before.append(sample)
                if sample[self.MEDTIMEPOINT] == self.AFTER_MED:
                    self.after.append(sample)
                if sample[self.MEDTIMEPOINT] == self.OTHER_MED:
                    self.other.append(sample)

        with open("./before2.csv", 'w') as before_file:
            filed_name = [self.HEALTHCODE, self.MEDTIMEPOINT, self.ZCR, self.SPECTAL_ENTROPY,
                          self.MFCC, self.ROLLOFF, self.ENERGY, self.FLUS, self.ENTROPY,
                          self.SPREAD, self.CENTROID, self.CHROMA_VECTOR, self.CHROMA_DEVIATION]
            writer = csv.DictWriter(before_file, fieldnames=filed_name)
            writer.writeheader()
            for sample in self.before:
                writer.writerow(sample)
        before_file.close()

        with open("./after2.csv", 'w') as after_file:
            filed_name = [self.HEALTHCODE, self.MEDTIMEPOINT, self.ZCR, self.SPECTAL_ENTROPY,
                          self.MFCC, self.ROLLOFF, self.ENERGY, self.FLUS, self.ENTROPY,
                          self.SPREAD, self.CENTROID, self.CHROMA_VECTOR, self.CHROMA_DEVIATION]
            writer = csv.DictWriter(after_file, fieldnames=filed_name)
            writer.writeheader()
            for sample in self.after:
                writer.writerow(sample)
        after_file.close()

        with open("./other2.csv", 'w') as other_file:
            filed_name = [self.HEALTHCODE, self.MEDTIMEPOINT, self.ZCR, self.SPECTAL_ENTROPY,
                          self.MFCC, self.ROLLOFF, self.ENERGY, self.FLUS, self.ENTROPY,
                          self.SPREAD, self.CENTROID, self.CHROMA_VECTOR, self.CHROMA_DEVIATION]
            writer = csv.DictWriter(other_file, fieldnames=filed_name)
            writer.writeheader()
            for sample in self.other:
                writer.writerow(sample)
        other_file.close()

    def convert_data(self, input_data):
        output_data = []
        for sample in input_data:
            sample = sample[1:-1]
            sample = sample.split(", ")
            sample = [float(_) for _ in sample]
            # map(float, sample)
            output_data.append(sample)
        output_data = np.array(output_data)
        return output_data

    def split_data_before(self):
        before = pd.read_csv('./before2.csv')
        # zcr = before[before.columns[2]]
        # zcr = np.array([_ for _ in zcr])
        # before2_zcr = self.convert_data(zcr)
        # np.save('./before2_zcr', before2_zcr)

        spectal_entropy = before[before.columns[3]]
        spectal_entropy = np.array([_ for _ in spectal_entropy])
        before2_spectal_entropy = self.convert_data(spectal_entropy)
        np.save('./before2_spectal_entropy', before2_spectal_entropy)

        # mfcc = before[before.columns[4]]
        # mfcc = np.array([_ for _ in mfcc])
        # before2_mfcc = self.convert_data(mfcc)
        # np.save('./before2_mfcc', before2_mfcc)

        rolloff = before[before.columns[5]]
        rolloff = np.array([_ for _ in rolloff])
        before2_rolloff = self.convert_data(rolloff)
        np.save('./before2_rolloff', before2_rolloff)

        # energy = before[before.columns[6]]
        # energy = np.array([_ for _ in energy])
        # before2_energy = self.convert_data(energy)
        # np.save('./before2_energy', before2_energy)

        # flus = before[before.columns[7]]
        # flus = np.array([_ for _ in flus])
        # before2_flus = self.convert_data(flus)
        # np.save('./before2_flus', before2_flus)

        entropy = before[before.columns[8]]
        entropy = np.array([_ for _ in entropy])
        before2_entropy = self.convert_data(entropy)
        np.save('./before2_entropy', before2_entropy)
        #
        # spread = before[before.columns[9]]
        # spread = np.array([_ for _ in spread])
        # before2_spread = self.convert_data(spread)
        # np.save('./before2_spread', before2_spread)
        #
        # centroid = before[before.columns[10]]
        # centroid = np.array([_ for _ in centroid])
        # before2_centroid = self.convert_data(centroid)
        # np.save('./before2_centroid', before2_centroid)
        #
        # chroma_vector = before[before.columns[11]]
        # chroma_vector = np.array([_ for _ in chroma_vector])
        # before2_chroma_vector = self.convert_data(chroma_vector)
        # np.save('./before2_chroma_vector', before2_chroma_vector)
        #
        # chroma_deviation = before[before.columns[12]]
        # chroma_deviation = np.array([_ for _ in chroma_deviation])
        # before2_chroma_deviation = self.convert_data(chroma_deviation)
        # np.save('./before2_chroma_deviation', before2_chroma_deviation)


    def split_data_after(self):
        after = pd.read_csv('./after2.csv')

        # zcr = after[after.columns[2]]
        # zcr = np.array([_ for _ in zcr])
        # after2_zcr = self.convert_data(zcr)
        # np.save('./after2_zcr', after2_zcr)

        spectal_entropy = after[after.columns[3]]
        spectal_entropy = np.array([_ for _ in spectal_entropy])
        after2_spectal_entropy = self.convert_data(spectal_entropy)
        np.save('./after2_spectal_entropy', after2_spectal_entropy)

        # mfcc = after[after.columns[4]]
        # mfcc = np.array([_ for _ in mfcc])
        # after2_mfcc = self.convert_data(mfcc)
        # np.save('./after2_mfcc', after2_mfcc)

        rolloff = after[after.columns[5]]
        rolloff = np.array([_ for _ in rolloff])
        after2_rolloff = self.convert_data(rolloff)
        np.save('./after2_rolloff', after2_rolloff)

        # energy = after[after.columns[6]]
        # energy = np.array([_ for _ in energy])
        # after2_energy = self.convert_data(energy)
        # np.save('./after2_energy', after2_energy)
        #
        # flus = after[after.columns[7]]
        # flus = np.array([_ for _ in flus])
        # after2_flus = self.convert_data(flus)
        # np.save('./after2_flus', after2_flus)

        entropy = after[after.columns[8]]
        entropy = np.array([_ for _ in entropy])
        after2_entropy = self.convert_data(entropy)
        np.save('./after2_entropy', after2_entropy)

        # spread = after[after.columns[9]]
        # spread = np.array([_ for _ in spread])
        # after2_spread = self.convert_data(spread)
        # np.save('./after2_spread', after2_spread)
        #
        # centroid = after[after.columns[10]]
        # centroid = np.array([_ for _ in centroid])
        # after2_centroid = self.convert_data(centroid)
        # np.save('./after2_centroid', after2_centroid)
        #
        # chroma_vector = after[after.columns[11]]
        # chroma_vector = np.array([_ for _ in chroma_vector])
        # after2_chroma_vector = self.convert_data(chroma_vector)
        # np.save('./after2_chroma_vector', after2_chroma_vector)
        #
        # chroma_deviation = after[after.columns[12]]
        # chroma_deviation = np.array([_ for _ in chroma_deviation])
        # after2_chroma_deviation = self.convert_data(chroma_deviation)
        # np.save('./after2_chroma_deviation', after2_chroma_deviation)

    def split_data_other(self):
        other = pd.read_csv('./other2.csv')

        # zcr = other[other.columns[2]]
        # zcr = np.array([_ for _ in zcr])
        # other2_zcr = self.convert_data(zcr)
        # np.save('./other2_zcr', other2_zcr)

        spectal_entropy = other[other.columns[3]]
        spectal_entropy = np.array([_ for _ in spectal_entropy])
        other2_spectal_entropy = self.convert_data(spectal_entropy)
        np.save('./other2_spectal_entropy', other2_spectal_entropy)

        # mfcc = other[other.columns[4]]
        # mfcc = np.array([_ for _ in mfcc])
        # other2_mfcc = self.convert_data(mfcc)
        # np.save('./other2_mfcc', other2_mfcc)

        rolloff = other[other.columns[5]]
        rolloff = np.array([_ for _ in rolloff])
        other2_rolloff = self.convert_data(rolloff)
        np.save('./other2_rolloff', other2_rolloff)
        #
        # energy = other[other.columns[6]]
        # energy = np.array([_ for _ in energy])
        # other2_energy = self.convert_data(energy)
        # np.save('./other2_energy', other2_energy)
        #
        # flus = other[other.columns[7]]
        # flus = np.array([_ for _ in flus])
        # other2_flus = self.convert_data(flus)
        # np.save('./other2_flus', other2_flus)

        entropy = other[other.columns[8]]
        entropy = np.array([_ for _ in entropy])
        other2_entropy = self.convert_data(entropy)
        np.save('./other2_entropy', other2_entropy)

        # spread = other[other.columns[9]]
        # spread = np.array([_ for _ in spread])
        # other2_spread = self.convert_data(spread)
        # np.save('./other2_spread', other2_spread)
        #
        # centroid = other[other.columns[10]]
        # centroid = np.array([_ for _ in centroid])
        # other2_centroid = self.convert_data(centroid)
        # np.save('./other2_centroid', other2_centroid)
        #
        # chroma_vector = other[other.columns[11]]
        # chroma_vector = np.array([_ for _ in chroma_vector])
        # other2_chroma_vector = self.convert_data(chroma_vector)
        # np.save('./other2_chroma_vector', other2_chroma_vector)
        #
        # chroma_deviation = other[other.columns[12]]
        # chroma_deviation = np.array([_ for _ in chroma_deviation])
        # other2_chroma_deviation = self.convert_data(chroma_deviation)
        # np.save('./other2_chroma_deviation', other2_chroma_deviation)

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

class Preprocess(object):
    def reduce_dim_pca(self):
        labels = np.load("./labels.npy")
        gene = np.load('./gene.npy')
        two_d_pca = PCA(n_components=100)
        two_d_pca.fit(gene)
        two_d_gene = two_d_pca.fit_transform(gene)
        pc1 = two_d_gene[:,0]
        pc2 = two_d_gene[:,1]
        pc3 = two_d_gene[:,2]
        plt.scatter(pc1, pc2, c=labels)
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc1, pc2, pc3, c=labels, marker='o')
        ax.set_xlabel('pc1 Label')
        ax.set_ylabel('pc2 Label')
        ax.set_zlabel('pc3 Label')
        plt.show()
        fig.savefig('pca_3d.pdf')


if __name__ == '__main__':

    c = CompareMedicine()
    # c.divide_sample_by_medtimepoint()
    # c.split_data_before()
    # c.split_data_after()
    # c.split_data_other()

    # c.truncation('./before2_zcr.npy', './before2_zcr.npy')
    # c.truncation('./before2_spectal_entropy.npy', './before2_spectal_entropy.npy')
    # c.truncation('./before2_rolloff.npy', './before2_rolloff.npy')
    # c.truncation('./before2_flus.npy', './before2_flus.npy')
    # c.truncation('./before2_spread.npy', './before2_spread.npy')
    # c.truncation('./before2_centroid.npy', './before2_centroid.npy')
    # c.truncation('./before2_chroma_deviation.npy', './before2_chroma_deviation.npy')
    # c.truncation('./before2_energy.npy', './before2_energy.npy')
    # c.truncation('./before2_entropy.npy', './before2_entropy.npy')
    #
    # c.truncation('./after2_zcr.npy', './after2_zcr.npy')
    # c.truncation('./after2_spectal_entropy.npy', './after2_spectal_entropy.npy')
    # c.truncation('./after2_rolloff.npy', './after2_rolloff.npy')
    # c.truncation('./after2_flus.npy', './after2_flus.npy')
    # c.truncation('./after2_spread.npy', './after2_spread.npy')
    # c.truncation('./after2_centroid.npy', './after2_centroid.npy')
    # c.truncation('./after2_chroma_deviation.npy', './after2_chroma_deviation.npy')
    # c.truncation('./after2_energy.npy', './after2_energy.npy')
    # c.truncation('./after2_entropy.npy', './after2_entropy.npy')

    # c.truncation('./other2_spectal_entropy.npy', './other2_spectal_entropy.npy')
    # c.truncation('./other2_rolloff.npy', './other2_rolloff.npy')
    # c.truncation('./other2_entropy.npy', './other2_entropy.npy')





