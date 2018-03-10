import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class MeasureWeight(object):

    def generate_sample(self, filename):
        sample_size = 100
        data = np.load(filename)
        matrix = data[np.random.randint(data.shape[0], size=sample_size),:]
        matrix = matrix.astype(float)
        mylist = np.var(matrix, axis=1)
        # for i in range(100):
        #     mylist[i] = matrix.var(axis=0)
        return mylist

    def label(self):
        a = np.full((100,),1)
        a = np.append(a, np.full((100,),2))
        a = np.append(a, np.full((100,),3))
        a = np.append(a, np.full((100,),4))
        a = np.append(a, np.full((100,),5))
        a = np.append(a, np.full((100,),6))
        a = np.append(a, np.full((100,),7))
        a = np.append(a, np.full((100,),8))
        a = np.append(a, np.full((100,),9))
        return a


class Preprocess(object):
    def reduce_dim_pca(self, array, label):
        # cmap = plt.get_cmap('RdBu', 18)
        labels = label
        gene = array
        two_d_pca = PCA(n_components=100)
        two_d_pca.fit(gene)
        two_d_gene = two_d_pca.fit_transform(gene)
        pc1 = two_d_gene[:,0]
        pc2 = two_d_gene[:,1]
        pc3 = two_d_gene[:,2]
        plt.scatter(pc1, pc2, c=labels)
        cbar = plt.colorbar(ticks=np.arange(18))
        cbar.set_label('leukemia types')
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.show()

        fig = plt.figure()
        cmap = plt.get_cmap('RdBu', 18)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc1, pc2, pc3, c=labels, marker='o', cmap = cmap)
        m = cm.ScalarMappable(cmap=plt.get_cmap('RdBu', 18))
        a = np.arange(18)
        m.set_array(a)
        cbar = plt.colorbar(m)
        cbar.set_label('leukemia types')
        ax.set_xlabel('pc1 Label')
        ax.set_ylabel('pc2 Label')
        ax.set_zlabel('pc3 Label')
        plt.show()
        fig.savefig('pca_3d.pdf')

if __name__ == '__main__':

    m = MeasureWeight()
    p = Preprocess()
    filenames = ['./after2_zcr.npy', './after2_spectal_entropy.npy', './after2_rolloff.npy', './after2_flus.npy',
                 './after2_spread.npy', './after2_centroid.npy', './after2_chroma_deviation.npy', './after2_energy.npy',
                 './after2_entropy.npy']
    filenames2 = ['./before2_zcr.npy', './before2_spectal_entropy.npy', './before2_rolloff.npy', './before2_flus.npy',
                 './before2_spread.npy', './before2_centroid.npy', './before2_chroma_deviation.npy',
                 './before2_energy.npy',
                 './before2_entropy.npy']

    all_val1 = []
    all_val2 = []
    for file in filenames:
        cur_var = m.generate_sample(file)
        all_val1.append(np.mean(cur_var))
    for file in filenames2:
        cur_var = m.generate_sample(file)
        all_val2.append(np.mean(cur_var))

    all_val1 = np.array(all_val1)
    log_val1 = 0 - np.log(all_val1)

    all_val2 = np.array(all_val2)
    log_val2 = 0 - np.log(all_val2)
    # print(log_val)
    p1 = plt.bar(np.arange(len(filenames)), log_val1, align='center', alpha=0.5)
    p2 = plt.bar(np.arange(len(filenames)), log_val2, bottom=log_val1, align='center', alpha=0.5)
    plt.xticks(np.arange(len(filenames)), ['zcr', 'spectral_entropy', 'rolloff', 'flus',
                                           'spread', 'centroid', 'chroma_deviation', 'energy', 'entropy'])
    plt.ylabel('Negative Log Variance')
    plt.title('Features')
    plt.legend((p1[0], p2[0]), ('after', 'before'))
    plt.show()
    # np.save(all_val, "all_var")

    # zcr = m.generate_sample('./after2_flus.npy')
    # print(zcr.shape)
    # print(np.mean(zcr))



    # output = np.append(m.generate_sample('./after2_zcr.npy'), m.generate_sample('./after2_spectal_entropy.npy'), axis=0)
    # output = np.append(output, m.generate_sample('./after2_rolloff.npy'), axis=0)
    # output = np.append(output, m.generate_sample('./after2_flus.npy'), axis=0)
    # output = np.append(output, m.generate_sample('./after2_spread.npy'), axis=0)
    # output = np.append(output, m.generate_sample('./after2_centroid.npy'), axis=0)
    # output = np.append(output, m.generate_sample('./after2_chroma_deviation.npy'), axis=0)
    # output = np.append(output, m.generate_sample('./after2_energy.npy'), axis=0)
    # output = np.append(output, m.generate_sample('./after2_entropy.npy'), axis=0)

    # p.reduce_dim_pca(output, m.label())




