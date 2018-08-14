import os
import csv
import json
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
# from pandas.io.json import json_normalize
#
# #modify before mfcc
# before_mfcc = np.load('./before_female_mfcc.npy')
# mfcc1 = []
# mfcc2 = []
# mfcc3 =[]
# mfcc4 = []
# mfcc5 = []
# mfcc6 = []
# mfcc7 = []
# mfcc8 = []
# mfcc9 = []
# mfcc10 = []
# mfcc11 = []
# mfcc12 = []
# mfcc13 = []
# list = [mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13]
#
#
#
# for i in range(before_mfcc.shape[0]):
#     k = 0
#     each_patient1 = []
#     each_patient2 = []
#     each_patient3 = []
#     each_patient4 = []
#     each_patient5 = []
#     each_patient6 = []
#     each_patient7 = []
#     each_patient8 = []
#     each_patient9 = []
#     each_patient10 = []
#     each_patient11 = []
#     each_patient12 = []
#     each_patient13 = []
#     list1 = [each_patient1, each_patient2, each_patient3, each_patient4, each_patient5, each_patient6,
#              each_patient7, each_patient8, each_patient9, each_patient10, each_patient11, each_patient12,
#              each_patient13]
#     for j in range(len(before_mfcc[i])):
#         if '[' in before_mfcc[i][j]:
#             list1[k].append(float(before_mfcc[i][j][1:]))
#         elif ']' in before_mfcc[i][j]:
#             list1[k].append(float(before_mfcc[i][j][0:-1]))
#             list[k].append(list1[k])
#             k = k+1
#         else:
#             list1[k].append(float(before_mfcc[i][j]))
#
#
# np.save('./before_female_mfcc1', mfcc1)
# np.save('./before_female_mfcc2', mfcc2)
# np.save('./before_female_mfcc3', mfcc3)
# np.save('./before_female_mfcc4', mfcc4)
# np.save('./before_female_mfcc5', mfcc5)
# np.save('./before_female_mfcc6', mfcc6)
# np.save('./before_female_mfcc7', mfcc7)
# np.save('./before_female_mfcc8', mfcc8)
# np.save('./before_female_mfcc9', mfcc9)
# np.save('./before_female_mfcc10', mfcc10)
# np.save('./before_female_mfcc11', mfcc11)
# np.save('./before_female_mfcc12', mfcc12)
# np.save('./before_female_mfcc13', mfcc13)



after_mfcc = np.load('./after_female_mfcc.npy')
mfcc1 = []
mfcc2 = []
mfcc3 =[]
mfcc4 = []
mfcc5 = []
mfcc6 = []
mfcc7 = []
mfcc8 = []
mfcc9 = []
mfcc10 = []
mfcc11 = []
mfcc12 = []
mfcc13 = []
list = [mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13]



for i in range(after_mfcc.shape[0]):
    k = 0
    each_patient1 = []
    each_patient2 = []
    each_patient3 = []
    each_patient4 = []
    each_patient5 = []
    each_patient6 = []
    each_patient7 = []
    each_patient8 = []
    each_patient9 = []
    each_patient10 = []
    each_patient11 = []
    each_patient12 = []
    each_patient13 = []
    list1 = [each_patient1, each_patient2, each_patient3, each_patient4, each_patient5, each_patient6,
             each_patient7, each_patient8, each_patient9, each_patient10, each_patient11, each_patient12,
             each_patient13]
    for j in range(len(after_mfcc[i])):
        if '[' in after_mfcc[i][j]:
            list1[k].append(float(after_mfcc[i][j][1:]))
        elif ']' in after_mfcc[i][j]:
            list1[k].append(float(after_mfcc[i][j][0:-1]))
            list[k].append(list1[k])
            k = k+1
        else:
            list1[k].append(float(after_mfcc[i][j]))


np.save('./after_female_mfcc1', mfcc1)
np.save('./after_female_mfcc2', mfcc2)
np.save('./after_female_mfcc3', mfcc3)
np.save('./after_female_mfcc4', mfcc4)
np.save('./after_female_mfcc5', mfcc5)
np.save('./after_female_mfcc6', mfcc6)
np.save('./after_female_mfcc7', mfcc7)
np.save('./after_female_mfcc8', mfcc8)
np.save('./after_female_mfcc9', mfcc9)
np.save('./after_female_mfcc10', mfcc10)
np.save('./after_female_mfcc11', mfcc11)
np.save('./after_female_mfcc12', mfcc12)
np.save('./after_female_mfcc13', mfcc13)


#
# after_chroma_vector = np.load('./before_female_chroma_vector.npy')
# chroma_vector1 = []
# chroma_vector2 = []
# chroma_vector3 =[]
# chroma_vector4 = []
# chroma_vector5 = []
# chroma_vector6 = []
# chroma_vector7 = []
# chroma_vector8 = []
# chroma_vector9 = []
# chroma_vector10 = []
# chroma_vector11 = []
# chroma_vector12 = []
# list = [chroma_vector1, chroma_vector2, chroma_vector3, chroma_vector4, chroma_vector5, chroma_vector6,
#         chroma_vector7, chroma_vector8, chroma_vector9, chroma_vector10, chroma_vector11, chroma_vector12]
#
#
#
# for i in range(after_chroma_vector.shape[0]):
#     k = 0
#     each_patient1 = []
#     each_patient2 = []
#     each_patient3 = []
#     each_patient4 = []
#     each_patient5 = []
#     each_patient6 = []
#     each_patient7 = []
#     each_patient8 = []
#     each_patient9 = []
#     each_patient10 = []
#     each_patient11 = []
#     each_patient12 = []
#     list1 = [each_patient1, each_patient2, each_patient3, each_patient4, each_patient5, each_patient6,
#              each_patient7, each_patient8, each_patient9, each_patient10, each_patient11, each_patient12]
#     for j in range(len(after_chroma_vector[i])):
#         if '[' in after_chroma_vector[i][j]:
#             list1[k].append(float(after_chroma_vector[i][j][1:]))
#         elif ']' in after_chroma_vector[i][j]:
#             list1[k].append(float(after_chroma_vector[i][j][0:-1]))
#             list[k].append(list1[k])
#             k = k+1
#         else:
#             list1[k].append(float(after_chroma_vector[i][j]))
#
#
# np.save('./before_female_chroma_vector1', chroma_vector1)
# np.save('./before_female_chroma_vector2', chroma_vector2)
# np.save('./before_female_chroma_vector3', chroma_vector3)
# np.save('./before_female_chroma_vector4', chroma_vector4)
# np.save('./before_female_chroma_vector5', chroma_vector5)
# np.save('./before_female_chroma_vector6', chroma_vector6)
# np.save('./before_female_chroma_vector7', chroma_vector7)
# np.save('./before_female_chroma_vector8', chroma_vector8)
# np.save('./before_female_chroma_vector9', chroma_vector9)
# np.save('./before_female_chroma_vector10', chroma_vector10)
# np.save('./before_female_chroma_vector11', chroma_vector11)
# np.save('./before_female_chroma_vector12', chroma_vector12)

after_chroma_vector = np.load('./after_female_chroma_vector.npy')
chroma_vector1 = []
chroma_vector2 = []
chroma_vector3 =[]
chroma_vector4 = []
chroma_vector5 = []
chroma_vector6 = []
chroma_vector7 = []
chroma_vector8 = []
chroma_vector9 = []
chroma_vector10 = []
chroma_vector11 = []
chroma_vector12 = []
list = [chroma_vector1, chroma_vector2, chroma_vector3, chroma_vector4, chroma_vector5, chroma_vector6, chroma_vector7, chroma_vector8, chroma_vector9, chroma_vector10, chroma_vector11, chroma_vector12]



for i in range(after_chroma_vector.shape[0]):
    k = 0
    each_patient1 = []
    each_patient2 = []
    each_patient3 = []
    each_patient4 = []
    each_patient5 = []
    each_patient6 = []
    each_patient7 = []
    each_patient8 = []
    each_patient9 = []
    each_patient10 = []
    each_patient11 = []
    each_patient12 = []
    list1 = [each_patient1, each_patient2, each_patient3, each_patient4, each_patient5, each_patient6,
             each_patient7, each_patient8, each_patient9, each_patient10, each_patient11, each_patient12]
    for j in range(len(after_chroma_vector[i])):
        if '[' in after_chroma_vector[i][j]:
            list1[k].append(float(after_chroma_vector[i][j][1:]))
        elif ']' in after_chroma_vector[i][j]:
            list1[k].append(float(after_chroma_vector[i][j][0:-1]))
            list[k].append(list1[k])
            k = k+1
        else:
            list1[k].append(float(after_chroma_vector[i][j]))


np.save('./after_female_chroma_vector1', chroma_vector1)
np.save('./after_female_chroma_vector2', chroma_vector2)
np.save('./after_female_chroma_vector3', chroma_vector3)
np.save('./after_female_chroma_vector4', chroma_vector4)
np.save('./after_female_chroma_vector5', chroma_vector5)
np.save('./after_female_chroma_vector6', chroma_vector6)
np.save('./after_female_chroma_vector7', chroma_vector7)
np.save('./after_female_chroma_vector8', chroma_vector8)
np.save('./after_female_chroma_vector9', chroma_vector9)
np.save('./after_female_chroma_vector10', chroma_vector10)
np.save('./after_female_chroma_vector11', chroma_vector11)
np.save('./after_female_chroma_vector12', chroma_vector12)

#
# after_mfcc = np.load('./other2_mfcc.npy')
# mfcc1 = []
# mfcc2 = []
# mfcc3 =[]
# mfcc4 = []
# mfcc5 = []
# mfcc6 = []
# mfcc7 = []
# mfcc8 = []
# mfcc9 = []
# mfcc10 = []
# mfcc11 = []
# mfcc12 = []
# mfcc13 = []
# list = [mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13]
#
#
#
# for i in range(after_mfcc.shape[0]):
#     k = 0
#     each_patient1 = []
#     each_patient2 = []
#     each_patient3 = []
#     each_patient4 = []
#     each_patient5 = []
#     each_patient6 = []
#     each_patient7 = []
#     each_patient8 = []
#     each_patient9 = []
#     each_patient10 = []
#     each_patient11 = []
#     each_patient12 = []
#     each_patient13 = []
#     list1 = [each_patient1, each_patient2, each_patient3, each_patient4, each_patient5, each_patient6,
#              each_patient7, each_patient8, each_patient9, each_patient10, each_patient11, each_patient12,
#              each_patient13]
#     for j in range(len(after_mfcc[i])):
#         if '[' in after_mfcc[i][j]:
#             list1[k].append(float(after_mfcc[i][j][1:]))
#         elif ']' in after_mfcc[i][j]:
#             list1[k].append(float(after_mfcc[i][j][0:-1]))
#             list[k].append(list1[k])
#             k = k+1
#         else:
#             list1[k].append(float(after_mfcc[i][j]))
#
#
# np.save('./other_mfcc1', mfcc1)
# np.save('./other_mfcc2', mfcc2)
# np.save('./other_mfcc3', mfcc3)
# np.save('./other_mfcc4', mfcc4)
# np.save('./other_mfcc5', mfcc5)
# np.save('./other_mfcc6', mfcc6)
# np.save('./other_mfcc7', mfcc7)
# np.save('./other_mfcc8', mfcc8)
# np.save('./other_mfcc9', mfcc9)
# np.save('./other_mfcc10', mfcc10)
# np.save('./other_mfcc11', mfcc11)
# np.save('./other_mfcc12', mfcc12)
# np.save('./other_mfcc13', mfcc13)



# after_chroma_vector = np.load('./other2_chroma_vector.npy')
# chroma_vector1 = []
# chroma_vector2 = []
# chroma_vector3 =[]
# chroma_vector4 = []
# chroma_vector5 = []
# chroma_vector6 = []
# chroma_vector7 = []
# chroma_vector8 = []
# chroma_vector9 = []
# chroma_vector10 = []
# chroma_vector11 = []
# chroma_vector12 = []
# list = [chroma_vector1, chroma_vector2, chroma_vector3, chroma_vector4, chroma_vector5, chroma_vector6,
#         chroma_vector7, chroma_vector8, chroma_vector9, chroma_vector10, chroma_vector11, chroma_vector12]
#
#
#
# for i in range(after_chroma_vector.shape[0]):
#     k = 0
#     each_patient1 = []
#     each_patient2 = []
#     each_patient3 = []
#     each_patient4 = []
#     each_patient5 = []
#     each_patient6 = []
#     each_patient7 = []
#     each_patient8 = []
#     each_patient9 = []
#     each_patient10 = []
#     each_patient11 = []
#     each_patient12 = []
#     list1 = [each_patient1, each_patient2, each_patient3, each_patient4, each_patient5, each_patient6,
#              each_patient7, each_patient8, each_patient9, each_patient10, each_patient11, each_patient12]
#     for j in range(len(after_chroma_vector[i])):
#         if '[' in after_chroma_vector[i][j]:
#             list1[k].append(float(after_chroma_vector[i][j][1:]))
#         elif ']' in after_chroma_vector[i][j]:
#             list1[k].append(float(after_chroma_vector[i][j][0:-1]))
#             list[k].append(list1[k])
#             k = k+1
#         else:
#             list1[k].append(float(after_chroma_vector[i][j]))
#
#
# np.save('./other_chroma_vector1', chroma_vector1)
# np.save('./other_chroma_vector2', chroma_vector2)
# np.save('./other_chroma_vector3', chroma_vector3)
# np.save('./other_chroma_vector4', chroma_vector4)
# np.save('./other_chroma_vector5', chroma_vector5)
# np.save('./other_chroma_vector6', chroma_vector6)
# np.save('./other_chroma_vector7', chroma_vector7)
# np.save('./other_chroma_vector8', chroma_vector8)
# np.save('./other_chroma_vector9', chroma_vector9)
# np.save('./other_chroma_vector10', chroma_vector10)
# np.save('./other_chroma_vector11', chroma_vector11)
# np.save('./other_chroma_vector12', chroma_vector12)











