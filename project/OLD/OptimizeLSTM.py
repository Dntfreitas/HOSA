import gc
import pickle

import numpy as np
import scipy.io as spio
import tensorflow as tf
from scipy import signal
from scipy.integrate import simps
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

#### Control variable

Begin = 0
numberSubjects = 18
patienteceValue = 1  # patience for validation

startEpochs = 0  # Subject to start the test: 0
Epochs = 1  # number of iterations, 19 for LOO, <11 for TFCV
BeginTest = 10  # 18 for LOOCV, 10 for TFCV

PercentageOfData = 1  # 0.25, 0.5, 0.75, 1

FeatureBasedModel = 0  # 1 to use feature based mode; 0 for deep learning
numberFeatures = 12  # 1,2,...,20; 18 for A1, 13 for A2, 4 for A3
Midlay = 5  # segmentation for symbolic dynamics

Subtype = 1  # 1 for A1, 2 for A2, 3 for A3
forA = 1  # for A phase choose Subtype=1 and forA=1
forNREM = 0  # for NREM estimation choose 1 and for A phase choose 1

timeStepsStart = 5  # 5, 15, 25, 35
timeStepsMax = 35  # 5, 15, 25, 35
timeStepsStep = 10  # 1, 5, 10
numbHideenStart = 100  # 100, 200, 300, 400
numbHideenMax = 400  # 100, 200, 300, 400
numbHideenStep = 100  # 50, 100, 200
thresholdAphase = 0.01  # thresold for the minimum increasse
ExaminedLayersMax = 5  # maximum number of LSTM to examine 2, 3, 4, 5, ...

EpochsWork = 10  # number of iterations 10 for TFCV or 50 for LOOCV
patience = patienteceValue

if FeatureBasedModel == 0:
    numberFeatures = 100

#### Load n
mat = spio.loadmat('n1eegminut2.mat', squeeze_me=True)
n1 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n2eegminut2.mat', squeeze_me=True)
n2 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n3eegminut2.mat', squeeze_me=True)
n3 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n4eegminut2.mat', squeeze_me=True)
n4 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n5eegminut2.mat', squeeze_me=True)
n5 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n6eegminut2.mat', squeeze_me=True)
n6 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n7eegminut2.mat', squeeze_me=True)
n7 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n8eegminut2.mat', squeeze_me=True)
n8 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n9eegminut2.mat', squeeze_me=True)
n9 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n10eegminut2.mat', squeeze_me=True)
n10 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n11eegminut2.mat', squeeze_me=True)
n11 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n13eegminut2.mat', squeeze_me=True)
n13 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n14eegminut2.mat', squeeze_me=True)
n14 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n15eegminut2.mat', squeeze_me=True)
n15 = mat.get('eegSensor')
del mat
mat = spio.loadmat('n16eegminut2.mat', squeeze_me=True)
n16 = mat.get('eegSensor')
del mat

mat = spio.loadmat('sdb1eegminut2.mat', squeeze_me=True)
sdb1 = mat.get('eegSensor')
del mat
mat = spio.loadmat('sdb2eegminut2.mat', squeeze_me=True)
sdb2 = mat.get('eegSensor')
del mat
mat = spio.loadmat('sdb3eegminut2.mat', squeeze_me=True)
sdb3 = mat.get('eegSensor')
del mat
mat = spio.loadmat('sdb4eegminut2.mat', squeeze_me=True)
sdb4 = mat.get('eegSensor')
del mat

#### Load labels
searchval = 1
mat = spio.loadmat('n1eegminutLable2.mat', squeeze_me=True)
nc1 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n2eegminutLable2.mat', squeeze_me=True)
nc2 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n3eegminutLable2.mat', squeeze_me=True)
nc3 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n4eegminutLable2.mat', squeeze_me=True)
nc4 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n5eegminutLable2.mat', squeeze_me=True)
nc5 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n6eegminutLable2.mat', squeeze_me=True)
nc6 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n7eegminutLable2.mat', squeeze_me=True)
nc7 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n8eegminutLable2.mat', squeeze_me=True)
nc8 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n9eegminutLable2.mat', squeeze_me=True)
nc9 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n10eegminutLable2.mat', squeeze_me=True)
nc10 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n11eegminutLable2.mat', squeeze_me=True)
nc11 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n13eegminutLable2.mat', squeeze_me=True)
nc13 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n14eegminutLable2.mat', squeeze_me=True)
nc14 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n15eegminutLable2.mat', squeeze_me=True)
nc15 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('n16eegminutLable2.mat', squeeze_me=True)
nc16 = mat.get('CAPlabel1')
del mat

mat = spio.loadmat('sdb1eegminutLable2.mat', squeeze_me=True)
sdbc1 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('sdb2eegminutLable2.mat', squeeze_me=True)
sdbc2 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('sdb3eegminutLable2.mat', squeeze_me=True)
sdbc3 = mat.get('CAPlabel1')
del mat
mat = spio.loadmat('sdb4eegminutLable2.mat', squeeze_me=True)
sdbc4 = mat.get('CAPlabel1')
del mat

mat = spio.loadmat('n1hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch1 = mat.get('Hipno')
nch1[nch1 == 5] = 0
nch1[nch1 > 0] = 1
del mat
mat = spio.loadmat('n2hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch2 = mat.get('Hipno')
nch2[nch2 == 5] = 0
nch2[nch2 > 0] = 1
del mat
mat = spio.loadmat('n3hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch3 = mat.get('Hipno')
nch3[nch3 == 5] = 0
nch3[nch3 > 0] = 1
del mat
mat = spio.loadmat('n4hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch4 = mat.get('Hipno')
nch4[nch4 == 5] = 0
nch4[nch4 > 0] = 1
del mat
mat = spio.loadmat('n5hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch5 = mat.get('Hipno')
nch5[nch5 == 5] = 0
nch5[nch5 > 0] = 1
del mat
mat = spio.loadmat('n6hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch6 = mat.get('Hipno')
nch6[nch6 == 5] = 0
nch6[nch6 > 0] = 1
del mat
mat = spio.loadmat('n7hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch7 = mat.get('Hipno')
nch7[nch7 == 5] = 0
nch7[nch7 > 0] = 1
del mat
mat = spio.loadmat('n8hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch8 = mat.get('Hipno')
nch8[nch8 == 5] = 0
nch8[nch8 > 0] = 1
del mat
mat = spio.loadmat('n9hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch9 = mat.get('Hipno')
nch9[nch9 == 5] = 0
nch9[nch9 > 0] = 1
del mat
mat = spio.loadmat('n10hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch10 = mat.get('Hipno')
nch10[nch10 == 5] = 0
nch10[nch10 > 0] = 1
del mat
mat = spio.loadmat('n11hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch11 = mat.get('Hipno')
nch11[nch11 == 5] = 0
nch11[nch11 > 0] = 1
del mat
mat = spio.loadmat('n13hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch13 = mat.get('Hipno')
nch13[nch13 == 5] = 0
nch13[nch13 > 0] = 1
del mat
mat = spio.loadmat('n14hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch14 = mat.get('Hipno')
nch14[nch14 == 5] = 0
nch14[nch14 > 0] = 1
del mat
mat = spio.loadmat('n15hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch15 = mat.get('Hipno')
nch15[nch15 == 5] = 0
nch15[nch15 > 0] = 1
del mat
mat = spio.loadmat('n16hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch16 = mat.get('Hipno')
nch16[nch16 == 5] = 0
nch16[nch16 > 0] = 1
del mat
mat = spio.loadmat('n16hypnoEEGminutLable2V2.mat', squeeze_me=True)
nch16 = mat.get('Hipno')
nch16[nch16 == 5] = 0
nch16[nch16 > 0] = 1
del mat

mat = spio.loadmat('sdb1hypnoEEGminutLable2V2.mat', squeeze_me=True)
sdbch1 = mat.get('Hipno')
sdbch1[sdbch1 == 5] = 0
sdbch1[sdbch1 > 0] = 1
del mat
mat = spio.loadmat('sdb2hypnoEEGminutLable2V2.mat', squeeze_me=True)
sdbch2 = mat.get('Hipno')
sdbch2[sdbch2 == 5] = 0
sdbch2[sdbch2 > 0] = 1
del mat
mat = spio.loadmat('sdb3hypnoEEGminutLable2V2.mat', squeeze_me=True)
sdbch3 = mat.get('Hipno')
sdbch3[sdbch3 == 5] = 0
sdbch3[sdbch3 > 0] = 1
del mat
mat = spio.loadmat('sdb4hypnoEEGminutLable2V2.mat', squeeze_me=True)
sdbch4 = mat.get('Hipno')
sdbch4[sdbch4 == 5] = 0
sdbch4[sdbch4 > 0] = 1
del mat

sigmaLine = np.std(n1)
CodedSequence = np.zeros(len(n1))
n1AAA = n1
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence1 = CodedSequence

sigmaLine = np.std(n2)
CodedSequence = np.zeros(len(n2))
n1AAA = n2
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence2 = CodedSequence

sigmaLine = np.std(n3)
CodedSequence = np.zeros(len(n3))
n1AAA = n3
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence3 = CodedSequence

sigmaLine = np.std(n4)
CodedSequence = np.zeros(len(n4))
n1AAA = n4
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence4 = CodedSequence

sigmaLine = np.std(n5)
CodedSequence = np.zeros(len(n5))
n1AAA = n5
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence5 = CodedSequence

sigmaLine = np.std(n6)
CodedSequence = np.zeros(len(n6))
n1AAA = n6
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence6 = CodedSequence

sigmaLine = np.std(n7)
CodedSequence = np.zeros(len(n7))
n1AAA = n7
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence7 = CodedSequence

sigmaLine = np.std(n8)
CodedSequence = np.zeros(len(n8))
n1AAA = n8
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence8 = CodedSequence

sigmaLine = np.std(n9)
CodedSequence = np.zeros(len(n9))
n1AAA = n9
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence9 = CodedSequence

sigmaLine = np.std(n10)
CodedSequence = np.zeros(len(n10))
n1AAA = n10
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence10 = CodedSequence

sigmaLine = np.std(n11)
CodedSequence = np.zeros(len(n11))
n1AAA = n11
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence11 = CodedSequence

sigmaLine = np.std(n13)
CodedSequence = np.zeros(len(n13))
n1AAA = n13
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence13 = CodedSequence

sigmaLine = np.std(n14)
CodedSequence = np.zeros(len(n14))
n1AAA = n14
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence14 = CodedSequence

sigmaLine = np.std(n15)
CodedSequence = np.zeros(len(n15))
n1AAA = n15
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence15 = CodedSequence

sigmaLine = np.std(n16)
CodedSequence = np.zeros(len(n16))
n1AAA = n16
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequence16 = CodedSequence

sigmaLine = np.std(sdb1)
CodedSequence = np.zeros(len(sdb1))
n1AAA = sdb1
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequencesdb1 = CodedSequence

sigmaLine = np.std(sdb2)
CodedSequence = np.zeros(len(sdb2))
n1AAA = sdb2
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequencesdb2 = CodedSequence

sigmaLine = np.std(sdb3)
CodedSequence = np.zeros(len(sdb3))
n1AAA = sdb3
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequencesdb3 = CodedSequence

sigmaLine = np.std(sdb4)
CodedSequence = np.zeros(len(sdb4))
n1AAA = sdb4
for i in range(len(n1AAA)):
    if n1AAA[i] > 0:
        if n1AAA[i] < sigmaLine:
            CodedSequence[i] = 4
        elif n1AAA[i] < Midlay * sigmaLine:
            CodedSequence[i] = 5
        else:
            CodedSequence[i] = 6

    else:
        if n1AAA[i] > -sigmaLine:
            CodedSequence[i] = 3
        elif n1AAA[i] > -Midlay * sigmaLine:
            CodedSequence[i] = 2
        else:
            CodedSequence[i] = 1

CodedSequencesdb4 = CodedSequence

##### Prepare n

AccAtEnd = np.zeros(EpochsWork)
SenAtEnd = np.zeros(EpochsWork)
SpeAtEnd = np.zeros(EpochsWork)
AUCAtEnd = np.zeros(EpochsWork)

AccAtEndM = np.zeros([4, 2 * ExaminedLayersMax *
                      np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SenAtEndM = np.zeros([4, 2 * ExaminedLayersMax *
                      np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SpeAtEndM = np.zeros([4, 2 * ExaminedLayersMax *
                      np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
AUCAtEndM = np.zeros([4, 2 * ExaminedLayersMax *
                      np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])

AccAtEndL = np.zeros([2, ExaminedLayersMax *
                      np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SenAtEndL = np.zeros([2, ExaminedLayersMax *
                      np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SpeAtEndL = np.zeros([2, ExaminedLayersMax *
                      np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
AUCAtEndL = np.zeros([2, ExaminedLayersMax *
                      np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])

AccAtEndG = np.zeros([ExaminedLayersMax, np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SenAtEndG = np.zeros([ExaminedLayersMax, np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SpeAtEndG = np.zeros([ExaminedLayersMax, np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
AUCAtEndG = np.zeros([ExaminedLayersMax, np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                      np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])

AccAtEndN = np.zeros([np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1), np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SenAtEndN = np.zeros([np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1), np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SpeAtEndN = np.zeros([np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1), np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
AUCAtEndN = np.zeros([np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1), np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])

AccAtEndT = np.zeros(np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1))
SenAtEndT = np.zeros(np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1))
SpeAtEndT = np.zeros(np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1))
AUCAtEndT = np.zeros(np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1))

AccAtEndLarg = np.zeros([2, ExaminedLayersMax *
                         np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                         np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SenAtEndLarg = np.zeros([2, ExaminedLayersMax *
                         np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                         np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SpeAtEndLarg = np.zeros([2, ExaminedLayersMax *
                         np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                         np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
AUCAtEndLarg = np.zeros([2, ExaminedLayersMax *
                         np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                         np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])

AccAtEndGarg = np.zeros([ExaminedLayersMax, np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                         np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SenAtEndGarg = np.zeros([ExaminedLayersMax, np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                         np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SpeAtEndGarg = np.zeros([ExaminedLayersMax, np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                         np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
AUCAtEndGarg = np.zeros([ExaminedLayersMax, np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1) *
                         np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])

AccAtEndNarg = np.zeros([np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1), np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SenAtEndNarg = np.zeros([np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1), np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
SpeAtEndNarg = np.zeros([np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1), np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])
AUCAtEndNarg = np.zeros([np.int((numbHideenMax - numbHideenStart) / numbHideenStep + 1), np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1)])

AccAtEndTarg = np.zeros(np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1))
SenAtEndTarg = np.zeros(np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1))
SpeAtEndTarg = np.zeros(np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1))
AUCAtEndTarg = np.zeros(np.int((timeStepsMax - timeStepsStart) / timeStepsStep + 1))

BestNet = np.zeros(5)  # 0->m, 1->L, 2->ExaminedLayers, 3->numbHideen, 4->timeSteps

n1 = (n1 - np.mean(n1)) / np.std(n1)
n2 = (n2 - np.mean(n2)) / np.std(n2)
n3 = (n3 - np.mean(n3)) / np.std(n3)
n4 = (n4 - np.mean(n4)) / np.std(n4)
n5 = (n5 - np.mean(n5)) / np.std(n5)
n6 = (n6 - np.mean(n6)) / np.std(n6)
n7 = (n7 - np.mean(n7)) / np.std(n7)
n8 = (n8 - np.mean(n8)) / np.std(n8)
n9 = (n9 - np.mean(n9)) / np.std(n9)
n10 = (n10 - np.mean(n10)) / np.std(n10)
n11 = (n11 - np.mean(n11)) / np.std(n11)
n13 = (n13 - np.mean(n13)) / np.std(n13)
n14 = (n14 - np.mean(n14)) / np.std(n14)
n15 = (n15 - np.mean(n15)) / np.std(n15)
n16 = (n16 - np.mean(n16)) / np.std(n16)

sdb1 = (sdb1 - np.mean(sdb1)) / np.std(sdb1)
sdb2 = (sdb2 - np.mean(sdb2)) / np.std(sdb2)
sdb3 = (sdb3 - np.mean(sdb3)) / np.std(sdb3)
sdb4 = (sdb4 - np.mean(sdb4)) / np.std(sdb4)

for ee in range(startEpochs, Epochs, 1):
    tf.keras.backend.clear_session()
    gc.collect()

    XTrain = [];
    XTest = [];
    YTrain = [];
    YTest = [];
    YTrainh = [];
    YTestHypno = [];
    XTrainSeq = []
    XTestSeq = []

    indexT = 0
    countM = 0
    countL = 0
    countG = 0
    countN = 0
    AUCmax = 0
    for timeSteps in range(timeStepsStart, timeStepsMax + timeStepsStep, timeStepsStep):

        if Epochs > 10:
            print('\n\n Using LOOCV \n \n')
            print('\n\n Subject: ', ee)

            # import pickle
            # normalSubjects = np.random.permutation(19) # choose subjects order

            if ee == 0:
                normalSubjects = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0])
            elif ee == 1:
                normalSubjects = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1])
            elif ee == 2:
                normalSubjects = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2])
            elif ee == 3:
                normalSubjects = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 3])
            elif ee == 4:
                normalSubjects = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4])
            elif ee == 5:
                normalSubjects = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 5])
            elif ee == 6:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 6])
            elif ee == 7:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 7])
            elif ee == 8:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 8])
            elif ee == 9:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 9])
            elif ee == 10:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 10])
            elif ee == 11:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 11])
            elif ee == 12:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 12])
            elif ee == 13:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 13])
            elif ee == 14:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 14])
            elif ee == 15:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 15])
            elif ee == 16:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 16])
            elif ee == 17:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 17])
            else:
                normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

            if normalSubjects[Begin] == 0:
                XTrain = n1
                YTrainh = nch1
                YTrain = nc1
                XTrainSeq = CodedSequence1
            if normalSubjects[Begin] == 1:
                XTrain = n2
                YTrainh = nch2
                YTrain = nc2
                XTrainSeq = CodedSequence2
            if normalSubjects[Begin] == 2:
                XTrain = n3
                YTrainh = nch3
                YTrain = nc3
                XTrainSeq = CodedSequence3
            if normalSubjects[Begin] == 3:
                XTrain = n4
                YTrainh = nch4
                YTrain = nc4
                XTrainSeq = CodedSequence4
            if normalSubjects[Begin] == 4:
                XTrain = n5
                YTrainh = nch5
                YTrain = nc5
                XTrainSeq = CodedSequence5
            if normalSubjects[Begin] == 5:
                XTrain = n6
                YTrainh = nch6
                YTrain = nc6
                XTrainSeq = CodedSequence6
            if normalSubjects[Begin] == 6:
                XTrain = n7
                YTrainh = nch7
                YTrain = nc7
                XTrainSeq = CodedSequence7
            if normalSubjects[Begin] == 7:
                XTrain = n8
                YTrainh = nch8
                YTrain = nc8
                XTrainSeq = CodedSequence8
            if normalSubjects[Begin] == 8:
                XTrain = n9
                YTrainh = nch9
                YTrain = nc9
                XTrainSeq = CodedSequence9
            if normalSubjects[Begin] == 9:
                XTrain = n10
                YTrainh = nch10
                YTrain = nc10
                XTrainSeq = CodedSequence10
            if normalSubjects[Begin] == 10:
                XTrain = n11
                YTrainh = nch11
                YTrain = nc11
                XTrainSeq = CodedSequence11
            if normalSubjects[Begin] == 11:
                XTrain = n13
                YTrainh = nch13
                YTrain = nc13
                XTrainSeq = CodedSequence13
            if normalSubjects[Begin] == 12:
                XTrain = n14
                YTrainh = nch14
                YTrain = nc14
                XTrainSeq = CodedSequence14
            if normalSubjects[Begin] == 13:
                XTrain = n15
                YTrainh = nch15
                YTrain = nc15
                XTrainSeq = CodedSequence15
            if normalSubjects[Begin] == 14:
                XTrain = n16
                YTrainh = nch16
                YTrain = nc16
                XTrainSeq = CodedSequence16
            if normalSubjects[Begin] == 15:
                XTrain = sdb1
                YTrainh = sdbch1
                YTrain = sdbc1
                XTrainSeq = CodedSequencesdb1
            if normalSubjects[Begin] == 16:
                XTrain = sdb2
                YTrainh = sdbch2
                YTrain = sdbc2
                XTrainSeq = CodedSequencesdb2
            if normalSubjects[Begin] == 17:
                XTrain = sdb3
                YTrainh = sdbch3
                YTrain = sdbc3
                XTrainSeq = CodedSequencesdb3
            if normalSubjects[Begin] == 18:
                XTrain = sdb4
                YTrainh = sdbch4
                YTrain = sdbc4
                XTrainSeq = CodedSequencesdb4

            if normalSubjects[BeginTest] == 0:
                XTest = n1
                YTestHypno = nch1
                YTest = nc1
                XTestSeq = CodedSequence1
            if normalSubjects[BeginTest] == 1:
                XTest = n2
                YTestHypno = nch2
                YTest = nc2
                XTestSeq = CodedSequence2
            if normalSubjects[BeginTest] == 2:
                XTest = n3
                YTestHypno = nch3
                YTest = nc3
                XTestSeq = CodedSequence3
            if normalSubjects[BeginTest] == 3:
                XTest = n4
                YTestHypno = nch4
                YTest = nc4
                XTestSeq = CodedSequence4
            if normalSubjects[BeginTest] == 4:
                XTest = n5
                YTestHypno = nch5
                YTest = nc5
                XTestSeq = CodedSequence5
            if normalSubjects[BeginTest] == 5:
                XTest = n6
                YTestHypno = nch6
                YTest = nc6
                XTestSeq = CodedSequence6
            if normalSubjects[BeginTest] == 6:
                XTest = n7
                YTestHypno = nch7
                YTest = nc7
                XTestSeq = CodedSequence7
            if normalSubjects[BeginTest] == 7:
                XTest = n8
                YTestHypno = nch8
                YTest = nc8
                XTestSeq = CodedSequence8
            if normalSubjects[BeginTest] == 8:
                XTest = n9
                YTestHypno = nch9
                YTest = nc9
                XTestSeq = CodedSequence9
            if normalSubjects[BeginTest] == 9:
                XTest = n10
                YTestHypno = nch10
                YTest = nc10
                XTestSeq = CodedSequence10
            if normalSubjects[BeginTest] == 10:
                XTest = n11
                YTestHypno = nch11
                YTest = nc11
                XTestSeq = CodedSequence11
            if normalSubjects[BeginTest] == 11:
                XTest = n13
                YTestHypno = nch13
                YTest = nc13
                XTestSeq = CodedSequence13
            if normalSubjects[BeginTest] == 12:
                XTest = n14
                YTestHypno = nch14
                YTest = nc14
                XTestSeq = CodedSequence14
            if normalSubjects[BeginTest] == 13:
                XTest = n15
                YTestHypno = nch15
                YTest = nc15
                XTestSeq = CodedSequence15
            if normalSubjects[BeginTest] == 14:
                XTest = n16
                YTestHypno = nch16
                YTest = nc16
                XTestSeq = CodedSequence16
            if normalSubjects[BeginTest] == 15:
                XTest = sdb1
                YTestHypno = sdbch1
                YTest = sdbc1
                XTestSeq = CodedSequencesdb1
            if normalSubjects[BeginTest] == 16:
                XTest = sdb2
                YTestHypno = sdbch2
                YTest = sdbc2
                XTestSeq = CodedSequencesdb2
            if normalSubjects[BeginTest] == 17:
                XTest = sdb3
                YTestHypno = sdbch3
                YTest = sdbc3
                XTestSeq = CodedSequencesdb3
            if normalSubjects[BeginTest] == 18:
                XTest = sdb4
                YTestHypno = sdbch4
                YTest = sdbc4
                XTestSeq = CodedSequencesdb4

            for x in range(20):
                if x < BeginTest and x > Begin:  # train
                    if normalSubjects[x] == 0:
                        YTrainh = np.concatenate((YTrainh, nch1), axis=0)
                        XTrain = np.concatenate((XTrain, n1), axis=0)
                        YTrain = np.concatenate((YTrain, nc1), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence1), axis=0)
                    if normalSubjects[x] == 1:
                        YTrainh = np.concatenate((YTrainh, nch2), axis=0)
                        XTrain = np.concatenate((XTrain, n2), axis=0)
                        YTrain = np.concatenate((YTrain, nc2), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence2), axis=0)
                    if normalSubjects[x] == 2:
                        YTrainh = np.concatenate((YTrainh, nch3), axis=0)
                        XTrain = np.concatenate((XTrain, n3), axis=0)
                        YTrain = np.concatenate((YTrain, nc3), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence3), axis=0)
                    if normalSubjects[x] == 3:
                        YTrainh = np.concatenate((YTrainh, nch4), axis=0)
                        XTrain = np.concatenate((XTrain, n4), axis=0)
                        YTrain = np.concatenate((YTrain, nc4), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence4), axis=0)
                    if normalSubjects[x] == 4:
                        YTrainh = np.concatenate((YTrainh, nch5), axis=0)
                        XTrain = np.concatenate((XTrain, n5), axis=0)
                        YTrain = np.concatenate((YTrain, nc5), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence5), axis=0)
                    if normalSubjects[x] == 5:
                        YTrainh = np.concatenate((YTrainh, nch6), axis=0)
                        XTrain = np.concatenate((XTrain, n6), axis=0)
                        YTrain = np.concatenate((YTrain, nc6), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence6), axis=0)
                    if normalSubjects[x] == 6:
                        YTrainh = np.concatenate((YTrainh, nch7), axis=0)
                        XTrain = np.concatenate((XTrain, n7), axis=0)
                        YTrain = np.concatenate((YTrain, nc7), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence7), axis=0)
                    if normalSubjects[x] == 7:
                        YTrainh = np.concatenate((YTrainh, nch8), axis=0)
                        XTrain = np.concatenate((XTrain, n8), axis=0)
                        YTrain = np.concatenate((YTrain, nc8), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence8), axis=0)
                    if normalSubjects[x] == 8:
                        YTrainh = np.concatenate((YTrainh, nch9), axis=0)
                        XTrain = np.concatenate((XTrain, n9), axis=0)
                        YTrain = np.concatenate((YTrain, nc9), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence9), axis=0)
                    if normalSubjects[x] == 9:
                        YTrainh = np.concatenate((YTrainh, nch10), axis=0)
                        XTrain = np.concatenate((XTrain, n10), axis=0)
                        YTrain = np.concatenate((YTrain, nc10), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence10), axis=0)
                    if normalSubjects[x] == 10:
                        YTrainh = np.concatenate((YTrainh, nch11), axis=0)
                        XTrain = np.concatenate((XTrain, n11), axis=0)
                        YTrain = np.concatenate((YTrain, nc11), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence11), axis=0)
                    if normalSubjects[x] == 11:
                        YTrainh = np.concatenate((YTrainh, nch13), axis=0)
                        XTrain = np.concatenate((XTrain, n13), axis=0)
                        YTrain = np.concatenate((YTrain, nc13), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence13), axis=0)
                    if normalSubjects[x] == 12:
                        YTrainh = np.concatenate((YTrainh, nch14), axis=0)
                        XTrain = np.concatenate((XTrain, n14), axis=0)
                        YTrain = np.concatenate((YTrain, nc14), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence14), axis=0)
                    if normalSubjects[x] == 13:
                        YTrainh = np.concatenate((YTrainh, nch15), axis=0)
                        XTrain = np.concatenate((XTrain, n15), axis=0)
                        YTrain = np.concatenate((YTrain, nc15), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence15), axis=0)
                    if normalSubjects[x] == 14:
                        YTrainh = np.concatenate((YTrainh, nch16), axis=0)
                        XTrain = np.concatenate((XTrain, n16), axis=0)
                        YTrain = np.concatenate((YTrain, nc16), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence16), axis=0)
                    if normalSubjects[x] == 15:
                        YTrainh = np.concatenate((YTrainh, sdbch1), axis=0)
                        XTrain = np.concatenate((XTrain, sdb1), axis=0)
                        YTrain = np.concatenate((YTrain, sdbc1), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequencesdb1), axis=0)
                    if normalSubjects[x] == 16:
                        YTrainh = np.concatenate((YTrainh, sdbch2), axis=0)
                        XTrain = np.concatenate((XTrain, sdb2), axis=0)
                        YTrain = np.concatenate((YTrain, sdbc2), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequencesdb2), axis=0)
                    if normalSubjects[x] == 17:
                        YTrainh = np.concatenate((YTrainh, sdbch3), axis=0)
                        XTrain = np.concatenate((XTrain, sdb3), axis=0)
                        YTrain = np.concatenate((YTrain, sdbc3), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequencesdb3), axis=0)
                    if normalSubjects[x] == 18:
                        YTrainh = np.concatenate((YTrainh, sdbch4), axis=0)
                        XTrain = np.concatenate((XTrain, sdb4), axis=0)
                        YTrain = np.concatenate((YTrain, sdbc4), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequencesdb4), axis=0)


        else:
            print('\n\n Using TFCV')
            print('\n\n Iteration: ', ee)

            normalSubjects = np.random.permutation(19)  # choose subjects order

            if normalSubjects[Begin] == 0:
                XTrain = n1
                YTrainh = nch1
                YTrain = nc1
                XTrainSeq = CodedSequence1
            if normalSubjects[Begin] == 1:
                XTrain = n2
                YTrainh = nch2
                YTrain = nc2
                XTrainSeq = CodedSequence2
            if normalSubjects[Begin] == 2:
                XTrain = n3
                YTrainh = nch3
                YTrain = nc3
                XTrainSeq = CodedSequence3
            if normalSubjects[Begin] == 3:
                XTrain = n4
                YTrainh = nch4
                YTrain = nc4
                XTrainSeq = CodedSequence4
            if normalSubjects[Begin] == 4:
                XTrain = n5
                YTrainh = nch5
                YTrain = nc5
                XTrainSeq = CodedSequence5
            if normalSubjects[Begin] == 5:
                XTrain = n6
                YTrainh = nch6
                YTrain = nc6
                XTrainSeq = CodedSequence6
            if normalSubjects[Begin] == 6:
                XTrain = n7
                YTrainh = nch7
                YTrain = nc7
                XTrainSeq = CodedSequence7
            if normalSubjects[Begin] == 7:
                XTrain = n8
                YTrainh = nch8
                YTrain = nc8
                XTrainSeq = CodedSequence8
            if normalSubjects[Begin] == 8:
                XTrain = n9
                YTrainh = nch9
                YTrain = nc9
                XTrainSeq = CodedSequence9
            if normalSubjects[Begin] == 9:
                XTrain = n10
                YTrainh = nch10
                YTrain = nc10
                XTrainSeq = CodedSequence10
            if normalSubjects[Begin] == 10:
                XTrain = n11
                YTrainh = nch11
                YTrain = nc11
                XTrainSeq = CodedSequence11
            if normalSubjects[Begin] == 11:
                XTrain = n13
                YTrainh = nch13
                YTrain = nc13
                XTrainSeq = CodedSequence13
            if normalSubjects[Begin] == 12:
                XTrain = n14
                YTrainh = nch14
                YTrain = nc14
                XTrainSeq = CodedSequence14
            if normalSubjects[Begin] == 13:
                XTrain = n15
                YTrainh = nch15
                YTrain = nc15
                XTrainSeq = CodedSequence15
            if normalSubjects[Begin] == 14:
                XTrain = n16
                YTrainh = nch16
                YTrain = nc16
                XTrainSeq = CodedSequence16
            if normalSubjects[Begin] == 15:
                XTrain = sdb1
                YTrainh = sdbch1
                YTrain = sdbc1
                XTrainSeq = CodedSequencesdb1
            if normalSubjects[Begin] == 16:
                XTrain = sdb2
                YTrainh = sdbch2
                YTrain = sdbc2
                XTrainSeq = CodedSequencesdb2
            if normalSubjects[Begin] == 17:
                XTrain = sdb3
                YTrainh = sdbch3
                YTrain = sdbc3
                XTrainSeq = CodedSequencesdb3
            if normalSubjects[Begin] == 18:
                XTrain = sdb4
                YTrainh = sdbch4
                YTrain = sdbc4
                XTrainSeq = CodedSequencesdb4

            if normalSubjects[BeginTest] == 0:
                XTest = n1
                YTestHypno = nch1
                YTest = nc1
                XTestSeq = CodedSequence1
            if normalSubjects[BeginTest] == 1:
                XTest = n2
                YTestHypno = nch2
                YTest = nc2
                XTestSeq = CodedSequence2
            if normalSubjects[BeginTest] == 2:
                XTest = n3
                YTestHypno = nch3
                YTest = nc3
                XTestSeq = CodedSequence3
            if normalSubjects[BeginTest] == 3:
                XTest = n4
                YTestHypno = nch4
                YTest = nc4
                XTestSeq = CodedSequence4
            if normalSubjects[BeginTest] == 4:
                XTest = n5
                YTestHypno = nch5
                YTest = nc5
                XTestSeq = CodedSequence5
            if normalSubjects[BeginTest] == 5:
                XTest = n6
                YTestHypno = nch6
                YTest = nc6
                XTestSeq = CodedSequence6
            if normalSubjects[BeginTest] == 6:
                XTest = n7
                YTestHypno = nch7
                YTest = nc7
                XTestSeq = CodedSequence7
            if normalSubjects[BeginTest] == 7:
                XTest = n8
                YTestHypno = nch8
                YTest = nc8
                XTestSeq = CodedSequence8
            if normalSubjects[BeginTest] == 8:
                XTest = n9
                YTestHypno = nch9
                YTest = nc9
                XTestSeq = CodedSequence9
            if normalSubjects[BeginTest] == 9:
                XTest = n10
                YTestHypno = nch10
                YTest = nc10
                XTestSeq = CodedSequence10
            if normalSubjects[BeginTest] == 10:
                XTest = n11
                YTestHypno = nch11
                YTest = nc11
                XTestSeq = CodedSequence11
            if normalSubjects[BeginTest] == 11:
                XTest = n13
                YTestHypno = nch13
                YTest = nc13
                XTestSeq = CodedSequence13
            if normalSubjects[BeginTest] == 12:
                XTest = n14
                YTestHypno = nch14
                YTest = nc14
                XTestSeq = CodedSequence14
            if normalSubjects[BeginTest] == 13:
                XTest = n15
                YTestHypno = nch15
                YTest = nc15
                XTestSeq = CodedSequence15
            if normalSubjects[BeginTest] == 14:
                XTest = n16
                YTestHypno = nch16
                YTest = nc16
                XTestSeq = CodedSequence16
            if normalSubjects[BeginTest] == 15:
                XTest = sdb1
                YTestHypno = sdbch1
                YTest = sdbc1
                XTestSeq = CodedSequencesdb1
            if normalSubjects[BeginTest] == 16:
                XTest = sdb2
                YTestHypno = sdbch2
                YTest = sdbc2
                XTestSeq = CodedSequencesdb2
            if normalSubjects[BeginTest] == 17:
                XTest = sdb3
                YTestHypno = sdbch3
                YTest = sdbc3
                XTestSeq = CodedSequencesdb3
            if normalSubjects[BeginTest] == 18:
                XTest = sdb4
                YTestHypno = sdbch4
                YTest = sdbc4
                XTestSeq = CodedSequencesdb4

            for x in range(21):
                if x < BeginTest and x > Begin:  # train
                    if normalSubjects[x] == 0:
                        YTrainh = np.concatenate((YTrainh, nch1), axis=0)
                        XTrain = np.concatenate((XTrain, n1), axis=0)
                        YTrain = np.concatenate((YTrain, nc1), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence1), axis=0)
                    if normalSubjects[x] == 1:
                        YTrainh = np.concatenate((YTrainh, nch2), axis=0)
                        XTrain = np.concatenate((XTrain, n2), axis=0)
                        YTrain = np.concatenate((YTrain, nc2), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence2), axis=0)
                    if normalSubjects[x] == 2:
                        YTrainh = np.concatenate((YTrainh, nch3), axis=0)
                        XTrain = np.concatenate((XTrain, n3), axis=0)
                        YTrain = np.concatenate((YTrain, nc3), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence3), axis=0)
                    if normalSubjects[x] == 3:
                        YTrainh = np.concatenate((YTrainh, nch4), axis=0)
                        XTrain = np.concatenate((XTrain, n4), axis=0)
                        YTrain = np.concatenate((YTrain, nc4), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence4), axis=0)
                    if normalSubjects[x] == 4:
                        YTrainh = np.concatenate((YTrainh, nch5), axis=0)
                        XTrain = np.concatenate((XTrain, n5), axis=0)
                        YTrain = np.concatenate((YTrain, nc5), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence5), axis=0)
                    if normalSubjects[x] == 5:
                        YTrainh = np.concatenate((YTrainh, nch6), axis=0)
                        XTrain = np.concatenate((XTrain, n6), axis=0)
                        YTrain = np.concatenate((YTrain, nc6), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence6), axis=0)
                    if normalSubjects[x] == 6:
                        YTrainh = np.concatenate((YTrainh, nch7), axis=0)
                        XTrain = np.concatenate((XTrain, n7), axis=0)
                        YTrain = np.concatenate((YTrain, nc7), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence7), axis=0)
                    if normalSubjects[x] == 7:
                        YTrainh = np.concatenate((YTrainh, nch8), axis=0)
                        XTrain = np.concatenate((XTrain, n8), axis=0)
                        YTrain = np.concatenate((YTrain, nc8), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence8), axis=0)
                    if normalSubjects[x] == 8:
                        YTrainh = np.concatenate((YTrainh, nch9), axis=0)
                        XTrain = np.concatenate((XTrain, n9), axis=0)
                        YTrain = np.concatenate((YTrain, nc9), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence9), axis=0)
                    if normalSubjects[x] == 9:
                        YTrainh = np.concatenate((YTrainh, nch10), axis=0)
                        XTrain = np.concatenate((XTrain, n10), axis=0)
                        YTrain = np.concatenate((YTrain, nc10), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence10), axis=0)
                    if normalSubjects[x] == 10:
                        YTrainh = np.concatenate((YTrainh, nch11), axis=0)
                        XTrain = np.concatenate((XTrain, n11), axis=0)
                        YTrain = np.concatenate((YTrain, nc11), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence11), axis=0)
                    if normalSubjects[x] == 11:
                        YTrainh = np.concatenate((YTrainh, nch13), axis=0)
                        XTrain = np.concatenate((XTrain, n13), axis=0)
                        YTrain = np.concatenate((YTrain, nc13), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence13), axis=0)
                    if normalSubjects[x] == 12:
                        YTrainh = np.concatenate((YTrainh, nch14), axis=0)
                        XTrain = np.concatenate((XTrain, n14), axis=0)
                        YTrain = np.concatenate((YTrain, nc14), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence14), axis=0)
                    if normalSubjects[x] == 13:
                        YTrainh = np.concatenate((YTrainh, nch15), axis=0)
                        XTrain = np.concatenate((XTrain, n15), axis=0)
                        YTrain = np.concatenate((YTrain, nc15), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence15), axis=0)
                    if normalSubjects[x] == 14:
                        YTrainh = np.concatenate((YTrainh, nch16), axis=0)
                        XTrain = np.concatenate((XTrain, n16), axis=0)
                        YTrain = np.concatenate((YTrain, nc16), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequence16), axis=0)
                    if normalSubjects[x] == 15:
                        YTrainh = np.concatenate((YTrainh, sdbch1), axis=0)
                        XTrain = np.concatenate((XTrain, sdb1), axis=0)
                        YTrain = np.concatenate((YTrain, sdbc1), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequencesdb1), axis=0)
                    if normalSubjects[x] == 16:
                        YTrainh = np.concatenate((YTrainh, sdbch2), axis=0)
                        XTrain = np.concatenate((XTrain, sdb2), axis=0)
                        YTrain = np.concatenate((YTrain, sdbc2), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequencesdb2), axis=0)
                    if normalSubjects[x] == 17:
                        YTrainh = np.concatenate((YTrainh, sdbch3), axis=0)
                        XTrain = np.concatenate((XTrain, sdb3), axis=0)
                        YTrain = np.concatenate((YTrain, sdbc3), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequencesdb3), axis=0)
                    if normalSubjects[x] == 18:
                        YTrainh = np.concatenate((YTrainh, sdbch4), axis=0)
                        XTrain = np.concatenate((XTrain, sdb4), axis=0)
                        YTrain = np.concatenate((YTrain, sdbc4), axis=0)
                        XTrainSeq = np.concatenate((XTrainSeq, CodedSequencesdb4), axis=0)

                if x <= numberSubjects and x >= BeginTest:  # test
                    if normalSubjects[x] == 1:
                        XTest = np.concatenate((XTest, n1), axis=0)
                        YTest = np.concatenate((YTest, nc1), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch1), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence1), axis=0)
                    if normalSubjects[x] == 2:
                        XTest = np.concatenate((XTest, n2), axis=0)
                        YTest = np.concatenate((YTest, nc2), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch2), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence2), axis=0)
                    if normalSubjects[x] == 3:
                        XTest = np.concatenate((XTest, n3), axis=0)
                        YTest = np.concatenate((YTest, nc3), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch3), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence3), axis=0)
                    if normalSubjects[x] == 4:
                        XTest = np.concatenate((XTest, n4), axis=0)
                        YTest = np.concatenate((YTest, nc4), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch4), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence4), axis=0)
                    if normalSubjects[x] == 5:
                        XTest = np.concatenate((XTest, n5), axis=0)
                        YTest = np.concatenate((YTest, nc5), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch5), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence5), axis=0)
                    if normalSubjects[x] == 6:
                        XTest = np.concatenate((XTest, n6), axis=0)
                        YTest = np.concatenate((YTest, nc6), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch6), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence6), axis=0)
                    if normalSubjects[x] == 7:
                        XTest = np.concatenate((XTest, n7), axis=0)
                        YTest = np.concatenate((YTest, nc7), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch7), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence7), axis=0)
                    if normalSubjects[x] == 8:
                        XTest = np.concatenate((XTest, n8), axis=0)
                        YTest = np.concatenate((YTest, nc8), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch8), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence8), axis=0)
                    if normalSubjects[x] == 9:
                        XTest = np.concatenate((XTest, n9), axis=0)
                        YTest = np.concatenate((YTest, nc9), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch9), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence9), axis=0)
                    if normalSubjects[x] == 10:
                        XTest = np.concatenate((XTest, n10), axis=0)
                        YTest = np.concatenate((YTest, nc10), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch10), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence10), axis=0)
                    if normalSubjects[x] == 11:
                        XTest = np.concatenate((XTest, n11), axis=0)
                        YTest = np.concatenate((YTest, nc11), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch11), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence11), axis=0)
                    if normalSubjects[x] == 13:
                        XTest = np.concatenate((XTest, n13), axis=0)
                        YTest = np.concatenate((YTest, nc13), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch13), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence13), axis=0)
                    if normalSubjects[x] == 14:
                        XTest = np.concatenate((XTest, n14), axis=0)
                        YTest = np.concatenate((YTest, nc14), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch14), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence14), axis=0)
                    if normalSubjects[x] == 15:
                        XTest = np.concatenate((XTest, n15), axis=0)
                        YTest = np.concatenate((YTest, nc15), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch15), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence15), axis=0)
                    if normalSubjects[x] == 16:
                        XTest = np.concatenate((XTest, n16), axis=0)
                        YTest = np.concatenate((YTest, nc16), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, nch16), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequence16), axis=0)
                    if normalSubjects[x] == 17:
                        XTest = np.concatenate((XTest, sdb1), axis=0)
                        YTest = np.concatenate((YTest, sdbc1), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, sdbch1), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequencesdb1), axis=0)
                    if normalSubjects[x] == 18:
                        XTest = np.concatenate((XTest, sdb2), axis=0)
                        YTest = np.concatenate((YTest, sdbc2), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, sdbch2), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequencesdb2), axis=0)
                    if normalSubjects[x] == 19:
                        XTest = np.concatenate((XTest, sdb3), axis=0)
                        YTest = np.concatenate((YTest, sdbc3), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, sdbch3), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequencesdb3), axis=0)
                    if normalSubjects[x] == 20:
                        XTest = np.concatenate((XTest, sdb4), axis=0)
                        YTest = np.concatenate((YTest, sdbc4), axis=0)
                        YTestHypno = np.concatenate((YTestHypno, sdbch4), axis=0)
                        XTestSeq = np.concatenate((XTestSeq, CodedSequencesdb4), axis=0)

        YTest2 = YTest + 1

        index = [range(round(len(YTrain) * PercentageOfData), len(YTrain))]
        YTrain = np.delete(YTrain, index)
        index = [range(round(len(YTest) * PercentageOfData), len(YTest))]
        YTest = np.delete(YTest, index)

        index = [range(round(len(YTrainh) * PercentageOfData), len(YTrainh))]
        YTrainh = np.delete(YTrainh, index)
        index = [range(round(len(YTestHypno) * PercentageOfData), len(YTestHypno))]
        YTestHypno = np.delete(YTestHypno, index)

        index = [range(round(len(XTrain) * PercentageOfData), len(XTrain))]
        XTrain = np.delete(XTrain, index)
        index = [range(round(len(XTest) * PercentageOfData), len(XTest))]
        XTest = np.delete(XTest, index)

        index = [range(round(len(XTrainSeq) * PercentageOfData), len(XTrainSeq))]
        XTrainSeq = np.delete(XTrainSeq, index)
        index = [range(round(len(XTestSeq) * PercentageOfData), len(XTestSeq))]
        XTestSeq = np.delete(XTestSeq, index)

        if PercentageOfData < 1:
            XTrain = np.delete(XTrain, [range(round(len(XTrain) - ((len(XTrain) / 100) - (len(XTrain) // 100)) * 100), len(XTrain))])
            XTest = np.delete(XTest, [range(round(len(XTest) - ((len(XTest) / 100) - (len(XTest) // 100)) * 100), len(XTest))])
            XTrainSeq = np.delete(XTrainSeq, [range(round(len(XTrainSeq) - ((len(XTrainSeq) / 100) - (len(XTrainSeq) // 100)) * 100), len(XTrainSeq))])
            XTestSeq = np.delete(XTestSeq, [range(round(len(XTestSeq) - ((len(XTestSeq) / 100) - (len(XTestSeq) // 100)) * 100), len(XTestSeq))])
        while round(len(XTrain) / 100) < len(YTrain):
            YTrain = np.delete(YTrain, -1)
            YTrainh = np.delete(YTrainh, -1)
        while round(len(XTest) / 100) < len(YTest):
            YTest = np.delete(YTest, -1)
            YTestHypno = np.delete(YTestHypno, -1)

        XTrain = XTrain.reshape(round(len(XTrain) / 100), 100)
        XTest = XTest.reshape(round(len(XTest) / 100), 100)
        XTrainSeq = XTrainSeq.reshape(round(len(XTrainSeq) / 100), 100)
        XTestSeq = XTestSeq.reshape(round(len(XTestSeq) / 100), 100)

        if FeatureBasedModel > 0:
            NewFeatures1 = np.zeros(len(XTrainSeq))  # symbolic dynamics -4 displacement
            NewFeatures2 = np.zeros(len(XTrainSeq))  # symbolic dynamics -3 displacement
            NewFeatures3 = np.zeros(len(XTrainSeq))  # symbolic dynamics -2 displacement
            NewFeatures4 = np.zeros(len(XTrainSeq))  # symbolic dynamics -1 displacement
            NewFeatures5 = np.zeros(len(XTrainSeq))  # symbolic dynamics 0 displacement
            NewFeatures6 = np.zeros(len(XTrainSeq))  # symbolic dynamics 1 displacement
            NewFeatures7 = np.zeros(len(XTrainSeq))  # symbolic dynamics 2 displacement
            NewFeatures8 = np.zeros(len(XTrainSeq))  # symbolic dynamics 3 displacement
            NewFeatures9 = np.zeros(len(XTrainSeq))  # symbolic dynamics 4 displacement
            NewFeatures10 = np.zeros(len(XTrainSeq))  # difference of max and the twp previous max (CAP start indicator)
            # NewFeatures11=np.zeros(len(XTrainSeq)) #highest peak
            # NewFeatures12=np.zeros(len(XTrainSeq)) #std

            NewFeatures13 = np.zeros(len(XTrainSeq))  # PSD delta (0.5–4 Hz)
            NewFeatures14 = np.zeros(len(XTrainSeq))  # PSD theta (4–8 Hz)
            NewFeatures15 = np.zeros(len(XTrainSeq))  # PSD alpha (8–12 Hz)
            NewFeatures16 = np.zeros(len(XTrainSeq))  # PSD sigma(12–15 Hz)
            NewFeatures17 = np.zeros(len(XTrainSeq))  # PSD beta (15–30 Hz) band

            NewFeatures18 = np.zeros(len(XTrainSeq))  # abs aplityde / PSD delta (0.5–4 Hz)
            NewFeatures19 = np.zeros(len(XTrainSeq))  # abs aplityde / PSD theta (4–8 Hz)
            NewFeatures20 = np.zeros(len(XTrainSeq))  # abs aplityde / PSD alpha (8–12 Hz)
            NewFeatures21 = np.zeros(len(XTrainSeq))  # abs aplityde / PSD sigma(12–15 Hz)
            NewFeatures22 = np.zeros(len(XTrainSeq))  # abs aplityde / PSD beta (15–30 Hz) band

            # Define window length
            sf = 100  # 100 Hz sampling freq
            win = sf

            for kk in range(len(XTrainSeq) - 1):
                DiffD = np.diff(XTrainSeq[kk, :])
                for k in range(99):
                    if DiffD[k] == -4:
                        NewFeatures1[kk] = NewFeatures1[kk] + 1
                    elif DiffD[k] == -3:
                        NewFeatures2[kk] = NewFeatures2[kk] + 1
                    elif DiffD[k] == -2:
                        NewFeatures3[kk] = NewFeatures3[kk] + 1
                    elif DiffD[k] == -1:
                        NewFeatures4[kk] = NewFeatures4[kk] + 1
                    elif DiffD[k] == 0:
                        NewFeatures5[kk] = NewFeatures5[kk] + 1
                    elif DiffD[k] == 1:
                        NewFeatures6[kk] = NewFeatures6[kk] + 1
                    elif DiffD[k] == 2:
                        NewFeatures7[kk] = NewFeatures7[kk] + 1
                    elif DiffD[k] == 3:
                        NewFeatures8[kk] = NewFeatures8[kk] + 1
                    elif DiffD[k] == 4:
                        NewFeatures9[kk] = NewFeatures9[kk] + 1
                if kk > 2:
                    NewFeatures10[kk] = np.max(XTrain[kk, :]) - np.mean(np.max(XTrain[kk - 1, :]) + np.max(XTrain[kk - 2, :]))
                #    NewFeatures11[kk]=np.max(XTrainSeq[kk,:])
                #    NewFeatures12[kk]=np.std(DiffD)
                freqs, psd = signal.welch(XTrain[kk, :] * 10, sf, nperseg=win)
                # Define delta lower and upper limits
                low, high = 0.5, 4
                # Find intersecting values in frequency vector
                idx_delta = np.logical_and(freqs >= low, freqs <= high)

                # Frequency resolution
                freq_res = freqs[1] - freqs[0]
                # Compute the absolute power by approximating the area under the curve
                delta_power = simps(psd[idx_delta], dx=freq_res)
                NewFeatures13[kk] = delta_power
                # Define theta lower and upper limits
                low, high = 4, 8
                idx_theta = np.logical_and(freqs >= low, freqs <= high)
                theta_power = simps(psd[idx_theta], dx=freq_res)
                NewFeatures14[kk] = theta_power
                # Define alpha lower and upper limits
                low, high = 8, 12
                idx_alpha = np.logical_and(freqs >= low, freqs <= high)
                alpha_power = simps(psd[idx_alpha], dx=freq_res)
                NewFeatures15[kk] = alpha_power
                # Define sigma lower and upper limits
                low, high = 12, 15
                idx_sigma = np.logical_and(freqs >= low, freqs <= high)
                sigma_power = simps(psd[idx_sigma], dx=freq_res)
                NewFeatures16[kk] = sigma_power
                # Define beta lower and upper limits
                low, high = 15, 30
                idx_beta = np.logical_and(freqs >= low, freqs <= high)
                beta_power = simps(psd[idx_beta], dx=freq_res)
                NewFeatures17[kk] = beta_power
                maximuvalu = sum(abs(XTrain[kk, :]))
                NewFeatures18[kk] = maximuvalu / (1 + delta_power)
                NewFeatures19[kk] = maximuvalu / (1 + theta_power)
                NewFeatures20[kk] = maximuvalu / (1 + alpha_power)
                NewFeatures21[kk] = maximuvalu / (1 + sigma_power)
                NewFeatures22[kk] = maximuvalu / (1 + beta_power)

            NewFeatures1T = np.zeros(len(XTestSeq))
            NewFeatures2T = np.zeros(len(XTestSeq))
            NewFeatures3T = np.zeros(len(XTestSeq))
            NewFeatures4T = np.zeros(len(XTestSeq))
            NewFeatures5T = np.zeros(len(XTestSeq))
            NewFeatures6T = np.zeros(len(XTestSeq))
            NewFeatures7T = np.zeros(len(XTestSeq))
            NewFeatures8T = np.zeros(len(XTestSeq))
            NewFeatures9T = np.zeros(len(XTestSeq))
            NewFeatures10T = np.zeros(len(XTestSeq))
            # NewFeatures11T=np.zeros(len(XTestSeq))
            # NewFeatures12T=np.zeros(len(XTestSeq))

            NewFeatures13T = np.zeros(len(XTestSeq))
            NewFeatures14T = np.zeros(len(XTestSeq))
            NewFeatures15T = np.zeros(len(XTestSeq))
            NewFeatures16T = np.zeros(len(XTestSeq))
            NewFeatures17T = np.zeros(len(XTestSeq))

            NewFeatures18T = np.zeros(len(XTestSeq))
            NewFeatures19T = np.zeros(len(XTestSeq))
            NewFeatures20T = np.zeros(len(XTestSeq))
            NewFeatures21T = np.zeros(len(XTestSeq))
            NewFeatures22T = np.zeros(len(XTestSeq))

            for kk in range(len(XTestSeq) - 1):
                DiffD = np.diff(XTestSeq[kk, :])
                for k in range(99):
                    if DiffD[k] == -4:
                        NewFeatures1T[kk] = NewFeatures1T[kk] + 1
                    elif DiffD[k] == -3:
                        NewFeatures2T[kk] = NewFeatures2T[kk] + 1
                    elif DiffD[k] == -2:
                        NewFeatures3T[kk] = NewFeatures3T[kk] + 1
                    elif DiffD[k] == -1:
                        NewFeatures4T[kk] = NewFeatures4T[kk] + 1
                    elif DiffD[k] == 0:
                        NewFeatures5T[kk] = NewFeatures5T[kk] + 1
                    elif DiffD[k] == 1:
                        NewFeatures6T[kk] = NewFeatures6T[kk] + 1
                    elif DiffD[k] == 2:
                        NewFeatures7T[kk] = NewFeatures7T[kk] + 1
                    elif DiffD[k] == 3:
                        NewFeatures8T[kk] = NewFeatures8T[kk] + 1
                    elif DiffD[k] == 4:
                        NewFeatures9T[kk] = NewFeatures9T[kk] + 1
                if kk > 2:
                    NewFeatures10T[kk] = np.max(XTest[kk, :]) - np.mean(np.max(XTest[kk - 1, :]) + np.max(XTest[kk - 2, :]))
                #    NewFeatures11T[kk]=np.max(XTestSeq[kk,:])
                #    NewFeatures12T[kk]=np.std(DiffD)
                freqs, psd = signal.welch(XTest[kk, :] * 10, sf, nperseg=win)
                # Define delta lower and upper limits
                low, high = 0.5, 4
                # Find intersecting values in frequency vector
                idx_delta = np.logical_and(freqs >= low, freqs <= high)
                # Frequency resolution
                freq_res = freqs[1] - freqs[0]
                # Compute the absolute power by approximating the area under the curve
                delta_power = simps(psd[idx_delta], dx=freq_res)
                NewFeatures13T[kk] = delta_power
                # Define theta lower and upper limits
                low, high = 4, 8
                idx_theta = np.logical_and(freqs >= low, freqs <= high)
                theta_power = simps(psd[idx_theta], dx=freq_res)
                NewFeatures14T[kk] = theta_power
                # Define alpha lower and upper limits
                low, high = 8, 12
                idx_alpha = np.logical_and(freqs >= low, freqs <= high)
                alpha_power = simps(psd[idx_alpha], dx=freq_res)
                NewFeatures15T[kk] = alpha_power
                # Define sigma lower and upper limits
                low, high = 12, 15
                idx_sigma = np.logical_and(freqs >= low, freqs <= high)
                sigma_power = simps(psd[idx_sigma], dx=freq_res)
                NewFeatures16T[kk] = sigma_power
                # Define beta lower and upper limits
                low, high = 15, 30
                idx_beta = np.logical_and(freqs >= low, freqs <= high)
                beta_power = simps(psd[idx_beta], dx=freq_res)
                NewFeatures17T[kk] = beta_power
                maximuvalu = sum(abs(XTest[kk, :]))
                NewFeatures18T[kk] = maximuvalu / (1 + delta_power)
                NewFeatures19T[kk] = maximuvalu / (1 + theta_power)
                NewFeatures20T[kk] = maximuvalu / (1 + alpha_power)
                NewFeatures21T[kk] = maximuvalu / (1 + sigma_power)
                NewFeatures22T[kk] = maximuvalu / (1 + beta_power)

        XTrain2 = XTrain;
        XTest2 = XTest;

        if numberFeatures > 1:
            features = numberFeatures
        else:
            features = numberFeatures + 1

        if Subtype == 1:
            ###### A1
            if forA == 1:
                for i in range(0, len(YTrain), 1):  # just A phase
                    if YTrain[i] > 0:
                        YTrain[i] = 1
                    else:
                        YTrain[i] = 0
                for i in range(0, len(YTest), 1):  # just A phase
                    if YTest[i] > 0:
                        YTest[i] = 1
                    else:
                        YTest[i] = 0

                if FeatureBasedModel > 0:
                    if numberFeatures == 1:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures13[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures13T[:, None], axis=1)
                    elif numberFeatures == 2:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    elif numberFeatures == 3:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    elif numberFeatures == 4:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    elif numberFeatures == 5:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    elif numberFeatures == 6:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    elif numberFeatures == 7:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    elif numberFeatures == 8:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    elif numberFeatures == 9:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    elif numberFeatures == 10:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    elif numberFeatures == 11:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    elif numberFeatures == 12:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    elif numberFeatures == 13:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    elif numberFeatures == 14:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    elif numberFeatures == 15:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    elif numberFeatures == 16:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    elif numberFeatures == 17:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    elif numberFeatures == 18:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                    elif numberFeatures == 19:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    else:
                        XTrain = NewFeatures13
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                        XTest = NewFeatures13T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)


            else:
                for i in range(0, len(YTrain), 1):  # just A phase
                    if YTrain[i] == 1:
                        YTrain[i] = 1
                    else:
                        YTrain[i] = 0
                for i in range(0, len(YTest), 1):  # just A phase
                    if YTest[i] == 1:
                        YTest[i] = 1
                    else:
                        YTest[i] = 0
                if FeatureBasedModel > 0:
                    if numberFeatures == 1:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures22[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures22T[:, None], axis=1)
                    elif numberFeatures == 2:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    elif numberFeatures == 3:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    elif numberFeatures == 4:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    elif numberFeatures == 5:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    elif numberFeatures == 6:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    elif numberFeatures == 7:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                    elif numberFeatures == 8:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    elif numberFeatures == 9:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    elif numberFeatures == 10:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    elif numberFeatures == 11:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    elif numberFeatures == 12:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    elif numberFeatures == 13:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    elif numberFeatures == 14:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    elif numberFeatures == 15:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    elif numberFeatures == 16:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    elif numberFeatures == 17:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)
                    elif numberFeatures == 18:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    elif numberFeatures == 19:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                    else:
                        XTrain = NewFeatures22
                        XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                        XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                        XTest = NewFeatures22T
                        XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                        XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)

        elif Subtype == 2:
            ####### A2
            for i in range(0, len(YTrain), 1):  # just A phase
                if YTrain[i] == 2:
                    YTrain[i] = 1
                else:
                    YTrain[i] = 0
            for i in range(0, len(YTest), 1):  # just A phase
                if YTest[i] == 2:
                    YTest[i] = 1
                else:
                    YTest[i] = 0

            if FeatureBasedModel > 0:
                if numberFeatures == 1:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures13[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures13T[:, None], axis=1)
                elif numberFeatures == 2:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                elif numberFeatures == 3:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                elif numberFeatures == 4:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                elif numberFeatures == 5:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                elif numberFeatures == 6:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                elif numberFeatures == 7:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                elif numberFeatures == 8:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                elif numberFeatures == 9:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                elif numberFeatures == 10:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                elif numberFeatures == 11:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                elif numberFeatures == 12:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                elif numberFeatures == 13:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                elif numberFeatures == 14:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                elif numberFeatures == 15:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                elif numberFeatures == 16:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                elif numberFeatures == 17:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                elif numberFeatures == 18:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                elif numberFeatures == 19:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                else:
                    XTrain = NewFeatures13
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures17[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                    XTest = NewFeatures13T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures17T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)


        else:
            ####### A3
            for i in range(0, len(YTrain), 1):  # just A phase
                if YTrain[i] == 3:
                    YTrain[i] = 1
                else:
                    YTrain[i] = 0
            for i in range(0, len(YTest), 1):  # just A phase
                if YTest[i] == 3:
                    YTest[i] = 1
                else:
                    YTest[i] = 0
            if FeatureBasedModel > 0:
                if numberFeatures == 1:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures17[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures17T[:, None], axis=1)
                elif numberFeatures == 2:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                elif numberFeatures == 3:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                elif numberFeatures == 4:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                elif numberFeatures == 5:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                elif numberFeatures == 6:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                elif numberFeatures == 7:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                elif numberFeatures == 8:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                elif numberFeatures == 9:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                elif numberFeatures == 10:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                elif numberFeatures == 11:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                elif numberFeatures == 12:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                elif numberFeatures == 13:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                elif numberFeatures == 14:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                elif numberFeatures == 15:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                elif numberFeatures == 16:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)
                elif numberFeatures == 17:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                elif numberFeatures == 18:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                elif numberFeatures == 19:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                else:
                    XTrain = NewFeatures17
                    XTrain = np.append(XTrain[:, None], NewFeatures10[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures3[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures4[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures7[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures15[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures22[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures16[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures2[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures8[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures6[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures19[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures9[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures18[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures20[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures5[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures1[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures14[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures13[:, None], axis=1)
                    XTrain = np.append(XTrain, NewFeatures21[:, None], axis=1)
                    XTest = NewFeatures17T
                    XTest = np.append(XTest[:, None], NewFeatures10T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures3T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures4T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures7T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures15T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures22T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures16T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures2T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures8T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures6T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures19T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures9T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures18T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures20T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures5T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures1T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures14T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures13T[:, None], axis=1)
                    XTest = np.append(XTest, NewFeatures21T[:, None], axis=1)

        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(YTrainh),
                                                          YTrain)
        class_weights = {i: class_weights[i] for i in range(2)}

        class_weights2 = class_weight.compute_class_weight('balanced',
                                                           np.unique(YTrainh),
                                                           YTrainh)
        class_weights2 = {i: class_weights2[i] for i in range(2)}

        # exapand dimensions

        for ii in range(timeSteps - 1):  # delete first labels because of the overlapping time step
            YTrain = np.delete(YTrain, 0, 0)
            YTest = np.delete(YTest, 0, 0)
            YTrainh = np.delete(YTrainh, 0, 0)
            YTestHypno = np.delete(YTestHypno, 0, 0)

        encoder = LabelEncoder()
        encoder.fit(YTrain)
        encoded_YTrain = encoder.transform(YTrain)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_YTrain = to_categorical(encoded_YTrain, 2)
        encoder = LabelEncoder()
        encoder.fit(YTest)
        encoded_YTest = encoder.transform(YTest)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_YTest = to_categorical(encoded_YTest, 2)
        YTrain = dummy_YTrain
        YTest = dummy_YTest

        encoder = LabelEncoder()
        encoder.fit(YTrainh)
        encoded_YTrain = encoder.transform(YTrainh)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_YTrain = to_categorical(encoded_YTrain, 2)
        encoder = LabelEncoder()
        encoder.fit(YTestHypno)
        encoded_YTest = encoder.transform(YTestHypno)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_YTest = to_categorical(encoded_YTest, 2)
        YTrainh = dummy_YTrain
        YTestHypno = dummy_YTest

        XTrain = np.repeat(XTrain, timeSteps, axis=0)
        XTest = np.repeat(XTest, timeSteps, axis=0)

        if timeSteps == 5:
            XTrain2 = np.zeros((int(len(XTrain) - timeSteps * 4), features))
            indi = 1
            for k in range(len(XTrain) - timeSteps * 4):
                if indi == 1:
                    XTrain2[k, :] = XTrain[k, :]
                    indi = 2
                elif indi == 2:
                    XTrain2[k, :] = XTrain[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrain2[k, :] = XTrain[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrain2[k, :] = XTrain[k + timeSteps * 3 - 1, :]
                    indi = 5
                else:
                    XTrain2[k, :] = XTrain[k + timeSteps * 4 - 1, :]
                    indi = 1

            XTrain = XTrain2.reshape(round(XTrain2.shape[0] / timeSteps), timeSteps, features)

            XTest2 = np.zeros((int(len(XTest) - timeSteps * 4), features))
            indi = 1
            for k in range(len(XTest) - timeSteps * 4):
                if indi == 1:
                    XTest2[k, :] = XTest[k, :]
                    indi = 2
                elif indi == 2:
                    XTest2[k, :] = XTest[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTest2[k, :] = XTest[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTest2[k, :] = XTest[k + timeSteps * 3 - 1, :]
                    indi = 5
                else:
                    XTest2[k, :] = XTest[k + timeSteps * 4 - 1, :]
                    indi = 1
            XTest = XTest2.reshape(round(XTest2.shape[0] / timeSteps), timeSteps, features)

        elif timeSteps == 15:

            XTrain2 = np.zeros((int(len(XTrain) - timeSteps * 14), features))
            indi = 1
            for k in range(len(XTrain) - timeSteps * 14):
                if indi == 1:
                    XTrain2[k, :] = XTrain[k, :]
                    indi = 2
                elif indi == 2:
                    XTrain2[k, :] = XTrain[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrain2[k, :] = XTrain[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrain2[k, :] = XTrain[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrain2[k, :] = XTrain[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrain2[k, :] = XTrain[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrain2[k, :] = XTrain[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrain2[k, :] = XTrain[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrain2[k, :] = XTrain[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrain2[k, :] = XTrain[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrain2[k, :] = XTrain[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrain2[k, :] = XTrain[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrain2[k, :] = XTrain[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrain2[k, :] = XTrain[k + timeSteps * 13 - 1, :]
                    indi = 15
                else:
                    XTrain2[k, :] = XTrain[k + timeSteps * 14 - 1, :]
                    indi = 1

            XTrain = XTrain2.reshape(round(XTrain2.shape[0] / timeSteps), timeSteps, features)

            XTest2 = np.zeros((int(len(XTest) - timeSteps * 14), features))
            indi = 1
            for k in range(len(XTest) - timeSteps * 14):
                if indi == 1:
                    XTest2[k, :] = XTest[k, :]
                    indi = 2
                elif indi == 2:
                    XTest2[k, :] = XTest[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTest2[k, :] = XTest[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTest2[k, :] = XTest[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTest2[k, :] = XTest[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTest2[k, :] = XTest[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTest2[k, :] = XTest[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTest2[k, :] = XTest[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTest2[k, :] = XTest[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTest2[k, :] = XTest[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTest2[k, :] = XTest[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTest2[k, :] = XTest[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTest2[k, :] = XTest[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTest2[k, :] = XTest[k + timeSteps * 13 - 1, :]
                    indi = 15
                else:
                    XTest2[k, :] = XTest[k + timeSteps * 14 - 1, :]
                    indi = 1
            XTest = XTest2.reshape(round(XTest2.shape[0] / timeSteps), timeSteps, features)


        elif timeSteps == 25:

            XTrain2 = np.zeros((int(len(XTrain) - timeSteps * 24), features))
            indi = 1
            for k in range(len(XTrain) - timeSteps * 24):
                if indi == 1:
                    XTrain2[k, :] = XTrain[k, :]
                    indi = 2
                elif indi == 2:
                    XTrain2[k, :] = XTrain[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrain2[k, :] = XTrain[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrain2[k, :] = XTrain[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrain2[k, :] = XTrain[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrain2[k, :] = XTrain[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrain2[k, :] = XTrain[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrain2[k, :] = XTrain[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrain2[k, :] = XTrain[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrain2[k, :] = XTrain[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrain2[k, :] = XTrain[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrain2[k, :] = XTrain[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrain2[k, :] = XTrain[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrain2[k, :] = XTrain[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTrain2[k, :] = XTrain[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTrain2[k, :] = XTrain[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTrain2[k, :] = XTrain[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTrain2[k, :] = XTrain[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTrain2[k, :] = XTrain[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTrain2[k, :] = XTrain[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTrain2[k, :] = XTrain[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTrain2[k, :] = XTrain[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTrain2[k, :] = XTrain[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTrain2[k, :] = XTrain[k + timeSteps * 23 - 1, :]
                    indi = 25
                else:
                    XTrain2[k, :] = XTrain[k + timeSteps * 24 - 1, :]
                    indi = 1

            XTrain = XTrain2.reshape(round(XTrain2.shape[0] / timeSteps), timeSteps, features)

            XTest2 = np.zeros((int(len(XTest) - timeSteps * 24), features))
            indi = 1
            for k in range(len(XTest) - timeSteps * 24):
                if indi == 1:
                    XTest2[k, :] = XTest[k, :]
                    indi = 2
                elif indi == 2:
                    XTest2[k, :] = XTest[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTest2[k, :] = XTest[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTest2[k, :] = XTest[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTest2[k, :] = XTest[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTest2[k, :] = XTest[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTest2[k, :] = XTest[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTest2[k, :] = XTest[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTest2[k, :] = XTest[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTest2[k, :] = XTest[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTest2[k, :] = XTest[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTest2[k, :] = XTest[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTest2[k, :] = XTest[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTest2[k, :] = XTest[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTest2[k, :] = XTest[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTest2[k, :] = XTest[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTest2[k, :] = XTest[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTest2[k, :] = XTest[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTest2[k, :] = XTest[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTest2[k, :] = XTest[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTest2[k, :] = XTest[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTest2[k, :] = XTest[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTest2[k, :] = XTest[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTest2[k, :] = XTest[k + timeSteps * 23 - 1, :]
                    indi = 25
                else:
                    XTest2[k, :] = XTest[k + timeSteps * 24 - 1, :]
                    indi = 1
            XTest = XTest2.reshape(round(XTest2.shape[0] / timeSteps), timeSteps, features)

        else:

            XTrain2 = np.zeros((int(len(XTrain) - timeSteps * 34), features))
            indi = 1
            for k in range(len(XTrain) - timeSteps * 34):
                if indi == 1:
                    XTrain2[k, :] = XTrain[k, :]
                    indi = 2
                elif indi == 2:
                    XTrain2[k, :] = XTrain[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrain2[k, :] = XTrain[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrain2[k, :] = XTrain[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrain2[k, :] = XTrain[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrain2[k, :] = XTrain[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrain2[k, :] = XTrain[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrain2[k, :] = XTrain[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrain2[k, :] = XTrain[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrain2[k, :] = XTrain[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrain2[k, :] = XTrain[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrain2[k, :] = XTrain[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrain2[k, :] = XTrain[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrain2[k, :] = XTrain[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTrain2[k, :] = XTrain[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTrain2[k, :] = XTrain[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTrain2[k, :] = XTrain[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTrain2[k, :] = XTrain[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTrain2[k, :] = XTrain[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTrain2[k, :] = XTrain[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTrain2[k, :] = XTrain[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTrain2[k, :] = XTrain[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTrain2[k, :] = XTrain[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTrain2[k, :] = XTrain[k + timeSteps * 23 - 1, :]
                    indi = 25
                elif indi == 25:
                    XTrain2[k, :] = XTrain[k + timeSteps * 24 - 1, :]
                    indi = 26
                elif indi == 26:
                    XTrain2[k, :] = XTrain[k + timeSteps * 25 - 1, :]
                    indi = 27
                elif indi == 27:
                    XTrain2[k, :] = XTrain[k + timeSteps * 26 - 1, :]
                    indi = 28
                elif indi == 28:
                    XTrain2[k, :] = XTrain[k + timeSteps * 27 - 1, :]
                    indi = 29
                elif indi == 29:
                    XTrain2[k, :] = XTrain[k + timeSteps * 28 - 1, :]
                    indi = 30
                elif indi == 30:
                    XTrain2[k, :] = XTrain[k + timeSteps * 29 - 1, :]
                    indi = 31
                elif indi == 31:
                    XTrain2[k, :] = XTrain[k + timeSteps * 30 - 1, :]
                    indi = 32
                elif indi == 32:
                    XTrain2[k, :] = XTrain[k + timeSteps * 31 - 1, :]
                    indi = 33
                elif indi == 33:
                    XTrain2[k, :] = XTrain[k + timeSteps * 32 - 1, :]
                    indi = 34
                elif indi == 34:
                    XTrain2[k, :] = XTrain[k + timeSteps * 33 - 1, :]
                    indi = 35
                else:
                    XTrain2[k, :] = XTrain[k + timeSteps * 34 - 1, :]
                    indi = 1

            XTrain = XTrain2.reshape(round(XTrain2.shape[0] / timeSteps), timeSteps, features)

            XTest2 = np.zeros((int(len(XTest) - timeSteps * 34), features))
            indi = 1
            for k in range(len(XTest) - timeSteps * 34):
                if indi == 1:
                    XTest2[k, :] = XTest[k, :]
                    indi = 2
                elif indi == 2:
                    XTest2[k, :] = XTest[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTest2[k, :] = XTest[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTest2[k, :] = XTest[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTest2[k, :] = XTest[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTest2[k, :] = XTest[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTest2[k, :] = XTest[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTest2[k, :] = XTest[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTest2[k, :] = XTest[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTest2[k, :] = XTest[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTest2[k, :] = XTest[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTest2[k, :] = XTest[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTest2[k, :] = XTest[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTest2[k, :] = XTest[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTest2[k, :] = XTest[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTest2[k, :] = XTest[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTest2[k, :] = XTest[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTest2[k, :] = XTest[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTest2[k, :] = XTest[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTest2[k, :] = XTest[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTest2[k, :] = XTest[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTest2[k, :] = XTest[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTest2[k, :] = XTest[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTest2[k, :] = XTest[k + timeSteps * 23 - 1, :]
                    indi = 25
                elif indi == 25:
                    XTest2[k, :] = XTest[k + timeSteps * 24 - 1, :]
                    indi = 26
                elif indi == 26:
                    XTest2[k, :] = XTest[k + timeSteps * 25 - 1, :]
                    indi = 27
                elif indi == 27:
                    XTest2[k, :] = XTest[k + timeSteps * 26 - 1, :]
                    indi = 28
                elif indi == 28:
                    XTest2[k, :] = XTest[k + timeSteps * 27 - 1, :]
                    indi = 29
                elif indi == 29:
                    XTest2[k, :] = XTest[k + timeSteps * 28 - 1, :]
                    indi = 30
                elif indi == 30:
                    XTest2[k, :] = XTest[k + timeSteps * 29 - 1, :]
                    indi = 31
                elif indi == 31:
                    XTest2[k, :] = XTest[k + timeSteps * 30 - 1, :]
                    indi = 32
                elif indi == 32:
                    XTest2[k, :] = XTest[k + timeSteps * 31 - 1, :]
                    indi = 33
                elif indi == 33:
                    XTest2[k, :] = XTest[k + timeSteps * 32 - 1, :]
                    indi = 34
                elif indi == 34:
                    XTest2[k, :] = XTest[k + timeSteps * 33 - 1, :]
                    indi = 35
                else:
                    XTest2[k, :] = XTest[k + timeSteps * 34 - 1, :]
                    indi = 1

            XTest = XTest2.reshape(round(XTest2.shape[0] / timeSteps), timeSteps, features)

        indexN = 0
        for numbHideen in range(numbHideenStart, numbHideenMax + numbHideenStep, numbHideenStep):
            indexG = 0
            stopAnalysis = 0
            for ExaminedLayers in range(0, ExaminedLayersMax + 1, 1):
                if stopAnalysis == 0:
                    for L in range(0, 2, 1):
                        for m in range(0, 4, 1):
                            for ff in range(EpochsWork):

                                print('\n\n Epoch Work: ', ff, ' for m = ', m, ', L = ', L,
                                      ', ExaminedLayers = ', ExaminedLayers, ', numbHideen = ', numbHideen, ', and timeSteps =', timeSteps)

                                if forNREM > 0:
                                    x_train, x_valid, y_train, y_valid = train_test_split(XTrain, YTrainh, test_size=0.33, shuffle=True)
                                else:
                                    x_train, x_valid, y_train, y_valid = train_test_split(XTrain, YTrain, test_size=0.33, shuffle=True)

                                if ExaminedLayers == 0:
                                    model = Sequential()
                                    if L == 0:
                                        model.add(LSTM(numbHideen, input_shape=(XTrain.shape[1], XTrain.shape[2])))
                                    else:
                                        model.add(Bidirectional(LSTM(numbHideen, input_shape=(XTrain.shape[1], XTrain.shape[2]))))
                                else:
                                    model = Sequential()
                                    if L == 0:
                                        model.add(LSTM(numbHideen, input_shape=(XTrain.shape[1], XTrain.shape[2]), return_sequences=True))  # True = many to many
                                    else:
                                        model.add(Bidirectional(LSTM(numbHideen, input_shape=(XTrain.shape[1], XTrain.shape[2]), return_sequences=True)))  # True = many to many
                                for z in range(0, ExaminedLayers, 1):
                                    if z == ExaminedLayers - 1:
                                        if L == 0:
                                            model.add(LSTM(numbHideen))  # True = many to many
                                        else:
                                            model.add(Bidirectional(LSTM(numbHideen)))  # True = many to many
                                    else:
                                        if L == 0:
                                            model.add(LSTM(numbHideen, return_sequences=True))  # True = many to many
                                        else:
                                            model.add(Bidirectional(LSTM(numbHideen, return_sequences=True)))  # True = many to many
                                model.add(Dropout(0.1))
                                if m == 0:
                                    model.add(Dense(np.int(np.floor(numbHideen / 2 + 0.5)), kernel_initializer='normal', activation='relu'))
                                elif m == 1:
                                    model.add(Dense(numbHideen, kernel_initializer='normal', activation='relu'))
                                elif m == 2:
                                    model.add(Dense(numbHideen * 2, kernel_initializer='normal', activation='relu'))
                                model.add(Dense(2, activation='softmax'))

                                import tensorflow as tf


                                class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):

                                    def __init__(self, patienteceValue, valid_data):
                                        super(EarlyStoppingAtMinLoss, self).__init__()
                                        self.patience = patienteceValue
                                        self.best_weights = None
                                        self.validation_data = valid_data

                                    def on_train_begin(self, logs=None):
                                        self.wait = 0
                                        self.stopped_epoch = 0
                                        self.best = 0.2
                                        self._data = []
                                        self.curentAUC = 0.2
                                        print('Train started')

                                    def on_epoch_end(self, epoch, logs=None):
                                        X_val, y_val = self.validation_data[0], self.validation_data[1]
                                        y_predict = np.asarray(model.predict(X_val))

                                        fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.argmax(y_val, axis=1), y_predict[:, 1])
                                        auc_keras = auc(fpr_keras, tpr_keras)
                                        self.curentAUC = auc_keras
                                        current = auc_keras

                                        print('AUC : ', current)

                                        if np.greater(self.curentAUC, self.best + thresholdAphase):  # np.less
                                            print('Update')
                                            self.best = self.curentAUC
                                            self.wait = 0
                                            self.best_weights = self.model.get_weights()
                                        else:
                                            self.wait += 1
                                            if self.wait >= self.patience:
                                                self.stopped_epoch = epoch
                                                self.model.stop_training = True
                                                print('Restoring model weights from the end of the best epoch.')
                                                self.model.set_weights(self.best_weights)

                                    def on_train_end(self, logs=None):
                                        if self.stopped_epoch > 0:
                                            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


                                model.compile(loss='binary_crossentropy',
                                              optimizer='adam',
                                              metrics=[tf.keras.metrics.AUC()])

                                model.fit(x_train, y_train,
                                          batch_size=1000,
                                          epochs=20,  # 10000
                                          validation_data=(x_valid, y_valid),
                                          verbose=1,
                                          class_weight=class_weights,
                                          callbacks=EarlyStoppingAtMinLoss(patienteceValue, (x_valid, y_valid)))

                                print("Testing")
                                proba = model.predict(XTest)
                                if forNREM > 0:
                                    YTestOneLine = np.zeros(len(YTestHypno));
                                    for x in range(len(YTestHypno)):
                                        if YTestHypno[x, 0] == 1:
                                            YTestOneLine[x] = 0
                                        else:
                                            YTestOneLine[x] = 1
                                else:
                                    YTestOneLine = np.zeros(len(YTest));
                                    for x in range(len(YTest)):
                                        if YTest[x, 0] == 1:
                                            YTestOneLine[x] = 0
                                        else:
                                            YTestOneLine[x] = 1

                                predictiony_pred = np.zeros(len(YTestOneLine));
                                for x in range(len(YTestOneLine)):
                                    if proba[x, 0] > 0.5:
                                        predictiony_pred[x] = 0
                                    else:
                                        predictiony_pred[x] = 1

                                tn, fp, fn, tp = confusion_matrix(YTestOneLine, predictiony_pred).ravel()
                                print(classification_report(YTestOneLine, predictiony_pred))
                                accuracy0 = (tp + tn) / (tp + tn + fp + fn)
                                sensitivity0 = tp / (tp + fn)
                                specificity0 = tn / (fp + tn)

                                fpr_keras, tpr_keras, thresholds_keras = roc_curve(YTestOneLine, proba[:, 1])
                                auc_keras = auc(fpr_keras, tpr_keras)
                                print('AUC : ', auc_keras)

                                capPredictedPredicted = predictiony_pred;
                                for k in range(len(capPredictedPredicted) - 1):
                                    if k > 0:
                                        if capPredictedPredicted[k - 1] == 0 and capPredictedPredicted[k] == 1 and capPredictedPredicted[k + 1] == 0:
                                            capPredictedPredicted[k] = 0

                                for k in range(len(capPredictedPredicted) - 1):
                                    if k > 0:
                                        if capPredictedPredicted[k - 1] == 1 and capPredictedPredicted[k] == 0 and capPredictedPredicted[k + 1] == 1:
                                            capPredictedPredicted[k] = 1

                                tn, fp, fn, tp = confusion_matrix(YTestOneLine, capPredictedPredicted).ravel()
                                print(classification_report(YTestOneLine, capPredictedPredicted))
                                accuracy0 = (tp + tn) / (tp + tn + fp + fn)
                                print('Accuracy : ', accuracy0)
                                sensitivity0 = tp / (tp + fn)
                                print('Sensitivity : ', sensitivity0)
                                specificity0 = tn / (fp + tn)
                                print('Specificity : ', specificity0)
                                AccAtEnd[ff] = accuracy0
                                SenAtEnd[ff] = sensitivity0
                                SpeAtEnd[ff] = specificity0
                                AUCAtEnd[ff] = auc_keras

                                # del model, x_train, x_valid, y_train, y_valid, XTrain, YTrain, XTest, YTest, YTestHypno, YTrainh

                            AccAtEndM[m, countM] = np.mean(AccAtEnd)
                            SenAtEndM[m, countM] = np.mean(SenAtEnd)
                            SpeAtEndM[m, countM] = np.mean(SpeAtEnd)
                            AUCAtEndM[m, countM] = np.mean(AUCAtEnd)
                            f = open("AccAtEndM.txt", 'ab')
                            pickle.dump(AccAtEndM, f)
                            f.close()
                            f = open("SenAtEndM.txt", 'ab')
                            pickle.dump(SenAtEndM, f)
                            f.close()
                            f = open("SpeAtEndM.txt", 'ab')
                            pickle.dump(SpeAtEndM, f)
                            f.close()
                            f = open("AUCAtEndM.txt", 'ab')
                            pickle.dump(AUCAtEndM, f)
                            f.close()
                            if AUCmax < np.mean(AUCAtEnd):  # found a better network
                                BestNet[0] = m
                                BestNet[1] = L
                                BestNet[2] = indexG
                                BestNet[3] = indexN
                                BestNet[4] = indexT
                                f = open("BestNet.txt", 'ab')
                                pickle.dump(BestNet, f)
                                f.close()
                                AUCmax = np.mean(AUCAtEnd)

                        countM += 1
                        AccAtEndL[L, countL] = np.max(AccAtEndM[:, countM - 1])
                        SenAtEndL[L, countL] = np.max(SenAtEndM[:, countM - 1])
                        SpeAtEndL[L, countL] = np.max(SpeAtEndM[:, countM - 1])
                        AUCAtEndL[L, countL] = np.max(AUCAtEndM[:, countM - 1])
                        f = open("AccAtEndL.txt", 'ab')
                        pickle.dump(AccAtEndL, f)
                        f.close()
                        f = open("SenAtEndL.txt", 'ab')
                        pickle.dump(SenAtEndL, f)
                        f.close()
                        f = open("SpeAtEndL.txt", 'ab')
                        pickle.dump(SpeAtEndL, f)
                        f.close()
                        f = open("AUCAtEndL.txt", 'ab')
                        pickle.dump(AUCAtEndL, f)
                        f.close()
                        AccAtEndLarg[L, countL] = np.argmax(AccAtEndM[:, countM - 1])
                        SenAtEndLarg[L, countL] = np.argmax(SenAtEndM[:, countM - 1])
                        SpeAtEndLarg[L, countL] = np.argmax(SpeAtEndM[:, countM - 1])
                        AUCAtEndLarg[L, countL] = np.argmax(AUCAtEndM[:, countM - 1])
                        f = open("AccAtEndLarg.txt", 'ab')
                        pickle.dump(AccAtEndLarg, f)
                        f.close()
                        f = open("SenAtEndLarg.txt", 'ab')
                        pickle.dump(SenAtEndLarg, f)
                        f.close()
                        f = open("SpeAtEndLarg.txt", 'ab')
                        pickle.dump(SpeAtEndLarg, f)
                        f.close()
                        f = open("AUCAtEndLarg.txt", 'ab')
                        pickle.dump(AUCAtEndLarg, f)
                        f.close()

                    countL += 1
                    AccAtEndG[indexG, countG] = np.max(AccAtEndL[:, countL - 1])
                    SenAtEndG[indexG, countG] = np.max(SenAtEndL[:, countL - 1])
                    SpeAtEndG[indexG, countG] = np.max(SpeAtEndL[:, countL - 1])
                    AUCAtEndG[indexG, countG] = np.max(AUCAtEndL[:, countL - 1])
                    f = open("AccAtEndG.txt", 'ab')
                    pickle.dump(AccAtEndG, f)
                    f.close()
                    f = open("SenAtEndG.txt", 'ab')
                    pickle.dump(SenAtEndG, f)
                    f.close()
                    f = open("SpeAtEndG.txt", 'ab')
                    pickle.dump(SpeAtEndG, f)
                    f.close()
                    f = open("AUCAtEndG.txt", 'ab')
                    pickle.dump(AUCAtEndG, f)
                    f.close()
                    AccAtEndGarg[indexG, countG] = np.max(AccAtEndL[:, countL - 1])
                    SenAtEndGarg[indexG, countG] = np.max(SenAtEndL[:, countL - 1])
                    SpeAtEndGarg[indexG, countG] = np.max(SpeAtEndL[:, countL - 1])
                    AUCAtEndGarg[indexG, countG] = np.max(AUCAtEndL[:, countL - 1])
                    f = open("AccAtEndGarg.txt", 'ab')
                    pickle.dump(AccAtEndGarg, f)
                    f.close()
                    f = open("SenAtEndGarg.txt", 'ab')
                    pickle.dump(SenAtEndGarg, f)
                    f.close()
                    f = open("SpeAtEndGarg.txt", 'ab')
                    pickle.dump(SpeAtEndGarg, f)
                    f.close()
                    f = open("AUCAtEndGarg.txt", 'ab')
                    pickle.dump(AUCAtEndGarg, f)
                    f.close()
                    if AUCAtEndG[indexG, countG] - AUCAtEndG[indexG - 1, countG] <= thresholdAphase:
                        stopAnalysis = 1
                    indexG += 1

            countG += 1
            AccAtEndN[indexN, countN] = np.max(AccAtEndG[:, countG - 1])
            SenAtEndN[indexN, countN] = np.max(SenAtEndG[:, countG - 1])
            SpeAtEndN[indexN, countN] = np.max(SpeAtEndG[:, countG - 1])
            AUCAtEndN[indexN, countN] = np.max(AUCAtEndG[:, countG - 1])
            f = open("AccAtEndN.txt", 'ab')
            pickle.dump(AccAtEndN, f)
            f.close()
            f = open("SenAtEndN.txt", 'ab')
            pickle.dump(SenAtEndN, f)
            f.close()
            f = open("SpeAtEndN.txt", 'ab')
            pickle.dump(SpeAtEndN, f)
            f.close()
            f = open("AUCAtEndN.txt", 'ab')
            pickle.dump(AUCAtEndN, f)
            f.close()
            AccAtEndNarg[indexN, countN] = np.max(AccAtEndG[:, countG - 1])
            SenAtEndNarg[indexN, countN] = np.max(SenAtEndG[:, countG - 1])
            SpeAtEndNarg[indexN, countN] = np.max(SpeAtEndG[:, countG - 1])
            AUCAtEndNarg[indexN, countN] = np.max(AUCAtEndG[:, countG - 1])
            f = open("AccAtEndNarg.txt", 'ab')
            pickle.dump(AccAtEndNarg, f)
            f.close()
            f = open("SenAtEndNarg.txt", 'ab')
            pickle.dump(SenAtEndNarg, f)
            f.close()
            f = open("SpeAtEndNarg.txt", 'ab')
            pickle.dump(SpeAtEndNarg, f)
            f.close()
            f = open("AUCAtEndNarg.txt", 'ab')
            pickle.dump(AUCAtEndNarg, f)
            f.close()
            indexN += 1

        countN += 1
        AccAtEndT[indexT] = np.max(AccAtEndN[:, countN - 1])
        SenAtEndT[indexT] = np.max(SenAtEndN[:, countN - 1])
        SpeAtEndT[indexT] = np.max(SpeAtEndN[:, countN - 1])
        AUCAtEndT[indexT] = np.max(AUCAtEndN[:, countN - 1])
        f = open("AccAtEndT.txt", 'ab')
        pickle.dump(AccAtEndT, f)
        f.close()
        f = open("SenAtEndT.txt", 'ab')
        pickle.dump(SenAtEndT, f)
        f.close()
        f = open("SpeAtEndT.txt", 'ab')
        pickle.dump(SpeAtEndT, f)
        f.close()
        f = open("AUCAtEndT.txt", 'ab')
        pickle.dump(AUCAtEndT, f)
        f.close()
        AccAtEndTarg[indexT] = np.max(AccAtEndN[:, countN - 1])
        SenAtEndTarg[indexT] = np.max(SenAtEndN[:, countN - 1])
        SpeAtEndTarg[indexT] = np.max(SpeAtEndN[:, countN - 1])
        AUCAtEndTarg[indexT] = np.max(AUCAtEndN[:, countN - 1])
        f = open("AccAtEndTarg.txt", 'ab')
        pickle.dump(AccAtEndTarg, f)
        f.close()
        f = open("SenAtEndTarg.txt", 'ab')
        pickle.dump(SenAtEndTarg, f)
        f.close()
        f = open("SpeAtEndTarg.txt", 'ab')
        pickle.dump(SpeAtEndTarg, f)
        f.close()
        f = open("AUCAtEndTarg.txt", 'ab')
        pickle.dump(AUCAtEndTarg, f)
        f.close()
        indexT += 1

print('Finished')
