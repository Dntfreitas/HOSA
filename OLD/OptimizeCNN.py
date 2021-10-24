import gc
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

#### Control variable

Begin = 0
BeginVal = 9
BeginTest = 10  # 10
numberSubjects = 18
patienteceValue = 1

numberSubjectsN = 14
numberSubjectsSDB = 3

Epochs = 10  # number of iterations: 10
PercentageOfData = 1  # 0.25, 0.5, 0.75, 1
thresholdAphase = 0.01  # 0.01 for A1 or A3, 0.02 for A2
patience = patienteceValue

overlappingStart = 0  # amount of overlapping to start: 0
overlappingMax = 18  # amount of overlapping to finish: the true value is overlappingEnd/2-1
overlappingStep = 2  # step of overlapping
ExaminedLayersMax = 5  # maximum number of LSTM to examine 2, 3, 4, 5, ...
kStart = 4  # start power of 2 for the number of kernels: 4
kMax = 7  # maximum power of 2 for the number of kernels: 7
NStart = 50  # starting value for the number of neurons of the dense layer: 50
NMax = 150  # maximum value for the number of neurons of the dense layer: 150
NStep = 50  # step value for the number of neurons of the dense layer
MULmax = 2  # maximum numbeer of multipliers for the network expanssion

AccAtEnd = np.zeros(Epochs)
SenAtEnd = np.zeros(Epochs)
SpeAtEnd = np.zeros(Epochs)
AUCAtEnd = np.zeros(Epochs)

AccAtEndMul = np.zeros([MULmax, np.int((NMax - NStart) / NStep + 1) *
                        np.int((kMax - kStart) + 1) *
                        np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                        ExaminedLayersMax * 3])
SenAtEndMul = np.zeros([MULmax, np.int((NMax - NStart) / NStep + 1) *
                        np.int((kMax - kStart) + 1) *
                        np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                        ExaminedLayersMax * 3])
SpeAtEndMul = np.zeros([MULmax, np.int((NMax - NStart) / NStep + 1) *
                        np.int((kMax - kStart) + 1) *
                        np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                        ExaminedLayersMax * 3])
AUCAtEndMul = np.zeros([MULmax, np.int((NMax - NStart) / NStep + 1) *
                        np.int((kMax - kStart) + 1) *
                        np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                        ExaminedLayersMax * 3])

AccAtEndA = np.zeros([3, np.int((NMax - NStart) / NStep + 1) *
                      np.int((kMax - kStart) + 1) *
                      np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])
SenAtEndA = np.zeros([3, np.int((NMax - NStart) / NStep + 1) *
                      np.int((kMax - kStart) + 1) *
                      np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])
SpeAtEndA = np.zeros([3, np.int((NMax - NStart) / NStep + 1) *
                      np.int((kMax - kStart) + 1) *
                      np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])
AUCAtEndA = np.zeros([3, np.int((NMax - NStart) / NStep + 1) *
                      np.int((kMax - kStart) + 1) *
                      np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])

AccAtEndN = np.zeros([np.int((NMax - NStart) / NStep + 1), np.int((kMax - kStart) + 1) *
                      np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])
SenAtEndN = np.zeros([np.int((NMax - NStart) / NStep + 1), np.int((kMax - kStart) + 1) *
                      np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])
SpeAtEndN = np.zeros([np.int((NMax - NStart) / NStep + 1), np.int((kMax - kStart) + 1) *
                      np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])
AUCAtEndN = np.zeros([np.int((NMax - NStart) / NStep + 1), np.int((kMax - kStart) + 1) *
                      np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])

AccAtEndK = np.zeros([np.int((kMax - kStart) + 1), np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])
SenAtEndK = np.zeros([np.int((kMax - kStart) + 1), np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])
SpeAtEndK = np.zeros([np.int((kMax - kStart) + 1), np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])
AUCAtEndK = np.zeros([np.int((kMax - kStart) + 1), np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                      ExaminedLayersMax])

AccAtEndO = np.zeros([np.int((overlappingMax - overlappingStart) / overlappingStep + 1), ExaminedLayersMax])
SenAtEndO = np.zeros([np.int((overlappingMax - overlappingStart) / overlappingStep + 1), ExaminedLayersMax])
SpeAtEndO = np.zeros([np.int((overlappingMax - overlappingStart) / overlappingStep + 1), ExaminedLayersMax])
AUCAtEndO = np.zeros([np.int((overlappingMax - overlappingStart) / overlappingStep + 1), ExaminedLayersMax])

AccAtEndG = np.zeros(ExaminedLayersMax)
SenAtEndG = np.zeros(ExaminedLayersMax)
SpeAtEndG = np.zeros(ExaminedLayersMax)
AUCAtEndG = np.zeros(ExaminedLayersMax)

AccAtEndAarg = np.zeros([3, np.int((NMax - NStart) / NStep + 1) *
                         np.int((kMax - kStart) + 1) *
                         np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])
SenAtEndAarg = np.zeros([3, np.int((NMax - NStart) / NStep + 1) *
                         np.int((kMax - kStart) + 1) *
                         np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])
SpeAtEndAarg = np.zeros([3, np.int((NMax - NStart) / NStep + 1) *
                         np.int((kMax - kStart) + 1) *
                         np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])
AUCAtEndAarg = np.zeros([3, np.int((NMax - NStart) / NStep + 1) *
                         np.int((kMax - kStart) + 1) *
                         np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])

AccAtEndNarg = np.zeros([np.int((NMax - NStart) / NStep + 1), np.int((kMax - kStart) + 1) *
                         np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])
SenAtEndNarg = np.zeros([np.int((NMax - NStart) / NStep + 1), np.int((kMax - kStart) + 1) *
                         np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])
SpeAtEndNarg = np.zeros([np.int((NMax - NStart) / NStep + 1), np.int((kMax - kStart) + 1) *
                         np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])
AUCAtEndNarg = np.zeros([np.int((NMax - NStart) / NStep + 1), np.int((kMax - kStart) + 1) *
                         np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])

AccAtEndKarg = np.zeros([np.int((kMax - kStart) + 1), np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])
SenAtEndKarg = np.zeros([np.int((kMax - kStart) + 1), np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])
SpeAtEndKarg = np.zeros([np.int((kMax - kStart) + 1), np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])
AUCAtEndKarg = np.zeros([np.int((kMax - kStart) + 1), np.int((overlappingMax - overlappingStart) / overlappingStep + 1) *
                         ExaminedLayersMax])

AccAtEndOarg = np.zeros([np.int((overlappingMax - overlappingStart) / overlappingStep + 1), ExaminedLayersMax])
SenAtEndOarg = np.zeros([np.int((overlappingMax - overlappingStart) / overlappingStep + 1), ExaminedLayersMax])
SpeAtEndOarg = np.zeros([np.int((overlappingMax - overlappingStart) / overlappingStep + 1), ExaminedLayersMax])
AUCAtEndOarg = np.zeros([np.int((overlappingMax - overlappingStart) / overlappingStep + 1), ExaminedLayersMax])

AccAtEndGarg = np.zeros(ExaminedLayersMax)
SenAtEndGarg = np.zeros(ExaminedLayersMax)
SpeAtEndGarg = np.zeros(ExaminedLayersMax)
AUCAtEndGarg = np.zeros(ExaminedLayersMax)

BestNet = np.zeros(6)  # 0->ExaminedLayers, 1->OverLap, 2->K, 3->N, 4->a, 5->mul

indexG = 0
countM = 0
countA = 0
countN = 0
countK = 0
countO = 0
AUCmax = 0
stopAnalysis = 0
for ExaminedLayers in range(0, ExaminedLayersMax + 1, 1):
    if stopAnalysis == 0:
        indexO = 0
        for OverLap in range(overlappingStart, overlappingMax + overlappingStep, overlappingStep):
            indexK = 0
            for KernelNumb in range(kStart, kMax + 1, 1):
                indexN = 0
                for n in range(NStart, NMax + NStep, NStep):

                    indexMul = 0
                    for mul in range(1, MULmax + 1, 1):
                        for ee in range(Epochs):

                            print('\n\n Epoch: ', ee, ' for mul ', mul,
                                  ', for A ', a, ', for N ', n,
                                  ', for K ', KernelNumb,
                                  ', for O ', OverLap,
                                  ', for G ', ExaminedLayers)

                            tf.keras.backend.clear_session()
                            gc.collect()

                            # TODO: train-test split

                            # TODO: Compute class weights
                            class_weights = class_weight.compute_class_weight('balanced', np.unique(YTrain), YTrain)
                            class_weights = {i: class_weights[i] for i in range(2)}

                            # TODO: to categorical
                            YTrain = to_categorical(YTrain)  # len(XTrain) labels with 1 label per epoch and 1 feature.
                            YTest = to_categorical(YTest)  # len(XTrain) labels with 1 label per epoch and 1 feature.

                            model = Sequential()

                            for z in range(0, ExaminedLayers + 1, 1):
                                if z == 0:
                                    model.add(Conv1D(2 ^ KernelNumb, 2, strides=1, activation='relu', input_shape=(features, 1)))
                                    model.add(MaxPooling1D(pool_size=2, strides=2))
                                    model.add(Dropout(0.1))
                                    KernelNumbprev = 2 ^ KernelNumb
                                else:
                                    model.add(Conv1D(KernelNumbprev * mul, 2, activation='relu'))
                                    model.add(MaxPooling1D(pool_size=2, strides=2))
                                    model.add(Dropout(0.1))
                            model.add(Flatten())
                            model.add(Dense(n, activation='relu'))
                            model.add(Dense(2, activation='softmax'))

                            model.compile(loss='binary_crossentropy',
                                          optimizer='adam',
                                          metrics=[tf.keras.metrics.AUC()])

                            x_train, x_valid, y_train, y_valid = train_test_split(XTrain, YTrain, test_size=0.33, shuffle=True)

                            model.fit(x_train, y_train,
                                      batch_size=1000,
                                      epochs=20,
                                      validation_data=(x_valid, y_valid),
                                      verbose=1,
                                      shuffle=True, class_weight=class_weights, callbacks=EarlyStoppingAtMinLoss(patienteceValue, (x_valid, y_valid)))

                            print("Testing")
                            proba = model.predict(XTest)
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
                            AccAtEnd[ee] = accuracy0
                            SenAtEnd[ee] = sensitivity0
                            SpeAtEnd[ee] = specificity0
                            AUCAtEnd[ee] = auc_keras

                            # del XTrain, YTrain, XTest, YTest, model

                        AccAtEndMul[indexMul, countM] = np.mean(AccAtEnd)
                        SenAtEndMul[indexMul, countM] = np.mean(SenAtEnd)
                        SpeAtEndMul[indexMul, countM] = np.mean(SpeAtEnd)
                        AUCAtEndMul[indexMul, countM] = np.mean(AUCAtEnd)
                        f = open("AccAtEndMul.txt", 'ab')
                        pickle.dump(AccAtEndMul, f)
                        f.close()
                        f = open("SenAtEndMul.txt", 'ab')
                        pickle.dump(SenAtEndMul, f)
                        f.close()
                        f = open("SpeAtEndMul.txt", 'ab')
                        pickle.dump(SpeAtEndMul, f)
                        f.close()
                        f = open("AUCAtEndMul.txt", 'ab')
                        pickle.dump(AUCAtEndMul, f)
                        f.close()
                        if AUCmax < np.mean(AUCAtEnd):  # found a better network
                            BestNet[0] = ExaminedLayers
                            BestNet[1] = OverLap
                            BestNet[2] = KernelNumb
                            BestNet[3] = n
                            BestNet[4] = a
                            BestNet[5] = mul
                            f = open("BestNet.txt", 'ab')
                            pickle.dump(BestNet, f)
                            f.close()
                            AUCmax = np.mean(AUCAtEnd)
                        indexMul += 1

                    countM += 1
                    AccAtEndA[indexA, countA] = np.max(AccAtEndMul[:, countM - 1])
                    SenAtEndA[indexA, countA] = np.max(SenAtEndMul[:, countM - 1])
                    SpeAtEndA[indexA, countA] = np.max(SpeAtEndMul[:, countM - 1])
                    AUCAtEndA[indexA, countA] = np.max(AUCAtEndMul[:, countM - 1])
                    f = open("AccAtEndA.txt", 'ab')
                    pickle.dump(AccAtEndA, f)
                    f.close()
                    f = open("SenAtEndA.txt", 'ab')
                    pickle.dump(SenAtEndA, f)
                    f.close()
                    f = open("SpeAtEndA.txt", 'ab')
                    pickle.dump(SpeAtEndA, f)
                    f.close()
                    f = open("AUCAtEndA.txt", 'ab')
                    pickle.dump(AUCAtEndA, f)
                    f.close()
                    AccAtEndAarg[indexA, countA] = np.argmax(AccAtEndMul[:, countM - 1])
                    SenAtEndAarg[indexA, countA] = np.argmax(SenAtEndMul[:, countM - 1])
                    SpeAtEndAarg[indexA, countA] = np.argmax(SpeAtEndMul[:, countM - 1])
                    AUCAtEndAarg[indexA, countA] = np.argmax(AUCAtEndMul[:, countM - 1])
                    f = open("AccAtEndAarg.txt", 'ab')
                    pickle.dump(AccAtEndAarg, f)
                    f.close()
                    f = open("SenAtEndAarg.txt", 'ab')
                    pickle.dump(SenAtEndAarg, f)
                    f.close()
                    f = open("SpeAtEndAarg.txt", 'ab')
                    pickle.dump(SpeAtEndAarg, f)
                    f.close()
                    f = open("AUCAtEndAarg.txt", 'ab')
                    pickle.dump(AUCAtEndAarg, f)
                    f.close()
                    indexA += 1

                countA += 1
                AccAtEndN[indexN, countN] = np.max(AccAtEndA[:, countA - 1])
                SenAtEndN[indexN, countN] = np.max(SenAtEndA[:, countA - 1])
                SpeAtEndN[indexN, countN] = np.max(SpeAtEndA[:, countA - 1])
                AUCAtEndN[indexN, countN] = np.max(AUCAtEndA[:, countA - 1])
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
                AccAtEndNarg[indexN, countN] = np.argmax(AccAtEndA[:, countA - 1])
                SenAtEndNarg[indexN, countN] = np.argmax(SenAtEndA[:, countA - 1])
                SpeAtEndNarg[indexN, countN] = np.argmax(SpeAtEndA[:, countA - 1])
                AUCAtEndNarg[indexN, countN] = np.argmax(AUCAtEndA[:, countA - 1])
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
            AccAtEndK[indexK, countK] = np.max(AccAtEndN[:, countN - 1])
            SenAtEndK[indexK, countK] = np.max(SenAtEndN[:, countN - 1])
            SpeAtEndK[indexK, countK] = np.max(SpeAtEndN[:, countN - 1])
            AUCAtEndK[indexK, countK] = np.max(AUCAtEndN[:, countN - 1])
            f = open("AccAtEndK.txt", 'ab')
            pickle.dump(AccAtEndK, f)
            f.close()
            f = open("SenAtEndK.txt", 'ab')
            pickle.dump(SenAtEndK, f)
            f.close()
            f = open("SpeAtEndK.txt", 'ab')
            pickle.dump(SpeAtEndK, f)
            f.close()
            f = open("AUCAtEndK.txt", 'ab')
            pickle.dump(AUCAtEndK, f)
            f.close()
            AccAtEndKarg[indexK, countK] = np.argmax(AccAtEndN[:, countN - 1])
            SenAtEndKarg[indexK, countK] = np.argmax(SenAtEndN[:, countN - 1])
            SpeAtEndKarg[indexK, countK] = np.argmax(SpeAtEndN[:, countN - 1])
            AUCAtEndKarg[indexK, countK] = np.argmax(AUCAtEndN[:, countN - 1])
            f = open("AccAtEndKarg.txt", 'ab')
            pickle.dump(AccAtEndKarg, f)
            f.close()
            f = open("SenAtEndKarg.txt", 'ab')
            pickle.dump(SenAtEndKarg, f)
            f.close()
            f = open("SpeAtEndKarg.txt", 'ab')
            pickle.dump(SpeAtEndKarg, f)
            f.close()
            f = open("AUCAtEndKarg.txt", 'ab')
            pickle.dump(AUCAtEndKarg, f)
            f.close()
            indexK += 1

        countK += 1
        AccAtEndO[indexO, countO] = np.max(AccAtEndK[:, countK - 1])
        SenAtEndO[indexO, countO] = np.max(SenAtEndK[:, countK - 1])
        SpeAtEndO[indexO, countO] = np.max(SpeAtEndK[:, countK - 1])
        AUCAtEndO[indexO, countO] = np.max(AUCAtEndK[:, countK - 1])
        f = open("AccAtEndO.txt", 'ab')
        pickle.dump(AccAtEndO, f)
        f.close()
        f = open("SenAtEndO.txt", 'ab')
        pickle.dump(SenAtEndO, f)
        f.close()
        f = open("SpeAtEndO.txt", 'ab')
        pickle.dump(SpeAtEndO, f)
        f.close()
        f = open("AUCAtEndO.txt", 'ab')
        pickle.dump(AUCAtEndO, f)
        f.close()
        AccAtEndOarg[indexO, countO] = np.argmax(AccAtEndK[:, countK - 1])
        SenAtEndOarg[indexO, countO] = np.argmax(SenAtEndK[:, countK - 1])
        SpeAtEndOarg[indexO, countO] = np.argmax(SpeAtEndK[:, countK - 1])
        AUCAtEndOarg[indexO, countO] = np.argmax(AUCAtEndK[:, countK - 1])
        f = open("AccAtEndOarg.txt", 'ab')
        pickle.dump(AccAtEndOarg, f)
        f.close()
        f = open("SenAtEndOarg.txt", 'ab')
        pickle.dump(SenAtEndOarg, f)
        f.close()
        f = open("SpeAtEndOarg.txt", 'ab')
        pickle.dump(SpeAtEndOarg, f)
        f.close()
        f = open("AUCAtEndOarg.txt", 'ab')
        pickle.dump(AUCAtEndOarg, f)
        f.close()
        indexO += 1

    countO += 1
    AccAtEndG[indexG] = np.max(AccAtEndO[:, countO - 1])
    SenAtEndG[indexG] = np.max(SenAtEndO[:, countO - 1])
    SpeAtEndG[indexG] = np.max(SpeAtEndO[:, countO - 1])
    AUCAtEndG[indexG] = np.max(AUCAtEndO[:, countO - 1])
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
    AccAtEndGarg[indexG] = np.argmax(AccAtEndO[:, countO - 1])
    SenAtEndGarg[indexG] = np.argmax(SenAtEndO[:, countO - 1])
    SpeAtEndGarg[indexG] = np.argmax(SpeAtEndO[:, countO - 1])
    AUCAtEndGarg[indexG] = np.argmax(AUCAtEndO[:, countO - 1])
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
    if indexG > 0:
        if AUCAtEndG[indexG] - AUCAtEndG[indexG - 1] <= thresholdAphase:
            stopAnalysis = 1
    indexG += 1

print('Finished')
