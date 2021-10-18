import scipy.io as spio
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import Activation
#from tensorflow.keras.models import model_from_json
#from sklearn.metrics import confusion_matrix
#from sklearn import metrics

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.datasets import imdb
# from keras import regularizers

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix            
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,concatenate
import tensorflow as tf
import gc

import pickle

#### Control variable

Begin=0
BeginVal=9
BeginTest=10 # 10
numberSubjects=18
patienteceValue=1

numberSubjectsN=14
numberSubjectsSDB=3

Epochs=10 # number of iterations: 10
PercentageOfData=1 # 0.25, 0.5, 0.75, 1
thresholdAphase=0.01 # 0.01 for A1 or A3, 0.02 for A2
patience=patienteceValue


overlappingStart=0 # amount of overlapping to start: 0
overlappingMax=18 # amount of overlapping to finish: the true value is overlappingEnd/2-1
overlappingStep=2 # step of overlapping 
ExaminedLayersMax=5 # maximum number of LSTM to examine 2, 3, 4, 5, ...
kStart=4 # start power of 2 for the number of kernels: 4
kMax=7 # maximum power of 2 for the number of kernels: 7
NStart=50 # starting value for the number of neurons of the dense layer: 50
NMax=150 # maximum value for the number of neurons of the dense layer: 150
NStep=50 # step value for the number of neurons of the dense layer
MULmax=2 # maximum numbeer of multipliers for the network expanssion



AccAtEnd=np.zeros(Epochs)
SenAtEnd=np.zeros(Epochs)
SpeAtEnd=np.zeros(Epochs)
AUCAtEnd=np.zeros(Epochs)

AccAtEndMul=np.zeros([MULmax,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax*3])
SenAtEndMul=np.zeros([MULmax,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax*3])
SpeAtEndMul=np.zeros([MULmax,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax*3])
AUCAtEndMul=np.zeros([MULmax,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax*3])

AccAtEndA=np.zeros([3,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SenAtEndA=np.zeros([3,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SpeAtEndA=np.zeros([3,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
AUCAtEndA=np.zeros([3,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])

AccAtEndN=np.zeros([np.int((NMax-NStart)/NStep+1),np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SenAtEndN=np.zeros([np.int((NMax-NStart)/NStep+1),np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SpeAtEndN=np.zeros([np.int((NMax-NStart)/NStep+1),np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
AUCAtEndN=np.zeros([np.int((NMax-NStart)/NStep+1),np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])

AccAtEndK=np.zeros([np.int((kMax-kStart)+1),np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SenAtEndK=np.zeros([np.int((kMax-kStart)+1),np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SpeAtEndK=np.zeros([np.int((kMax-kStart)+1),np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
AUCAtEndK=np.zeros([np.int((kMax-kStart)+1),np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])

AccAtEndO=np.zeros([np.int((overlappingMax-overlappingStart)/overlappingStep+1),ExaminedLayersMax])
SenAtEndO=np.zeros([np.int((overlappingMax-overlappingStart)/overlappingStep+1),ExaminedLayersMax])
SpeAtEndO=np.zeros([np.int((overlappingMax-overlappingStart)/overlappingStep+1),ExaminedLayersMax])
AUCAtEndO=np.zeros([np.int((overlappingMax-overlappingStart)/overlappingStep+1),ExaminedLayersMax])

AccAtEndG=np.zeros(ExaminedLayersMax)
SenAtEndG=np.zeros(ExaminedLayersMax)
SpeAtEndG=np.zeros(ExaminedLayersMax)
AUCAtEndG=np.zeros(ExaminedLayersMax)


AccAtEndAarg=np.zeros([3,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SenAtEndAarg=np.zeros([3,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SpeAtEndAarg=np.zeros([3,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
AUCAtEndAarg=np.zeros([3,np.int((NMax-NStart)/NStep+1)*
                   np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])

AccAtEndNarg=np.zeros([np.int((NMax-NStart)/NStep+1),np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SenAtEndNarg=np.zeros([np.int((NMax-NStart)/NStep+1),np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SpeAtEndNarg=np.zeros([np.int((NMax-NStart)/NStep+1),np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
AUCAtEndNarg=np.zeros([np.int((NMax-NStart)/NStep+1),np.int((kMax-kStart)+1)*
                   np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])

AccAtEndKarg=np.zeros([np.int((kMax-kStart)+1),np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SenAtEndKarg=np.zeros([np.int((kMax-kStart)+1),np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
SpeAtEndKarg=np.zeros([np.int((kMax-kStart)+1),np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])
AUCAtEndKarg=np.zeros([np.int((kMax-kStart)+1),np.int((overlappingMax-overlappingStart)/overlappingStep+1)*
                   ExaminedLayersMax])

AccAtEndOarg=np.zeros([np.int((overlappingMax-overlappingStart)/overlappingStep+1),ExaminedLayersMax])
SenAtEndOarg=np.zeros([np.int((overlappingMax-overlappingStart)/overlappingStep+1),ExaminedLayersMax])
SpeAtEndOarg=np.zeros([np.int((overlappingMax-overlappingStart)/overlappingStep+1),ExaminedLayersMax])
AUCAtEndOarg=np.zeros([np.int((overlappingMax-overlappingStart)/overlappingStep+1),ExaminedLayersMax])

AccAtEndGarg=np.zeros(ExaminedLayersMax)
SenAtEndGarg=np.zeros(ExaminedLayersMax)
SpeAtEndGarg=np.zeros(ExaminedLayersMax)
AUCAtEndGarg=np.zeros(ExaminedLayersMax)


BestNet=np.zeros(6) # 0->ExaminedLayers, 1->OverLap, 2->K, 3->N, 4->a, 5->mul

indexG=0
countM=0
countA=0
countN=0
countK=0
countO=0
AUCmax=0
stopAnalysis=0
for ExaminedLayers in range (0, ExaminedLayersMax + 1, 1):
    if stopAnalysis == 0:
        indexO=0
        for OverLap in range (overlappingStart,overlappingMax + overlappingStep,overlappingStep):
            indexK=0
            for KernelNumb in range (kStart, kMax + 1, 1):
                indexN=0
                for n in range (NStart, NMax  + NStep, NStep):
     
                    if OverLap == 0:
                        Overlapping=0
                    else:
                        Overlapping=OverLap-1
                    
                    if OverLap > 0:
                        overlapingSide=[0,1,2]
                    else:
                        overlapingSide=[1]
                    indexA=0
                    for a in range (0, len(overlapingSide), 1):
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
                        
                        if overlapingSide[a] == 0:
                            for k in range(numberSubjectsN+1):
                                if k < 11:
                                    dataName="n"+str(k+1)
                                    labName="nc"+str(k+1)   
                                    Datadata=eval(dataName)
                                    Lablab=eval(labName)
                                    DatadataV2=np.zeros(((int(len(Datadata)/100)-Overlapping*2),Overlapping*2*100+100))
                                    counting=0
                                    for x in range(0, int((len(Datadata)/100-Overlapping*2-Overlapping)), 1): # Overlapping
                                        DatadataV2[counting,]=Datadata[(x*100):(x*100+100)+Overlapping*100*2]
                                        counting=counting+1
                                    if k == 0:
                                        n1=DatadataV2
                                        nc1=nc1[0:len(nc1)-Overlapping*2]
                                    elif k == 1:
                                        n2=DatadataV2
                                        nc2=nc2[0:len(nc2)-Overlapping*2]
                                    elif k == 2:
                                        n3=DatadataV2
                                        nc3=nc3[0:len(nc3)-Overlapping*2]
                                    elif k == 3:
                                        n4=DatadataV2
                                        nc4=nc4[0:len(nc4)-Overlapping*2]
                                    elif k == 4:
                                        n5=DatadataV2
                                        nc5=nc5[0:len(nc5)-Overlapping*2]
                                    elif k == 5:
                                        n6=DatadataV2
                                        nc6=nc6[0:len(nc6)-Overlapping*2]
                                    elif k == 6:
                                        n7=DatadataV2
                                        nc7=nc7[0:len(nc7)-Overlapping*2]
                                    elif k == 7:
                                        n8=DatadataV2
                                        nc8=nc8[0:len(nc8)-Overlapping*2]
                                    elif k == 8:
                                        n9=DatadataV2
                                        nc9=nc9[0:len(nc9)-Overlapping*2]
                                    elif k == 9:
                                        n10=DatadataV2
                                        nc10=nc10[0:len(nc10)-Overlapping*2]
                                    else:
                                        n11=DatadataV2
                                        nc11=nc11[0:len(nc11)-Overlapping*2]
                                else:
                                    dataName="n"+str(k+2)
                                    labName="nc"+str(k+2)   
                                    Datadata=eval(dataName)
                                    Lablab=eval(labName)
                                    DatadataV2=np.zeros(((int(len(Datadata)/100)-Overlapping*2),Overlapping*2*100+100))
                                    counting=0
                                    for x in range(0, int((len(Datadata)/100-Overlapping*2-Overlapping)), 1): # Overlapping
                                        DatadataV2[counting,]=Datadata[(x*100):(x*100+100)+Overlapping*100*2]
                                        counting=counting+1
                                    if k == 11:
                                        n13=DatadataV2
                                        nc13=nc13[0:len(nc13)-Overlapping*2]
                                    elif k == 12:
                                        n14=DatadataV2
                                        nc14=nc14[0:len(nc14)-Overlapping*2]
                                    elif k == 13:
                                        n15=DatadataV2
                                        nc15=nc15[0:len(nc15)-Overlapping*2]
                                    else:
                                        n16=DatadataV2
                                        nc16=nc16[0:len(nc16)-Overlapping*2]
                            for k in range(numberSubjectsSDB+1):
                                dataName="sdb"+str(k+1)
                                labName="sdbc"+str(k+1)   
                                Datadata=eval(dataName)
                                Lablab=eval(labName)
                                DatadataV2=np.zeros(((int(len(Datadata)/100)-Overlapping*2),Overlapping*2*100+100))
                                counting=0
                                for x in range(0, int((len(Datadata)/100-Overlapping*2-Overlapping)), 1): # Overlapping
                                    DatadataV2[counting,]=Datadata[(x*100):(x*100+100)+Overlapping*100*2]
                                    counting=counting+1
                                if k == 0:
                                    sdb1=DatadataV2
                                    sdbc1=sdbc1[0:len(sdbc1)-Overlapping*2]
                                    # a=np.where(~n1.any(axis=1))[0] # find all zeros windows
                                elif k == 1:
                                    sdb2=DatadataV2
                                    sdbc2=sdbc2[0:len(sdbc2)-Overlapping*2]
                                elif k == 2:
                                    sdb3=DatadataV2
                                    sdbc3=sdbc3[0:len(sdbc3)-Overlapping*2]
                                else:
                                    sdb4=DatadataV2
                                    sdbc4=sdbc4[0:len(sdbc4)-Overlapping*2]                 
                                
                        elif overlapingSide[a] == 1:    
                            for k in range(numberSubjectsN+1):
                                if k < 11:
                                    dataName="n"+str(k+1)
                                    labName="nc"+str(k+1)   
                                    Datadata=eval(dataName)
                                    Lablab=eval(labName)
                                    DatadataV2=np.zeros(((int(len(Datadata)/100)-Overlapping*2),Overlapping*2*100+100))
                                    counting=0
                                    for x in range(Overlapping, int((len(Datadata)/100-Overlapping*2)), 1): # Overlapping
                                        DatadataV2[counting,]=Datadata[(x*100)-Overlapping*100:(x*100+100)+Overlapping*100]
                                        counting=counting+1
                                    if k == 0:
                                        n1=DatadataV2
                                        nc1=nc1[Overlapping:len(nc1)-Overlapping]
                                        # a=np.where(~n1.any(axis=1))[0] # find all zeros windows
                                    elif k == 1:
                                        n2=DatadataV2
                                        nc2=nc2[Overlapping:len(nc2)-Overlapping]
                                    elif k == 2:
                                        n3=DatadataV2
                                        nc3=nc3[Overlapping:len(nc3)-Overlapping]
                                    elif k == 3:
                                        n4=DatadataV2
                                        nc4=nc4[Overlapping:len(nc4)-Overlapping]
                                    elif k == 4:
                                        n5=DatadataV2
                                        nc5=nc5[Overlapping:len(nc5)-Overlapping]
                                    elif k == 5:
                                        n6=DatadataV2
                                        nc6=nc6[Overlapping:len(nc6)-Overlapping]
                                    elif k == 6:
                                        n7=DatadataV2
                                        nc7=nc7[Overlapping:len(nc7)-Overlapping]
                                    elif k == 7:
                                        n8=DatadataV2
                                        nc8=nc8[Overlapping:len(nc8)-Overlapping]
                                    elif k == 8:
                                        n9=DatadataV2
                                        nc9=nc9[Overlapping:len(nc9)-Overlapping]
                                    elif k == 9:
                                        n10=DatadataV2
                                        nc10=nc10[Overlapping:len(nc10)-Overlapping]
                                    else:
                                        n11=DatadataV2
                                        nc11=nc11[Overlapping:len(nc11)-Overlapping]
                                else:
                                    dataName="n"+str(k+2)
                                    labName="nc"+str(k+2)   
                                    Datadata=eval(dataName)
                                    Lablab=eval(labName)
                                    DatadataV2=np.zeros(((int(len(Datadata)/100)-Overlapping*2),Overlapping*2*100+100))
                                    counting=0
                                    for x in range(Overlapping, int((len(Datadata)/100-Overlapping*2)), 1): # Overlapping
                                        DatadataV2[counting,]=Datadata[(x*100)-Overlapping*100:(x*100+100)+Overlapping*100]
                                        counting=counting+1
                                    if k == 11:
                                        n13=DatadataV2
                                        nc13=nc13[Overlapping:len(nc13)-Overlapping]
                                    elif k == 12:
                                        n14=DatadataV2
                                        nc14=nc14[Overlapping:len(nc14)-Overlapping]
                                    elif k == 13:
                                        n15=DatadataV2
                                        nc15=nc15[Overlapping:len(nc15)-Overlapping]
                                    else:
                                        n16=DatadataV2
                                        nc16=nc16[Overlapping:len(nc16)-Overlapping]
                            for k in range(numberSubjectsSDB+1):
                                dataName="sdb"+str(k+1)
                                labName="sdbc"+str(k+1)   
                                Datadata=eval(dataName)
                                Lablab=eval(labName)
                                DatadataV2=np.zeros(((int(len(Datadata)/100)-Overlapping*2),Overlapping*2*100+100))
                                counting=0
                                for x in range(Overlapping, int((len(Datadata)/100-Overlapping*2)), 1): # Overlapping
                                    DatadataV2[counting,]=Datadata[(x*100)-Overlapping*100:(x*100+100)+Overlapping*100]
                                    counting=counting+1
                                if k == 0:
                                    sdb1=DatadataV2
                                    sdbc1=sdbc1[Overlapping:len(sdbc1)-Overlapping]
                                    # a=np.where(~n1.any(axis=1))[0] # find all zeros windows
                                elif k == 1:
                                    sdb2=DatadataV2
                                    sdbc2=sdbc2[Overlapping:len(sdbc2)-Overlapping]
                                elif k == 2:
                                    sdb3=DatadataV2
                                    sdbc3=sdbc3[Overlapping:len(sdbc3)-Overlapping]
                                else:
                                    sdb4=DatadataV2
                                    sdbc4=sdbc4[Overlapping:len(sdbc4)-Overlapping]  
                        else:
                            for k in range(numberSubjectsN+1):
                                if k < 11:
                                    dataName="n"+str(k+1)
                                    labName="nc"+str(k+1)   
                                    Datadata=eval(dataName)
                                    Lablab=eval(labName)
                                    DatadataV2=np.zeros(((int(len(Datadata)/100)-Overlapping*2),Overlapping*2*100+100))
                                    counting=0
                                    for x in range(0, int((len(Datadata)/100-Overlapping*2-Overlapping)), 1): # Overlapping
                                        DatadataV2[counting,]=Datadata[(x*100):(x*100+100)+Overlapping*100*2]
                                        counting=counting+1
                                    if k == 0:
                                        n1=DatadataV2
                                        nc1=nc1[Overlapping*2:len(nc1)]
                                        # a=np.where(~n1.any(axis=1))[0] # find all zeros windows
                                    elif k == 1:
                                        n2=DatadataV2
                                        nc2=nc2[Overlapping*2:len(nc2)]
                                    elif k == 2:
                                        n3=DatadataV2
                                        nc3=nc3[Overlapping*2:len(nc3)]
                                    elif k == 3:
                                        n4=DatadataV2
                                        nc4=nc4[Overlapping*2:len(nc4)]
                                    elif k == 4:
                                        n5=DatadataV2
                                        nc5=nc5[Overlapping*2:len(nc5)]
                                    elif k == 5:
                                        n6=DatadataV2
                                        nc6=nc6[Overlapping*2:len(nc6)]
                                    elif k == 6:
                                        n7=DatadataV2
                                        nc7=nc7[Overlapping*2:len(nc7)]
                                    elif k == 7:
                                        n8=DatadataV2
                                        nc8=nc8[Overlapping*2:len(nc8)]
                                    elif k == 8:
                                        n9=DatadataV2
                                        nc9=nc9[Overlapping*2:len(nc9)]
                                    elif k == 9:
                                        n10=DatadataV2
                                        nc10=nc10[Overlapping*2:len(nc10)]
                                    else:
                                        n11=DatadataV2
                                        nc11=nc11[Overlapping*2:len(nc11)]
                                else:
                                    dataName="n"+str(k+2)
                                    labName="nc"+str(k+2)   
                                    Datadata=eval(dataName)
                                    Lablab=eval(labName)
                                    DatadataV2=np.zeros(((int(len(Datadata)/100)-Overlapping*2),Overlapping*2*100+100))
                                    counting=0
                                    for x in range(0, int((len(Datadata)/100-Overlapping*2-Overlapping)), 1): # Overlapping
                                        DatadataV2[counting,]=Datadata[(x*100):(x*100+100)+Overlapping*100*2]
                                        counting=counting+1
                                    if k == 11:
                                        n13=DatadataV2
                                        nc13=nc13[Overlapping*2:len(nc13)]
                                    elif k == 12:
                                        n14=DatadataV2
                                        nc14=nc14[Overlapping*2:len(nc14)]
                                    elif k == 13:
                                        n15=DatadataV2
                                        nc15=nc15[Overlapping*2:len(nc15)]
                                    else:
                                        n16=DatadataV2
                                        nc16=nc16[Overlapping*2:len(nc16)]
                           
                            for k in range(numberSubjectsSDB+1):
                                dataName="sdb"+str(k+1)
                                labName="sdbc"+str(k+1)   
                                Datadata=eval(dataName)
                                Lablab=eval(labName)
                                DatadataV2=np.zeros(((int(len(Datadata)/100)-Overlapping*2),Overlapping*2*100+100))
                                counting=0
                                for x in range(0, int((len(Datadata)/100-Overlapping*2-Overlapping)), 1): # Overlapping
                                    DatadataV2[counting,]=Datadata[(x*100):(x*100+100)+Overlapping*100*2]
                                if k == 0:
                                    sdb1=DatadataV2
                                    sdbc1=sdbc1[Overlapping*2:len(sdbc1)]
                                elif k == 1:
                                    sdb2=DatadataV2
                                    sdbc2=sdbc2[Overlapping*2:len(sdbc2)]
                                elif k == 2:
                                    sdb3=DatadataV2
                                    sdbc3=sdbc3[Overlapping*2:len(sdbc3)]
                                else:
                                    sdb4=DatadataV2
                                    sdbc4=sdbc4[Overlapping*2:len(sdbc4)]              
                        indexMul=0                            
                        for mul in range (1,MULmax+1,1):
                            for ee in range (Epochs):
                                
                                print ('\n\n Epoch: ', ee, ' for mul ', mul,
                                       ', for A ', a, ', for N ', n,
                                       ', for K ', KernelNumb,
                                       ', for O ', OverLap,
                                       ', for G ', ExaminedLayers)
    
                                
                                tf.keras.backend.clear_session()
                                gc.collect()
                            
                                normalSubjects = np.random.permutation(19) # choose subjects order
                                
                                XTrain=[];
                                XTest=[];
                                YTrain=[];
                                YTest=[];        
                                
                                if normalSubjects[Begin] == 0:
                                    XTrain=n1
                                    YTrain=nc1
                                if normalSubjects[Begin] == 1:
                                    XTrain=n2
                                    YTrain=nc2
                                if normalSubjects[Begin] == 2:
                                    XTrain=n3
                                    YTrain=nc3
                                if normalSubjects[Begin] == 3:
                                    XTrain=n4
                                    YTrain=nc4
                                if normalSubjects[Begin] == 4:
                                    XTrain=n5
                                    YTrain=nc5
                                if normalSubjects[Begin] == 5:
                                    XTrain=n6
                                    YTrain=nc6
                                if normalSubjects[Begin] == 6:
                                    XTrain=n7
                                    YTrain=nc7
                                if normalSubjects[Begin] == 7:
                                    XTrain=n8
                                    YTrain=nc8
                                if normalSubjects[Begin] == 8:
                                    XTrain=n9
                                    YTrain=nc9
                                if normalSubjects[Begin] == 9:
                                    XTrain=n10
                                    YTrain=nc10
                                if normalSubjects[Begin] == 10:
                                    XTrain=n11
                                    YTrain=nc11
                                if normalSubjects[Begin] == 11:
                                    XTrain=n13
                                    YTrain=nc13
                                if normalSubjects[Begin] == 12:
                                    XTrain=n14
                                    YTrain=nc14
                                if normalSubjects[Begin] == 13:
                                    XTrain=n15
                                    YTrain=nc15
                                if normalSubjects[Begin] == 14:
                                    XTrain=n16
                                    YTrain=nc16
                                if normalSubjects[Begin] == 15:
                                    XTrain=sdb1
                                    YTrain=sdbc1
                                if normalSubjects[Begin] == 16:
                                    XTrain=sdb2
                                    YTrain=sdbc2
                                if normalSubjects[Begin] == 17:
                                    XTrain=sdb3
                                    YTrain=sdbc3
                                if normalSubjects[Begin] == 18:
                                    XTrain=sdb4
                                    YTrain=sdbc4
                                    
                                if normalSubjects[BeginTest] == 0:
                                    XTest=n1
                                    YTest=nc1
                                if normalSubjects[BeginTest] == 1:
                                    XTest=n2
                                    YTest=nc2
                                if normalSubjects[BeginTest] == 2:
                                    XTest=n3
                                    YTest=nc3
                                if normalSubjects[BeginTest] == 3:
                                    XTest=n4
                                    YTest=nc4
                                if normalSubjects[BeginTest] == 4:
                                    XTest=n5
                                    YTest=nc5
                                if normalSubjects[BeginTest] == 5:
                                    XTest=n6
                                    YTest=nc6
                                if normalSubjects[BeginTest] == 6:
                                    XTest=n7
                                    YTest=nc7
                                if normalSubjects[BeginTest] == 7:
                                    XTest=n8
                                    YTest=nc8
                                if normalSubjects[BeginTest] == 8:
                                    XTest=n9
                                    YTest=nc9
                                if normalSubjects[BeginTest] == 9:
                                    XTest=n10
                                    YTest=nc10
                                if normalSubjects[BeginTest] == 10:
                                    XTest=n11
                                    YTest=nc11
                                if normalSubjects[BeginTest] == 11:
                                    XTest=n13
                                    YTest=nc13
                                if normalSubjects[BeginTest] == 12:
                                    XTest=n14
                                    YTest=nc14
                                if normalSubjects[BeginTest] == 13:
                                    XTest=n15
                                    YTest=nc15
                                if normalSubjects[BeginTest] == 14:
                                    XTest=n16
                                    YTest=nc16
                                if normalSubjects[BeginTest] == 15:
                                    XTest=sdb1
                                    YTest=sdbc1
                                if normalSubjects[BeginTest] == 16:
                                    XTest=sdb2
                                    YTest=sdbc2
                                if normalSubjects[BeginTest] == 17:
                                    XTest=sdb3
                                    YTest=sdbc3
                                if normalSubjects[BeginTest] == 18:
                                    XTest=sdb4
                                    YTest=sdbc4
                                
                                for x in range(20):
                                    if x < BeginTest and x > Begin : # train
                                        if normalSubjects[x] == 0:
                                            XTrain=np.concatenate((XTrain, n1), axis=0)
                                            YTrain=np.concatenate((YTrain, nc1), axis=0)
                                        if normalSubjects[x] == 1:
                                            XTrain=np.concatenate((XTrain, n2), axis=0)
                                            YTrain=np.concatenate((YTrain, nc2), axis=0)
                                        if normalSubjects[x] == 2:
                                            XTrain=np.concatenate((XTrain, n3), axis=0)
                                            YTrain=np.concatenate((YTrain, nc3), axis=0)
                                        if normalSubjects[x] == 3:
                                            XTrain=np.concatenate((XTrain, n4), axis=0)
                                            YTrain=np.concatenate((YTrain, nc4), axis=0)
                                        if normalSubjects[x] == 4:
                                            XTrain=np.concatenate((XTrain, n5), axis=0)
                                            YTrain=np.concatenate((YTrain, nc5), axis=0)
                                        if normalSubjects[x] == 5:
                                            XTrain=np.concatenate((XTrain, n6), axis=0)
                                            YTrain=np.concatenate((YTrain, nc6), axis=0)
                                        if normalSubjects[x] == 6:
                                            XTrain=np.concatenate((XTrain, n7), axis=0)
                                            YTrain=np.concatenate((YTrain, nc7), axis=0)
                                        if normalSubjects[x] == 7:
                                            XTrain=np.concatenate((XTrain, n8), axis=0)
                                            YTrain=np.concatenate((YTrain, nc8), axis=0)
                                        if normalSubjects[x] == 8:
                                            XTrain=np.concatenate((XTrain, n9), axis=0)
                                            YTrain=np.concatenate((YTrain, nc9), axis=0)
                                        if normalSubjects[x] == 9:
                                            XTrain=np.concatenate((XTrain, n10), axis=0)
                                            YTrain=np.concatenate((YTrain, nc10), axis=0)
                                        if normalSubjects[x] == 10:
                                            XTrain=np.concatenate((XTrain, n11), axis=0)
                                            YTrain=np.concatenate((YTrain, nc11), axis=0)
                                        if normalSubjects[x] == 11:
                                            XTrain=np.concatenate((XTrain, n13), axis=0)
                                            YTrain=np.concatenate((YTrain, nc13), axis=0)
                                        if normalSubjects[x] == 12:
                                            XTrain=np.concatenate((XTrain, n14), axis=0)
                                            YTrain=np.concatenate((YTrain, nc14), axis=0)
                                        if normalSubjects[x] == 13:
                                            XTrain=np.concatenate((XTrain, n15), axis=0)
                                            YTrain=np.concatenate((YTrain, nc15), axis=0)
                                        if normalSubjects[x] == 14:
                                            XTrain=np.concatenate((XTrain, n16), axis=0)
                                            YTrain=np.concatenate((YTrain, nc16), axis=0)               
                                        if normalSubjects[x] == 15:
                                            XTrain=np.concatenate((XTrain, sdb1), axis=0)
                                            YTrain=np.concatenate((YTrain, sdbc1), axis=0)
                                        if normalSubjects[x] == 16:
                                            XTrain=np.concatenate((XTrain, sdb2), axis=0)
                                            YTrain=np.concatenate((YTrain, sdbc2), axis=0)
                                        if normalSubjects[x] == 17:
                                            XTrain=np.concatenate((XTrain, sdb3), axis=0)
                                            YTrain=np.concatenate((YTrain, sdbc3), axis=0)
                                        if normalSubjects[x] == 18:
                                            XTrain=np.concatenate((XTrain, sdb4), axis=0)
                                            YTrain=np.concatenate((YTrain, sdbc4), axis=0)
                            
                                    if x <= numberSubjects and x >= BeginTest: # test
                                        if normalSubjects[x] == 0:
                                            XTest=np.concatenate((XTest, n1), axis=0)
                                            YTest=np.concatenate((YTest, nc1), axis=0)
                                        if normalSubjects[x] == 1:
                                            XTest=np.concatenate((XTest, n2), axis=0)
                                            YTest=np.concatenate((YTest, nc2), axis=0)
                                        if normalSubjects[x] == 2:
                                            XTest=np.concatenate((XTest, n3), axis=0)
                                            YTest=np.concatenate((YTest, nc3), axis=0)
                                        if normalSubjects[x] == 3:
                                            XTest=np.concatenate((XTest, n4), axis=0)
                                            YTest=np.concatenate((YTest, nc4), axis=0)
                                        if normalSubjects[x] == 4:
                                            XTest=np.concatenate((XTest, n5), axis=0)
                                            YTest=np.concatenate((YTest, nc5), axis=0)
                                        if normalSubjects[x] == 5:
                                            XTest=np.concatenate((XTest, n6), axis=0)
                                            YTest=np.concatenate((YTest, nc6), axis=0)
                                        if normalSubjects[x] == 6:
                                            XTest=np.concatenate((XTest, n7), axis=0)
                                            YTest=np.concatenate((YTest, nc7), axis=0)
                                        if normalSubjects[x] == 7:
                                            XTest=np.concatenate((XTest, n8), axis=0)
                                            YTest=np.concatenate((YTest, nc8), axis=0)
                                        if normalSubjects[x] == 8:
                                            XTest=np.concatenate((XTest, n9), axis=0)
                                            YTest=np.concatenate((YTest, nc9), axis=0)
                                        if normalSubjects[x] == 9:
                                            XTest=np.concatenate((XTest, n10), axis=0)
                                            YTest=np.concatenate((YTest, nc10), axis=0)
                                        if normalSubjects[x] == 10:
                                            XTest=np.concatenate((XTest, n11), axis=0)
                                            YTest=np.concatenate((YTest, nc11), axis=0)
                                        if normalSubjects[x] == 11:
                                            XTest=np.concatenate((XTest, n13), axis=0)
                                            YTest=np.concatenate((YTest, nc13), axis=0)
                                        if normalSubjects[x] == 12:
                                            XTest=np.concatenate((XTest, n14), axis=0)
                                            YTest=np.concatenate((YTest, nc14), axis=0)
                                        if normalSubjects[x] == 13:
                                            XTest=np.concatenate((XTest, n15), axis=0)
                                            YTest=np.concatenate((YTest, nc15), axis=0)
                                        if normalSubjects[x] == 14:
                                            XTest=np.concatenate((XTest, n16), axis=0)
                                            YTest=np.concatenate((YTest, nc16), axis=0) 
                                        if normalSubjects[x] == 15:
                                            XTest=np.concatenate((XTest, sdb1), axis=0)
                                            YTest=np.concatenate((YTest, sdbc1), axis=0) 
                                        if normalSubjects[x] == 16:
                                            XTest=np.concatenate((XTest, sdb2), axis=0)
                                            YTest=np.concatenate((YTest, sdbc2), axis=0) 
                                        if normalSubjects[x] == 17:
                                            XTest=np.concatenate((XTest, sdb3), axis=0)
                                            YTest=np.concatenate((YTest, sdbc3), axis=0) 
                                        if normalSubjects[x] == 18:
                                            XTest=np.concatenate((XTest, sdb4), axis=0)
                                            YTest=np.concatenate((YTest, sdbc4), axis=0) 
                                            
                                
                                index = [range(round(len(YTrain)*PercentageOfData),len(YTrain))]
                                YTrain = np.delete(YTrain, index)
                                index = [range(round(len(YTest)*PercentageOfData),len(YTest))]
                                YTest = np.delete(YTest, index)
                                
                            
                                index = [range(round(len(XTrain)*PercentageOfData),len(XTrain))]
                                XTrain = np.delete(XTrain, index,0)
                                index = [range(round(len(XTest)*PercentageOfData),len(XTest))]
                                XTest = np.delete(XTest, index,0)
                                
                                
                                if PercentageOfData < 1:
                                    XTrain = np.delete(XTrain, [range(round(len(XTrain)-((len(XTrain)/100)-(len(XTrain)//100))*100),len(XTrain))],0)
                                    XTest = np.delete(XTest, [range(round(len(XTest)-((len(XTest)/100)-(len(XTest)//100))*100),len(XTest))],0)
                                while len(XTrain) < len (YTrain):
                                    YTrain = np.delete(YTrain, -1)
                                while len(XTest) < len (YTest):
                                    YTest = np.delete(YTest, -1)
                        
                                for i in range(0,len(YTrain),1): # just A phase
                                    if YTrain[i]>0:
                                        YTrain[i]=1
                                    else:
                                        YTrain[i]=0
                                for i in range(0,len(YTest),1): # just A phase
                                    if YTest[i]>0:
                                        YTest[i]=1
                                    else:
                                        YTest[i]=0
                        
                            
                                class_weights = class_weight.compute_class_weight('balanced',
                                                                                  np.unique(YTrain),
                                                                                  YTrain)
                                class_weights = {i : class_weights[i] for i in range(2)}
                                                            
                                features=100*(Overlapping*2+1)
                                                 
                                XTrain = XTrain.reshape(len(XTrain), features, 1) # len(XTrain) samples with 1 time step and 100 feature.
                                XTest = XTest.reshape(len(XTest), features, 1) # len(XTrain) samples with 1 time step and 100 feature.   
                                
                                YTrain = to_categorical(YTrain) # len(XTrain) labels with 1 label per epoch and 1 feature.
                                YTest = to_categorical(YTest) # len(XTrain) labels with 1 label per epoch and 1 feature.
    
                                model = Sequential()
                                
                                for z in range (0, ExaminedLayers+1, 1):
                                    if z == 0:
                                        model.add(Conv1D(2^KernelNumb, 2, strides=1, activation='relu', input_shape=(features,1)))
                                        model.add(MaxPooling1D(pool_size=2, strides=2))
                                        model.add(Dropout(0.1))
                                        KernelNumbprev=2^KernelNumb
                                    else:
                                        model.add(Conv1D(KernelNumbprev*mul, 2, activation='relu')) 
                                        model.add(MaxPooling1D(pool_size=2, strides=2))
                                        model.add(Dropout(0.1))
                                model.add(Flatten())
                                model.add(Dense(n, activation='relu'))
                                model.add(Dense(2, activation='softmax'))
                     
                                
                                class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
                                  """Stop training when the loss is at its min, i.e. the loss stops decreasing.
                                
                                  Arguments:
                                      patience: Number of epochs to wait after min has been hit. After this
                                      number of no improvement, training stops.
                                  """
                                
                                  def __init__(self, patienteceValue, valid_data):
                                    super(EarlyStoppingAtMinLoss, self).__init__()
                                    self.patience = patienteceValue
                                    self.best_weights = None
                                    self.validation_data = valid_data
                                
                                  def on_train_begin(self, logs=None):
                                    # The number of epoch it has waited when loss is no longer minimum.
                                    self.wait = 0
                                    # The epoch the training stops at.
                                    self.stopped_epoch = 0
                                    # Initialize the best as infinity.
                                #    self.best = np.Inf
                                    self.best = 0.2
                                    self._data = []
                                    self.curentAUC = 0.2
                                    print ('Train started')
                                
                                  def on_epoch_end(self, epoch, logs=None):
                                    X_val, y_val = self.validation_data[0], self.validation_data[1]
                                    y_predict = np.asarray(model.predict(X_val))
                                    
                                    
                                    fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.argmax(y_val, axis=1), y_predict[:,1])
                                    auc_keras = auc(fpr_keras, tpr_keras)
                                    self.curentAUC = auc_keras
                                    current = auc_keras
                                #    self.curentAUC = current
                                #    print('AUC %05d' % (self.bestAUC))
                                    print('AUC : ',current)
                                    
                                #    current = logs.get('loss')
                                    if np.greater(self.curentAUC, self.best+thresholdAphase):  # np.less
                                        print('Update')
                                        self.best = self.curentAUC
                                        self.wait = 0
                                        # Record the best weights if current results is better (less).
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
                        
                                
                                x_train, x_valid, y_train, y_valid = train_test_split(XTrain, YTrain, test_size=0.33, shuffle= True)
                                
                                model.fit(x_train, y_train,
                                                  batch_size=1000,
                                                  epochs=20, 
                                                  validation_data=(x_valid, y_valid),
                                                  verbose=1,
                                                  shuffle=True, class_weight=class_weights,callbacks=EarlyStoppingAtMinLoss(patienteceValue,(x_valid, y_valid)))
                                
                                print("Testing")
                                proba = model.predict(XTest)
                                YTestOneLine=np.zeros(len(YTest));
                                for x in range(len(YTest)):
                                    if YTest[x,0] == 1:
                                        YTestOneLine[x]=0
                                    else:
                                        YTestOneLine[x]=1
                                
                                predictiony_pred=np.zeros(len(YTestOneLine));
                                for x in range(len(YTestOneLine)):
                                    if proba[x,0] > 0.5:
                                        predictiony_pred[x]=0
                                    else:
                                        predictiony_pred[x]=1
                                        
                                tn, fp, fn, tp = confusion_matrix(YTestOneLine, predictiony_pred).ravel()
                                print(classification_report(YTestOneLine, predictiony_pred))
                                accuracy0=(tp+tn)/(tp+tn+fp+fn)
                                sensitivity0 = tp/(tp+fn)
                                specificity0 = tn/(fp+tn)       
                                fpr_keras, tpr_keras, thresholds_keras = roc_curve(YTestOneLine, proba[:,1])
                                auc_keras = auc(fpr_keras, tpr_keras)
                                print('AUC : ', auc_keras)
                                
                                capPredictedPredicted=predictiony_pred;
                                for k in range (len(capPredictedPredicted)-1):
                                    if k > 0:
                                        if capPredictedPredicted[k-1]==0 and capPredictedPredicted[k]==1 and capPredictedPredicted[k+1]==0:
                                            capPredictedPredicted[k]=0
                                            
                                for k in range (len(capPredictedPredicted)-1):
                                    if k > 0:
                                        if capPredictedPredicted[k-1]==1 and capPredictedPredicted[k]==0 and capPredictedPredicted[k+1]==1:
                                            capPredictedPredicted[k]=1
                            
                                tn, fp, fn, tp = confusion_matrix(YTestOneLine, capPredictedPredicted).ravel()
                                print(classification_report(YTestOneLine, capPredictedPredicted))
                                accuracy0=(tp+tn)/(tp+tn+fp+fn)
                                print ('Accuracy : ', accuracy0)
                                sensitivity0 = tp/(tp+fn)
                                print('Sensitivity : ', sensitivity0)
                                specificity0 = tn/(fp+tn)
                                print('Specificity : ', specificity0)
                                AccAtEnd[ee]=accuracy0
                                SenAtEnd[ee]=sensitivity0
                                SpeAtEnd[ee]=specificity0
                                AUCAtEnd[ee]=auc_keras
    
                                # del XTrain, YTrain, XTest, YTest, model
                                
                            AccAtEndMul[indexMul,countM]=np.mean(AccAtEnd)
                            SenAtEndMul[indexMul,countM]=np.mean(SenAtEnd)
                            SpeAtEndMul[indexMul,countM]=np.mean(SpeAtEnd)
                            AUCAtEndMul[indexMul,countM]=np.mean(AUCAtEnd)
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
                            if AUCmax < np.mean(AUCAtEnd): # found a better network
                                BestNet[0]=ExaminedLayers
                                BestNet[1]=OverLap
                                BestNet[2]=KernelNumb
                                BestNet[3]=n
                                BestNet[4]=a
                                BestNet[5]=mul
                                f = open("BestNet.txt", 'ab')
                                pickle.dump(BestNet, f)
                                f.close() 
                                AUCmax=np.mean(AUCAtEnd)
                            indexMul+=1
                            
                        countM+=1
                        AccAtEndA[indexA,countA]=np.max(AccAtEndMul[:,countM-1])
                        SenAtEndA[indexA,countA]=np.max(SenAtEndMul[:,countM-1])
                        SpeAtEndA[indexA,countA]=np.max(SpeAtEndMul[:,countM-1])
                        AUCAtEndA[indexA,countA]=np.max(AUCAtEndMul[:,countM-1])
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
                        AccAtEndAarg[indexA,countA]=np.argmax(AccAtEndMul[:,countM-1])
                        SenAtEndAarg[indexA,countA]=np.argmax(SenAtEndMul[:,countM-1])
                        SpeAtEndAarg[indexA,countA]=np.argmax(SpeAtEndMul[:,countM-1])
                        AUCAtEndAarg[indexA,countA]=np.argmax(AUCAtEndMul[:,countM-1])
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
                        indexA+=1
                        
                    countA+=1
                    AccAtEndN[indexN,countN]=np.max(AccAtEndA[:,countA-1])
                    SenAtEndN[indexN,countN]=np.max(SenAtEndA[:,countA-1])
                    SpeAtEndN[indexN,countN]=np.max(SpeAtEndA[:,countA-1])
                    AUCAtEndN[indexN,countN]=np.max(AUCAtEndA[:,countA-1])
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
                    AccAtEndNarg[indexN,countN]=np.argmax(AccAtEndA[:,countA-1])
                    SenAtEndNarg[indexN,countN]=np.argmax(SenAtEndA[:,countA-1])
                    SpeAtEndNarg[indexN,countN]=np.argmax(SpeAtEndA[:,countA-1])
                    AUCAtEndNarg[indexN,countN]=np.argmax(AUCAtEndA[:,countA-1])
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
                    indexN+=1
                    
                countN+=1
                AccAtEndK[indexK,countK]=np.max(AccAtEndN[:,countN-1])
                SenAtEndK[indexK,countK]=np.max(SenAtEndN[:,countN-1])
                SpeAtEndK[indexK,countK]=np.max(SpeAtEndN[:,countN-1])
                AUCAtEndK[indexK,countK]=np.max(AUCAtEndN[:,countN-1])
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
                AccAtEndKarg[indexK,countK]=np.argmax(AccAtEndN[:,countN-1])
                SenAtEndKarg[indexK,countK]=np.argmax(SenAtEndN[:,countN-1])
                SpeAtEndKarg[indexK,countK]=np.argmax(SpeAtEndN[:,countN-1])
                AUCAtEndKarg[indexK,countK]=np.argmax(AUCAtEndN[:,countN-1])
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
                indexK+=1
                
            countK+=1
            AccAtEndO[indexO,countO]=np.max(AccAtEndK[:,countK-1])
            SenAtEndO[indexO,countO]=np.max(SenAtEndK[:,countK-1])
            SpeAtEndO[indexO,countO]=np.max(SpeAtEndK[:,countK-1])
            AUCAtEndO[indexO,countO]=np.max(AUCAtEndK[:,countK-1])
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
            AccAtEndOarg[indexO,countO]=np.argmax(AccAtEndK[:,countK-1])
            SenAtEndOarg[indexO,countO]=np.argmax(SenAtEndK[:,countK-1])
            SpeAtEndOarg[indexO,countO]=np.argmax(SpeAtEndK[:,countK-1])
            AUCAtEndOarg[indexO,countO]=np.argmax(AUCAtEndK[:,countK-1])
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
            indexO+=1
            
        countO+=1
        AccAtEndG[indexG]=np.max(AccAtEndO[:,countO-1])
        SenAtEndG[indexG]=np.max(SenAtEndO[:,countO-1])
        SpeAtEndG[indexG]=np.max(SpeAtEndO[:,countO-1])
        AUCAtEndG[indexG]=np.max(AUCAtEndO[:,countO-1])
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
        AccAtEndGarg[indexG]=np.argmax(AccAtEndO[:,countO-1])
        SenAtEndGarg[indexG]=np.argmax(SenAtEndO[:,countO-1])
        SpeAtEndGarg[indexG]=np.argmax(SpeAtEndO[:,countO-1])
        AUCAtEndGarg[indexG]=np.argmax(AUCAtEndO[:,countO-1])
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
            if AUCAtEndG[indexG] - AUCAtEndG[indexG-1] <= thresholdAphase:
                stopAnalysis=1
        indexG+=1

print ('Finished')
