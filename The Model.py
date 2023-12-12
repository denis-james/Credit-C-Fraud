# Denis James
# Submitted on:10-Dec-2022


### Importing Modules
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn import metrics

#Importing Datasets
Transaction_Info=pd.read_csv("data-new/transactions_obf.csv")
Reported_Frauds=pd.read_csv("data-new/labels_obf.csv")
Transaction_Info['merchantCountry']

#Splitting Date/Time Feature
Transaction_Info["transactionDate"]=pd.to_datetime(Transaction_Info["transactionTime"]).dt.date
Transaction_Info["transactionTime"]=pd.to_datetime(Transaction_Info["transactionTime"]).dt.time
DateSplit=pd.DataFrame([str(i).split('-') for i in Transaction_Info.transactionDate],columns=['transactionYear','transactionMonth','transactionDay'])
TimeSplit=pd.DataFrame([str(i).split(':')[0:3] for i in Transaction_Info.transactionTime],columns=['transactionHour','transactionMinute','transactionSecond'])
Transaction_Info[['transactionHour','transactionMinute','transactionSecond']]=TimeSplit.astype(int)
Transaction_Info[['transactionYear','transactionMonth','transactionDay']]=DateSplit.astype(int)

#Defining Fraud Labels
Transaction_Info['fraudLabel']=False
for i in Reported_Frauds.eventId:
    Transaction_Info.loc[Transaction_Info.eventId==i,'fraudLabel']=True

#Cleaning Data off of missing and noisy values
Transaction_Info=Transaction_Info[Transaction_Info.posEntryMode!=79]
Transaction_Info=Transaction_Info[Transaction_Info.transactionAmount>0]
Transaction_Info.iloc[Transaction_Info.merchantZip.isnull(),6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="..",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="...",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="....",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip==".....",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="**",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="***",6]='0'


#Reseting Index after cleaning
Transaction_Info=Transaction_Info.reset_index()
del Transaction_Info['index']


# Assigns each Account Number, merchant Id and merchantZip a unique integer which makes it easier for the algorithm to work on.
AccountNumberCategories=pd.Categorical(Transaction_Info.accountNumber)
MerchantIdCategories=pd.Categorical(Transaction_Info.accountNumber)
MerchantZip=pd.Categorical(Transaction_Info.merchantZip)
Transaction_Info['accountNumberCodes']=AccountNumberCategories.codes
Transaction_Info['merchantIdCodes']=MerchantIdCategories.codes
Transaction_Info['MerchantZipCodes']=MerchantZip.codes




# Splitting Dataset into False and True Test/Train Sets
TrueTrainCases=Transaction_Info[Transaction_Info.fraudLabel==True].sample(800)
TrueTestCases=pd.concat([Transaction_Info[Transaction_Info.fraudLabel==True],TrueTrainCases]).drop_duplicates(keep=False)#Transaction_Info-TrueTrainCases
FalseTrainCases=Transaction_Info[Transaction_Info.fraudLabel==False].sample(1800)
FalseTestCases=pd.concat([Transaction_Info[Transaction_Info.fraudLabel==False],FalseTrainCases]).drop_duplicates(keep=False)#Transaction_Info-FalseTrainCases

#Features that contribute to the analysis
TrainingIndex=['mcc','posEntryMode','transactionAmount','availableCash','accountNumberCodes','merchantIdCodes','MerchantZipCodes','transactionHour','transactionMinute','transactionSecond','transactionYear','transactionMonth','transactionDay']
TheTrainData=pd.concat([TrueTrainCases,FalseTrainCases])[TrainingIndex+['fraudLabel']]


# Defining and training the model

#Defining the AI Model and  Gridsearch Parameters list 
n_estimators=[i for i in range(5,20,4)]
criterion=["gini", "entropy"]
max_depth=[i for i in range(5,25,4)]
min_samples_split=[i for i in range(5,30,4)]
min_samples_leaf=[i for i in range(3,10)]
max_features=['auto', 'sqrt']
bootstrap=[False]
Parameters= {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap,
    'criterion':criterion
    }
#n_jobs=,random_state=,verbose=,warm_start=,class_weight=,max_samples=}
model=RandomForestClassifier()
random_model=GridSearchCV(estimator=model,param_grid=Parameters,verbose=4)


#Fitting the model onto the Train Data  
random_model.fit(TheTrainData[TrainingIndex],TheTrainData['fraudLabel'])
print(random_model.best_params_)
# Preparing the Testing Dataset, none of who's entries coincide with that of the training Dataset.

#Preparing Test Data
TheTestData=pd.concat([TrueTestCases,FalseTestCases.sample(50)])[TrainingIndex+['fraudLabel']]

#Test Predictions
Predictions=pd.Series(random_model.predict(TheTrainData[TrainingIndex]))
Results=TheTrainData.fraudLabel;    Results=Results.reset_index().fraudLabel
print(Predictions)


#Performance of the Classifier
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(list(Results),list(Predictions)),display_labels=["False","True"]).plot(cmap='Greys')

print(random_model.score(TheTrainData[TrainingIndex],TheTrainData.fraudLabel))
print(random_model.score(TheTestData[TrainingIndex],TheTestData.fraudLabel))