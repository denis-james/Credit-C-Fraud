# Denis James


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
All_Other_True_Cases=pd.concat([Transaction_Info[Transaction_Info.fraudLabel==True],TrueTrainCases]).drop_duplicates(keep=False)#Transaction_Info-TrueTrainCases
TrueTestCases=pd.concat([All_Other_True_Cases,TrueTrainCases.sample(50)])
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
TheTestData=pd.concat([TrueTestCases,FalseTestCases.sample(100)])[TrainingIndex+['fraudLabel']]

#Test Predictions

Testing_On_Train_Data=pd.concat([TrueTrainCases,FalseTrainCases.sample(800)])[TrainingIndex+['fraudLabel']]
Predictions=pd.Series(random_model.predict(Testing_On_Train_Data[TrainingIndex]))
Results=Testing_On_Train_Data.fraudLabel;    Results=Results.reset_index().fraudLabel
print(Predictions)

#Performance of the Classifier on Training Data
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(list(Results),list(Predictions)),display_labels=["Legitimate","Fraudulent"]).plot(cmap='Greys')


#Test Predictions
Predictions=pd.Series(random_model.predict(TheTestData[TrainingIndex]))
Results=TheTestData.fraudLabel;    Results=Results.reset_index().fraudLabel
print(Predictions)


#Performance of the Classifier on Test Data
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(list(Results),list(Predictions)),display_labels=["Legitimate","Fraudulent"]).plot(cmap='Greys')

Train_Accuracy=random_model.score(Testing_On_Train_Data[TrainingIndex],Testing_On_Train_Data.fraudLabel)
Test_Accuracy=random_model.score(TheTestData[TrainingIndex],TheTestData.fraudLabel)

fig,ax=plt.subplots()
bar1=ax.bar('Test',Test_Accuracy*100)
ax.bar_label(bar1)
bar2=ax.bar('Train',Train_Accuracy*100)
ax.bar_label(bar2)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Results After Training')

weight_counts = {
    'Legitimate':[sum(Transaction_Info.fraudLabel==False)*100/len(Transaction_Info)],
    'Fraudulent':[sum(Transaction_Info.fraudLabel==True)*100/len(Transaction_Info)]
}

fig, ax = plt.subplots(figsize=(8,1))
ax.barh('Data',weight_counts['Legitimate'],label='Legitimate',left=weight_counts['Fraudulent'],color='green')

ax.barh('Data',weight_counts['Fraudulent'],label='Fraudulent',left=0,color='red')

ax.set_title("Class Proportion")
ax.legend(loc="upper right")

plt.show()