# Denis James
# Submitted on:10-Dec-2022


#importing all the necessary modules
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn import metrics

# importing the dataset
Transaction_Info=pd.read_csv("C://Users/denis/OneDrive/Desktop/DS Task/data-new/transactions_obf.csv")
Reported_Frauds=pd.read_csv("C://Users/denis/OneDrive/Desktop/DS Task/data-new/labels_obf.csv")

# Splitting the time(YYYY-MM-DDThh:mm:ss) into two columns Date(YYYY-MM-DD) and Time(hh:mm:ss)
Transaction_Info["transactionDate"]=pd.to_datetime(Transaction_Info["transactionTime"]).dt.date
Transaction_Info["transactionTime"]=pd.to_datetime(Transaction_Info["transactionTime"]).dt.time

# Assigns labels to each transaction: True if fraudulent, False otherwise
for i in Reported_Frauds.eventId:
    if i in list(Transaction_Info.eventId):
        Transaction_Info.loc[Transaction_Info.eventId==i,'fraudLabel']=True
Transaction_Info.loc[Transaction_Info.fraudLabel!=1,'fraudLabel']=False


# Clearing all the noise in the data
Transaction_Info=Transaction_Info[Transaction_Info.posEntryMode!=79]
Transaction_Info=Transaction_Info[Transaction_Info.transactionAmount>0]
Transaction_Info.iloc[Transaction_Info.merchantZip.isnull(),6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="..",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="...",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="....",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip==".....",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="**",6]='0'
Transaction_Info.iloc[Transaction_Info.merchantZip=="***",6]='0'


Transaction_Info['FraudCommittedPreviously']=0
Transaction_Info['FraudedPreviously']=0


Transaction_Info=Transaction_Info.reset_index()
del Transaction_Info['index']


# this loop creates a new feature for every transaction in the dataset. It stores the number of times the payee account has been frauded prior to that transaction. (takes atmost 40 mins to execute)
for i in range(len(Transaction_Info)):
    temporary=Transaction_Info.iloc[0:i+1,[2,11]]
    anothertemporary=temporary[temporary.accountNumber==list(temporary.accountNumber)[-1]]
    Transaction_Info.iloc[i,13]=len(anothertemporary[anothertemporary.fraudLabel==1])
    del temporary,anothertemporary

    print((i/118621)*100,"%/ rewritten")


# this loop creates a new feature for every transaction in the dataset. It stores the number of times the merchant account has been involved in a fraudulent transaction. (takes atmost 40 mins to execute)
for i in range(len(Transaction_Info)):
    temporary=Transaction_Info.iloc[0:i+1,[3,11]]
    anothertemporary=temporary[temporary.merchantId==list(temporary.merchantId)[-1]]
    Transaction_Info.iloc[i,12]=len(anothertemporary[anothertemporary.fraudLabel==1])
    del temporary,anothertemporary
    print((i/118621)*100,"%/ rewritten")


# Assigns each Account Number, merchant Id and merchantZip a unique integer which makes it easier for the algorithm to work on.
AccountNumberCategories=pd.Categorical(Transaction_Info.accountNumber)
MerchantIdCategories=pd.Categorical(Transaction_Info.accountNumber)
MerchantZip=pd.Categorical(Transaction_Info.merchantZip)
Transaction_Info['accountNumberCodes']=AccountNumberCategories.codes
Transaction_Info['merchantIdCodes']=MerchantIdCategories.codes
Transaction_Info['MerchantZipCodes']=MerchantZip.codes


# Splits the 'transactionTime' and 'transactionDate' column into 'transactionHour','transactionMinute','transactionYear','transactionMonth','transactionDay',
# It is assumed, while training the model that the exact second at which the transaction was made would not make a difference on it's prediction. 
DateSplit=pd.DataFrame([str(i).split('-',) for i in Transaction_Info.transactionDate],columns=['transactionYear','transactionMonth','transactionDay'])
TimeSplit=pd.DataFrame([str(i).split(':')[0:2] for i in Transaction_Info.transactionTime],columns=['transactionHour','transactionMinute'])
Transaction_Info[['transactionHour','transactionMinute']]=TimeSplit.astype(int)
Transaction_Info[['transactionYear','transactionMonth','transactionDay']]=DateSplit.astype(int)



#Splitting the clean dataset into Train and Test Dataset.
TrueTrainCases=Transaction_Info[Transaction_Info.fraudLabel==True].sample(800)
TrueTestCases=pd.concat([Transaction_Info[Transaction_Info.fraudLabel==True],TrueTrainCases]).drop_duplicates(keep=False)
FalseTrainCases=Transaction_Info[Transaction_Info.fraudLabel==False].sample(4800)
FalseTestCases=pd.concat([Transaction_Info[Transaction_Info.fraudLabel==False],FalseTrainCases]).drop_duplicates(keep=False)



# Preparing the Training Dataset
TheTrainData=pd.concat([TrueTrainCases,FalseTrainCases])[['mcc','posEntryMode','transactionAmount','availableCash','FraudCommittedPreviously','FraudedPreviously','accountNumberCodes','merchantIdCodes','MerchantZipCodes','transactionHour','transactionMinute','transactionYear','transactionMonth','transactionDay','fraudLabel']]


# Defining and training the model

# model=IsolationForest(n_estimators=53,max_samples=1600,contamination=.142,max_features=10,verbose=1)
model=RandomForestClassifier(n_estimators=37,criterion='entropy')


model.fit(TheTrainData[['mcc','posEntryMode','transactionAmount','availableCash','FraudCommittedPreviously','FraudedPreviously','accountNumberCodes','merchantIdCodes','MerchantZipCodes','transactionHour','transactionMinute','transactionYear','transactionMonth','transactionDay']],TheTrainData['fraudLabel'].astype('bool'))

# Preparing the Testing Dataset, none of who's entries coincide with that of the training Dataset.

TheTestData=pd.concat([TrueTestCases,FalseTestCases.sample(100)])[['mcc','posEntryMode','transactionAmount','availableCash','FraudCommittedPreviously','FraudedPreviously','accountNumberCodes','merchantIdCodes','MerchantZipCodes','transactionHour','transactionMinute','transactionYear','transactionMonth','transactionDay','fraudLabel']]

# Predicting the Test Data Lables
Predictions=pd.Series(model.predict(TheTestData[['mcc','posEntryMode','transactionAmount','availableCash','FraudCommittedPreviously','FraudedPreviously','accountNumberCodes','merchantIdCodes','MerchantZipCodes','transactionHour','transactionMinute','transactionYear','transactionMonth','transactionDay']]))
Results=TheTestData.fraudLabel
Results=Results.reset_index().fraudLabel
print(Predictions)

# Displaying the confusion Matrix.
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(list(Results),list(Predictions)),display_labels=["False","True"]).plot(cmap='Greys')

print(model.feature_importance)