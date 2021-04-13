# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

#Reading .csv data using pandas function ""pd.read_csv("")""
train_data = pd.read_csv('/content/Train_aBjfeNk.csv')
test_data = pd.read_csv('/content/Test_LqhgPWU.csv')
sample_data = pd.read_csv('/content/sample_submission_wyi0h0z.csv')

#how train_data looks?
print('Training Data')
train_data

#test_data
print("Testing Data ")
test_data

#Sample data
print('Sample data')
sample_data

# Now lets look the shape of data.
print('Train Data Shape : {}'.format(train_data.shape))
print('Test Data Shape  : {}'.format(test_data.shape))
print('Sample Data Shape: {}'.format(sample_data.shape))

# Disply information of data using info() method of pandas.DataFrame
train_data.info()

# Describe your training data 
print('The columns are missing here: its because they are "object", which means kind of "string".')
train_data.describe()

# count the missing value
print(train_data.isnull().sum())
print('\r\nThere are missing  values in 6 columns.')

# Handle missing values
miss_data_col = train_data[['Ever_Married','Graduated','Profession','Work_Experience','Family_Size','Var_1']]
miss_data_col

#count the number of NaN values in each column
miss_data_col.isnull().sum()

# Now fill missing value with 
impute = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
train_data['Ever_Married'] =impute.fit_transform(train_data[['Ever_Married']])
train_data['Graduated'] =impute.fit_transform(train_data[['Graduated']])
train_data['Profession'] =impute.fit_transform(train_data[['Profession']])
train_data['Work_Experience'] =impute.fit_transform(train_data[['Work_Experience']])
train_data['Family_Size'] =impute.fit_transform(train_data[['Family_Size']])
train_data['Var_1'] =impute.fit_transform(train_data[['Var_1']])

# Varify wheather imputation is done or not.
train_data.isnull().sum()

# Now split the data
train,validation = train_test_split(train_data,test_size = 0.2)
print('Train Shape {}'.format(train.shape))
print('Validation Shape {}'.format(validation.shape))

# saperating string dtypes
train_obj = train_data.select_dtypes(include='object')
train_obj

validation

# Encoder : object to int/float
lbl_enc = LabelEncoder()
train['Gender']= lbl_enc.fit_transform(train['Gender'])
train['Ever_Married']=lbl_enc.fit_transform(train['Ever_Married'])
train['Graduated']=lbl_enc.fit_transform(train['Graduated'])
train['Profession']=lbl_enc.fit_transform(train['Profession'])
train['Spending_Score']=lbl_enc.fit_transform(train['Spending_Score'])
train['Var_1']=lbl_enc.fit_transform(train['Var_1'])
#train['Segmentation']=lbl_enc.fit_transform(train['Segmentation'])
train.info()

validation.info()

validation['Gender']= lbl_enc.fit_transform(validation['Gender'])
validation['Graduated']=lbl_enc.fit_transform(validation['Graduated'])
validation['Profession']=lbl_enc.fit_transform(validation['Profession'])
validation['Var_1']=lbl_enc.fit_transform(validation['Var_1'])
#validation['Segmentation']=lbl_enc.fit_transform(validation['Segmentation'])
validation['Ever_Married']=lbl_enc.fit_transform(validation['Ever_Married'])
validation['Spending_Score']=lbl_enc.fit_transform(validation['Spending_Score'])

validation.info()

train = train.drop(labels='ID',axis = 1)
train

validation = validation.drop(labels='ID',axis = 1)
validation

train_X = train[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size','Var_1']]
train_y = train['Segmentation']

#train your time
clf = LGBMClassifier()
clf.fit(train_X,train_y)
#train_X.shape,train_y.shape,test_data.shape

# Time for training your data
#model = LGBMClassifier()
#model.fit(train.iloc[:,1:-1],train.iloc[:,-1])

# Predicting
prediction = model.predict(validation.iloc[:,1:-1]) 
prediction

validation.shape,train.shape,test_data.shape

#Evaluation Metric
print('Accuracy Score : {}'.format(accuracy_score(validation.iloc[:,-1],prediction)))
