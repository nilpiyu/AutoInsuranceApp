import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import sklearn
import joblib

st.image('header.png')
st.title(':blue[Insurance Fraud Prediction App]')
st.write("""-- This app predicts a Claim has Fraud Reported or not --

""")
st.write(':point_left: (click arrow sign for hide and unhide form) :green[Please Fillup the input field of left side for Prediction.] :sunglasses:')
# st.download_button('Download Sample file link for check', 'https://github.com/ripon2488/Insurace-app-fraud-detection/blob/main/AutoInsuranceFraudDetection.csv')
st.sidebar.header('Please Input Claim Data')

# Collects user input features into dataframe

def user_input_features():
    policy_number = st.sidebar.text_input('Policy Number: ')
    age = st.sidebar.text_input('Age of persons: ')    
    months_as_customer = st.sidebar.text_input('Month as customer')
    policy_bind_date = st.sidebar.text_input("Policy Bind Date", value="28-12-1998")
    policy_state =st.sidebar.selectbox('Policy State: ', ( 'OH','IN','IL'))
    policy_csl =st.sidebar.selectbox('Policy CSL: ', ( '100/300','250/500','500/1000'))
    policy_deductable = st.sidebar.selectbox('Policy Deductable: ', ( '500','1000','2000'))
    policy_annual_premium =  st.sidebar.text_input('Policy Annual Premium: ')
    umbrella_limit = st.sidebar.text_input('Umbrella Limit: ')
    insured_sex = st.sidebar.selectbox('Gender of person ',('MALE', 'FEMALE'))
    insured_zip = st.sidebar.text_input('Insured Zip: ')
    insured_education_level =st.sidebar.selectbox('Insured Education: ', ( 'Associate','College','High School','JD', 'Masters', 'MD', 'PhD'))
    insured_occupation = st.sidebar.selectbox('Insured OccupationL: ', ('transport-moving', 'armed-forces','craft-repair','arm-celerical','sales', 'tech-support', 'exec-managerial', 'prof-speciality','other services'))
    insured_hobbies = st.sidebar.selectbox('Insured Hobbies: ', ('basketball', 'board-games','base-jumping','bungie-jumping','camping', 'chess', 'dance', 'golf','hiking','video-games','reading','sleeping','other'))
    insured_relationship = st.sidebar.selectbox('Insured Relationship:', ( 'husband','unmarried','wife','own-child','other-relative','not-in-family'))
    capital_gains = st.sidebar.text_input('Capital Gain: ')
    capital_loss = st.sidebar.text_input('Capital Loss: ')
    incident_type = st.sidebar.selectbox('Incident Type: ',( 'Multi-vehicle Collision','Single Vehicle Collision','Parked Car', 'Vehicle Theft'))
    collision_type =  st.sidebar.selectbox('Collision  Type:', ( 'Rear Collision','Front Collision','Side Collision'))
    incident_severity = st.sidebar.selectbox('Incident Severity: ', ( 'Major Damage', 'Minor Damage','Total Loss', 'Trivial Damage'))
    authorities_contacted = st.sidebar.selectbox('Authorities Contacted:',('Police', 'Fire', 'Ambulance', 'None', 'Other'))
    incident_hour_of_the_day = st.sidebar.text_input(' Incident Hour of Day: ')
    incident_date = st.sidebar.text_input("Incident Date")
    incident_state = st.sidebar.selectbox('Incident State:',('SC','VA','NY','OH','WV', 'PA', 'NC'))
    incident_city = st.sidebar.selectbox('Incident City:',('Arlignton','Columbus','Hillsdale', 'Northbrook','Springfield', 'Arlington'))
    incident_location = st.sidebar.text_input('Incident Location: ')
    number_of_vehicles_involved = st.sidebar.text_input('Number of Vehicles Involved: ')
    property_damage = st.sidebar.selectbox('Property Damage:',('YES','NO','?'))
    bodily_injuries = st.sidebar.text_input('Bodily Injuries')
    witnesses = st.sidebar.text_input('Witness')
    police_report_available = st.sidebar.selectbox('Police Report Available:',('YES','NO'))
    total_claim_amount = st.sidebar.text_input('Total Claim Amount')
    injury_claim = st.sidebar.text_input('Injury Claim')
    property_claim = st.sidebar.text_input('Property Claim')
    vehicle_claim =st.sidebar.text_input('Vehicle Claim')
    auto_make = st.sidebar.text_input('Auto Make')
    auto_model = st.sidebar.text_input('Auto Model')
    auto_year = st.sidebar.text_input('Auto year')    


    data = {'months_as_customer' :months_as_customer,
'age' :age,
'policy_number' :policy_number,
'policy_bind_date' :policy_bind_date,
'policy_state' :policy_state,
'policy_csl' :policy_csl,
'policy_deductable' :policy_deductable,
'policy_annual_premium' :policy_annual_premium,
'umbrella_limit' :umbrella_limit,
'insured_zip' :insured_zip,
'insured_sex' :insured_sex,
'insured_education_level' :insured_education_level,
'insured_occupation' :insured_occupation,
'insured_hobbies' :insured_hobbies,
'insured_relationship' :insured_relationship,
'capital-gains' :capital_gains,
'capital-loss' :capital_loss,
'incident_date' :incident_date,
'incident_type' :incident_type,
'collision_type' :collision_type,
'incident_severity' :incident_severity,
'authorities_contacted' :authorities_contacted,
'incident_state' :incident_state,
'incident_city' :incident_city,
'incident_location' :incident_location,
'incident_hour_of_the_day' :incident_hour_of_the_day,
'number_of_vehicles_involved' :number_of_vehicles_involved,
'property_damage' :property_damage,
'bodily_injuries' :bodily_injuries,
'witnesses' :witnesses,
'police_report_available' :police_report_available,
'total_claim_amount' :total_claim_amount,
'injury_claim' :injury_claim,
'property_claim' :property_claim,
'vehicle_claim' :vehicle_claim,
'auto_make' :auto_make,
'auto_model' :auto_model,
'auto_year' :auto_year
}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# st.write(input_df)

def predict(data):
    clf = joblib.load("model_RFC.sav")
    return clf.predict(data)

# df_new_data =pd.read_csv('Automobile_insurance_fraud_v1.csv')
df_new_data = input_df
# Apply model to make predictions
st.write(df_new_data)
# preprocessing steps
import pandas as pd
# For handling date variables
from datetime import date
# For plotting graphs

# For plotting graphs

# For splitting dataset 
from sklearn.model_selection import train_test_split
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,cross_val_score 
from sklearn.metrics import accuracy_score, recall_score, classification_report, cohen_kappa_score,confusion_matrix,precision_score,f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder # Use this instead of pd.getDummies
import pickle
import os


ohe_2 =  joblib.load("Pickle_OHE_Model.sav")

## convert Date fields 
df_new_data['policy_bind_date'] = pd.to_datetime(df_new_data['policy_bind_date'],dayfirst=True)
df_new_data['incident_date'] = pd.to_datetime(df_new_data['incident_date'],dayfirst=True)

## Create new columns
df_new_data['Policy_Tenure_Days'] = df_new_data['policy_bind_date'].rsub(pd.Timestamp('now').floor('d')).dt.days
##df_new_data['vehicle_age']=int(date.today().strftime('%Y'))-df_new_data['auto_year']
df_new_data['vehicle_age'] = int(int(2023) - int(df_new_data['auto_year']))
df_new_data['csl_per_person'] = df_new_data.policy_csl.str.split('/', expand=True)[0]
df_new_data['csl_per_accident'] = df_new_data.policy_csl.str.split('/', expand=True)[1]
df_new_data['csl_per_person']  = df_new_data['csl_per_person'].astype('int32')
df_new_data['csl_per_accident']  = df_new_data['csl_per_accident'].astype('int32')


df_new_data['authorities_contacted'].fillna('Not_Contacted', inplace = True)

## Set Categorical fields
df_new_data['policy_number'] = df_new_data['policy_number'].astype('category')
df_new_data['policy_state'] = df_new_data['policy_state'].astype('category')
df_new_data['policy_csl'] = df_new_data['policy_csl'].astype('category')
df_new_data['insured_zip'] = df_new_data['insured_zip'].astype('category')
df_new_data['police_report_available'] = df_new_data['police_report_available'].astype('category')
df_new_data['auto_make'] = df_new_data['auto_make'].astype('category')
df_new_data['auto_model'] = df_new_data['auto_model'].astype('category')
df_new_data['insured_sex'] = df_new_data['insured_sex'].astype('category')
df_new_data['insured_education_level'] = df_new_data['insured_education_level'].astype('category')
df_new_data['insured_occupation'] = df_new_data['insured_occupation'].astype('category')
df_new_data['insured_hobbies'] = df_new_data['insured_hobbies'].astype('category')
df_new_data['insured_relationship'] = df_new_data['insured_relationship'].astype('category')
df_new_data['incident_type'] = df_new_data['incident_type'].astype('category')
df_new_data['collision_type'] = df_new_data['collision_type'].astype('category')
df_new_data['incident_severity'] = df_new_data['incident_severity'].astype('category')
df_new_data['authorities_contacted'] = df_new_data['authorities_contacted'].astype('category')
df_new_data['incident_state'] = df_new_data['incident_state'].astype('category')
df_new_data['incident_city'] = df_new_data['incident_city'].astype('category')
df_new_data['incident_location'] = df_new_data['incident_location'].astype('category')
df_new_data['property_damage'] = df_new_data['property_damage'].astype('category')

## Remove Columns that are not significant for fraud detection

### Policy Number ---- Only potential information we can get is how old is the policy, we already have months_as_customer.
## Hence it's redundant. ---> DROPPING
df_new_data=df_new_data.drop(['policy_number'], axis=1)
df_new_data=df_new_data.drop(['policy_bind_date'], axis=1)
df_new_data=df_new_data.drop(['auto_year'],axis=1)
df_new_data=df_new_data.drop(['policy_csl'], axis=1)
df_new_data=df_new_data.drop(['insured_zip'], axis=1)
df_new_data=df_new_data.drop(['incident_date'], axis=1)
df_new_data=df_new_data.drop(['incident_location'], axis=1)
df_new_data=df_new_data.drop(['auto_make'], axis=1)
df_new_data=df_new_data.drop(['auto_model'], axis=1)


## Applying one-hot encoding to convert all categorical variables except out target variables
dummies = ohe_2.transform(df_new_data[[
'policy_state',
'insured_sex',
'insured_education_level',
'insured_occupation',
'insured_hobbies',
'insured_relationship',
'incident_type',
'collision_type',
'incident_severity',
'authorities_contacted',
'incident_state',
'incident_city',
'property_damage',
'police_report_available']])

dummies=pd.DataFrame(dummies.toarray(), columns = ohe_2.get_feature_names_out())

## Prepare final Data Set for buliding model
df_new_data_X1 = df_new_data[['months_as_customer',
'age',
'policy_deductable',
'policy_annual_premium',
'umbrella_limit',
'capital-gains',
'capital-loss',
'incident_hour_of_the_day',
'number_of_vehicles_involved',
'bodily_injuries',
'witnesses',
'total_claim_amount',
'injury_claim',
'property_claim',
'vehicle_claim',
'vehicle_age',
'Policy_Tenure_Days',
'csl_per_person',
'csl_per_accident']]

df_new_data_X1 = df_new_data_X1.join(dummies)

scaler = StandardScaler(with_mean=False)
df_new_data_X1_scaled = scaler.fit_transform(df_new_data_X1)


if st.button("Click here to Predict Fraud in Claim Submission"):
   # result = predict(input_df)
    result = predict(df_new_data_X1_scaled)
    if (result[0]== 0):
        st.subheader('The Claim :green[Fraud Not Detected]')
    else:
        st.subheader('The Claim :red[Fraud Detected] ')
