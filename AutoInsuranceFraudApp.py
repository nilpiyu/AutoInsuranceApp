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
st.download_button('Download Sample file link for check', 'https://github.com/ripon2488/Insurace-app-fraud-detection/blob/main/AutoInsuranceFraudDetection.csv')
st.sidebar.header('Please Input Features Value')

# Collects user input features into dataframe

def user_input_features():
    policy_number = st.sidebar.number_input('Policy Number: ')
    age = st.sidebar.number_input('Age of persons: ')    
    months_as_customer = st.sidebar.number_input('Month as customer')
    policy_bind_date = st.date_input("Policy Bind Date")
    policy_state =st.sidebar.selectbox('Policy State: ', ( 'OH','IN','IL'))
    policy_csl =st.sidebar.selectbox('Policy CSL: ', ( '100/300','250/500','500/1000'))
    policy_deductable = st.sidebar.selectbox('Policy Deductable: ', ( '500','1000','2000'))
    policy_annual_premium =  st.sidebar.number_input('Polish Annual Premium: ')
    umbrella_limit = st.sidebar.number_input('Umbrella Limit: ')
    insured_sex = st.sidebar.selectbox('Gender of persons 0=Female, 1=Male: ',(0,1))
    insured_zip = st.sidebar.number_input('Insured Zip: ')
    insured_education_level =st.sidebar.selectbox('Insured Education: ', ( 'Associate','College','High School','JD', 'Masters', 'MD', 'PhD'))
    insured_occupation = st.sidebar.selectbox('Insured OccupationL: ', ('transport-moving', 'armed-forces','craft-repair','arm-celerical','sales', 'tech-support', 'exec-managerial', 'prof-speciality','other services'))
    insured_hobbies = st.sidebar.selectbox('Insured Hobbies: ', ('basketball', 'board-games','base-jumping','bungie-jumping','camping', 'chess', 'dance', 'golf','hiking','video-games','reading','sleeping','other'))
    insured_relationship = st.sidebar.selectbox('Insured Relationship:', ( 'husband','unmarried','wife','own-child','other-relative','not-in-family'))
    capital_gains = st.sidebar.number_input('Capital Gain: ')
    capital_loss = st.sidebar.number_input('Capital Loss: ')
    incident_type = st.sidebar.selectbox('Incident Type: ',( 'Multi-vehicle Collision','Single Vehicle Collision','Parked Car', 'Vehicle Theft'))
    collision_type =  st.sidebar.selectbox('Collision  Type:', ( 'Rear Collision','Front Collision','Side Collision'))
    incident_severity = st.sidebar.selectbox('Incident Severity: ', ( 'Major Damage', 'Minor Damage','Total Loss', 'Trivial Damage'))
    authorities_contacted = st.sidebar.selectbox('Authorities Contacted:',('Police', 'Fire', 'Ambulance', 'None', 'Other'))
    incident_hour_of_the_day = st.sidebar.number_input(' Incident Hour of Day: ')
    incident_date = st.date_input("Incident Date")
    incident_state = st.sidebar.selectbox('Incident State:',('SC','VA','NY','OH','WV', 'PA', 'NC'))
    incident_city = st.sidebar.selectbox('Incident City:',('Arlignton','Columbus','Hillsdale', 'Northbrook','Springfield', 'Arlington'))
    incident_location = st.sidebar.text_input('Incident Location: ')
    number_of_vehicles_involved = st.sidebar.number_input('Number of Vehicles Involved: ')
    property_damage = st.sidebar.selectbox('Property Damage:',('YES','NO'))
    bodily_injuries = st.sidebar.number_input('Bidily Injuries')
    witnesses = st.sidebar.number_input('Witness')
    police_report_available = st.sidebar.selectbox('Police Report Available:',('YES','NO'))
    total_claim_amount = st.sidebar.number_input('Total Claim Amount')
    injury_claim = st.sidebar.number_input('Injury Claim')
    property_claim = st.sidebar.number_input('Property Claim')
    vehicle_claim =st.sidebar.number_input('Vehicle Claim')
    auto_make = st.sidebar.text_input('Auto Make')
    auto_model = st.sidebar.text_input('Auto Model')
    auto_year = st.sidebar.number_input('Auto year')    


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
'capital-gains' :capital-gains,
'capital-loss' :capital-loss,
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

st.write(input_df)

def predict(data):
    clf = joblib.load("model_RFC.sav")
    return clf.predict(data)


# Apply model to make predictions

if st.button("Click here to Predict Fraud in Claim Submission"):
    result = predict(input_df)

    if (result[0]== 0):
        st.subheader('The Claim :green[Fraud Not Detected] :sunglasses: 	:sparkling_heart:')
    else:
        st.subheader('The Claim :red[Fraud Detected] :worried:')
