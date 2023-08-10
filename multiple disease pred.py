import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#Loading the saved models

diabetes_model=pickle.load(open('diabetes_model.sav', 'rb'))
diabetes_scaler=pickle.load(open('diabetes_scaler.sav', 'rb'))

heart_model=pickle.load(open('heart_model.sav', 'rb'))
heart_scaler=pickle.load(open('heartscaler.sav', 'rb'))

parkinson_model=pickle.load(open('parkinson_model.sav', 'rb'))
parkinson_scaler=pickle.load(open('parkinson_scaler.sav','rb'))


# sidebar for navigation
with st.sidebar:
    selected=option_menu('Multiple Disease Prediction System',
                         
                         ['Diabetes Predication',
                          'Heart Disease Prediction',
                          'Parkinson Prediction'],
                         
                         icons=['activity','heart','person'],
                         
                         default_index=0)
    

#Diabetes Predication Page
if selected=='Diabetes Predication':
    
    # page title
    st.title('Diabetes Predication')
    
    # Getting the input data from the user
    col1,col2,col3=st.columns(3)
    
    with col1:
        Pregnancies=st.text_input('Number of Pregancies')
    
    with col2:
        Glucose=st.text_input('Glucose Level')
    
    with col3:
        BloodPressure=st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness=st.text_input('Skin Thickness value')
    
    with col2:
        Insulin=st.text_input('Insulin Level')
    
    with col3:
        BMI=st.text_input('BMI value')
        
    with col1:
        DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function Value')
    
    with col2:
        Age =st.text_input('Age of the Person')
        
    # Code for prediction
    
    diab_dignosis= ''
    
    # creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        
        input_data=([[Pregnancies,Glucose
                    ,BloodPressure,
                       SkinThickness,Insulin,
                          BMI,DiabetesPedigreeFunction,Age]])
        input_numpy=np.asarray(input_data).reshape(1,-1)
        scaler=diabetes_scaler.transform(input_numpy)
        diab_predication=   diabetes_model.predict(scaler)
        if (diab_predication[0]==1):
            diab_dignosis='The Preson is Diabetic'
        else:
            diab_dignosis='The Preson is not Diabetic'
    st.success(diab_dignosis)        
           
if (selected=='Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction')
    
    # Getting the user inputs
    col1,col2,col3=st.columns(3)
    
    with col1:
        Age=st.text_input('Age')
    with col2:
        Sex=st.text_input('Sex')
    with col3:
        cp=st.text_input('Chest Pain Type')
    with col1:
        trestbps=st.text_input('Resting Blood Pressure')
    with col2:
        chol=st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs=st.text_input('Fasting Blood Sugar >120 mg/dl')
    with col1:
        restecg=st.text_input('Resting Electrocardiagraphic results')
    with col2:
        thalach=st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exange=st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak=st.text_input('ST depression induced by exercise')
    with col2:
        slope=st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca=st.text_input('number of major vessels (0-3) colored by flourosop')
    with col1:
        thal=st.text_input('The names and social security')
        
    
    
    # Code for prediction
    
    heart_diag=''
    
    # creating a button for prediction
    if st.button('Heart Test Results'):
        input_user=([[Age,Sex,cp,trestbps,chol,fbs,restecg,thalach,exange, oldpeak,slope,ca,thal]])
        
        #Convert data into array
        input_data_array=np.asarray(input_user).reshape(1,-1)
        
        #convert into scaler
        input_heart_scaler=heart_scaler.fit_transform(input_data_array)
        # print(input_data_array)
        predication=heart_model.predict(input_heart_scaler)
        
        if (predication[0]==1):
            heart_diag='This Person Postive Heart Disease'
        else:
            heart_diag='This Person not have  Heart Disease' 
    st.success(heart_diag)
              
     

if (selected=='Parkinson Prediction'):
    
    # page title
    st.title('Parkinson Prediction')
    
    #Inputdata
    col1,col2,col3=st.columns(3)
    
    
    with col1:
        Mdvpfohz=st.text_input('Average vocal fundamental frequency')
    
    with col2:
        Mdvpfhihz=st.text_input('Maximum vocal fundamental frequency')
        
    with col3:
        mdvpflohz=st.text_input('Minimum vocal fundamental frequency')
    
    with col1:
        mdvpjitter=st.text_input('Percentage of variation in fundamental frequency') 
        
    with col2:
        mdvpjitterabs=st.text_input('Absoulte of variation in fundamental frequency')
        
    with col3:
        mdvprap=st.text_input('Rap')
    
    with col1:
        mdvpppq=st.text_input('Ppq') #
    with col2:
        jitterddp=st.text_input('Ddp') #
    with col3:
        mdvpshimmer=st.text_input('Shimmer')#
        
    with col1:#
        mdvpshimmerdb=st.text_input('ShimmerDB')
        
    with col2:#
        mdvpapq=st.text_input('Measures of APQ')
        
    with col3:#
        ShimmerDDA=st.text_input('ShimmerDDA')     
        
    with col1:#
         Nhr=st.text_input('Measures of NHR')
    
    with col2:#
        Hnr=st.text_input('Measures of HNR')
    
    with col3:#
        Rpde=st.text_input('Dynamical Complexity measures')
        
    with col1:#
        Dfa=st.text_input('Signal fractal scaling exponent')
        
    with col2:#
        spread1=st.text_input('Spread Fundamental')
        
    with col3:#
        spread2=st.text_input('Spread Frequency')
    
    with col1:
        d2=st.text_input('D2 Measure')
    with col2:
        ppe=st.text_input('PPE')
    with col3:
        text1=st.text_input('text1')
    with col1:
        text2=st.text_input('text2')
    with col2:
        text3=st.text_input('text3')
        
        
        
        # Code for predication
        
        parkinson_result=''
        
        if st.button('Parkinson Result'):
            
            input_data=([[Mdvpfohz,Mdvpfhihz,mdvpflohz,mdvpjitter,mdvpjitterabs,mdvprap,
                          mdvpppq,jitterddp,mdvpshimmer,mdvpshimmerdb,mdvpapq,ShimmerDDA,
                          Nhr,Hnr,Dfa,spread1,spread2,d2,ppe,text1,text2,text3]])
            
            #Convert into array
            
            input_parkinson_array=np.asarray(input_data).reshape(1,-1)
            
            #Convert into scaler
            
            input_scaler=parkinson_scaler.fit_transform(input_parkinson_array)
            
            
            predication_parkinson=parkinson_model.predict(input_scaler)
            
            if predication_parkinson[0]==1:
                parkinson_result='This Person has parkinson disease'
            else:
                parkinson_result='This Person does not have parkinson disease'
        st.success(parkinson_result)
                