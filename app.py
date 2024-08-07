import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu

diabetes_model=pickle.load(open('diabetic_model.pkl','rb'))
heart_model=pickle.load(open('heart_disease_model.pkl','rb'))
parkinson_model=pickle.load(open('Parkinson_model.pkl','rb'))
parkinson_scaler=pickle.load(open('parkinson_scaler.pkl','rb'))

with st.sidebar:
    selected=option_menu('Multiple Disease Prediction System',
                         ['Diabetes Prediction',
                          'Heart Disease Prediction',
                          'Parkinson Prediction'],
                         icons=['activity','heart','person'],
                         default_index=0)

#Diabetes Page
if selected=='Diabetes Prediction':
    def diabetes_prediction(input_data):
        #changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = diabetes_model.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
            st.success('The person is not diabetic')
        else:
            st.warning('The person may be diabetic')

    st.title('Diabetes Prediction')

    col1,col2,col3=st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

#Heart Disease Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.selectbox('Sex', ['0', '1'])

    with col3:
        cp = st.selectbox('Chest Pain types', ['0', '1', '2', '3'])

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['0', '1'])

    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results', ['0', '1', '2'])

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['0', '1'])

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', ['0', '1', '2'])

    with col3:
        ca = st.selectbox('Major vessels colored by flourosopy', ['0', '1', '2', '3'])

    with col1:
        thal = st.selectbox('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', ['0', '1', '2', '3'])

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        # Convert input to a list of floats
        heart_diagnosis = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg),
                           float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]

        # Make prediction
        heart_prediction = heart_model.predict([heart_diagnosis])

        if heart_prediction[0] == 1:
            st.error('The person is predicted to have heart disease.')
        else:
            st.success('The person is predicted to not have heart disease.')


# Parkinson Page
if selected=='Parkinson Prediction':
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo_input = st.text_input('MDVP: Fo(Hz)')
        fo = float(fo_input.strip()) if fo_input.strip() else 0.0

    with col2:
        fhi_input = st.text_input('MDVP: Fhi(Hz)')
        fhi = float(fhi_input.strip()) if fhi_input.strip() else 0.0

    with col3:
        flo_input = st.text_input('MDVP: Flo(Hz)')
        flo = float(flo_input.strip()) if flo_input.strip() else 0.0

    with col4:
        Jitter_percent_input = st.text_input('MDVP: Jitter(%)')
        Jitter_percent = float(Jitter_percent_input.strip()) if Jitter_percent_input.strip() else 0.0

    with col5:
        Jitter_Abs_input = st.text_input('MDVP: Jitter(Abs)')
        Jitter_Abs = float(Jitter_Abs_input.strip()) if Jitter_Abs_input.strip() else 0.0

    with col1:
        RAP_input = st.text_input('MDVP: RAP')
        RAP = float(RAP_input.strip()) if RAP_input.strip() else 0.0

    with col2:
        PPQ_input = st.text_input('MDVP: PPQ')
        PPQ = float(PPQ_input.strip()) if PPQ_input.strip() else 0.0

    with col3:
        DDP_input = st.text_input('Jitter: DDP')
        DDP = float(DDP_input.strip()) if DDP_input.strip() else 0.0

    with col4:
        Shimmer_input = st.text_input('MDVP: Shimmer')
        Shimmer = float(Shimmer_input.strip()) if Shimmer_input.strip() else 0.0

    with col5:
        Shimmer_dB_input = st.text_input('MDVP: Shimmer(dB)')
        Shimmer_dB = float(Shimmer_dB_input.strip()) if Shimmer_dB_input.strip() else 0.0

    with col1:
        APQ3_input = st.text_input('Shimmer: APQ3')
        APQ3 = float(APQ3_input.strip()) if APQ3_input.strip() else 0.0

    with col2:
        APQ5_input = st.text_input('Shimmer: APQ5')
        APQ5 = float(APQ5_input.strip()) if APQ5_input.strip() else 0.0

    with col3:
        APQ_input = st.text_input('MDVP: APQ')
        APQ = float(APQ_input.strip()) if APQ_input.strip() else 0.0

    with col4:
        DDA_input = st.text_input('Shimmer: DDA')
        DDA = float(DDA_input.strip()) if DDA_input.strip() else 0.0

    with col5:
        NHR_input = st.text_input('NHR')
        NHR = float(NHR_input.strip()) if NHR_input.strip() else 0.0

    with col1:
        HNR_input = st.text_input('HNR')
        HNR = float(HNR_input.strip()) if HNR_input.strip() else 0.0

    with col2:
        RPDE_input = st.text_input('RPDE')
        RPDE = float(RPDE_input.strip()) if RPDE_input.strip() else 0.0

    with col3:
        DFA_input = st.text_input('DFA')
        DFA = float(DFA_input.strip()) if DFA_input.strip() else 0.0

    with col4:
        spread1_input = st.text_input('spread1')
        spread1 = float(spread1_input.strip()) if spread1_input.strip() else 0.0

    with col5:
        spread2_input = st.text_input('spread2')
        spread2 = float(spread2_input.strip()) if spread2_input.strip() else 0.0

    with col1:
        D2_input = st.text_input('D2')
        D2 = float(D2_input.strip()) if D2_input.strip() else 0.0

    with col2:
        PPE_input = st.text_input('PPE')
        PPE = float(PPE_input.strip()) if PPE_input.strip() else 0.0

    # code for Prediction
    parkinsons_diagnosis_list = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE,
                                                           DFA, spread1, spread2, D2, PPE]
    parkinsons_diagnosis_array=np.asarray(parkinsons_diagnosis_list)
    parkinsons_diagnosis=parkinsons_diagnosis_array.reshape(1,-1)
    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        # Scale the input data
        parkinsons_diagnosis_scaled = parkinson_scaler.transform(parkinsons_diagnosis)
        # Make prediction
        parkinsons_prediction = parkinson_model.predict(parkinsons_diagnosis_scaled)

        if (parkinsons_prediction[0] == 1):
            st.warning("The person has Parkinson's disease")
        else:
            st.success("The person does not have Parkinson's disease")
