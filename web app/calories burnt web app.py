from markupsafe import string
import numpy as np 
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('D:/Studies/academics/intern placement prep/Projects/machine learning/calories burnt prediction/trained_model.sav','rb'))

#creating a function for prediction

def calories_prediction(input_data):
    

    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=object)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    
    st.title('Calories Burnt Prediction')
    
    #getting the input data from the user
    
    Gender = st.text_input('Enter your gender: ')
    Age = st.text_input('Enter your age: ')
    Height = st.text_input('Enter your Height: ')
    Weight= st.text_input('Enter your weight: ')
    Duration = st.text_input('Enter the duration of the exercise: ')
    Heart_Rate = st.text_input('Enter your Heart Rate: ')
    Body_Temp = st.text_input('Enter your body temperature: ')
    
    #code for the prediction
    diagnosis=''
    
    #creating a button for prediction
    if st.button('How much calories did I burnt?'):
        diagnosis = calories_prediction([Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp])
    
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()