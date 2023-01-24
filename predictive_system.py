import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('D:/Studies/academics/intern placement prep/Projects/machine learning/calories burnt prediction/trained_model.sav','rb'))

input_data =(0,69,179.0,79.0,5.0,88.0,38.7)


#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)