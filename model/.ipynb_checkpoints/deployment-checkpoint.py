import streamlit as st
import numpy as np
import pickle
import json

# Load the trained model
with open('banglorehomepricesmodel.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the columns info for one-hot encoding
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# List of locations extracted from columns (all except initial features)
locations = data_columns[3:]

def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1
    
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    
    return model.predict([x])[0]

def main():

    st.title('Bangalore Real Estate Price Prediction')

    location = st.selectbox('Location', locations)
    sqft = st.number_input('Total Square Feet', min_value=300.0, max_value=10000.0, step=10.0)
    bath = st.number_input('Number of Bathrooms', min_value=1, max_value=10, step=1)
    bhk = st.number_input('Number of Bedrooms (BHK)', min_value=1, max_value=10, step=1)

    if st.button('Predict Price'):
        price = predict_price(location, sqft, bath, bhk)
        st.success(f'Estimated price: ₹ {price:.2f} Lakhs')

if __name__ == '__main__':
    main()
