import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle

#TITLE

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://tse2.mm.bing.net/th/id/OIP.tiYRfo0QkmNmCS73xlbtAgAAAA?rs=1&pid=ImgDetMain&o=7&rm=3', width = 700)

st.header('A model of housing prices to predict median house values in California', divider = True)
st.header('''User must enter given values to predict Price:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Features 🏠')
st.sidebar.image('https://images.pexels.com/photos/186077/pexels-photo-186077.jpeg')

# read_data
temp_df = pd.read_csv('california.csv')
random.seed(12)
all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min', 'max'])
    
    var = st.sidebar.slider(f'Select {i} Value', int(min_value), int(max_value),
                            random.randint(int(min_value),int(max_value)))
    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])


with open('House_Price_Pred_Ridge_Model.pkl', 'rb') as f:
    chatgpt = pickle.load(f)
    


price = chatgpt.predict(final_value)[0]

import time
st.write(pd.DataFrame(zip(col,all_values)),index =[1])

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price')

place = st.empty()
place.image('https://i.pinimg.com/originals/9f/36/29/9f36292ea8634d4472ae384ed2181640.gif', width = 200)


if price>0:
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
        
    
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    st.success(body)

else:
    body = 'Invalid House features values'
    st.warning(body)