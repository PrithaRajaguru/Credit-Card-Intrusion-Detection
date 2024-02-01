import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
import pyttsx3

def voice_out(havetosay):
    initiate = pyttsx3.init()
    initiate.setProperty('rate', 190) 
    voices = initiate.getProperty('voices')
    initiate.setProperty('voice', voices[0].id) 
    initiate.say(havetosay)
    initiate.runAndWait()

st.set_page_config(layout='wide', page_title='Credit Card Intrusion Detection System', page_icon='💳')

data = pd.read_csv("creditcard.csv")

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

model = RandomForestClassifier() 
model.fit(X_train, y_train)

train_acc = accuracy_score(model.predict(X_train), y_train)*100
test_acc = accuracy_score(model.predict(X_test), y_test)*100

st.title("Credit Card Intrusion Detection System")

random_list = []

col3, col4, col5 = st.columns([6,0.5,3.5])
with col5:
    image_path = "image.png"
    img = Image.open(image_path)
    st.image(img)
with col3:
    col1, col2 = st.columns([8,2])
    with col1:
        st.subheader("Generate Random Values")
    with col2:
        ''
        random = st.button('Random')
    if random:
        for column in X.columns:
            min_value = X[column].min()
            max_value = X[column].max()
            random_number = np.random.uniform(min_value, max_value)
            random_list.append(random_number)
        random_values = ', '.join(map(str,random_list))
        st.code(random_values)
    ''

    input_df = st.text_input('Enter All the Features')
    col6, col7 = st.columns([8,2])
    with col7:
        input_df_lst = input_df.split(',')
        submit = st.button("Submit")
    if submit:  
        features = np.array(input_df_lst, dtype=np.float64)
        prediction = model.predict(features.reshape(1,-1))

        if prediction[0] == 0:
            st.info("The Input Features represent the **LEGITIMATE** Transaction")
            havetosay = "The Input Features represent the LEGITIMATE Transaction"
            voice_out(havetosay)
        else:
            st.info("The Input Features represent the **FRAUDULENT** Transaction")
            havetosay = "The Input Features represent the FRAUDULENT Transaction"
            voice_out(havetosay)