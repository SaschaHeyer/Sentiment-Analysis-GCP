import streamlit as st
import requests
import json

predictionEndpoint = 'https://sentiment-nnlg3yqwxa-ue.a.run.app/sentiment'

@st.cache(suppress_st_warning=True)
def getPrediction(text):
    data = {'text': text}
    response=requests.post(predictionEndpoint, json = data)
    responseTime=response.elapsed.total_seconds()*1000
    statusCode=response.status_code

    return responseTime, statusCode, response

st.title('Sentiment Analysis')
st.write('Sentiment Analysis takes the text and identifies the emotional meanings behind it. This is useful in situations such as identifying negative support requests, analyzing how people feel about your brand or product, and better understanding customer needs in general.')
st.markdown('The model training and prediction code is covered in the corresponding GitHub repository https://github.com/SaschaHeyer/Sentiment-Analysis-GCP.')
text_input = st.text_area('Text:')

st.write("The first request might take a while (service scales to zero). Subsequent requests are faster.")

if st.button("Predict"):
    responseTime, statusCode, response = getPrediction(text_input)
        
    st.subheader('Prediction')
    st.write("Response time: ", responseTime, 'ms')
    st.write("Response code:", statusCode)

    prediction = json.loads(response.text)
    st.write(prediction)