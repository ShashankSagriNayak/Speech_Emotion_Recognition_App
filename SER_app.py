import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load models
bilstm_model = load_model('bilstm_model.h5')
crnn_model = load_model('crnn_model.h5')
#transformer_model = load_model('transformer_model.h5')
resnet_model = load_model('resnet_model.h5')

# Define emotion labels
emotion_labels = ['Anger','Disgust','Fear','Happiness','Neutral','Sadness']  

# Preprocess the audio
def preprocess_audio(file, sr=16000, n_mfcc=40):
    y, sr = librosa.load(file, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    mfcc_features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    mfcc_scaled = np.mean(mfcc_features.T, axis=0)
    return np.expand_dims(mfcc_scaled, axis=0)

# Predict emotion
def predict_emotion(file_path, model):
    processed_audio = preprocess_audio(file_path)
    prediction = model.predict(processed_audio)
    return emotion_labels[np.argmax(prediction)]

# Streamlit interface
st.title('Speech Emotion Recognition App')

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Select model
    model_option = st.selectbox('Choose a model', ('BiLSTM', 'CRNN', 'Transformer', 'ResNet'))

    if st.button('Predict'):
        if model_option == 'BiLSTM':
            emotion = predict_emotion(uploaded_file, bilstm_model)
        elif model_option == 'CRNN':
            emotion = predict_emotion(uploaded_file, crnn_model)
       # elif model_option == 'Transformer':
          #  emotion = predict_emotion(uploaded_file, transformer_model)
        elif model_option == 'ResNet':
            emotion = predict_emotion(uploaded_file, resnet_model)
        
        st.write(f'Predicted Emotion: {emotion}')