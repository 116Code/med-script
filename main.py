import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model.eval()

def predict_disease_category(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits)
    predicted_labels = [model.config.id2label[_id] for _id in (predictions > 0.3).nonzero()[:, 1].tolist()]
    return predicted_labels

# Streamlit UI Design
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="centered")

# Custom CSS for modern look
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        h1 {color: #343a40; text-align: center;}
        .stButton>button {width: 100%; padding: 10px; font-size: 18px;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""<h1>🩺 Early Disease Detection</h1>""", unsafe_allow_html=True)
st.write("Enter a medical transcript to predict possible disease categories using AI-powered analysis.")

# Input box with placeholder
text_input = st.text_area("**Medical Record Transcript**", "", placeholder="Type or paste the medical record transcript here...")

# Prediction button
if st.button("🔍 Predict Disease Category"):
    if text_input:
        with st.spinner("Analyzing... 🔬"):
            categories = predict_disease_category(text_input)
        
        # Display results
        st.subheader("Predicted Categories:")
        if categories:
            st.success("✔ " + ", ".join(categories))
        else:
            st.warning("⚠ No significant disease category detected.")
    else:
        st.error("⚠ Please enter text before predicting.")

# Footer
# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Developed @2025</p>
""", unsafe_allow_html=True)
