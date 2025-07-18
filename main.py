import streamlit as st
import torch
import requests
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Load model diagnosis ---
@st.cache_resource
def load_diagnosis_model():
    tokenizer = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
    model = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
    model.eval()
    return tokenizer, model

tokenizer_en, model_en = load_diagnosis_model()

# --- Terjemahan via LibreTranslate.de ---
def translate_mymemory(text, source_lang="id", target_lang="en"):
    try:
        url = "https://api.mymemory.translated.net/get"
        params = {
            "q": text,
            "langpair": f"{source_lang}|{target_lang}"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return data['responseData']['translatedText']
    except Exception as e:
        return f"[Error Translating: {e}]"



# --- Prediksi penyakit ---
def predict_disease_category(text_en):
    inputs = tokenizer_en(text_en, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_en(**inputs)
    probs = torch.sigmoid(outputs.logits)
    predicted = (probs > 0.3).nonzero()[:, 1].tolist()
    labels = [model_en.config.id2label[i] for i in predicted]
    return labels

# --- UI Streamlit ---
st.set_page_config(page_title="Multilingual Disease Prediction", layout="centered")
st.markdown("<h1 style='text-align:center;'>🩺 Disease Prediction from Medical Text</h1>", unsafe_allow_html=True)
st.write("Enter medical text in **Indonesian**, **Spanish**, or **English**. The system predict the disease category.")

text_input = st.text_area("📝 Medical Text:", placeholder="Example: the patient feels short of breath")

if st.button("🔍 Predict"):
    if not text_input.strip():
        st.error("⚠ Please enter the text first.")
    else:
        with st.spinner("🚀 Processing..."):

            lang = detect(text_input)  # ex: 'es', 'id', 'en'

            # Translasi ke Inggris jika bukan English
            if lang == "id":
                text_en = translate_mymemory(text_input, source_lang="id", target_lang="en")
            elif lang == "es":
                text_en = translate_mymemory(text_input, source_lang="es", target_lang="en")
            else:
                text_en = text_input

            # Prediksi penyakit
            categories = predict_disease_category(text_en)

            # Translate hasil balik ke bahasa awal jika perlu
            if categories:
                result_text = ", ".join(categories)
                if lang == "id":
                    result_text = translate_mymemory(result_text, source_lang="en", target_lang="id")
                elif lang == "es":
                    result_text = translate_mymemory(result_text, source_lang="en", target_lang="es")

                st.success("✔ Category of Disease Detected:")
                st.markdown(f"**🗂️ {result_text}**")
            else:
                st.warning("⚠ No significant disease categories were found.")
