from deep_translator import GoogleTranslator
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model hanya sekali
diag_tokenizer = AutoTokenizer.from_pretrained("kamalkraj/bioelectra-base-discriminator-bioNER")
diag_model = AutoModelForSequenceClassification.from_pretrained("kamalkraj/bioelectra-base-discriminator-bioNER")

def detect_and_translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_to_original(text, target_lang):
    return GoogleTranslator(source='en', target=target_lang).translate(text)

def predict_disease_category(text_en):
    inputs = diag_tokenizer(text_en, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = diag_model(**inputs)
    predictions = torch.sigmoid(outputs.logits)
    predicted_labels = [diag_model.config.id2label[_id] for _id in (predictions > 0.3).nonzero()[:, 1].tolist()]
    return predicted_labels

# Streamlit UI
st.title("Medical Transcript Classification (Multilingual)")

text_input = st.text_area("Masukkan teks medis (Bahasa Indonesia / Spanyol / Inggris):")

if st.button("Prediksi"):
    with st.spinner("Sedang memproses..."):
        # Deteksi bahasa dan translate ke Inggris
        translated_text = detect_and_translate_to_english(text_input)

        # Proses prediksi dengan teks berbahasa Inggris
        results = predict_disease_category(translated_text)

        # Translate hasil label kembali ke bahasa awal (opsional)
        detected_lang = GoogleTranslator(source='auto', target='en').detect_language(text_input)
        translated_labels = [translate_to_original(label, detected_lang) for label in results]

    st.success("Kategori Prediksi:")
    for label in translated_labels:
        st.write(f"- {label}")
