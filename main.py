import streamlit as st
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    M2M100Tokenizer, M2M100ForConditionalGeneration
)

# Load translation model
trans_model_name = "facebook/m2m100_418M"
trans_tokenizer = M2M100Tokenizer.from_pretrained(trans_model_name)
trans_model = M2M100ForConditionalGeneration.from_pretrained(trans_model_name)

# Load disease prediction model (English)
diag_tokenizer = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
diag_model = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
diag_model.eval()

# Language detection helper
from langdetect import detect

def translate(text, src_lang, tgt_lang):
    trans_tokenizer.src_lang = src_lang
    encoded = trans_tokenizer(text, return_tensors="pt")
    generated = trans_model.generate(
        **encoded,
        forced_bos_token_id=trans_tokenizer.lang_code_to_id[tgt_lang]
    )
    return trans_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def predict_disease_category(text):
    # Translate ke Inggris jika perlu
    lang_code_map = {"id": "ind_Latn", "es": "spa_Latn", "en": "eng_Latn"}
    detected_lang = detect(text)

    if detected_lang != "en":
        src_lang = lang_code_map.get(detected_lang, "eng_Latn")
        text_en = translate(text, src_lang, "eng_Latn")
    else:
        text_en = text

    # Diagnosis
    inputs = diag_tokenizer(text_en, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = diag_model(**inputs)
    predictions = torch.sigmoid(outputs.logits)
    predicted_labels = [diag_model.config.id2label[_id] for _id in (predictions > 0.3).nonzero()[:, 1].tolist()]

    # Translate hasil kembali ke bahasa awal
    if detected_lang != "en" and predicted_labels:
        predicted_labels_translated = [translate(label, "eng_Latn", lang_code_map[detected_lang]) for label in predicted_labels]
    else:
        predicted_labels_translated = predicted_labels

    return predicted_labels_translated

# Streamlit UI
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="centered")
st.markdown("""<h1 style='text-align: center;'>🩺 Early Disease Detection</h1>""", unsafe_allow_html=True)
st.write("Enter a medical transcript in Indonesian, Spanish, or English. The system will detect the language, analyze, and return predicted disease categories.")

text_input = st.text_area("**Medical Record Transcript**", "", placeholder="Tulis teks medis dalam bahasa Indonesia, Spanyol, atau Inggris...")

if st.button("🔍 Predict Disease Category"):
    if text_input.strip():
        with st.spinner("Analyzing... 🔬"):
            categories = predict_disease_category(text_input)
        st.subheader("Predicted Categories:")
        if categories:
            st.success("✔ " + ", ".join(categories))
        else:
            st.warning("⚠ No significant disease category detected.")
    else:
        st.error("⚠ Please enter some text.")

st.markdown("<hr><p style='text-align: center;'>Developed @2025</p>", unsafe_allow_html=True)
