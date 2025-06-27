import streamlit as st
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    MarianMTModel, MarianTokenizer, pipeline
)

# Load model untuk prediksi penyakit
tokenizer_en = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model_en = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model_en.eval()

# Load model translasi
id_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-id-en")
id_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-id-en")
en_id_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")
en_id_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-id")

es_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
es_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")
en_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en_es_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

# Deteksi bahasa otomatis
lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

# Fungsi translate dua arah
def translate(text, direction):
    if direction == "id-en":
        tokenizer, model = id_en_tokenizer, id_en_model
    elif direction == "en-id":
        tokenizer, model = en_id_tokenizer, en_id_model
    elif direction == "es-en":
        tokenizer, model = es_en_tokenizer, es_en_model
    elif direction == "en-es":
        tokenizer, model = en_es_tokenizer, en_es_model
    else:
        return text

    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Fungsi prediksi kategori penyakit
def predict_disease_category(text_en):
    inputs = tokenizer_en(text_en, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_en(**inputs)
    probs = torch.sigmoid(outputs.logits)
    predicted = (probs > 0.3).nonzero()[:, 1].tolist()
    labels = [model_en.config.id2label[i] for i in predicted]
    return labels

# UI Streamlit
st.set_page_config(page_title="Prediksi Penyakit Multibahasa", layout="centered")

st.markdown("<h1 style='text-align:center;'>🩺 Prediksi Penyakit dari Teks Medis</h1>", unsafe_allow_html=True)
st.write("Masukkan teks medis dalam Bahasa Indonesia, Spanyol, atau Inggris. Sistem akan memproses dan menampilkan hasilnya.")

text_input = st.text_area("📝 Teks Medis:", placeholder="Contoh: Pasien mengalami sesak napas dan nyeri dada...")

if st.button("🔍 Prediksi"):
    if not text_input.strip():
        st.error("⚠ Harap masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Memproses..."):
            # Deteksi bahasa
            lang = lang_detector(text_input)[0]['label']
            lang = lang.lower()

            # Translate ke Inggris jika perlu
            if lang == "id":
                text_en = translate(text_input, "id-en")
            elif lang == "es":
                text_en = translate(text_input, "es-en")
            else:
                text_en = text_input  # Sudah bahasa Inggris

            # Prediksi penyakit
            categories = predict_disease_category(text_en)

            # Translate balik hasil ke bahasa asli
            if categories:
                result_text = ", ".join(categories)
                if lang == "id":
                    result_text = translate(result_text, "en-id")
                elif lang == "es":
                    result_text = translate(result_text, "en-es")

                st.success("✔ Kategori Penyakit Terdeteksi:")
                st.markdown(f"**🗂️ {result_text}**")
            else:
                st.warning("⚠ Tidak ditemukan kategori penyakit yang signifikan.")
