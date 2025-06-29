import streamlit as st
import torch
import requests
import sys
import types

# Workaround untuk bug torch.classes path
if isinstance(getattr(__import__('torch'), 'classes', None), types.ModuleType):
    setattr(sys.modules['torch.classes'], '__path__', [])

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline
)

# Load model untuk prediksi penyakit
tokenizer_en = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model_en = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model_en.eval()

# Deteksi bahasa otomatis
from langdetect import detect

# Fungsi translate dengan LibreTranslate
def translate_libretranslate(text, source_lang="id", target_lang="en"):
    try:
        url = "https://de.libretranslate.com/translate"
        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        return response.json()["translatedText"]
    except Exception as e:
        return f"[Error Translating: {str(e)}]"



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
            # lang = lang_detector(text_input)[0]['label'].lower()
            
            lang = detect(text_input)  # output langsung kode: 'en', 'id', 'es', dll

            # Translate ke Inggris jika perlu
            if lang == "id":
                text_en = translate_libretranslate(text_input, "id", "en")
            elif lang == "es":
                text_en = translate_libretranslate(text_input, "es", "en")
            else:
                text_en = text_input

            # Prediksi penyakit
            categories = predict_disease_category(text_en)

            # Translate balik hasil ke bahasa asli
            if categories:
                result_text = ", ".join(categories)
                if lang == "id":
                    result_text = translate_libretranslate(result_text, "en", "id")
                elif lang == "es":
                    result_text = translate_libretranslate(result_text, "en", "es")

                st.success("✔ Kategori Penyakit Terdeteksi:")
                st.markdown(f"**🗂️ {result_text}**")
            else:
                st.warning("⚠ Tidak ditemukan kategori penyakit yang signifikan.")
