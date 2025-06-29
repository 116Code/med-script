import streamlit as st
import torch
import sys
import types
import argostranslate.package
import argostranslate.translate
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Workaround untuk bug torch.classes path (jika diperlukan)
if isinstance(getattr(__import__('torch'), 'classes', None), types.ModuleType):
    setattr(sys.modules['torch.classes'], '__path__', [])

# --- Load model diagnosis ---
@st.cache_resource
def load_diagnosis_model():
    tokenizer = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
    model = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
    model.eval()
    return tokenizer, model

tokenizer_en, model_en = load_diagnosis_model()

# --- Fungsi Translate Offline dengan Argos Translate ---
def translate_argos(text, from_code="es", to_code="en"):
    try:
        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next((lang for lang in installed_languages if lang.code == from_code), None)
        to_lang = next((lang for lang in installed_languages if lang.code == to_code), None)

        if not from_lang or not to_lang:
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package = next(
                filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages)
            )
            argostranslate.package.install_from_path(package.download())
            installed_languages = argostranslate.translate.get_installed_languages()
            from_lang = next((lang for lang in installed_languages if lang.code == from_code), None)
            to_lang = next((lang for lang in installed_languages if lang.code == to_code), None)

        translation = from_lang.get_translation(to_lang)
        return translation.translate(text)

    except Exception as e:
        # Tampilkan traceback atau error sebenarnya
        import traceback
        st.error("Terjadi kesalahan translasi:")
        st.text(traceback.format_exc())
        return "[Terjadi Error Translasi]"

# --- Fungsi Prediksi Penyakit ---
def predict_disease_category(text_en):
    inputs = tokenizer_en(text_en, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_en(**inputs)
    probs = torch.sigmoid(outputs.logits)
    predicted = (probs > 0.3).nonzero()[:, 1].tolist()
    labels = [model_en.config.id2label[i] for i in predicted]
    return labels

# --- UI Streamlit ---
st.set_page_config(page_title="Prediksi Penyakit Multibahasa", layout="centered")
st.markdown("<h1 style='text-align:center;'>🩺 Prediksi Penyakit dari Teks Medis (Multibahasa & Offline)</h1>", unsafe_allow_html=True)
st.write("Masukkan teks medis dalam Bahasa **Indonesia**, **Spanyol**, atau **Inggris**. Sistem akan menerjemahkan otomatis dan memprediksi kategori penyakit.")

text_input = st.text_area("📝 Teks Medis:", placeholder="Contoh: el paciente siente que le falta el aire")

if st.button("🔍 Prediksi"):
    if not text_input.strip():
        st.error("⚠ Harap masukkan teks terlebih dahulu.")
    else:
        with st.spinner("🚀 Memproses..."):

            lang = detect(text_input)  # 'id', 'en', 'es', etc.

            # Translasi ke Inggris jika perlu
            if lang == "id":
                text_en = translate_argos(text_input, from_code="id", to_code="en")
            elif lang == "es":
                text_en = translate_argos(text_input, from_code="es", to_code="en")
            else:
                text_en = text_input

            # Prediksi penyakit
            categories = predict_disease_category(text_en)

            # Translate hasil balik
            if categories:
                result_text = ", ".join(categories)
                if lang == "id":
                    result_text = translate_argos(result_text, from_code="en", to_code="id")
                elif lang == "es":
                    result_text = translate_argos(result_text, from_code="en", to_code="es")

                st.success("✔ Kategori Penyakit Terdeteksi:")
                st.markdown(f"**🗂️ {result_text}**")
            else:
                st.warning("⚠ Tidak ditemukan kategori penyakit yang signifikan.")
