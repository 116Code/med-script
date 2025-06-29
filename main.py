import streamlit as st
import torch
import sys
import types

# Workaround untuk bug torch.classes path
if isinstance(getattr(__import__('torch'), 'classes', None), types.ModuleType):
    setattr(sys.modules['torch.classes'], '__path__', [])

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM, pipeline
)

# Load model prediksi penyakit (sudah dalam bahasa Inggris)
tokenizer_en = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model_en = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model_en.eval()

# Load model NLLB untuk translasi multibahasa (tanpa sentencepiece)
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Kode bahasa NLLB
lang_code_map = {
    "id": "ind_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn"
}

# Deteksi bahasa otomatis
lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

# Fungsi translasi dua arah pakai NLLB
def translate_nllb(text, src_lang, tgt_lang):
    src_code = lang_code_map.get(src_lang)
    tgt_code = lang_code_map.get(tgt_lang)

    if not src_code or not tgt_code:
        return text  # fallback jika kode tidak ditemukan

    tokenizer = nllb_tokenizer
    model = nllb_model

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs["input_ids"][0][0] = tokenizer.lang_code_to_id[src_code]

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
        max_length=512
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Fungsi prediksi penyakit
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
            lang = lang_detector(text_input)[0]['label'].lower()

            # Translate ke Inggris jika perlu
            if lang in ["id", "es"]:
                text_en = translate_nllb(text_input, lang, "en")
            else:
                text_en = text_input  # Sudah bahasa Inggris

            # Prediksi kategori penyakit
            categories = predict_disease_category(text_en)

            # Translate balik hasil jika perlu
            if categories:
                result_text = ", ".join(categories)
                if lang in ["id", "es"]:
                    result_text = translate_nllb(result_text, "en", lang)

                st.success("✔ Kategori Penyakit Terdeteksi:")
                st.markdown(f"**🗂️ {result_text}**")
            else:
                st.warning("⚠ Tidak ditemukan kategori penyakit yang signifikan.")
