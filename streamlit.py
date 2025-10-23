# app.py
import streamlit as st
import pandas as pd
import re
import emoji
import html
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from transformers import pipeline
import os

# --- Download NLTK data ---
@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk()

# --- Slangwords Dictionary ---
slangwords = {
    "@": "di", "abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar",
    "bgt": "banget", "maks": "maksimal", "bgtt": "banget", "bngt": "banget", "bgtu": "begitu",
    "tp": "tapi", "dg": "dengan", "dgn": "dengan", "dlm": "dalam", "utk": "untuk", "yg": "yang",
    "gk": "tidak", "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "nggaknya": "tidaknya",
    "dr": "dari", "kalo": "kalau", "kl": "kalau", "klo": "kalau", "sm": "sama", "sama2": "sama-sama",
    "tdk": "tidak", "blm": "belum", "sdh": "sudah", "udh": "sudah", "lg": "lagi",
    "knp": "kenapa", "krn": "karena", "karna": "karena", "sll": "selalu", "slalu": "selalu",
    "bkn": "bukan", "bkan": "bukan", "dpt": "dapat", "dptnya": "dapatnya", "jd": "jadi",
    "jdi": "jadi", "jg": "juga", "jga": "juga", "sja": "saja", "sj": "saja", "tpi": "tapi",
    "kpn": "kapan", "bs": "bisa", "bisa2": "bisa-bisa", "smpe": "sampai", "ampe": "sampai",
    "smp": "sampai", "org": "orang", "pke": "pakai", "pke2": "pakai-pakai", "skrg": "sekarang",
    "skrng": "sekarang", "lgsg": "langsung", "lgsg2": "langsung-langsung", "nih": "ini", "ni": "ini",
    "itu": "itu", "itu2": "itu-itu", "kek": "kayak", "ky": "kayak", "kyk": "kayak", "kayak": "kayak",
    "kn": "akan", "knp": "kenapa", "knpa": "kenapa", "knpsi": "kenapa sih", "lo": "kamu", "lu": "kamu",
    "loe": "kamu", "gua": "saya", "gw": "saya", "gue": "saya", "aku": "saya", "ak": "aku",
    "sy": "saya", "sya": "saya", "sia": "saya", "si": "sih", "sihh": "sih", "sihhh": "sih",
    "banget": "banget", "bgt": "banget", "bgtu": "begitu", "bego": "bodoh", "bego2": "bodoh-bodoh",
    "bnyk": "banyak", "bnyak": "banyak", "byk": "banyak", "cpt": "cepat", "cepet": "cepat",
    "cm": "cuma", "cuman": "cuma", "doang": "saja", "doank": "saja", "gitu": "begitu",
    "gt": "begitu", "gtu": "begitu", "gmn": "bagaimana", "gimana": "bagaimana", "gini": "begini",
    "hrs": "harus", "hrus": "harus", "jgn": "jangan", "km": "kamu", "kmrn": "kemarin",
    "krg": "kurang", "lbh": "lebih", "msh": "masih", "nggak": "tidak", "ngga": "tidak",
    "nih": "ini", "ni": "ini", "nnti": "nanti", "ntar": "nanti", "pake": "pakai", "pke": "pakai",
    "sbg": "sebagai", "sblm": "sebelum", "skr": "sekarang", "skrg": "sekarang", "slm": "selama",
    "sm": "sama", "sma": "sama", "smpe": "sampai", "smp": "sampai", "spy": "supaya", "tdk": "tidak",
    "trs": "terus", "trus": "terus", "ttg": "tentang", "ttp": "tetap", "utk": "untuk", "wkt": "waktu",
    "y": "ya", "ya": "ya", "yaa": "ya", "yaaa": "ya", "yup": "ya", "yups": "ya"
}

# --- Preprocessing Functions ---
def cleaningText(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = emoji.replace_emoji(text, "")
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+|https\S+|www\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def casefoldingText(text):
    return text.lower()

def fix_slangwords(text):
    words = text.split()
    fixed_words = [slangwords.get(word.lower(), word) for word in words]
    return ' '.join(fixed_words)

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(tokens):
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords.update(stopwords.words('english'))
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy"])
    return [word for word in tokens if word not in listStopwords]

@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

def stemmingText(text):
    stemmer = get_stemmer()
    return stemmer.stem(text)

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = cleaningText(text)
    text = casefoldingText(text)
    text = fix_slangwords(text)
    tokens = tokenizingText(text)
    tokens = filteringText(tokens)
    text = ' '.join(tokens)
    if text.strip() == "":
        return ""
    text = stemmingText(text)
    return text.strip()

# --- Load Model ---
@st.cache_resource
def load_model():
    with st.spinner("Memuat model sentiment analysis..."):
        try:
            nlp = pipeline(
                "sentiment-analysis",
                model="w11wo/indonesian-roberta-base-sentiment-classifier",
                tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier",
                truncation=True,
                max_length=512
            )
            return nlp
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None

nlp = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Klasifikasi Sentimen Teks", layout="centered")

st.title("🧠 Klasifikasi Sentimen Teks Bahasa Indonesia")
st.markdown('Developer by Sidqiamn')
st.markdown("Masukkan teks ulasan, aplikasi akan memprediksi apakah **positif**, **negatif**, atau **netral**.")

# Input teks
user_input = st.text_area("Masukkan teks di sini:", height=150, placeholder="Contoh: game ini seru banget tapi sering lag")

# Tombol prediksi
if st.button("🔍 Prediksi Sentimen"):
    if not user_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Sedang memproses..."):
            clean_text = preprocess_text(user_input)
            
            if not clean_text:
                st.error("Teks terlalu pendek atau tidak valid setelah preprocessing.")
            else:
                try:
                    result = nlp(clean_text)[0]
                    raw_label = result['label']
                    raw_score = result['score']

                    # Tampilkan hasil
                    st.metric("Label", raw_label.upper())
                    st.progress(raw_score)
                    st.write(f"**Confidence Score:** `{raw_score:.4f}`")

                    # Warna berdasarkan label
                    color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(raw_label, "⚪")
                    st.markdown(f"### {color} **Sentimen: {raw_label.upper()}**")

                    with st.expander("Lihat detail preprocessing"):
                        st.write("**Teks Asli:**", user_input)
                        st.write("**Teks Bersih:**", clean_text)

                    # Simpan riwayat (opsional)
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        'teks': user_input,
                        'clean': clean_text,
                        'label': raw_label,
                        'score': raw_score
                    })

                except Exception as e:
                    st.error(f"Error saat prediksi: {e}")

# --- Riwayat Prediksi ---
if 'history' in st.session_state and st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Riwayat Prediksi")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df[['teks', 'label', 'score']].head(10), use_container_width=True)

    if st.button("🗑️ Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()

# Footer
st.markdown("---")
st.caption("Model: `w11wo/indonesian-roberta-base-sentiment-classifier` | Preprocessing: Sastrawi + Slang Fix")
