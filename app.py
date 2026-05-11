#python -m streamlit run app.py ile calıstır terminalde

import streamlit as st
import pandas as pd
import numpy as np
import os
import pydicom
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Karaciğer Dijital İkiz", layout="wide", page_icon="🩺")

# --- MODEL EĞİTİMİ (Önbelleğe alınır, böylece her kaydırmada baştan eğitilmez) ---
@st.cache_resource
def model_egit():
    try:
        df1 = pd.read_csv('liver_cirrhosis.csv')
        df1['Age_Y'] = (df1['Age'] / 365).astype(int)
        df1_clean = df1[['Age_Y', 'Sex', 'Bilirubin', 'Albumin', 'Stage']].copy()
        df1_clean.columns = ['Age', 'Gender', 'Bilirubin', 'Albumin', 'Target']

        df2 = pd.read_csv('indian_liver_patient.csv')
        df2['Target'] = df2['Dataset'].map({2: 1.0, 1: 3.0})
        df2_clean = df2[['Age', 'Gender', 'Total_Bilirubin', 'Albumin', 'Target']].copy()
        df2_clean.columns = ['Age', 'Gender', 'Bilirubin', 'Albumin', 'Target']

        df_final = pd.concat([df1_clean, df2_clean], axis=0).reset_index(drop=True)
        df_final = df_final.fillna(df_final.median(numeric_only=True))

        np.random.seed(42)
        df_final['Image_Score'] = np.random.uniform(0.1, 0.9, size=len(df_final))

        le = LabelEncoder()
        df_final['Gender'] = le.fit_transform(df_final['Gender'])

        X = df_final[['Age', 'Gender', 'Bilirubin', 'Albumin', 'Image_Score']]
        y = df_final['Target'].astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test, = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        return model, scaler
    except Exception as e:
        st.error(f"Veri yükleme hatası: Lütfen CSV dosyalarının klasörde olduğundan emin olun. Detay: {e}")
        return None, None

# --- GÖRÜNTÜ ANALİZİ (Önbelleğe alınır) ---
@st.cache_data
def goruntu_analizi_yap(path):
    files = glob.glob(f"{path}/**/*.dcm", recursive=True)
    if not files: return 0.5, None
    ds = pydicom.dcmread(files[0])
    img = ds.pixel_array
    score = np.mean(img) / np.max(img)
    return score, img

# Modeli ve Verileri Yükle
model, scaler = model_egit()
goruntu_skoru, bt_goruntusu = goruntu_analizi_yap("./karaciger_veri_seti_mini")

# --- ARAYÜZ TASARIMI ---
st.title("🩺 Karaciğer Hibrit Dijital İkiz Simülatörü")
st.markdown("Klinik kan değerleri ile radyolojik BT görüntülerini birleştiren Karar Destek Sistemi.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Klinik Girdiler")
    yas = st.slider("Yaş", 18, 90, 25)
    cinsiyet = st.radio("Cinsiyet", ["Kadın", "Erkek"])
    bili = st.slider("Bilirubin Seviyesi (Normal: 0.5 - 1.1)", 0.1, 10.0, 0.8, step=0.1)
    albu = st.slider("Albumin Seviyesi (Normal: 3.5 - 5.0)", 1.0, 6.0, 4.0, step=0.1)
    
    st.divider()
    st.subheader("📷 Radyolojik Veri")
    if bt_goruntusu is not None:
        st.info(f"Arşivden BT Kesiti Yüklendi. Doku Skoru: **%{goruntu_skoru*100:.2f}**")
        fig, ax = plt.subplots()
        # Pencereleme (Windowing) ile daha net görüntü
        ax.imshow(bt_goruntusu, cmap=plt.cm.bone, vmin=0, vmax=150)
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("BT Görüntüsü bulunamadı. Lütfen klasör yolunu kontrol edin.")

with col2:
    st.subheader("🔬 Dijital İkiz Analiz Sonucu")
    
    if model and st.button("Tanı Senaryosunu Çalıştır", type="primary", use_container_width=True):
        gen_code = 1 if cinsiyet == "Erkek" else 0
        input_data = scaler.transform([[yas, gen_code, bili, albu, goruntu_skoru]])
        tahmin = model.predict(input_data)[0]
        
        durum_detaylari = {
            1: {"Baslik": "EVRE 1 (SAĞLIKLI / NORMAL DOKU)", "Renk": "success", "Aciklama": "Değerler normal sınırlardadır. Hücresel hasar veya sertleşme tespit edilmemiştir.", "Tavsiye": "Rutin kontrollere devam edilmeli."},
            2: {"Baslik": "EVRE 2 (FİBROZİS BAŞLANGICI)", "Renk": "warning", "Aciklama": "Hafif doku sertleşmesi (fibrozis) sinyalleri mevcuttur. Hasar geri döndürülebilir.", "Tavsiye": "Karaciğeri yoran toksinlerden uzak durulmalı, kilo kontrolü sağlanmalı."},
            3: {"Baslik": "EVRE 3 (İLERİ HASAR)", "Renk": "error", "Aciklama": "Belirgin doku bozulmaları ve kronik yetmezlik bulguları saptanmıştır.", "Tavsiye": "Uzman kontrolünde tedaviye başlanmalı, özel diyet uygulanmalı."},
            4: {"Baslik": "EVRE 4 (SİROZ)", "Renk": "error", "Aciklama": "Karaciğer dokusu yapısal bütünlüğünü kaybetmiş ve kronik siroz evresine geçilmiştir.", "Tavsiye": "Durum kritiktir, acil tam teşekküllü hastane müdahalesi gereklidir."}
        }
        
        sonuc = durum_detaylari.get(tahmin)
        
        if sonuc["Renk"] == "success":
            st.success(f"### {sonuc['Baslik']}")
        elif sonuc["Renk"] == "warning":
            st.warning(f"### {sonuc['Baslik']}")
        else:
            st.error(f"### {sonuc['Baslik']}")
            
        st.markdown(f"**Açıklama:** {sonuc['Aciklama']}")
        st.markdown(f"**Protokol:** {sonuc['Tavsiye']}")
        
        if tahmin > 1:
            st.info("⚠️ **Projeksiyon:** Parametrelerdeki ufak bozulmalar dijital ikiz durumunu bir sonraki risk evresine taşıyabilir.")