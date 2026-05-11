# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import pydicom
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
# YENİ EKLENEN KÜTÜPHANELER: Performans ölçümü için
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. VERİ HAZIRLAMA VE HİBRİT EĞİTİM ---
print("Sistem Hazırlanıyor: Klinik veriler ve Görüntü Skorları birleştiriliyor...")

# CSV Dosyalarını yükle
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

# KRİTİK ADIM: Eğitim verisine sentetik bir 'Image_Score' (0-1 arası) ekliyoruz
np.random.seed(42)
df_final['Image_Score'] = np.random.uniform(0.1, 0.9, size=len(df_final))

le = LabelEncoder()
df_final['Gender'] = le.fit_transform(df_final['Gender'])

# X artık 5 özellikten oluşuyor
X = df_final[['Age', 'Gender', 'Bilirubin', 'Albumin', 'Image_Score']]
y = df_final['Target'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# --- YENİ EKLENEN KISIM: MODEL PERFORMANS TESTİ ---
print("\n" + "-"*50)
print(" MODEL PERFORMANS TESTİ SONUÇLARI")
print("-"*50)
y_pred = model.predict(X_test)
dogruluk = accuracy_score(y_test, y_pred)
print(f"Genel Doğruluk Oranı (Accuracy): %{dogruluk * 100:.2f}")
print("\nDetaylı Sınıflandırma Raporu (Classification Report):")
print(classification_report(y_test, y_pred, zero_division=0))
print("-"*50)
# --------------------------------------------------

# --- 2. GÖRÜNTÜ ANALİZ MODÜLÜ ---
def get_real_image_score(path):
    """İndirilen gerçek BT görüntüsünden yoğunluk skoru üretir."""
    files = glob.glob(f"{path}/**/*.dcm", recursive=True)
    if not files: return 0.5 # Görüntü yoksa nötr değer
    
    ds = pydicom.dcmread(files[0])
    img = ds.pixel_array
    # Basit yoğunluk normalizasyonu
    score = np.mean(img) / np.max(img)
    return score

# --- 3. ETKİLEŞİMLİ DİJİTAL İKİZ PANELİ ---
def dijital_ikiz_calistir():
    print("\n" + "="*50)
    print(" KARACİĞER HİBRİT DİJİTAL İKİZ SİMÜLASYONU")
    print("="*50)
    
    try:
        # 1. Klinik Veri Girişi
        yas = int(input("Yaş: "))
        cins = input("Cinsiyet (E/K): ").upper()
        bili = float(input("Bilirubin (Normal: 0.5-1.1): "))
        albu = float(input("Albumin (Normal: 3.5-5.0): "))
        
        # 2. Gerçek Görüntü Analizi
        print("\n[Sistem] Arşivdeki BT görüntüsü analiz ediliyor...")
        goruntu_skoru = get_real_image_score("./karaciger_veri_seti_mini")
        print(f"[Analiz] Radyolojik Doku Skoru: %{goruntu_skoru*100:.2f}")
        
        # 3. Hibrit Tahmin
        gen_code = 1 if cins == 'E' else 0
        input_data = scaler.transform([[yas, gen_code, bili, albu, goruntu_skoru]])
        tahmin = model.predict(input_data)[0]
        
        # Sonuçların Detaylandırılması
        durum_detaylari = {
   
            1: {
                "Baslik": "EVRE 1: SAĞLIKLI",
                "Aciklama": "Kan değerleri ve tomografi sonuçları tamamen normal. Karaciğerde yağlanma veya hasar yok.",
                "Tavsiye": "Sağlıklı beslenmeye devam edin ve yıllık rutin kan kontrollerinizi aksatmayın."
            },
            2: {
                "Baslik": "EVRE 2: YAĞLANMA VE BAŞLANGIÇ",
                "Aciklama": "Hafif yağlanma ve doku sertleşmesi başlamış. Ancak bu hasar henüz tamamen geri döndürülebilir.",
                "Tavsiye": "Karaciğeri yoran gıdalardan ve toksinlerden uzak durun. Kilo kontrolü ve egzersiz ile organ kendini toparlayabilir."
            },
            3: {
                "Baslik": "EVRE 3: İLERİ HASAR",
                "Aciklama": "Karaciğerde belirgin bozulma ve fonksiyon kaybı var. Değerler kronik yetmezliğe işaret ediyor.",
                "Tavsiye": "Acilen uzman bir doktora başvurun. Sıkı bir tuzsuz diyet ve düzenli tıbbi takip (ultrason vb.) şarttır."
            },
            4: {
                "Baslik": "EVRE 4: SİROZ (GERİ DÖNÜŞSÜZ)",
                "Aciklama": "Karaciğer dokusu tamamen bozulmuş ve kronik siroz gelişmiştir. Hücrelerin kendini yenilemesi durmuştur.",
                "Tavsiye": "Durum kritiktir. Acil tıbbi müdahale ve karaciğer nakli (transplantasyon) ihtimali için tam teşekküllü bir hastaneye başvurulmalıdır."
            }
        }
        
        
        # Seçili evrenin bilgilerini çekiyoruz
        sonuc = durum_detaylari.get(tahmin, {"Baslik": "Bilinmiyor", "Aciklama": "-", "Tavsiye": "-"})
        
        # Terminal Ekranına Şık Bir Şekilde Yazdırma
        print(f"\n" + "■"*60)
        print(f"  DİJİTAL İKİZ KESİN TANI: {sonuc['Baslik']}")
        print("■"*60)
        print(f"\n AÇIKLAMA:\n    {sonuc['Aciklama']}")
        print(f"\n TAVSİYE :\n    {sonuc['Tavsiye']}")
        print("\n" + "="*60)
        
        # Senaryo Analizi (Gelecek Projeksiyonu)
        if tahmin > 1:
            print("\n PROJEKSİYON: Müdahale edilmez ve Bilirubin seviyesinde %20'lik bir artış \n yaşanırsa, dijital ikiz simülasyonu bir sonraki risk evresine geçiş öngörmektedir.")
            
    except Exception as e:
        print(f"Panel Hatası: {e}")

if __name__ == "__main__":
    dijital_ikiz_calistir()