import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. VERİ SETLERİNİ YÜKLE VE EŞİTLE
print("Sistem Hazırlanıyor, lütfen bekleyin...")

# Set 1: Mayo Clinic (Siroz Odaklı)
df1 = pd.read_csv('liver_cirrhosis.csv')
df1['Age_Y'] = (df1['Age'] / 365).astype(int)
df1_clean = df1[['Age_Y', 'Sex', 'Bilirubin', 'Albumin', 'Stage']].copy()
df1_clean.columns = ['Age', 'Gender', 'Bilirubin', 'Albumin', 'Target']

# Set 2: Indian Liver Patient (Denge Odaklı)
df2 = pd.read_csv('indian_liver_patient.csv')
# Dataset sütunu: 1 (Hasta), 2 (Sağlıklı). 
# Sağlıklıları Evre 1, Hastaları Evre 3 gibi kabul ederek havuzu genişletiyoruz.
df2['Target'] = df2['Dataset'].map({2: 1.0, 1: 3.0})
df2_clean = df2[['Age', 'Gender', 'Total_Bilirubin', 'Albumin', 'Target']].copy()
df2_clean.columns = ['Age', 'Gender', 'Bilirubin', 'Albumin', 'Target']

# İki Veri Setini Birleştir (Hibrit Havuz)
df_final = pd.concat([df1_clean, df2_clean], axis=0).reset_index(drop=True)

# 2. ÖN İŞLEME
df_final = df_final.fillna(df_final.median(numeric_only=True))
le = LabelEncoder()
df_final['Gender'] = le.fit_transform(df_final['Gender']) # F=0, M=1 gibi

X = df_final[['Age', 'Gender', 'Bilirubin', 'Albumin']]
y = df_final['Target'].astype(int)

# Veriyi Ölçeklendir (Standartlaştırma)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. MODELİ EĞİT (Daha Güçlü Parametreler)
model = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 4. BAŞARI RAPORU VE AYRINTILI YORUMLAR
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"\nHibrit Model Başarı Oranı: %{accuracy*100:.2f}")

def klinik_detaylandir(evre):
    bilgi = {
        1: {
            "Durum": "SAĞLIKLI / EVRE 1",
            "Detay": "Karaciğer dokusu normal fonksiyonlarını sürdürüyor. Ciddi bir yağlanma veya sertleşme bulgusu yok.",
            "Tavsiye": "Düzenli egzersiz ve dengeli beslenmeye devam edilmelidir."
        },
        2: {
            "Durum": "EVRE 2 (FİBROZİS)",
            "Detay": "Karaciğerde hafif skar (yara) dokusu oluşumu başlamış. Fonksiyonlar hala büyük oranda korunuyor.",
            "Tavsiye": "Alkol ve karaciğeri yoran gereksiz ilaç kullanımından kaçınılmalıdır."
        },
        3: {
            "Durum": "EVRE 3 (İLERİ HASAR)",
            "Detay": "Ciddi doku sertleşmesi ve fonksiyon kaybı riski. Karaciğer kanı süzmekte zorlanıyor olabilir.",
            "Tavsiye": "Mutlaka gastroenteroloji uzmanı takibi ve özel diyet programı uygulanmalıdır."
        },
        4: {
            "Durum": "EVRE 4 (SİROZ)",
            "Detay": "Geri dönüşü zor doku hasarı. Komplikasyon riski (ödem, varis vb.) yüksektir.",
            "Tavsiye": "Kişiselleştirilmiş dozaj ayarı ve çok sıkı klinik takip zorunludur."
        }
    }
    return bilgi.get(evre, {"Durum": "Bilinmiyor", "Detay": "Veri eksik.", "Tavsiye": "-"})

# 5. ETKİLEŞİMLİ PANEL
print("\n--- Kişiselleştirilmiş Karaciğer Dijital İkiz Analizi ---")
try:
    in_age = int(input("Yaş: "))
    in_gen = input("Cinsiyet (E/K): ").upper()
    in_bili = float(input("Bilirubin (Normal: 0.5 - 1.1): "))
    in_albu = float(input("Albumin (Normal: 3.5 - 5.0): "))

    gen_code = 1 if in_gen == 'E' else 0
    input_data = scaler.transform([[in_age, gen_code, in_bili, in_albu]])
    
    tahmin = int(model.predict(input_data)[0])
    res = klinik_detaylandir(tahmin)

    print(f"\n" + "="*50)
    print(f" ANALİZ SONUCU: {res['Durum']}")
    print(f" DETAY: {res['Detay']}")
    print(f" TAVSİYE: {res['Tavsiye']}")
    print("="*50)

except Exception as e:
    print(f"Giriş hatası: {e}")