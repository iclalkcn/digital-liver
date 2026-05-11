import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. VERİ SETİNİ YÜKLEME VE DERİNLEMESİNE ÖĞRETME
df = pd.read_csv('liver_cirrhosis.csv')

# Eksik verileri daha akıllıca dolduralım (Medyan ve Mod kullanarak)
cols_to_fix = ['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'Platelets']
for col in cols_to_fix:
    df[col] = df[col].fillna(df[col].median())
df['Stage'] = df['Stage'].fillna(df['Stage'].mode()[0])

# ÖZELLİK MÜHENDİSLİĞİ: Daha fazla veriyi öğretiyoruz
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Age_Years'] = (df['Age'] / 365).astype(int)

# 2. MODELİ GÜÇLENDİRME (Daha fazla parametre)
# Bakır (Copper) ve Trombosit (Platelets) karaciğer için çok kritiktir
features = ['Age_Years', 'Sex', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Platelets']
X = df[features]
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Algoritma ayarlarını (Hyperparameters) optimize ediyoruz
model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# 3. BAŞARI RAPORU
accuracy = accuracy_score(y_test, y_pred := model.predict(X_test))
print(f"Geliştirilmiş Model Başarı Oranı: %{accuracy*100:.2f}")

# 4. KLİNİK YORUMLAYICI FONKSİYONU
def evre_yorumu(evre):
    yorumlar = {
        1: "Evre 1: Karaciğerde hafif iltihaplanma var ancak doku hasarı yok. Erken teşhis! Yaşam tarzı değişikliği ile iyileşme oranı yüksektir.",
        2: "Evre 2: Fibrozis (dokuda sertleşme) başlamış. İlaç dozajı dikkatle ayarlanmalı, karaciğeri yoran maddelerden kaçınılmalıdır.",
        3: "Evre 3: İleri düzey sertleşme. Karaciğer fonksiyonları azalmış. Acil klinik takip ve kişiselleştirilmiş tedavi şarttır.",
        4: "Evre 4 (Siroz): Ciddi doku hasarı. Komplikasyon riski yüksek. Dozajlar minimum seviyede tutulmalı, uzman doktor denetimi zorunludur."
    }
    return yorumlar.get(evre, "Bilinmeyen Evre")

# 5. TAHMİN VE BİLGİLENDİRME
print("\n--- Hasta Analiz ve Karar Destek Sistemi ---")
try:
    age = int(input("Yaş: "))
    sex = 1 if input("Cinsiyet (E/K): ").upper() == 'E' else 0
    bili = float(input("Bilirubin (Örn 1.4): "))
    plat = float(input("Trombosit (Platelets - Örn 250): "))
    
    # Modelin görmediği diğer değerler için veri seti ortalamasını veriyoruz
    tahmin = model.predict([[age, sex, bili, 260, 3.5, 64, plat]])
    
    print(f"\n[TAHMİN EDİLEN EVRE]: {int(tahmin[0])}")
    print(f"[KLİNİK BİLGİ]: {evre_yorumu(int(tahmin[0]))}")
except Exception as e:
    print(f"Giriş hatası: {e}")