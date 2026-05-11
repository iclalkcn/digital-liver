import numpy as np
import matplotlib.pyplot as plt

# 1. Kullanıcıdan Veri Girişi (İnteraktif Bölüm)
try:
    user_age = int(input("Hastanın Yaşı: "))
    user_weight = float(input("Hastanın Kilosu (kg): "))
except ValueError:
    print("Geçersiz giriş! Varsayılan değerler kullanılıyor.")
    user_age, user_weight = 30, 70

# 2. Fizyolojik Katsayıların Hesaplanması
# Yaşlandıkça difüzyon (temizleme hızı) azalır
age_factor = max(0.05, 0.6 - (user_age / 140)) 

# Kilo arttıkça dağılım alanı (simülasyon alanı) genişler ama yoğunluk azalabilir
# Burada kilo, ilacın dokudaki ilk konsantrasyon etkisini değiştirecek
dosage_impact = 100 / user_weight  # Kilo arttıkça dozajın birim alandaki etkisi azalır

grid_size = 50
liver = np.zeros((grid_size, grid_size))
dt = 0.1
steps = 200
entry_x, entry_y = 25, 0

# 3. Simülasyon Döngüsü
for t in range(steps):
    new_liver = liver.copy()
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            new_liver[i, j] += age_factor * dt * (
                liver[i+1, j] + liver[i-1, j] + 
                liver[i, j+1] + liver[i, j-1] - 4*liver[i, j]
            )
    # Kilo etkili dozaj girişi
    dosage_amount = 2.0 * dosage_impact
    new_liver[entry_x, entry_y] += dosage_amount
    liver = new_liver

# 4. Görselleştirme
plt.figure(figsize=(10, 6))
plt.imshow(liver, cmap='magma')
plt.colorbar(label='İlaç Konsantrasyon Yoğunluğu')
plt.title(f"Dijital İkiz Analizi | Yaş: {user_age} | Kilo: {user_weight} kg")
plt.show()

print(f"\n--- Klinik Karar Destek Analizi ---")
print(f"Hesaplanan Dağılım Katsayısı: {age_factor:.3f}")
if user_age > 65 and user_weight < 60:
    print("KRİTİK UYARI: Düşük kilo ve ileri yaş kombinasyonu yüksek toksisite riski taşır!")
elif user_weight > 100:
    print("BİLGİ: Yüksek kilo nedeniyle ilaç dağılımı daha geniş bir alana yayılmaktadır.")