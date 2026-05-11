import numpy as np
import matplotlib.pyplot as plt

# 1. Karaciğer Ortamı ve Parametreler (Digital Phantom Mantığı)
grid_size = 50  # Karaciğer dokusu 50x50 bir hücre ağı olarak temsil edilir
liver = np.zeros((grid_size, grid_size))
dt = 0.1        # Zaman adımı
diffusion_rate = 0.5  # İlacın doku içindeki yayılma hızı

# 2. İlaç Giriş Noktası (Portal Ven / Damar Girişi)
entry_x, entry_y = 25, 0  # İlacın karaciğere girdiği nokta

# 3. Simülasyon Döngüsü (PBPK - Fizyoloji Tabanlı Modelleme)
def simulate_dose(dosage_amount, steps=100):
    global liver
    results = []
    
    for t in range(steps):
        # Difüzyon denklemi (Fick Kanunu simülasyonu)
        new_liver = liver.copy()
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                # İlacın çevre hücrelere yayılması
                new_liver[i, j] += diffusion_rate * dt * (
                    liver[i+1, j] + liver[i-1, j] + 
                    liver[i, j+1] + liver[i, j-1] - 4*liver[i, j]
                )
        
        # Giriş noktasından sürekli ilaç akışı
        new_liver[entry_x, entry_y] += dosage_amount
        liver = new_liver
        
        # Belirli adımlarda çıktı kaydet
        if t % 20 == 0:
            results.append(liver.copy())
    return results

# 150mg Dozaj Simülasyonu
snapshots = simulate_dose(dosage_amount=1.5, steps=200)

# 4. Görselleştirme (Hocaya Gösterilecek Çıktı)
plt.figure(figsize=(10, 6))
plt.imshow(snapshots[-1], cmap='hot', interpolation='nearest')
plt.colorbar(label='İlaç Konsantrasyonu (mg/cm³)')
plt.title('Karaciğer Dijital İkizi: İlaç Dağılım Isı Haritası (T=200)')
plt.xlabel('Doku Genişliği')
plt.ylabel('Doku Derinliği')
plt.show()