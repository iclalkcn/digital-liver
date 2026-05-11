# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt

def dijital_ikiz_bt_analizi(veri_klasoru):
    print(f"Klasör taranıyor: {veri_klasoru}")
    
    # Bütün alt klasörlerdeki .dcm (DICOM) dosyalarını bul
    dcm_dosyalari = glob.glob(f"{veri_klasoru}/**/*.dcm", recursive=True)
    
    if not dcm_dosyalari:
        print("Hata: Klasörde hiç DICOM dosyası bulunamadı!")
        return None
        
    print(f"Toplam {len(dcm_dosyalari)} adet kesit bulundu. İlk kesit analiz ediliyor...")
    
    # İlk dosyayı örnek olarak seçip oku
    hedef_dosya = dcm_dosyalari[0]
    ds = pydicom.dcmread(hedef_dosya)
    
    # Görüntüyü sayısal matrise (piksellere) çevir
    goruntu_matrisi = ds.pixel_array
    
    # --- BASİT ÖZNİTELİK ÇIKARIMI (DOKU ANALİZİ) ---
    # Gerçek projede (örn. TÜBİTAK/SIU) burada bir CNN modeli veya Segmentasyon (U-Net) olur.
    # Şimdilik piksel yoğunluğuna (Hounsfield Unit simülasyonu) bakarak bir hasar skoru üretiyoruz.
    ortalama_yogunluk = np.mean(goruntu_matrisi)
    max_yogunluk = np.max(goruntu_matrisi)
    
    # Karaciğer yağlandıkça/sertleştikçe yoğunluk değişir
    doku_skoru = (ortalama_yogunluk / max_yogunluk) * 100
    
    print("-" * 50)
    print(f"DİJİTAL İKİZ GÖRÜNTÜ ANALİZ SONUCU")
    print(f"Hasta/Çekim ID: {ds.PatientID if 'PatientID' in ds else 'Bilinmiyor'}")
    print(f"Doku Yoğunluk İndeksi: %{doku_skoru:.2f}")
    print("-" * 50)
    
    # GÖRÜNTÜYÜ EKRANA BASMA
    plt.figure(figsize=(8, 6))
    # vmin (minimum) ve vmax (maksimum) değerleriyle oynayarak kontrastı ayarlayabilirsin
    plt.imshow(goruntu_matrisi, cmap=plt.cm.bone)
    plt.title(f"Dijital İkiz - Karaciğer Kesiti\nDoku Skoru: %{doku_skoru:.2f}")
    plt.axis('off') # Eksenleri gizle
    
    # Modelin için ekstra bir katman olarak bu skoru ekranda göster
    plt.text(10, 20, f"Analiz: {'Riskli' if doku_skoru < 20 else 'Normal'}", 
             color='red' if doku_skoru < 20 else 'green', 
             fontsize=12, backgroundcolor='white')
             
    plt.show()
    
    return doku_skoru

# Kodu çalıştır (Klasör adının doğruluğunu kontrol et)
klasor_adi = "./karaciger_veri_seti_mini" 
analiz_sonucu = dijital_ikiz_bt_analizi(klasor_adi)