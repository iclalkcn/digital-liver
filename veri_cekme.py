# -*- coding: utf-8 -*-
import os
from tcia_utils import nbia

# --- AYARLAR ---
koleksiyon_adi = "TCGA-LIHC"
output_path = "./karaciger_veri_seti_mini"

# 4-5 GB için güvenli sınır: 20-25 arası seri
indirme_limiti = 20 

if not os.path.exists(output_path):
    os.makedirs(output_path)

print(f"[{koleksiyon_adi}] koleksiyonuna bağlanılıyor...")

try:
    # 1. Bütün listeyi (sadece metin formatında) çek
    tum_seriler = nbia.getSeries(collection=koleksiyon_adi)

    if isinstance(tum_seriler, list) and len(tum_seriler) > 0:
        print(f"Sistemde toplam {len(tum_seriler)} adet seri bulundu.")
        
        # 2. LİMİTLEME (50 GB olmaması için listeyi kesiyoruz)
        hedef_seriler = tum_seriler[:indirme_limiti]
        
        print(f"Kota ayarlandı: Sadece {indirme_limiti} adet seri (yaklaşık 4-5 GB) indirilecek.")
        print("İndirme işlemi başlıyor, arkana yaslan ve bekle...")
        
        # 3. İNDİRME 
        # Hata Çözümü: Direkt 'hedef_seriler' listesini gönderiyoruz, kütüphane içinden kendi ayıklıyor.
        nbia.downloadSeries(hedef_seriler, path=output_path)
        
        print(f"\nİşlem Başarıyla Tamamlandı!")
        print(f"Dosyalar şurada: {os.path.abspath(output_path)}")
        
    else:
        print("Hata: Koleksiyon listesi boş döndü.")

except Exception as e:
    print(f"\nBeklenmedik bir hata oluştu: {e}")