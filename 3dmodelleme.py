# -*- coding: utf-8 -*-
import os
import glob
import pydicom
import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes
from collections import Counter

# --- AYARLAR ---
veri_klasoru = "./karaciger_veri_seti_mini"

print("1. Z Aşaması: DICOM kesitleri aranıyor...")
dcm_dosyalari = glob.glob(f"{veri_klasoru}/**/*.dcm", recursive=True)

if not dcm_dosyalari:
    print("Hata: Klasörde DICOM dosyası bulunamadı!")
    exit()

# 2. KESİTLERİ ÜST ÜSTE DİZME VE AYIKLAMA
kesitler = []
sekiller = [] 

for dosya in dcm_dosyalari:
    try:
        ds = pydicom.dcmread(dosya)
        if hasattr(ds, 'SliceLocation') or hasattr(ds, 'ImagePositionPatient'):
            kesitler.append(ds)
            sekiller.append(ds.pixel_array.shape)
    except Exception as e:
        pass # Okunamayan bozuk dosyaları atla

print(f"Başlangıçta {len(kesitler)} kesit var.")

# En yaygın çözünürlüğü bul (Genelde 512x512)
en_yaygin_boyut = Counter(sekiller).most_common(1)[0][0]
print(f"Sistemin ana çözünürlüğü tespit edildi: {en_yaygin_boyut}")

# Sadece ana çözünürlüğe sahip kesitleri tut
filtrelenmis_kesitler = [s for s in kesitler if s.pixel_array.shape == en_yaygin_boyut]
Atilan_sayisi = len(kesitler) - len(filtrelenmis_kesitler)

print(f"Farklı boyuttaki {Atilan_sayisi} adet uyumsuz kesit çöpe atıldı.")
print(f"Kalan sağlam kesit sayısı: {len(filtrelenmis_kesitler)}. Z eksenine göre diziliyor...")

# Z eksenine göre küçükten büyüğe sırala
filtrelenmis_kesitler.sort(key=lambda x: float(x.ImagePositionPatient[2]))

# 3. 3D MATRİS OLUŞTURMA (Hacim Çıkarma)
print("2. Aşama: 3D Hacim Matrisi (Voxel) oluşturuluyor...")
# İŞTE DÜZELTİLEN SATIR BURASI: Artık 'kesitler' değil, 'filtrelenmis_kesitler' kullanılıyor
hacim_3d = np.stack([s.pixel_array for s in filtrelenmis_kesitler])

# 4. MARCHING CUBES İLE 3D MESH (YÜZEY) ÇIKARMA
print("3. Aşama: Yürüyen Küpler algoritması ile organ yüzeyi hesaplanıyor...")
print("(Bu işlem bilgisayarının hızına göre 10-30 saniye sürebilir, lütfen bekle...)")

esik_degeri = 50 
verts, faces, normals, values = marching_cubes(hacim_3d, level=esik_degeri, step_size=2)

z_oran = filtrelenmis_kesitler[0].SliceThickness if hasattr(filtrelenmis_kesitler[0], 'SliceThickness') else 1.0
verts[:, 0] = verts[:, 0] * z_oran 

# 5. PLOTLY İLE 3D İNTERAKTİF ÇİZİM
print("4. Aşama: 3D Model tarayıcıda oluşturuluyor...")
fig = go.Figure(data=[go.Mesh3d(
    z=verts[:, 0], 
    y=verts[:, 1], 
    x=verts[:, 2], 
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    opacity=0.5, 
    color='darkred', 
    flatshading=True
)])

fig.update_layout(
    title="Dijital İkiz: 3D Doku ve Organ Rekonstrüksiyonu",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="rgb(20, 24, 30)" 
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

html_dosyasi = "dijital_ikiz_3d_model.html"
fig.write_html(html_dosyasi, auto_open=True)
print(f"\nİşlem Tamam! Model '{html_dosyasi}' adıyla kaydedildi ve tarayıcında açılıyor.")