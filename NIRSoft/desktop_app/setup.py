import os
import sys
import PyInstaller.__main__

# Çalıştırılabilir dosyanın yanında olacak data dosyaları
datas = [
    ('applogo.ico', '.'),  # Logo dosyası
    ('monke.png', '.'),    # Easter egg resim
    ('BMW.mp4', '.'),      # Easter egg video
    ('acibadem.png', '.'), # Raporda kullanılan logo
]

# Main script (uygulamanın giriş noktası)
main_script = 'another.py'

# PyInstaller argümanları
args = [
    main_script,
    '--name=NIRSoft',
    '--windowed',  # GUI uygulaması olduğu için konsol penceresi gösterme
    '--onefile',   # Tek bir exe dosyası oluştur
    '--clean',     # Önceki derleme artıklarını temizle
    '--icon=applogo.ico',  # İkon dosyası
]

# Data dosyalarını ekle
for src, dst in datas:
    args.append(f'--add-data={src}{os.pathsep}{dst}')

# PyInstaller'ı çalıştır
PyInstaller.__main__.run(args)