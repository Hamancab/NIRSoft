import os
import re
import numpy as np
import pandas as pd
import scipy.io
import sys
import PyInstaller.__main__
import h5py
from scipy.integrate import trapezoid
import matplotlib as mpl
mpl.use('Agg')            # ← Burada GUI dışı backend'e geçiyoruz
import matplotlib.pyplot as plt  # ← Artık Agg backend'i kullanacak
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from PIL import Image
import nibabel as nib
from nilearn import plotting
from nilearn.image import smooth_img
import matplotlib.pyplot as plt
from nilearn import datasets, surface
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Tüm figürlerin arka plan rengi
mpl.rcParams['figure.facecolor'] = "#FFFFF0"
# Kaydedilen dosyaların da bu renkle kaydedilmesi için
mpl.rcParams['savefig.facecolor'] = '#FFFFF0'
# Tüm eksenlerin (axes) arka planı
mpl.rcParams['axes.facecolor']   = '#FFFFF0'
# Grid çizgileri için kontrast renk
mpl.rcParams['grid.color']       = '#cccccc'
# Yazı ve tick renkleri (isteğe bağlı)
mpl.rcParams['text.color']       = '#333333'
mpl.rcParams['axes.labelcolor']  = '#333333'
mpl.rcParams['xtick.color']      = '#333333'
mpl.rcParams['ytick.color']      = '#333333'


def plot_fnirs_brain_overlay(nirs_path, probe_mat_path, save_path):
    """
    nirs_path: .nirs dosyası
    probe_mat_path: probeInfo içeren .mat dosyası
    save_path: Kaydedilecek PNG dosya yolu
    """
    probe_mat = scipy.io.loadmat(probe_mat_path, squeeze_me=True, struct_as_record=False)
    coords_c3 = probe_mat['probeInfo'].probes.coords_c3

    nirs_mat = scipy.io.loadmat(nirs_path)
    d = nirs_mat['d']
    d_850 = d[:, -20:]
    od = np.log(np.max(d_850, axis=0) / d_850)
    mean_vals = np.mean(od, axis=0)

    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    mesh_left = fsaverage.pial_left
    mesh_right = fsaverage.pial_right

    coords_left, faces_left = surface.load_surf_mesh(mesh_left)
    coords_right, faces_right = surface.load_surf_mesh(mesh_right)

    all_coords = np.vstack([coords_left, coords_right])
    brain_center = all_coords[:, :2].mean(axis=0)
    chan_center = coords_c3[:, :2].mean(axis=0)
    brain_span = np.ptp(all_coords[:, :2], axis=0)
    chan_span = np.ptp(coords_c3[:, :2], axis=0)
    scale_brain = 0.7
    scale_factor = min(brain_span / chan_span) * scale_brain

    coords_left_scaled = (coords_left[:, :2] - brain_center) * scale_brain + brain_center
    coords_right_scaled = (coords_right[:, :2] - brain_center) * scale_brain + brain_center
    coords_c3_scaled = (coords_c3[:, :2] - chan_center) * scale_factor + brain_center

    fig, ax = plt.subplots(figsize=(10, 8))
    for tri in faces_left:
        polygon = coords_left_scaled[tri]
        ax.fill(polygon[:, 0], polygon[:, 1], color='lightgrey', alpha=0.7, linewidth=0)
    for tri in faces_right:
        polygon = coords_right_scaled[tri]
        ax.fill(polygon[:, 0], polygon[:, 1], color='lightgrey', alpha=0.7, linewidth=0)
    gamma = 0.2
    mean_vals_norm = (mean_vals - mean_vals.min()) / (mean_vals.max() - mean_vals.min())
    mean_vals_gamma = mean_vals_norm ** gamma

    sc = ax.scatter(coords_c3_scaled[:, 0], coords_c3_scaled[:, 1],
                    c=mean_vals_gamma, cmap='hot_r', s=1500, edgecolor='k', zorder=10)

    for i in range(coords_c3.shape[0]):
        ax.text(coords_c3_scaled[i, 0], coords_c3_scaled[i, 1], str(i+1),
                color='black', fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=12)

    plt.colorbar(sc, ax=ax, shrink=0.7, label='Mean HbO')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal')
    ax.set_title('2D Top View: Both Hemispheres + Channels (Scaled)')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_fnirs_volumetric_heatmap(nirs_path, probe_mat_path, save_path, offset=[0, -20, 25]):
    probe_mat = scipy.io.loadmat(probe_mat_path, squeeze_me=True, struct_as_record=False)
    coords_c3 = probe_mat['probeInfo'].probes.coords_c3

    nirs_mat = scipy.io.loadmat(nirs_path)
    d = nirs_mat['d']
    d_850 = d[:, -20:]
    od = np.log(np.max(d_850, axis=0) / d_850)
    mean_vals = np.mean(od, axis=0)

    coords_norm = (coords_c3 - coords_c3.mean(axis=0))
    mni_min = np.array([-60, -80, -30])
    mni_max = np.array([60, 80, 60])
    mni_range = mni_max - mni_min
    coords_range = np.ptp(coords_norm, axis=0)
    scale_factor = mni_range / coords_range * 0.6
    coords_scaled_mni = coords_norm * scale_factor + (mni_min + mni_max) / 2
    coords_scaled_mni_offset = coords_scaled_mni + np.array(offset)

    shape = (91, 109, 91)
    affine = np.array([
        [-2, 0, 0, 90],
        [0, 2, 0, -126],
        [0, 0, 2, -72],
        [0, 0, 0, 1]
    ])
    data = np.zeros(shape)
    voxel_coords = np.linalg.inv(affine).dot(np.vstack([coords_scaled_mni_offset.T, np.ones(coords_scaled_mni_offset.shape[0])]))[:3].T
    voxel_coords = np.round(voxel_coords).astype(int)
    for i, (x, y, z) in enumerate(voxel_coords):
        if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            data[x, y, z] = mean_vals[i]
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)
    gamma = 0.7
    data_gamma = data_norm ** gamma
    threshold = 0.4
    data_thresholded = np.where(data_gamma >= threshold, data_gamma, 0)
    img = nib.Nifti1Image(data_thresholded, affine)
    smoothed_img = smooth_img(img, fwhm=30)

    # PNG olarak kaydet
    display = plotting.plot_glass_brain(
        smoothed_img, display_mode='ortho', colorbar=True,
        title='fNIRS Volumetric Heatmap'
    )
    display.savefig(save_path)
    display.close()

def plot_and_save_stat_graphs(vals, channels, title, color, save_dir):
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    upper = mean_val + std_val
    lower = mean_val - std_val

    plt.figure(figsize=(12, 4))
    plt.bar(channels, vals, color="lightcoral", edgecolor='black')
    plt.axhline(upper, color='r', linestyle='--', label='Upper Threshold')
    plt.axhline(lower, color='g', linestyle='--', label='Lower Threshold')
    plt.axhline(mean_val, color='b', linestyle='--', label='Mean')
    plt.title(title)
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    normal_fn = save_fig_with_title(save_dir)

    z_vals = (vals - mean_val) / std_val
    z_mean = np.mean(z_vals)
    z_std = np.std(z_vals)
    plt.figure(figsize=(12, 4))
    plt.bar(channels, z_vals, color='steelblue', edgecolor='black')
    plt.axhline(z_mean + z_std, color='r', linestyle='--', label='Upper Threshold')
    plt.axhline(z_mean - z_std, color='g', linestyle='--', label='Lower Threshold')
    plt.axhline(z_mean, color='b', linestyle='--', label='Mean')
    plt.title("Z-score " + title)
    plt.ylabel("Z-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    z_fn = save_fig_with_title(save_dir)

    return normal_fn, z_fn



def save_fig_with_title(save_dir, name=None):
    fig = plt.gcf()
    axs = fig.axes
    title = name if name else (axs[0].get_title() if axs else 'figure')
    if not title:
        title = 'figure'
    # Geçerli dosya adı oluştur ve özel karakterleri kaldır
    title = re.sub(r'[^\w\.\- ]', '', title).strip().replace(' ', '_')[:50]  # Fixed regex
    # Kayıt yolunu oluştur
    file_path = os.path.join(save_dir, f"{title}.png")
    # Grafiği kaydet
    fig.savefig(file_path, bbox_inches='tight', dpi=300)  # dpi'yi artırmak kaliteyi iyileştirebilir
    print(f"Saved: {file_path}")
    plt.close(fig)  # Her grafiği kapattığınızdan emin olun
    return title + ".png"  # Dosya adını döndür

def get_trigger_indices(nirs_file_path):
    mat = scipy.io.loadmat(nirs_file_path)
    t = mat['t'].squeeze()
    s = mat['s'] if 's' in mat else None
    trigger_indices = sorted({
        idx
        for ch in range(s.shape[1])
        for idx in np.where(s[:, ch] == 1)[0]
    })
    trigger_times = [t[idx] for idx in trigger_indices]
    return trigger_indices, trigger_times

def resource_path(relative_path):
    """
    PyInstaller ile paketlendiğinde de kaynak dosyalarına erişim sağlar
    """
    try:
        base_path = sys._MEIPASS  # PyInstaller ile paketlenmişse
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))  # Çalışma dizini

    return os.path.join(base_path, relative_path)

def full_analysys(
    nirs_file_path: str,
    hasta_adi: str = None,
    patient_id: str = None,
    date_of_birth: str = None,
    gender: str = None,
    protocol_name: str = None,
    diagnosis: str = None,
    notes: str = None,
    save_dir: str = None,
    probe_mat_path: str = None,
    analysis_date: str = None,
    start_trigger: int = None,
    end_trigger: int = None,
):
    # Kullanıcıdan bilgi al (parametre verilmemişse)
    # input() ile terminalden bilgi isteme, eksikse None/boş olarak devam et
    """
    nirs_file_path: .nirs dosyasının tam yolu
    hasta_adi: Hasta adı-soyadı metni
    sidx, eidx: Trigger indeksleri (1-based) arasını analiz eder
    save_dir: Grafikleri ve raporu kaydedeceği klasör (default: script'in altındaki 'graph')
    """
    # Kayıt dizini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if save_dir is None:
        save_dir = os.path.join(current_dir, "graph")
    os.makedirs(save_dir, exist_ok=True)

    # 1) .nirs dosyasını yükle
    mat = scipy.io.loadmat(nirs_file_path)
    t = mat['t'].squeeze()
    d = mat['d']
    s = mat['s'] if 's' in mat else None

    # 2) Trigger indekslerini bul
    trigger_indices, trigger_times = get_trigger_indices(nirs_file_path)

    # Kullanıcıya trigger seçeneklerini sun
    print("Available triggers:")
    for i, time in enumerate(trigger_times):
        print(f"Trigger{i + 1}: {time:.2f} seconds")

    # Kullanıcıdan başlangıç ve bitiş triggerlarını seçmesini iste
    if start_trigger is None:
        start_trigger = int(input("Select the start trigger (e.g., 1 for Trigger1): ")) - 1
    if end_trigger is None:
        end_trigger = int(input("Select the end trigger (e.g., 7 for Trigger7): ")) - 1

    # Kullanıcının seçtiği başlangıç ve bitiş triggerları arasında kalanları filtrele
    if start_trigger < 0 or end_trigger >= len(trigger_indices) or start_trigger >= end_trigger:
        raise ValueError("Invalid trigger selection. Please ensure start_trigger < end_trigger and within range.")

    # Seçilen trigger aralığını al
    selected_triggers = list(trigger_indices[start_trigger:end_trigger + 1])
    selected_trigger_times = list(trigger_times[start_trigger:end_trigger + 1])

    # Ensure selected_trigger_times is a NumPy array
    selected_trigger_times = np.array(selected_trigger_times, dtype=float)

    # Print selected_triggers and selected_trigger_times for debugging
    print("Trigger Indices:", trigger_indices)
    print("Trigger Times:", trigger_times)
    print("Selected Triggers:", selected_triggers)
    print("Selected Trigger Times:", selected_trigger_times)

    # Yürüme ve dinlenme segmentlerini ayır
    segments = []
    for i in range(len(selected_triggers) - 1):
        segment_type = "Walk" if i % 2 == 0 else "Rest"
        segments.append({
            "type": segment_type,
            "start": selected_triggers[i],
            "end": selected_triggers[i + 1]
        })

    # Sadece "Walk" segmentlerini birleştir
    walk_segments = [seg for seg in segments if seg['type'] == "Walk"]
    t_concat = np.concatenate([t[seg['start']:seg['end'] + 1] for seg in walk_segments])
    d_concat = np.concatenate([d[seg['start']:seg['end'] + 1, :] for seg in walk_segments], axis=0)

    # 4) Sadece 850nm kanallar (son 20 kanal)
    d_seg_850 = d_concat[:, -20:]
    od = np.log(np.max(d_seg_850, axis=0) / d_seg_850)
    channels = [f"Ch{i+1}" for i in range(od.shape[1])]

############################################333

    # 'segments' listesi zaten hem Walk hem Rest segmentlerini içeriyor
    t_all = np.concatenate([t[seg['start']:seg['end']+1] for seg in segments])
    d_all = np.concatenate([d[seg['start']:seg['end']+1, -20:] for seg in segments], axis=0)

    # # OD hesaplamasını tekrarla
    # od_all = np.log(np.max(d_all, axis=0) / d_all)
    # Segmentler için od_all_segments
    od_all_segments = np.log(np.max(d_all, axis=0) / d_all)
    od_all_sel = np.log(np.max(d_all, axis=0) / d_all)
    mean_hbo_all = np.mean(od_all_sel, axis=1)

    # # ——— Yeni grafik: Walk+Rest dönemi ortalama HbO ———
    # plt.figure(figsize=(12, 5))
    # plt.plot(t_all, mean_hbo_all, color='darkblue', linewidth=2, label='Walk+Rest Ortalama')


########################################333333333
    # Analizler ve grafikler
    max_vals = np.max(od, axis=0)
    mean_vals = np.mean(od, axis=0)
    auc_vals = trapezoid(od, x=t_concat, axis=0)
    range_vals = np.ptp(od, axis=0)
    power_vals = np.mean(od**2, axis=0)
    lagmax_indices = np.argmax(od, axis=0)
    lagmax_times = t_concat[lagmax_indices]
    lagmax_relative = lagmax_times - t_concat[0]

    max_fn, max_z_fn = plot_and_save_stat_graphs(max_vals, channels, "Maximum HbO", "blue", save_dir)
    mean_fn, mean_z_fn = plot_and_save_stat_graphs(mean_vals, channels, "Mean HbO", "green", save_dir)
    auc_fn, auc_z_fn = plot_and_save_stat_graphs(auc_vals, channels, "AUC HbO", "purple", save_dir)
    range_fn, range_z_fn = plot_and_save_stat_graphs(range_vals, channels, "Range HbO", "orange", save_dir)
    power_fn, power_z_fn = plot_and_save_stat_graphs(power_vals, channels, "Power HbO", "red", save_dir)
    lagmax_fn, lagmax_z_fn = plot_and_save_stat_graphs(lagmax_relative, channels, "Lagmax (Time to Maximum)", "dodgerblue", save_dir)
    # Calculate Z-scores for mean, lagmax, power, range
    mean_z = (mean_vals - np.mean(mean_vals)) / np.std(mean_vals)
    lagmax_z = (lagmax_relative - np.mean(lagmax_relative)) / np.std(lagmax_relative)
    power_z = (power_vals - np.mean(power_vals)) / np.std(power_vals)
    range_z = (range_vals - np.mean(range_vals)) / np.std(range_vals)

    # # --- Tüm veriyle ortalama HbO zaman grafiği ve trigger çizgileri ---
    # d_850_all = d[:, -20:]
    # od_all = np.log(np.max(d_850_all, axis=0) / d_850_all)
    # Tüm veri için od_all_full
    d_850_all = d[:, -20:]
    od_all_full = np.log(np.max(d_850_all, axis=0) / d_850_all)

    mean_hbo_time_all = np.mean(od_all_full, axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(t, mean_hbo_time_all, color="black", linewidth=2, zorder=0)
    # Triggerlar arası örüntü: kırmızı saydam ve net alanlar
    plt.axvspan(t[0], t[trigger_indices[0]], color="#FFFFF0", alpha=0.3, zorder=1)  # İlk triggera kadar kırmızı saydam alan
    for i in range(len(trigger_indices) - 1):
        t_start = t[trigger_indices[i]]
        t_end = t[trigger_indices[i + 1]]
        if i % 2 == 0:
            # Net alan (boş bırak)
            plt.axvspan(t_start, t_end, color="#8DEDA3", alpha=0.3, zorder=1)  # Tamamen şeffaf alan
        else:
            # Kırmızı saydam alan
            plt.axvspan(t_start, t_end, color="#FFFFF0", alpha=0.3, zorder=1)
    # Trigger noktalarını kırmızı dik çizgilerle ekle
    for trig in trigger_times:
        plt.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)
    plt.title("Mean HbO Time Series (All Channels, with All Triggers)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean OD (HbO)")
    plt.grid(True)
    plt.tight_layout()
    mean_hbo_time_all_fn = save_fig_with_title(save_dir)


############################333
    plt.figure(figsize=(12, 5))
    plt.plot(t_all, mean_hbo_all, color='black', linewidth=2, label='Walk+Rest Ortalama', zorder=0)

    # Aynı gölgelendirmeleri ve trigger çizgilerini uygula
    plt.axvspan(t_all[0], selected_trigger_times[0], color="#FFFFF0", alpha=0.3, zorder=1)
    for i in range(len(selected_trigger_times) - 1):
        t_start = selected_trigger_times[i]
        t_end = selected_trigger_times[i + 1]
        if i % 2 == 0:
            plt.axvspan(t_start, t_end, color="#8DEDA3", alpha=0.3, zorder=1)
        else:
            plt.axvspan(t_start, t_end, color="#FFFFF0", alpha=0.3, zorder=1)
    for trig in selected_trigger_times:
        plt.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)

    plt.title("Mean HbO Time Series (Selected Triggers) — Walk + Rest")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean OD (HbO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Görseli kaydet
    mean_hbo_all_fn = save_fig_with_title(save_dir)


    # mean_hbo_time = np.mean(od, axis=1)
    # plt.figure(figsize=(12, 5))
    # plt.plot(t_concat, mean_hbo_time, color='black', linewidth=2)
    # # Triggerlar arası örüntü: kırmızı saydam ve net alanlar
    # plt.axvspan(t[0], selected_trigger_times[0], color="#080808", alpha=0.3, zorder=0)  # İlk triggera kadar kırmızı saydam alan
    # for i in range(len(selected_trigger_times) - 1):
    #     t_start = selected_trigger_times[i]
    #     t_end = selected_trigger_times[i + 1]
    #     if i % 2 == 0:
    #         # Net alan (boş bırak)
    #         plt.axvspan(t_start, t_end, color="white", alpha=0.0, zorder=0)  # Tamamen şeffaf alan
    #     else:
    #         # Kırmızı saydam alan
    #         plt.axvspan(t_start, t_end, color="#080808", alpha=0.3, zorder=0)
    # # Trigger noktalarını kırmızı dik çizgilerle ekle
    # for trig in selected_trigger_times:
    #     plt.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)
    # plt.title("Mean HbO Time Series (Selected Triggers)")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Mean OD (HbO)")
    # plt.grid(True)
    # plt.tight_layout()
    # mean_hbo_time_fn = save_fig_with_title(save_dir)

    mean_first10 = np.mean(mean_vals[:10])
    mean_last10 = np.mean(mean_vals[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [mean_first10, mean_last10], color=['royalblue', 'crimson'])
    # Dinamik y-axis: min/max değerler arası biraz boşluk bırak
    min_val = min(mean_first10, mean_last10)
    max_val = max(mean_first10, mean_last10)
    y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.5
    plt.ylim(min_val - y_margin, max_val + y_margin)
    plt.ylabel("Mean HbO (Mean OD)")
    plt.title("Mean HbO: First 10 vs Last 10 Channels")
    plt.tight_layout()
    mean_hbo_compare_fn = save_fig_with_title(save_dir)

    mean_z_first10 = np.mean(mean_z[:10])
    mean_z_last10 = np.mean(mean_z[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [mean_z_first10, mean_z_last10], color=['royalblue', 'crimson'])
    min_val = min(mean_z_first10, mean_z_last10)
    max_val = max(mean_z_first10, mean_z_last10)
    y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.5
    plt.ylim(min_val - y_margin, max_val + y_margin)
    plt.ylabel("Mean HbO Z-score")
    plt.title("Mean HbO Z-score: First 10 vs Last 10 Channels")
    plt.tight_layout()
    mean_z_compare_fn = save_fig_with_title(save_dir)

    lagmax_first10 = np.mean(lagmax_relative[:10])
    lagmax_last10 = np.mean(lagmax_relative[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [lagmax_first10, lagmax_last10], color=['royalblue', 'crimson'])
    min_val = min(lagmax_first10, lagmax_last10)
    max_val = max(lagmax_first10, lagmax_last10)
    y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.5
    plt.ylim(min_val - y_margin, max_val + y_margin)
    plt.ylabel("Lagmax (s)")
    plt.title("Lagmax: First 10 vs Last 10 Channels")
    plt.tight_layout()
    lagmax_compare_fn = save_fig_with_title(save_dir)

    lagmax_z_first10 = np.mean(lagmax_z[:10])
    lagmax_z_last10 = np.mean(lagmax_z[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [lagmax_z_first10, lagmax_z_last10], color=['royalblue', 'crimson'])
    min_val = min(lagmax_z_first10, lagmax_z_last10)
    max_val = max(lagmax_z_first10, lagmax_z_last10)
    y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.5
    plt.ylim(min_val - y_margin, max_val + y_margin)
    plt.ylabel("Lagmax Z-score")
    plt.title("Lagmax Z-score: First 10 vs Last 10 Channels")
    plt.tight_layout()
    lagmax_z_compare_fn = save_fig_with_title(save_dir)

    power_first10 = np.mean(power_vals[:10])
    power_last10 = np.mean(power_vals[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [power_first10, power_last10], color=['royalblue', 'crimson'])
    min_val = min(power_first10, power_last10)
    max_val = max(power_first10, power_last10)
    y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.5
    plt.ylim(min_val - y_margin, max_val + y_margin)
    plt.ylabel("Power HbO")
    plt.title("Power HbO: First 10 vs Last 10 Channels")
    plt.tight_layout()
    power_compare_fn = save_fig_with_title(save_dir)

    power_z_first10 = np.mean(power_z[:10])
    power_z_last10 = np.mean(power_z[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [power_z_first10, power_z_last10], color=['royalblue', 'crimson'])
    min_val = min(power_z_first10, power_z_last10)
    max_val = max(power_z_first10, power_z_last10)
    y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.5
    plt.ylim(min_val - y_margin, max_val + y_margin)
    plt.ylabel("Power HbO Z-score")
    plt.title("Power HbO Z-score: First 10 vs Last 10 Channels")
    plt.tight_layout()
    power_z_compare_fn = save_fig_with_title(save_dir)

    range_first10 = np.mean(range_vals[:10])
    range_last10 = np.mean(range_vals[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [range_first10, range_last10], color=['royalblue', 'crimson'])
    min_val = min(range_first10, range_last10)
    max_val = max(range_first10, range_last10)
    y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.5
    plt.ylim(min_val - y_margin, max_val + y_margin)
    plt.ylabel("Range HbO")
    plt.title("Range HbO: First 10 vs Last 10 Channels")
    plt.tight_layout()
    range_compare_fn = save_fig_with_title(save_dir)

    range_z_first10 = np.mean(range_z[:10])
    range_z_last10 = np.mean(range_z[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [range_z_first10, range_z_last10], color=['royalblue', 'crimson'])
    min_val = min(range_z_first10, range_z_last10)
    max_val = max(range_z_first10, range_z_last10)
    y_margin = (max_val - min_val) * 0.2 if max_val != min_val else 0.5
    plt.ylim(min_val - y_margin, max_val + y_margin)
    plt.ylabel("Range HbO Z-score")
    plt.title("Range HbO Z-score: First 10 vs Last 10 Channels")
    plt.tight_layout()
    range_z_compare_fn = save_fig_with_title(save_dir)


##############################3

    # sag_kortex_all = od_all_full[:, :10]
    sol_kortex_all = od_all_segments[:, 10:]
    # segment bazlı od_all_segments kullan
    sag_kortex_all = od_all_segments[:, :10]

    mean_sag_all = np.mean(sag_kortex_all, axis=1)
    mean_sol_all = np.mean(sol_kortex_all, axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(t_all, mean_sag_all, label='Right Cortex', linewidth=2, zorder=0, color='crimson')
    plt.plot(t_all, mean_sol_all, label='Left Cortex', linewidth=2, zorder=0, color='royalblue')

    # Aynı gölgelendirme ve trigger çizgileri
    plt.axvspan(t_all[0], selected_trigger_times[0], color="#FFFFF0", alpha=0.3, zorder=1)
    for i in range(len(selected_trigger_times) - 1):
        t_start = selected_trigger_times[i]
        t_end = selected_trigger_times[i + 1]
        if i % 2 == 0:
            plt.axvspan(t_start, t_end, color="#8DEDA3", alpha=0.3, zorder=1)
        else:
            plt.axvspan(t_start, t_end, color="#FFFFF0", alpha=0.3, zorder=1)
    for trig in selected_trigger_times:
        plt.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)

    plt.title("Right and Left Cortex Mean HbO Time Series — Walk + Rest")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean OD (HbO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Kaydetme
    sag_sol_all_fn = save_fig_with_title(save_dir)


    # sag_kortex = od[:, :10]
    # sol_kortex = od[:, 10:]
    # mean_sag = np.mean(sag_kortex, axis=1)
    # mean_sol = np.mean(sol_kortex, axis=1)
    # plt.figure(figsize=(12, 5))
    # plt.plot(t_concat, mean_sag, label='Right Cortex', color='royalblue', linewidth=2)
    # plt.plot(t_concat, mean_sol, label='Left Cortex', color='crimson', linewidth=2)
    # # Triggerlar arası örüntü: kırmızı saydam ve net alanlar
    # plt.axvspan(t[0], selected_trigger_times[0], color="#080808", alpha=0.3, zorder=0)  # İlk triggera kadar kırmızı saydam alan
    # for i in range(len(selected_trigger_times) - 1):
    #     t_start = selected_trigger_times[i]
    #     t_end = selected_trigger_times[i + 1]
    #     if i % 2 == 0:
    #         # Net alan (boş bırak)
    #         plt.axvspan(t_start, t_end, color="white", alpha=0.0, zorder=0)  # Tamamen şeffaf alan
    #     else:
    #         # Kırmızı saydam alan
    #         plt.axvspan(t_start, t_end, color="#080808", alpha=0.3, zorder=0)
    # # Trigger noktalarını kırmızı dik çizgilerle ekle
    # for trig in selected_trigger_times:
    #     plt.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)
    # plt.title("Right and Left Cortex Mean HbO Time Series")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Mean OD (HbO)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # sag_sol_time_fn = save_fig_with_title(save_dir)


#################################

    # Sağ ve sol korteksin tüm bloklardaki ortalama z-score’u
    sag_kortex_all = od_all_segments[:, :10]
    sol_kortex_all = od_all_segments[:, 10:]
    mean_sag_all = np.mean(sag_kortex_all, axis=1)
    mean_sol_all = np.mean(sol_kortex_all, axis=1)
    mean_sag_all_z = (mean_sag_all - np.mean(mean_sag_all)) / np.std(mean_sag_all)
    mean_sol_all_z = (mean_sol_all - np.mean(mean_sol_all)) / np.std(mean_sol_all)
    plt.figure(figsize=(12, 5))
    plt.plot(t_all, mean_sag_all_z, label='Right Cortex (Z)', linewidth=2, color='crimson', zorder=0)
    plt.plot(t_all, mean_sol_all_z, label='Left Cortex (Z)', linewidth=2, color='royalblue', zorder=0)
    # Aynı gölgelendirme ve trigger çizgileri
    plt.axvspan(t_all[0], selected_trigger_times[0], color="#FFFFF0", alpha=0.3, zorder=1)
    for i in range(len(selected_trigger_times) - 1):
        t_start = selected_trigger_times[i]
        t_end = selected_trigger_times[i + 1]
        if i % 2 == 0:
            plt.axvspan(t_start, t_end, color="#8DEDA3", alpha=0.3, zorder=1)
        else:
            plt.axvspan(t_start, t_end, color="#FFFFF0", alpha=0.3, zorder=1)
    for trig in selected_trigger_times:
        plt.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)
    plt.title("Right and Left Cortex Mean HbO (Z-score) Time Series — Walk + Rest")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Kaydetme
    sag_sol_all_z_fn = save_fig_with_title(save_dir, name="Right and Left Cortex Mean HbO (Z-score) Time Series — Walk + Rest")



    # mean_sag_z = (mean_sag - np.mean(mean_sag)) / np.std(mean_sag)
    # mean_sol_z = (mean_sol - np.mean(mean_sol)) / np.std(mean_sol)
    # plt.figure(figsize=(12, 5))
    # plt.plot(t_concat, mean_sag_z, label='Right Cortex (Z)', color='royalblue', linewidth=2)
    # plt.plot(t_concat, mean_sol_z, label='Left Cortex (Z)', color='crimson', linewidth=2)
    # # Triggerlar arası örüntü: kırmızı saydam ve net alanlar
    # plt.axvspan(t[0], selected_trigger_times[0], color="#080808", alpha=0.3, zorder=0)  # İlk triggera kadar kırmızı saydam alan
    # for i in range(len(selected_trigger_times) - 1):
    #     t_start = selected_trigger_times[i]
    #     t_end = selected_trigger_times[i + 1]
    #     if i % 2 == 0:
    #         # Net alan (boş bırak)
    #         plt.axvspan(t_start, t_end, color="white", alpha=0.0, zorder=0)  # Tamamen şeffaf alan
    #     else:
    #         # Kırmızı saydam alan
    #         plt.axvspan(t_start, t_end, color="#080808", alpha=0.3, zorder=0)
    # # Trigger noktalarını kırmızı dik çizgilerle ekle
    # for trig in selected_trigger_times:
    #     plt.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)
    # plt.title("Right and Left Cortex Mean HbO (z-score) Time Series")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Z-score")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # sag_sol_time_z_fn = save_fig_with_title(save_dir, name="Sag and Sol Cortex Mean HbO (Z-score) Time Series")




###########33



    # — Raw HbO sinyalleri (Walk + Rest) —
    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    fig.suptitle("HbO Signals - All Channels (Walk + Rest)", fontsize=16)
    for idx, ax in enumerate(axes.flat):
        if idx >= 20:
            ax.axis('off')
            continue
        # # <-- burayı değiştiriyoruz -->
        # channel_hbo = od_all_full[:, idx]
        # ax.plot(t_all, channel_hbo, linewidth=1.5, zorder=2)
        # --> segment verisi kullan
        channel_hbo = od_all_segments[:, idx]
        ax.plot(t_all, channel_hbo, linewidth=1.5, zorder=0, color='#1f77b4')


        # Gölgelendirme & trigger çizgileri (aynı eski mantık)
        ax.axvspan(t_all[0], selected_trigger_times[0], color="#FFFFF0", alpha=0.3, zorder=1)
        for i in range(len(selected_trigger_times)-1):
            t0, t1 = selected_trigger_times[i], selected_trigger_times[i+1]
            color = "#8DEDA3" if i%2==0 else "#FFFFF0"
            ax.axvspan(t0, t1, color=color, alpha=0.3 if color!="#ffffff" else 0.0, zorder=1)
        for trig in selected_trigger_times:
            ax.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)

        ax.set_title(f"Channel {idx+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("HbO (OD)")
        ax.grid(True)
    plt.tight_layout(rect=[0,0,1,0.95])
    all_channels_time_all_fn = save_fig_with_title(save_dir,
        name="HbO Signals - All Channels Walk+Rest")



    # — Z-score HbO sinyalleri (Walk + Rest) —
    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    fig.suptitle("HbO Signals (Z-score) - All Channels (Walk + Rest)", fontsize=16)
    for idx, ax in enumerate(axes.flat):
        if idx >= 20:
            ax.axis('off')
            continue
        # yine od_all’ı kullan
        # channel = od_all_full[:, idx]
        channel = od_all_segments[:, idx]
        channel_z = (channel - channel.mean()) / channel.std()
        ax.plot(t_all, channel_z, linewidth=1.5, zorder=0, color='#d62728')

        # gölgelendirme ve triggere aynı mantık
        ax.axvspan(t_all[0], selected_trigger_times[0], color="#FFFFF0", alpha=0.3, zorder=1)
        for i in range(len(selected_trigger_times)-1):
            t0, t1 = selected_trigger_times[i], selected_trigger_times[i+1]
            color = "#8DEDA3" if i%2==0 else "#FFFFF0"
            ax.axvspan(t0, t1, color=color, alpha=0.3 if color!="#ffffff" else 0.0, zorder=1)
        for trig in selected_trigger_times:
            ax.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)

        ax.set_title(f"Z-score Channel {idx+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("HbO (Z-score)")
        ax.grid(True)
    plt.tight_layout(rect=[0,0,1,0.95])
    all_channels_time_all_z_fn = save_fig_with_title(save_dir,
        name="HbO Signals (Z-score) - All Channels Walk+Rest")


    # fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    # fig.suptitle("HbO Signals - All Channels", fontsize=16)
    # for idx, ax in enumerate(axes.flat):
    #     if idx >= 20:
    #         ax.axis('off')  # Turn off unused subplots
    #         continue
    #     channel_hbo = od[:, idx]
    #     ax.plot(t_concat, channel_hbo, color='crimson', linewidth=1.5)
    #     # Triggerlar arası örüntü: kırmızı saydam ve net alanlar
    #     ax.axvspan(t[0], selected_trigger_times[0], color="#080808", alpha=0.3, zorder=0)
    #     for i in range(len(selected_trigger_times) - 1):
    #         t_start = selected_trigger_times[i]
    #         t_end = selected_trigger_times[i + 1]
    #         if i % 2 == 0:
    #             ax.axvspan(t_start, t_end, color="white", alpha=0.0, zorder=0)
    #         else:
    #             ax.axvspan(t_start, t_end, color="#080808", alpha=0.3, zorder=0)
    #     for trig in selected_trigger_times:
    #         ax.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)
    #     ax.set_title(f"Channel {idx + 1}")
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("HbO (OD)")
    #     ax.grid(True)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # all_channels_time_fn = save_fig_with_title(save_dir, name="HbO Signals - All Channels")

    # fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    # fig.suptitle("HbO Signals (Z-score) - All Channels", fontsize=16)
    # for idx, ax in enumerate(axes.flat):
    #     if idx >= 20:
    #         ax.axis('off')  # Turn off unused subplots
    #         continue
    #     channel_hbo = od[:, idx]
    #     channel_hbo_z = (channel_hbo - np.mean(channel_hbo)) / np.std(channel_hbo)
    #     ax.plot(t_concat, channel_hbo_z, color='royalblue', linewidth=1.5)
    #     # Triggerlar arası örüntü: kırmızı saydam ve net alanlar
    #     ax.axvspan(t[0], selected_trigger_times[0], color="#080808", alpha=0.3, zorder=0)
    #     for i in range(len(selected_trigger_times) - 1):
    #         t_start = selected_trigger_times[i]
    #         t_end = selected_trigger_times[i + 1]
    #         if i % 2 == 0:
    #             ax.axvspan(t_start, t_end, color="white", alpha=0.0, zorder=0)
    #         else:
    #             ax.axvspan(t_start, t_end, color="#080808", alpha=0.3, zorder=0)
    #     for trig in selected_trigger_times:
    #         ax.axvline(x=trig, color='red', linestyle='--', alpha=0.8, zorder=2)
    #     ax.set_title(f"Z-score Channel {idx + 1}")
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("HbO (Z-score)")
    #     ax.grid(True)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # all_channels_time_z_fn = save_fig_with_title(save_dir, name="HbO Signals (Z-score) - All Channels")


    # Kanal isimleri ve X/Y koordinatları
    channels = [
        'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10',
        'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16', 'Ch17', 'Ch18', 'Ch19', 'Ch20'
    ]
    X = [16.1, 16.2, 17.8, 17.6, 18.9, 19.6, 17.7, 19.1, 20.1, 20.5, 
         9.5, 9.2, 7.8, 8.3, 6.6, 5.8, 7.7, 6.4, 5.2, 4.9]
    Y = [12.1, 15.3, 13.7, 10.5, 12.0, 10.2, 16.8, 15.0, 13.3, 11.5, 
         11.9, 15.1, 13.4, 10.3, 11.7, 10.0, 16.7, 15.0, 13.2, 11.4]

    # Ham değerlerle scatter plot (örneğin mean_vals)
    values_raw = mean_vals
    plt.figure(figsize=(12, 5))
    sc = plt.scatter(X, Y, c=values_raw, cmap='coolwarm', s=400, edgecolor='black', zorder=2)
    for i, txt in enumerate(channels):
        plt.text(X[i], Y[i]+0.3, txt, color='black', fontsize=12, ha='center', va='bottom', fontweight='bold')
    plt.title('Channel Placement and HbO Heatmap (Raw Values)', fontsize=16)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.grid(True, alpha=0.4)
    plt.colorbar(sc, label='HbO (Mean)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    scatter_heatmap_fn = save_fig_with_title(save_dir)

    # Z-score ile scatter plot
    values_z = (values_raw - np.mean(values_raw)) / np.std(values_raw)
    plt.figure(figsize=(12, 5))
    sc = plt.scatter(X, Y, c=values_z, cmap='coolwarm', s=400, edgecolor='black', zorder=2)
    for i, txt in enumerate(channels):
        plt.text(X[i], Y[i]+0.3, txt, color='black', fontsize=12, ha='center', va='bottom', fontweight='bold')
    plt.title('Channel Placement and HbO Heatmap (Z-score)', fontsize=16)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.grid(True, alpha=0.4)
    plt.colorbar(sc, label='HbO (Z-score)')
    plt.gca().set_aspect('equal', adjustable='box')
    scatter_heatmap_z_fn = save_fig_with_title(save_dir)

#####################################DEBUGSOFAR

    # ---- DEBUG: PNG boyutlarını yazdır ----
    import glob
    from PIL import Image

    print("=== Kaydedilen PNG boyutları ===")
    for img in glob.glob(os.path.join(save_dir, "*.png")):
        try:
            with Image.open(img) as im:
                print(f"{os.path.basename(img)}: {im.size[0]}×{im.size[1]} px")
        except Exception as e:
            print(f"{os.path.basename(img)} açılamadı: {e}")
    # ---- DEBUG BİTTİ ----





###########################################################################################################################
    # 7) PDF rapor oluşturma
    # PDF'i sonuç klasörüne kaydet
    results_dir = save_dir  # Grafiklerin kaydedildiği klasör
    pdf_path = os.path.join(results_dir, "rapor.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4
    
    # İlk olarak logo bilgilerini tanımla
    logo_path = os.path.join(current_dir, "acibadem.png")
    
    # add_logo_and_image fonksiyonu - logo boyutları için düzeltildi
    def add_logo_and_image(page_img):
        # Logo
        logo_path = resource_path("acibadem.png")
        if os.path.exists(logo_path):
            # Logo varsa, boyutlarını al
            with Image.open(logo_path) as logo:
                iw, ih = logo.size
            c.drawImage(logo_path, x=(w-10*cm)/2, y=h-4*cm-2*cm, width=10*cm, height=4*cm, preserveAspectRatio=True)
            
        # Analiz görseli (sadece geçerli bir dosya adı verilmişse)
        if page_img:
            img_path = os.path.join(save_dir, page_img)
            if os.path.exists(img_path):
                with Image.open(img_path) as im:
                    ow, oh = im.size
                tw = w - 2*cm
                scale = tw / ow
                th = oh * scale
                c.drawImage(img_path,
                            x=(w-tw)/2,
                            y=h-4*cm-2*cm-th-1*cm,
                            width=tw,
                            height=th,
                            preserveAspectRatio=True)

    def sayfa_numarasi():
        c.setFont("Helvetica", 9)
        text = f"Page {c.getPageNumber()}"
        tw = c.stringWidth(text, "Helvetica", 9)
        c.drawString((w-tw)/2, 1*cm, text)

    # 1. sayfa: Üstten margin ile logo, altında başlık
    # ------------------------------------------------
    margin = 2 * cm

    # Logo boyutlarını al ve ölçekle
    if os.path.exists(logo_path):
        # Orijinal logo boyutları
        with Image.open(logo_path) as logo:
            iw, ih = logo.size

        # Sayfada kullanabileceğimiz maksimum genişlik ve yükseklik
        max_w = w - 2 * margin
        # Başlığın da altına sığması için biraz fazla boşluk bırakıyoruz
        max_h = h - 3 * margin  

        # Ölçekleme faktörü
        scale = min(max_w / iw, max_h / ih)
        img_w, img_h = iw * scale, ih * scale

        # X koordinatı: yatay ortalama
        x = (w - img_w) / 2
        # Y koordinatı: üstten margin kadar aşağı in, sonra resmi çiz
        y = h - margin - img_h
        c.drawImage(
            logo_path,
            x, y,
            width=img_w, height=img_h,
            preserveAspectRatio=True
        )

    # Başlığı resmin altına çiz
    c.setFont("Helvetica-Bold", 14)
    title = f"fNIRS Analysis on {hasta_adi}"
    tw = c.stringWidth(title, "Helvetica-Bold", 14)
    # Resmin altından 1 cm daha aşağıya yerleştir
    text_y = y - 1 * cm
    c.drawString((w - tw) / 2, text_y, title)

    sayfa_numarasi()
    c.showPage()
    # ------------------------------------------------

    # 2. sayfa hasta bilgileri
    add_logo_and_image(None)
    info = [
        ("Patient Name:",   hasta_adi),
        ("Date of Analysis:", analysis_date if analysis_date else pd.Timestamp.now().strftime("%d-%m-%Y")),
        ("Patient ID:",      patient_id),
        ("Date of Birth:",   date_of_birth),
        ("Gender:",          gender),
        ("Protocol Name:",   protocol_name),
        ("Diagnosis:",       diagnosis)
    ]
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, h-6*cm, "fNIRS Analysis")
    c.setFont("Helvetica", 10)
    for i, (lbl, val) in enumerate(info):
        c.drawString(50, h-7*cm - 15*i, lbl)
        c.drawString(200, h-7*cm - 15*i, val)
    
    # Notlar kısmı
    if notes:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y_pos, "Notes:")
        y_pos -= 18
        # Metni kutu içinde, kenarlardan eşit boşluklu ve ortalanmış şekilde yaz
        from reportlab.lib.enums import TA_JUSTIFY
        styles = getSampleStyleSheet()
        note_style = ParagraphStyle(
            'note',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            alignment=TA_JUSTIFY,
            leftIndent=0,
            rightIndent=0,
            spaceAfter=0,
            spaceBefore=0,
            leading=13,
        )
        # Sayfa kenar boşlukları
        left_margin = 50
        right_margin = 50
        box_width = w - left_margin - right_margin
        box_height = 120  # Gerekirse artırılabilir
        note_para = Paragraph(notes.replace("\n", "<br/>"), note_style)
        note_frame = Frame(left_margin, y_pos - box_height + 10, box_width, box_height, showBoundary=0)
        note_frame.addFromList([note_para], c)
        y_pos -= box_height

    sayfa_numarasi()
    c.showPage()


    # Her sayfanın başına logo eklemek için:
    def draw_logo():
        logo_path = resource_path("acibadem.png")
        print(f"Logo path: {logo_path}")  # Debugging için logo yolunu yazdır
        if os.path.exists(logo_path):
            try:
                with Image.open(logo_path) as logo:
                    iw, ih = logo.size
                logo_w = 10 * cm
                logo_h = 4 * cm
                x_pos = (w - logo_w) / 2  # Ortalanmış yatay konum
                y_pos = h - logo_h - 2 * cm  # Sayfanın üstünden 2 cm aşağıda
                c.drawImage(logo_path, x=x_pos, y=y_pos, width=logo_w, height=logo_h, preserveAspectRatio=True)
                print(f"Logo drawn at x={x_pos}, y={y_pos}, width={logo_w}, height={logo_h}")
            except Exception as e:
                print(f"Error drawing logo: {e}")  # Hata mesajını yazdır
        else:
            print("Logo file not found at the specified path.")

    # Tek grafikli sayfa
    def add_graph_page(img_fn, title):
        img_path = os.path.join(save_dir, img_fn)
        if os.path.exists(img_path):
            with Image.open(img_path) as im:
                ow, oh = im.size
            tw = w - 2*cm
            scale = tw / ow
            th = oh * scale
            y_img = (h - th) / 2  # Sayfanın ortasına yerleştir
            c.setFont("Helvetica-Bold", 13)
            tw_title = c.stringWidth(title, "Helvetica-Bold", 13)
            c.drawString((w-tw_title)/2, h-3*cm, title)
            c.drawImage(img_path, x=(w-tw)/2, y=y_img, width=tw, height=th, preserveAspectRatio=True)
        sayfa_numarasi()
        c.showPage()

    # İki grafikli sayfa
    def add_double_graph_page(img_fn1, title1, img_fn2, title2):
        img_path1 = os.path.join(save_dir, img_fn1)
        img_path2 = os.path.join(save_dir, img_fn2)
        y_top = h - 4*cm
        y_gap = 1*cm
        img_height = (h - 7*cm - y_gap) / 2
        tw = w - 2*cm

        # 1. grafik
        if os.path.exists(img_path1):
            with Image.open(img_path1) as im1:
                ow1, oh1 = im1.size
            scale1 = tw / ow1
            th1 = oh1 * scale1
            th1 = min(th1, img_height)
            c.setFont("Helvetica-Bold", 13)
            tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
            c.drawString((w-tw_title1)/2, y_top, title1)
            c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

        # 2. grafik
        if os.path.exists(img_path2):
            with Image.open(img_path2) as im2:
                ow2, oh2 = im2.size
            scale2 = tw / ow2
            th2 = oh2 * scale2
            th2 = min(th2, img_height)
            y2 = y_top-th2-1.5*cm-th2
            c.setFont("Helvetica-Bold", 13)
            tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
            c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
            c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)

        sayfa_numarasi()
        c.showPage()

    # 2.5 sayfa: all_channels_time_fn
    draw_logo()
    img_path = os.path.join(save_dir, r"C:\Users\Hamancab\Documents\VSCODE\NIRSoft\desktop_app\secondpg.jpg")
    if os.path.exists(img_path):
        with Image.open(img_path) as im:
            ow, oh = im.size
        tw = w - 2*cm
        scale = tw / ow
        th = oh * scale
        # Logonun altından başlasın
        y_img = h - 4*cm - 2*cm - th - 1*cm
        c.setFont("Helvetica-Bold", 13)
        # title = "Tüm Kanalların HbO Zaman Grafiği"
        # tw_title = c.stringWidth(title, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title)/2, y_img + th + 0.5*cm, title)
        c.drawImage(img_path, x=(w-tw)/2, y=y_img, width=tw, height=th, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    
    # 3. sayfa: all_channels_time_fn
    draw_logo()
    img_path = os.path.join(save_dir, all_channels_time_all_fn)
    if os.path.exists(img_path):
        with Image.open(img_path) as im:
            ow, oh = im.size
        tw = w - 2*cm
        scale = tw / ow
        th = oh * scale
        # Logonun altından başlasın
        y_img = h - 4*cm - 2*cm - th - 1*cm
        c.setFont("Helvetica-Bold", 13)
        # title = "Tüm Kanalların HbO Zaman Grafiği"
        # tw_title = c.stringWidth(title, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title)/2, y_img + th + 0.5*cm, title)
        c.drawImage(img_path, x=(w-tw)/2, y=y_img, width=tw, height=th, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    # 4. sayfa: all_channels_time_z_fn
    draw_logo()
    img_path1 = os.path.join(save_dir, sag_sol_all_fn)
    img_path2 = os.path.join(save_dir, mean_hbo_time_all_fn)
    y_top = h - 4*cm - 2*cm  # logonun altı
    y_gap = 1*cm
    img_height = (h - 7*cm - y_gap) / 2
    tw = w - 2*cm

    # 1. grafik
    if os.path.exists(img_path1):
        with Image.open(img_path1) as im1:
            ow1, oh1 = im1.size
        scale1 = tw / ow1
        th1 = oh1 * scale1
        th1 = min(th1, img_height)
        c.setFont("Helvetica-Bold", 13)
        # title1 = "Sağ ve Sol Korteks Ortalama HbO Zaman Grafiği"
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    # 2. grafik
    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th2-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # title2 = "Sağ ve Sol Korteks Ortalama HbO Zaman Grafiği (Z-score)"
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    # 5. sayfa: Sağ ve Sol Korteks HbO zaman grafiği + Z-score hali (aynı sayfa)
    draw_logo()
    img_path1 = os.path.join(save_dir, lagmax_fn)
    img_path2 = os.path.join(save_dir, max_fn)
    y_top = h - 4*cm - 2*cm  # logonun altı
    y_gap = 1*cm
    img_height = (h - 7*cm - y_gap) / 2
    tw = w - 2*cm

    # 1. grafik
    if os.path.exists(img_path1):
        with Image.open(img_path1) as im1:
            ow1, oh1 = im1.size
        scale1 = tw / ow1
        th1 = oh1 * scale1
        th1 = min(th1, img_height)
        c.setFont("Helvetica-Bold", 13)
        # title1 = "Sağ ve Sol Korteks Ortalama HbO Zaman Grafiği"
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    # 2. grafik
    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th2-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # title2 = "Sağ ve Sol Korteks Ortalama HbO Zaman Grafiği (Z-score)"
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    # 6. sayfa: Tüm kanallar subplot HbO zaman grafiği + Z-score hali (aynı sayfa)
    draw_logo()
    img_path1 = os.path.join(save_dir, mean_fn)
    img_path2 = os.path.join(save_dir, auc_fn)
    y_top = h - 4*cm - 2*cm  # logonun altı
    y_gap = 1*cm
    img_height = (h - 7*cm - y_gap) / 2
    tw = w - 2*cm

    # 1. grafik
    if os.path.exists(img_path1):
        with Image.open(img_path1) as im1:
            ow1, oh1 = im1.size
        scale1 = tw / ow1
        th1 = oh1 * scale1
        th1 = min(th1, img_height)
        c.setFont("Helvetica-Bold", 13)
        # title1 = "Tüm Kanalların HbO Zaman Grafiği"
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    # 2. grafik
    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th2-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # title2 = "Tüm Kanalların HbO Zaman Grafiği (Z-score)"
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    # 7. sayfa: lagmax ve z-scored lagmax
    draw_logo()
    img_path1 = os.path.join(save_dir, range_fn)
    img_path2 = os.path.join(save_dir, power_fn)
    y_top = h - 4*cm - 2*cm
    y_gap = 1*cm
    img_height = (h - 7*cm - y_gap) / 2
    tw = w - 2*cm

    if os.path.exists(img_path1):
        with Image.open(img_path1) as im1:
            ow1, oh1 = im1.size
        scale1 = tw / ow1
        th1 = oh1 * scale1
        th1 = min(th1, img_height)
        c.setFont("Helvetica-Bold", 13)
        # title1 = "Lagmax (Maksimuma Ulaşma Gecikmesi)"
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th2-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # title2 = "Lagmax (Z-score)"
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    # 8. sayfa: max_fn ve mean_fn
    draw_logo()
    img_path1 = os.path.join(save_dir, mean_z_fn)
    img_path2 = os.path.join(save_dir, auc_z_fn)
    y_top = h - 4*cm - 2*cm
    y_gap = 1*cm
    img_height = (h - 7*cm - y_gap) / 2
    tw = w - 2*cm

    if os.path.exists(img_path1):
        with Image.open(img_path1) as im1:
            ow1, oh1 = im1.size
        scale1 = tw / ow1
        th1 = oh1 * scale1
        th1 = min(th1, img_height)
        c.setFont("Helvetica-Bold", 13)
        # title1 = "Maksimum HbO"
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th2-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # title2 = "Ortalama HbO"
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    # 9. sayfa: auc ve range
    draw_logo()
    img_path1 = os.path.join(save_dir, range_z_fn)
    img_path2 = os.path.join(save_dir, power_z_fn)
    y_top = h - 4*cm - 2*cm
    y_gap = 1*cm
    img_height = (h - 7*cm - y_gap) / 2
    tw = w - 2*cm

    if os.path.exists(img_path1):
        with Image.open(img_path1) as im1:
            ow1, oh1 = im1.size
        scale1 = tw / ow1
        th1 = oh1 * scale1
        th1 = min(th1, img_height)
        c.setFont("Helvetica-Bold", 13)
        # title1 = "AUC HbO"
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th2-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # title2 = "Range HbO"
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    # # 10. sayfa: power
    # draw_logo()
    # img_path = os.path.join(save_dir, power_fn)
    # if os.path.exists(img_path):
    #     with Image.open(img_path) as im:
    #         ow, oh = im.size
    #     tw = w - 2*cm
    #     scale = tw / ow
    #     th = oh * scale
    #     y_img = h - 4*cm - 2*cm - th - 1*cm
    #     c.setFont("Helvetica-Bold", 13)
    #     # title = "Power HbO"
    #     # tw_title = c.stringWidth(title, "Helvetica-Bold", 13)
    #     # c.drawString((w-tw_title)/2, y_img + th + 0.5*cm, title)
    #     c.drawImage(img_path, x=(w-tw)/2, y=y_img, width=tw, height=th, preserveAspectRatio=True)
    # sayfa_numarasi()
    # c.showPage()

    # # 11. sayfa: max_fn ve mean_fn (Z-score)
    # draw_logo()
    # img_path1 = os.path.join(save_dir, max_z_fn)
    # img_path2 = os.path.join(save_dir, mean_z_fn)
    # y_top = h - 4*cm - 2*cm  # logonun altı
    # y_gap = 1*cm
    # img_height = (h - 7*cm - y_gap) / 2
    # tw = w - 2*cm

    # # 1. grafik
    # if os.path.exists(img_path1):
    #     with Image.open(img_path1) as im1:
    #         ow1, oh1 = im1.size
    #     scale1 = tw / ow1
    #     th1 = oh1 * scale1
    #     th1 = min(th1, img_height)
    #     c.setFont("Helvetica-Bold", 13)
    #     # title1 = "Maksimum HbO (Z-score)"
    #     # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
    #     # c.drawString((w-tw_title1)/2, y_top, title1)
    #     c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    # # 2. grafik
    # if os.path.exists(img_path2):
    #     with Image.open(img_path2) as im2:
    #         ow2, oh2 = im2.size
    #     scale2 = tw / ow2
    #     th2 = oh2 * scale2
    #     th2 = min(th2, img_height)
    #     y2 = y_top-th2-1.5*cm-th2
    #     c.setFont("Helvetica-Bold", 13)
    #     # title2 = "Ortalama HbO (Z-score)"
    #     # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
    #     # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
    #     c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    # sayfa_numarasi()
    # c.showPage()

    # # 12. sayfa: auc ve range (Z-score)
    # draw_logo()
    # img_path1 = os.path.join(save_dir, auc_z_fn)
    # img_path2 = os.path.join(save_dir, range_z_fn)
    # y_top = h - 4*cm - 2*cm
    # y_gap = 1*cm
    # img_height = (h - 7*cm - y_gap) / 2
    # tw = w - 2*cm

    # # 1. grafik
    # if os.path.exists(img_path1):
    #     with Image.open(img_path1) as im1:
    #         ow1, oh1 = im1.size
    #     scale1 = tw / ow1
    #     th1 = oh1 * scale1
    #     th1 = min(th1, img_height)
    #     c.setFont("Helvetica-Bold", 13)
    #     # title1 = "AUC HbO (Z-score)"
    #     # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
    #     # c.drawString((w-tw_title1)/2, y_top, title1)
    #     c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    # # 2. grafik
    # if os.path.exists(img_path2):
    #     with Image.open(img_path2) as im2:
    #         ow2, oh2 = im2.size
    #     scale2 = tw / ow2
    #     th2 = oh2 * scale2
    #     th2 = min(th2, img_height)
    #     y2 = y_top-th2-1.5*cm-th2
    #     c.setFont("Helvetica-Bold", 13)
    #     # title2 = "Range HbO (Z-score)"
    #     # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
    #     # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
    #     c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    # sayfa_numarasi()
    # c.showPage()

    # # # Sonraki sayfalarda sırayla tüm grafikler
    # # grafikler = [
    # #     max_fn, max_z_fn,
    # #     mean_fn, mean_z_fn,
    # #     auc_fn, auc_z_fn,
    # #     range_fn, range_z_fn,
    # #     power_fn, power_z_fn,
    # #     lagmax_fn, lagmax_z_fn,
    # #     mean_hbo_compare_fn, mean_z_compare_fn,
    # #     lagmax_compare_fn, lagmax_z_compare_fn,
    # #     power_compare_fn, power_z_compare_fn,
    # #     range_compare_fn, range_z_compare_fn,
    # #     mean_hbo_time_fn, mean_hbo_time_z_fn,
    # #     sag_sol_time_fn, sag_sol_time_z_fn,
    # #     all_channels_time_fn, all_channels_time_z_fn,
    # #     scatter_heatmap_fn, scatter_heatmap_z_fn,
    # # ]

    # # # if brain_overlay_img:
    # # #     grafikler.append(brain_overlay_img)

    # # # if volumetric_heatmap_img:
    # # #     grafikler.append(volumetric_heatmap_img)
    
    # # for img in grafikler:
    # #     add_logo_and_image(img)
    # #     sayfa_numarasi()
    # #     c.showPage()

    c.save()
    print(f"✅ PDF oluşturuldu: {pdf_path}")
    return pdf_path

    # Ensure image dimensions are within limits before saving
    def resize_image_if_needed(image_path, max_width=65535, max_height=65535):
        with Image.open(image_path) as img:
            width, height = img.size
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.ANTIALIAS)
                img.save(image_path)

    # Apply resizing to all saved images
    for img_path in glob.glob(os.path.join(save_dir, "*.png")):
        resize_image_if_needed(img_path)