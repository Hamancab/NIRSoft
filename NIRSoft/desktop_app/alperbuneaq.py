import os
import re
import numpy as np
import pandas as pd
import scipy.io
import sys
import h5py
import PyInstaller.__main__
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')            
import matplotlib.pyplot as plt  
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


def plot_fnirs_brain_overlay(nirs_path, probe_mat_path, save_path):
    """
    nirs_path: .nirs dosyasi
    probe_mat_path: probeInfo içeren .mat dosyasi
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



def save_fig_with_title(save_dir):
    fig = plt.gcf()
    axs = fig.axes
    title = axs[0].get_title() if axs else 'figure'
    if not title:
        title = 'figure'
    
    title = re.sub(r'[^\w\-. ]', '', title).strip().replace(' ', '_')[:50]

    file_path = os.path.join(save_dir, f"{title}.png")
    
    fig.savefig(file_path, bbox_inches='tight', dpi=300)  
    print(f"Saved: {file_path}")
    plt.close(fig)  
    return title + ".png"  

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
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
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
):

    """
    nirs_file_path: .nirs dosyasinin tam yolu
    hasta_adi: Hasta adi-soyadi metni
    sidx, eidx: Trigger indeksleri (1-based) arasini analiz eder
    save_dir: Grafikleri ve raporu kaydedeceği klasör (default: script'in altindaki 'graph')
    """
    
    current_dir = os.path.dirname(os.path.abspath(file))
    if save_dir is None:
        save_dir = os.path.join(current_dir, "graph")
    os.makedirs(save_dir, exist_ok=True)

    mat = scipy.io.loadmat(nirs_file_path)
    t = mat['t'].squeeze()
    d = mat['d']
    s = mat['s'] if 's' in mat else None

    snirf_path = os.path.join(current_dir, 'converted_data.snirf')
    with h5py.File(snirf_path, 'w') as f:
        f.create_dataset('nirs/t', data=t)
        f.create_dataset('nirs/d', data=d)
        f.create_dataset('nirs/s', data=s)
    with h5py.File(snirf_path, 'r') as f:
        t = f['nirs/t'][:]
        d = f['nirs/d'][:]
        s = f['nirs/s'][:]

    trigger_indices = sorted({
        idx
        for ch in range(s.shape[1])
        for idx in np.where(s[:, ch] == 1)[0]
    })
    if len(trigger_indices) < 7:
        raise ValueError("En az 7 trigger bulunmali.")

    walk1_start, walk1_end = trigger_indices[0], trigger_indices[1]
    walk2_start, walk2_end = trigger_indices[2], trigger_indices[3]
    walk3_start, walk3_end = trigger_indices[4], trigger_indices[5]

    t_walk1 = t[walk1_start:walk1_end+1]
    d_walk1 = d[walk1_start:walk1_end+1, :]
    t_walk2 = t[walk2_start:walk2_end+1]
    d_walk2 = d[walk2_start:walk2_end+1, :]
    t_walk3 = t[walk3_start:walk3_end+1]
    d_walk3 = d[walk3_start:walk3_end+1, :]

    t_concat = np.concatenate([t_walk1, t_walk2, t_walk3])
    d_concat = np.concatenate([d_walk1, d_walk2, d_walk3], axis=0)

    d_seg_850 = d_concat[:, -20:]
    od = np.log(np.max(d_seg_850, axis=0) / d_seg_850)
    channels = [f"Ch{i+1}" for i in range(od.shape[1])]


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

    d_850_all = d[:, -20:]
    od_all = np.log(np.max(d_850_all, axis=0) / d_850_all)
    mean_hbo_time_all = np.mean(od_all, axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(t, mean_hbo_time_all, color="#959595FF", linewidth=2)
    for idx in trigger_indices:
        plt.axvline(t[idx], color="#8F14149B", linestyle='--', alpha=0.7)
        
    highlight_color = "#000000"  
    for i in [0, 2, 4]:
        if i + 1 < len(trigger_indices):
            t_start = t[trigger_indices[i]]
            t_end = t[trigger_indices[i + 1]]
            i1 = trigger_indices[i]
            i2 = trigger_indices[i + 1]
            
            plt.axvspan(t_start, t_end, color="#D8D8D875", alpha=0.3)

            plt.plot(t[i1:i2+1], mean_hbo_time_all[i1:i2+1], color=highlight_color, linewidth=3)
    plt.title("Mean HbO Time Series (All Channels, with Triggers)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean OD (HbO)")
    plt.grid(True)
    plt.tight_layout()
    mean_hbo_time_all_fn = save_fig_with_title(save_dir)

    mean_first10 = np.mean(mean_vals[:10])
    mean_last10 = np.mean(mean_vals[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [mean_first10, mean_last10], color=['royalblue', 'crimson'])
    plt.ylabel("Mean HbO (Mean OD)")
    plt.title("Mean HbO: First 10 vs Last 10 Channels")
    plt.tight_layout()
    mean_hbo_compare_fn = save_fig_with_title(save_dir)

    mean_z_first10 = np.mean(mean_z[:10])
    mean_z_last10 = np.mean(mean_z[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [mean_z_first10, mean_z_last10], color=['royalblue', 'crimson'])
    plt.ylabel("Mean HbO Z-score")
    plt.title("Mean HbO Z-score: First 10 vs Last 10 Channels")
    plt.tight_layout()
    mean_z_compare_fn = save_fig_with_title(save_dir)

    lagmax_first10 = np.mean(lagmax_relative[:10])
    lagmax_last10 = np.mean(lagmax_relative[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [lagmax_first10, lagmax_last10], color=['royalblue', 'crimson'])
    plt.ylabel("Lagmax (s)")
    plt.title("Lagmax: First 10 vs Last 10 Channels")
    plt.tight_layout()
    lagmax_compare_fn = save_fig_with_title(save_dir)

    lagmax_z_first10 = np.mean(lagmax_z[:10])
    lagmax_z_last10 = np.mean(lagmax_z[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [lagmax_z_first10, lagmax_z_last10], color=['royalblue', 'crimson'])
    plt.ylabel("Lagmax Z-score")
    plt.title("Lagmax Z-score: First 10 vs Last 10 Channels")
    plt.tight_layout()
    lagmax_z_compare_fn = save_fig_with_title(save_dir)

    power_first10 = np.mean(power_vals[:10])
    power_last10 = np.mean(power_vals[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [power_first10, power_last10], color=['royalblue', 'crimson'])
    plt.ylabel("Power HbO")
    plt.title("Power HbO: First 10 vs Last 10 Channels")
    plt.tight_layout()
    power_compare_fn = save_fig_with_title(save_dir)

    power_z_first10 = np.mean(power_z[:10])
    power_z_last10 = np.mean(power_z[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [power_z_first10, power_z_last10], color=['royalblue', 'crimson'])
    plt.ylabel("Power HbO Z-score")
    plt.title("Power HbO Z-score: First 10 vs Last 10 Channels")
    plt.tight_layout()
    power_z_compare_fn = save_fig_with_title(save_dir)

    range_first10 = np.mean(range_vals[:10])
    range_last10 = np.mean(range_vals[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [range_first10, range_last10], color=['royalblue', 'crimson'])
    plt.ylabel("Range HbO")
    plt.title("Range HbO: First 10 vs Last 10 Channels")
    plt.tight_layout()
    range_compare_fn = save_fig_with_title(save_dir)

    range_z_first10 = np.mean(range_z[:10])
    range_z_last10 = np.mean(range_z[10:])
    plt.figure(figsize=(7,5))
    plt.bar(['First 10 Channels', 'Last 10 Channels'], [range_z_first10, range_z_last10], color=['royalblue', 'crimson'])
    plt.ylabel("Range HbO Z-score")
    plt.title("Range HbO Z-score: First 10 vs Last 10 Channels")
    plt.tight_layout()
    range_z_compare_fn = save_fig_with_title(save_dir)

    mean_hbo_time = np.mean(od, axis=1)
    plt.figure(figsize=(12, 5))
    plt.plot(t_concat, mean_hbo_time, color='black', linewidth=2)
    plt.title("Mean HbO Time Series (Selected Segments)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean OD (HbO)")
    plt.grid(True)
    plt.tight_layout()
    mean_hbo_time_fn = save_fig_with_title(save_dir)

    mean_hbo_time_z = (mean_hbo_time - np.mean(mean_hbo_time)) / np.std(mean_hbo_time)
    plt.figure(figsize=(12, 5))
    plt.plot(t_concat, mean_hbo_time_z, color='royalblue', linewidth=2)
    plt.title("Mean HbO Time Series (Z-score, Selected Segments)")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-score")
    plt.grid(True)
    plt.tight_layout()
    mean_hbo_time_z_fn = save_fig_with_title(save_dir)

    sag_kortex = od[:, :10]
    sol_kortex = od[:, 10:]
    mean_sag = np.mean(sag_kortex, axis=1)
    mean_sol = np.mean(sol_kortex, axis=1)
    plt.figure(figsize=(12, 5))
    plt.plot(t_concat, mean_sag, label='Right Cortex', color='royalblue', linewidth=2)
    plt.plot(t_concat, mean_sol, label='Left Cortex', color='crimson', linewidth=2)
    plt.title("Right and Left Cortex Mean HbO Time Series")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean OD (HbO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    sag_sol_time_fn = save_fig_with_title(save_dir)

    mean_sag_z = (mean_sag - np.mean(mean_sag)) / np.std(mean_sag)
    mean_sol_z = (mean_sol - np.mean(mean_sol)) / np.std(mean_sol)
    plt.figure(figsize=(12, 5))
    plt.plot(t_concat, mean_sag_z, label='Right Cortex (Z)', color='royalblue', linewidth=2)
    plt.plot(t_concat, mean_sol_z, label='Left Cortex (Z)', color='crimson', linewidth=2)
    plt.title("Right and Left Cortex Mean HbO Time Series (Z-score)")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    sag_sol_time_z_fn = save_fig_with_title(save_dir)

    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    fig.suptitle("HbO Signals - All Channels", fontsize=16)
    for i in range(20):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        channel_hbo = od[:, i]
        ax.plot(t_concat, channel_hbo, color='crimson', linewidth=1.5)
        ax.set_title(f"Channel {i + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("HbO (OD)")
        ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    all_channels_time_fn = save_fig_with_title(save_dir)

    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    fig.suptitle("HbO Signals (Z-score) - All Channels", fontsize=16)
    for i in range(20):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        channel_hbo = od[:, i]
        channel_hbo_z = (channel_hbo - np.mean(channel_hbo)) / np.std(channel_hbo)
        ax.plot(t_concat, channel_hbo_z, color='royalblue', linewidth=1.5)
        ax.set_title(f"Z-score Channel {i + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("HbO (Z-score)")
        ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    all_channels_time_z_fn = save_fig_with_title(save_dir)

    channels = [
        'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10',
        'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16', 'Ch17', 'Ch18', 'Ch19', 'Ch20'
    ]
    X = [16.1, 16.2, 17.8, 17.6, 18.9, 19.6, 17.7, 19.1, 20.1, 20.5, 
         9.5, 9.2, 7.8, 8.3, 6.6, 5.8, 7.7, 6.4, 5.2, 4.9]
    Y = [12.1, 15.3, 13.7, 10.5, 12.0, 10.2, 16.8, 15.0, 13.3, 11.5, 
         11.9, 15.1, 13.4, 10.3, 11.7, 10.0, 16.7, 15.0, 13.2, 11.4]

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
    plt.tight_layout()
    scatter_heatmap_z_fn = save_fig_with_title(save_dir)

###########################################################################################################################
    # 7) PDF report
    pdf_path = os.path.join(current_dir, "rapor.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4
    
    logo_path = os.path.join(current_dir, "acibadem.png")
    
    # Set a default y position in case logo does not exist
    margin = 2 * cm
    y = h - margin  # Default: top margin
    def add_logo_and_image(page_img):
        # Logo
        logo_path = resource_path("acibadem.png")
        if os.path.exists(logo_path):
            with Image.open(logo_path) as logo:
                iw, ih = logo.size
            c.drawImage(logo_path, x=(w-10*cm)/2, y=h-4*cm-2*cm, width=10*cm, height=4*cm, preserveAspectRatio=True)
            
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

    # ------------------------------------------------
    margin = 2 * cm

    if os.path.exists(logo_path):
        # Orijinal logo boyutları
        with Image.open(logo_path) as logo:
            iw, ih = logo.size

        max_w = w - 2 * margin
        max_h = h - 3 * margin  

        scale = min(max_w / iw, max_h / ih)
        logo_w_scaled = iw * scale
        logo_h_scaled = ih * scale

        x = (w - logo_w_scaled) / 2
        y = h - margin - logo_h_scaled
        c.drawImage(logo_path, x, y, width=logo_w_scaled, height=logo_h_scaled, preserveAspectRatio=True)
        y = y  # y is now the bottom of the logo
    else:
        y = h - margin
    c.setFont("Helvetica-Bold", 14)
    title = f"fNIRS Analysis on {hasta_adi}"
    tw = c.stringWidth(title, "Helvetica-Bold", 14)
    text_y = y - 1 * cm
    c.drawString((w - tw) / 2, text_y, title)

    sayfa_numarasi()
    c.showPage()
    # ------------------------------------------------
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
    
    y_pos = h-7*cm - 15*len(info) - 20 
    if notes:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y_pos, "Notes:")
        y_pos -= 18
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
        left_margin = 50
        right_margin = 50
        box_width = w - left_margin - right_margin
        box_height = 120 
        note_para = Paragraph(notes.replace("\n", "<br/>"), note_style)
        note_frame = Frame(left_margin, y_pos - box_height + 10, box_width, box_height, showBoundary=0)
        note_frame.addFromList([note_para], c)
        y_pos -= box_height

    sayfa_numarasi()
    c.showPage()

    def draw_logo():
        logo_path = resource_path("acibadem.png")
        if os.path.exists(logo_path):
            with Image.open(logo_path) as logo:
                iw, ih = logo.size
            logo_w = 10*cm
            logo_h = 4*cm
            c.drawImage(logo_path, x=(w-logo_w)/2, y=h-logo_h-2*cm, width=logo_w, height=logo_h, preserveAspectRatio=True)

    def add_graph_page(img_fn, title):
        img_path = os.path.join(save_dir, img_fn)
        if os.path.exists(img_path):
            with Image.open(img_path) as im:
                ow, oh = im.size
            tw = w - 2*cm
            scale = tw / ow
            th = oh * scale
            y_img = (h - th) / 2 
            c.setFont("Helvetica-Bold", 13)
            tw_title = c.stringWidth(title, "Helvetica-Bold", 13)
            c.drawString((w-tw_title)/2, h-3*cm, title)
            c.drawImage(img_path, x=(w-tw)/2, y=y_img, width=tw, height=th, preserveAspectRatio=True)
        sayfa_numarasi()
        c.showPage()

    def add_double_graph_page(img_fn1, title1, img_fn2, title2):
        img_path1 = os.path.join(save_dir, img_fn1)
        img_path2 = os.path.join(save_dir, img_fn2)
        y_top = h - 4*cm
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
            tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
            c.drawString((w-tw_title1)/2, y_top, title1)
            c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

        if os.path.exists(img_path2):
            with Image.open(img_path2) as im2:
                ow2, oh2 = im2.size
            scale2 = tw / ow2
            th2 = oh2 * scale2
            th2 = min(th2, img_height)
            y2 = y_top-th1-1.5*cm-th2
            c.setFont("Helvetica-Bold", 13)
            tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
            c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
            c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)

        sayfa_numarasi()
        c.showPage()

    draw_logo()
    img_path = os.path.join(save_dir, all_channels_time_fn)
    if os.path.exists(img_path):
        with Image.open(img_path) as im:
            ow, oh = im.size
        tw = w - 2*cm
        scale = tw / ow
        th = oh * scale
        y_img = h - 4*cm - 2*cm - th - 1*cm
        c.setFont("Helvetica-Bold", 13)
        # tw_title = c.stringWidth(title, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title)/2, y_img + th + 0.5*cm, title)
        c.drawImage(img_path, x=(w-tw)/2, y=y_img, width=tw, height=th, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    draw_logo()
    img_path1 = os.path.join(save_dir, sag_sol_time_fn)
    img_path2 = os.path.join(save_dir, mean_hbo_time_all_fn)
    y_top = h - 4*cm - 2*cm  # logonun altı
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
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th1-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    draw_logo()
    img_path1 = os.path.join(save_dir, lagmax_fn)
    img_path2 = os.path.join(save_dir, max_fn)
    y_top = h - 4*cm - 2*cm  # logonun altı
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
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th1-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    draw_logo()
    img_path1 = os.path.join(save_dir, mean_fn)
    img_path2 = os.path.join(save_dir, auc_fn)
    y_top = h - 4*cm - 2*cm  # logonun altı
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
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th1-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

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
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th1-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # title2 = "Lagmax (Z-score)"
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

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
        # title1 = "Max HbO"
        # tw_title1 = c.stringWidth(title1, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title1)/2, y_top, title1)
        c.drawImage(img_path1, x=(w-tw)/2, y=y_top-th1-0.5*cm, width=tw, height=th1, preserveAspectRatio=True)

    if os.path.exists(img_path2):
        with Image.open(img_path2) as im2:
            ow2, oh2 = im2.size
        scale2 = tw / ow2
        th2 = oh2 * scale2
        th2 = min(th2, img_height)
        y2 = y_top-th1-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

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
        y2 = y_top-th1-1.5*cm-th2
        c.setFont("Helvetica-Bold", 13)
        # title2 = "Range HbO"
        # tw_title2 = c.stringWidth(title2, "Helvetica-Bold", 13)
        # c.drawString((w-tw_title2)/2, y2+th2+0.2*cm, title2)
        c.drawImage(img_path2, x=(w-tw)/2, y=y2, width=tw, height=th2, preserveAspectRatio=True)
    sayfa_numarasi()
    c.showPage()

    c.save()
    print(f"✅ PDF oluşturuldu: {pdf_path}")
    return pdf_path