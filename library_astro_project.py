import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz, SkyCoord
import astropy.units as u
from scipy.stats import binned_statistic_2d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass, shift

# ==========================================
# 1. CORE PROCESSING & CLEANING (Full Diagnostic Version)
# ==========================================
def analyze_trajectory_and_clean(filename, lat_str, lon_str, height_m, 
                                 window_length=15, poly_order=3, sigma_threshold=3.5,
                                 box_margin=1.2, tail_trim_threshold=0.8,
                                 plot_result=True,          # Show trajectory plots?
                                 plot_absolute=False,       # Plot Absolute (Az/El) or Relative (Offsets)?
                                 plot_rfi=True,             # Show BBC Signal plots?
                                 bad_bbcs=None):            # List of bad channels to ignore
    """
    MASTER FUNCTION: Performs Trajectory Cleaning, Signal Curation (RFI), 
    and Flexible Plotting (Absolute vs Relative).
    Includes UINT16 Saturation Check.
    """
    
    # --- 1. Load Data ---
    try:
        with fits.open(filename, ignore_missing_end=True) as hdul:
            if hdul[1].data is None:
                print(f"❌ Error: File {filename.split('/')[-1]} contains no data table.")
                return None
            
            t = Table(hdul[1].data)
            
        # SAFETY CHECK: Is the table empty?
        if len(t) == 0:
            print(f"❌ Error: File {filename.split('/')[-1]} has 0 rows (Empty).")
            return None
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

    # --- 1.5. SIGNAL CURATION & SATURATION CHECK ---
    # Identify BBC columns
    bbc_cols = [col for col in t.colnames if 'BBC' in col]
    
    # --- NEW: UINT16 Saturation Check ---
    # Digital receivers often clip at 65535 (2^16 - 1). 
    # If we hit this, the data is non-linear and cannot be calibrated.
    saturation_limit = 65500 # A bit below 65535 to be safe
    
    for col in bbc_cols + ['RIGHT_POL']:
        if col in t.colnames:
            max_val = np.max(t[col])
            if max_val > saturation_limit:
                print(f"⚠️  WARNING: SATURATION DETECTED in {col}!")
                print(f"    Max Value: {max_val} (Limit: 65535)")
                print(f"    This scan is likely invalid for Flux Calibration.")

    # Plot BBCs if requested (to find RFI)
    if plot_rfi and len(bbc_cols) > 0:
        plt.figure(figsize=(12, 4))
        plt.title(f"Signal Check: Individual BBCs ({filename.split('/')[-1]})")
        for bbc in bbc_cols:
            if bad_bbcs and bbc in bad_bbcs: continue # Don't plot the ones we know are bad
            plt.plot(t[bbc], label=bbc, alpha=0.5, lw=0.8)
        
        # Add a red line to visualize the limit
        plt.axhline(65535, color='red', linestyle='--', label='Saturation Limit')
        
        plt.xlabel("Sample Number")
        plt.ylabel("Power (Arbitrary Units)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Recalculate Intensity (Handling Bad BBCs)
    if bad_bbcs:
        print(f"   -> Recalculating Intensity excluding: {bad_bbcs}")
        good_bbcs = [col for col in bbc_cols if col not in bad_bbcs]
        if not good_bbcs:
            print("Error: All BBCs marked as bad!")
            return None
        data_stack = np.column_stack([t[col] for col in good_bbcs])
        intensity = np.sum(data_stack, axis=1)
    else:
        intensity = t['RIGHT_POL'].astype(np.float64)

    # --- 2. Coordinates & Time ---
    location = EarthLocation(lat=lat_str, lon=lon_str, height=height_m*u.m)
    obs_times = Time(t['JD'], format='jd')
    
    time_sec = (obs_times - obs_times[0]).to(u.s).value
    time_min = (obs_times - obs_times[0]).to(u.min).value
    dt = np.median(np.diff(time_sec))
    if dt == 0: return None

    sun_icrs = get_sun(obs_times)
    sun_altaz = sun_icrs.transform_to(AltAz(obstime=obs_times, location=location))
    
    az_quant = t['Azimuth'] if t['Azimuth'].unit else t['Azimuth'] * u.deg
    el_quant = t['Elevation'] if t['Elevation'].unit else t['Elevation'] * u.deg

    # --- NEW: Astropy Crash Prevention ---
    # Hardware encoders sometimes overshoot the zenith slightly (e.g., 90.04 deg).
    # Astropy strictly requires Elevation to be between -90 and +90.
    el_quant = np.clip(el_quant.value, -90.0, 90.0) * u.deg
    
    # Calculate Offsets
    d_az = ((az_quant.to(u.deg).value - sun_altaz.az.deg) * np.cos(np.radians(sun_altaz.alt.deg))).astype(np.float64)
    d_el = (el_quant.to(u.deg).value - sun_altaz.alt.deg).astype(np.float64)

    tel_altaz = SkyCoord(alt=el_quant, az=az_quant, frame='altaz', obstime=obs_times, location=location)
    tel_icrs = tel_altaz.transform_to('icrs')
    d_ra = ((tel_icrs.ra.deg - sun_icrs.ra.deg) * np.cos(np.radians(sun_icrs.dec.deg))).astype(np.float64)
    d_dec = (tel_icrs.dec.deg - sun_icrs.dec.deg).astype(np.float64)

    # --- 3. Filter: Velocity ---
    if plot_result: print(f"Calculating Velocity...")
    vel_az = savgol_filter(d_az, window_length, poly_order, deriv=1, delta=dt)
    vel_el = savgol_filter(d_el, window_length, poly_order, deriv=1, delta=dt)
    total_vel = np.sqrt(vel_az**2 + vel_el**2)

    median_vel = np.median(total_vel)
    mad_vel = np.median(np.abs(total_vel - median_vel))
    threshold_vel = median_vel + (sigma_threshold * 1.4826 * mad_vel)
    
    is_anomaly = total_vel > threshold_vel

    # --- 4. Filter: Segment Analysis (Tail Chopper) ---
    valid_mask = ~is_anomaly
    change_indices = np.where(np.diff(valid_mask.astype(int)) != 0)[0] + 1
    split_indices = np.split(np.arange(len(valid_mask)), change_indices)
    valid_chunks = [chunk for chunk in split_indices if valid_mask[chunk[0]]]
    
    if len(valid_chunks) > 2:
        chunk_lengths = []
        for chunk in valid_chunks:
            dra = d_ra[chunk[-1]] - d_ra[chunk[0]]
            ddec = d_dec[chunk[-1]] - d_dec[chunk[0]]
            dist = np.sqrt(dra**2 + ddec**2)
            chunk_lengths.append(dist)
        
        last_length = chunk_lengths[-1]
        median_length = np.median(chunk_lengths)
        
        if last_length < (median_length * tail_trim_threshold):
            if plot_result: print(f"Detected Return Tail: Length {last_length:.2f} vs Median {median_length:.2f}. Removing.")
            is_anomaly[valid_chunks[-1]] = True
            
    # --- 5. Prepare Output (With Correct Intensity) ---
    full_df = pd.DataFrame({
        'Time_Min': time_min.astype(np.float64),
        'Az_Abs': az_quant.to(u.deg).value.astype(np.float64),
        'El_Abs': el_quant.to(u.deg).value.astype(np.float64),
        'RA_Abs': tel_icrs.ra.deg.astype(np.float64),
        'Dec_Abs': tel_icrs.dec.deg.astype(np.float64),
        'Az_Offset': d_az.astype(np.float64),
        'El_Offset': d_el.astype(np.float64),
        'RA_Offset': d_ra.astype(np.float64),
        'Dec_Offset': d_dec.astype(np.float64),
        'Intensity': intensity, # <--- The cleaned signal
        'Is_Anomaly': is_anomaly
    })
    
    clean_df = full_df[~is_anomaly].copy().reset_index(drop=True)

    # --- 6. Plotting (With Absolute/Relative Switch) ---
    if plot_result:
        # Determine what to plot based on user flag
        if plot_absolute:
            col_x_mech, col_y_mech = 'Az_Abs', 'El_Abs'
            col_x_cel, col_y_cel = 'RA_Abs', 'Dec_Abs'
            lbl_x_mech, lbl_y_mech = 'Azimuth (deg)', 'Elevation (deg)'
            lbl_x_cel, lbl_y_cel = 'Right Ascension (deg)', 'Declination (deg)'
            title_suffix = "(Absolute Coordinates)"
        else:
            col_x_mech, col_y_mech = 'Az_Offset', 'El_Offset'
            col_x_cel, col_y_cel = 'RA_Offset', 'Dec_Offset'
            lbl_x_mech, lbl_y_mech = 'Az Offset (deg)', 'El Offset (deg)'
            lbl_x_cel, lbl_y_cel = 'RA Offset (deg)', 'Dec Offset (deg)'
            title_suffix = "(Relative to Sun)"

        fig = plt.figure(figsize=(15, 14))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1.5], hspace=0.35)

        # ROW 1: Velocity
        ax_vel = fig.add_subplot(gs[0, :])
        ax_vel.plot(time_min, total_vel, color='black', lw=0.8, label='Velocity')
        ax_vel.axhline(threshold_vel, color='red', linestyle='--', label='Threshold')
        ax_vel.scatter(time_min[is_anomaly], total_vel[is_anomaly], color='red', s=10, zorder=5, label='Anomalies')
        ax_vel.set_ylabel("Speed (deg/s)")
        ax_vel.set_xlabel("Time (min)")
        ax_vel.set_title("1. Kinematic Analysis: Velocity Detection")
        ax_vel.legend(loc='upper right')
        ax_vel.grid(True, alpha=0.3)

        sc_mappable = None

        def plot_map_panel(ax, data_points, data_anoms, start_end_source, x_col, y_col, title, invert_x=False):
            nonlocal sc_mappable
            sc = ax.scatter(data_points[x_col], data_points[y_col], c=data_points['Time_Min'], 
                            cmap='viridis', s=5, alpha=0.7, label='Valid Scan')
            if sc_mappable is None: sc_mappable = sc
            
            if data_anoms is not None and not data_anoms.empty:
                ax.scatter(data_anoms[x_col], data_anoms[y_col], color='red', marker='x', s=20, alpha=0.5, label='Flagged')

            if not start_end_source.empty:
                ax.scatter(start_end_source[x_col].iloc[0], start_end_source[y_col].iloc[0], 
                           marker='*', s=250, facecolor='lime', edgecolor='k', zorder=10, label='Start')
                ax.scatter(start_end_source[x_col].iloc[-1], start_end_source[y_col].iloc[-1], 
                           marker='X', s=200, color='red', edgecolor='white', zorder=10, label='End')

            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            # Only force equal aspect ratio for offsets; absolute coords might be distorted
            if not plot_absolute: ax.axis('equal') 
            if invert_x: ax.invert_xaxis()
            ax.legend(loc='upper right', fontsize='small', framealpha=0.9)

        # Split Data
        df_good = full_df[~full_df['Is_Anomaly']]
        df_bad = full_df[full_df['Is_Anomaly']]

        # ROW 2: Raw Maps
        ax_az_raw = fig.add_subplot(gs[1, 0])
        plot_map_panel(ax_az_raw, df_good, df_bad, full_df, col_x_mech, col_y_mech, 
                       f"2a. Raw Mechanical Scan {title_suffix}", invert_x=False) 
        ax_az_raw.set_ylabel(lbl_y_mech)

        ax_ra_raw = fig.add_subplot(gs[1, 1])
        plot_map_panel(ax_ra_raw, df_good, df_bad, full_df, col_x_cel, col_y_cel, 
                       f"2b. Raw Celestial Scan {title_suffix}", invert_x=True)
        ax_ra_raw.set_ylabel(lbl_y_cel)

        # ROW 3: Clean Maps
        ax_az_clean = fig.add_subplot(gs[2, 0])
        plot_map_panel(ax_az_clean, df_good, None, clean_df, col_x_mech, col_y_mech, 
                       f"3a. Clean Mechanical Map {title_suffix}", invert_x=False)
        ax_az_clean.set_xlabel(lbl_x_mech)
        ax_az_clean.set_ylabel(lbl_y_mech)

        ax_ra_clean = fig.add_subplot(gs[2, 1])
        plot_map_panel(ax_ra_clean, df_good, None, clean_df, col_x_cel, col_y_cel, 
                       f"3b. Clean Celestial Map {title_suffix}", invert_x=True)
        ax_ra_clean.set_xlabel(lbl_x_cel)

        # Shared Colorbar
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(sc_mappable, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Time Elapsed (Minutes)')

        plt.subplots_adjust(bottom=0.12)
        plt.suptitle(f"Trajectory Cleaning Report: {filename.split('/')[-1]}", fontsize=16)
        plt.show()
    
    return clean_df


def subtract_baseline(signal_array, poly_order=1, sigma_threshold=3.0, plot_fit=False):
    """
    Removes low-frequency drift (baseline).
    """
    median = np.median(signal_array)
    std = np.std(signal_array)
    is_background = signal_array < (median + sigma_threshold * std)
    
    x = np.arange(len(signal_array))
    x_bg = x[is_background]
    y_bg = signal_array[is_background]
    
    if len(x_bg) < len(x) * 0.1:
        return signal_array, np.zeros_like(signal_array)
        
    try:
        z = np.polyfit(x_bg, y_bg, poly_order)
        p = np.poly1d(z)
        baseline = p(x)
        corrected_signal = signal_array - baseline
    except:
        return signal_array, np.zeros_like(signal_array)
    
    if plot_fit:
        plt.figure(figsize=(10, 3))
        plt.plot(x, signal_array, 'k-', alpha=0.3, label='Raw')
        plt.plot(x, baseline, 'r--', label='Baseline')
        plt.legend()
        plt.show()
    
    return corrected_signal, baseline

# ==========================================
# 2. MAPPING (Unified)
# ==========================================

def generate_tod_map(filename, lat_str, lon_str, height_m, 
                     num_pixels=45, show_cleaning_plots=False, 
                     bad_bbcs=None, use_absolute=False, show_map=True,
                     do_baseline_sub=True): # Defaulting to True is good practice
    """
    UNIFIED PIPELINE: Clean -> Baseline Sub -> Map.
    """
    print(f"--- Processing {filename.split('/')[-1]} ---")
    
    # 1. Clean
    clean_df = analyze_trajectory_and_clean(
        filename, lat_str, lon_str, height_m, 
        plot_result=show_cleaning_plots, plot_rfi=show_cleaning_plots, bad_bbcs=bad_bbcs
    )
    
    if clean_df is None or clean_df.empty: return None, None, None, None

    # 2. Baseline Subtraction
    if do_baseline_sub:
        raw_signal = clean_df['Intensity'].values
        corrected_signal, _ = subtract_baseline(raw_signal, poly_order=1, plot_fit=show_cleaning_plots)
        clean_df['Intensity'] = corrected_signal

    # 3. Coordinate Selection
    if use_absolute:
        x, y = clean_df['RA_Abs'], clean_df['Dec_Abs']
        xlabel, ylabel, title_type = "RA (deg)", "Dec (deg)", "Sky Map (Absolute)"
    else:
        x, y = clean_df['RA_Offset'], clean_df['Dec_Offset']
        xlabel, ylabel, title_type = "RA Offset (deg)", "Dec Offset (deg)", "Solar Image (Relative)"

    # 4. Binning
    x_bins = np.linspace(x.min(), x.max(), num_pixels + 1)
    y_bins = np.linspace(y.min(), y.max(), num_pixels + 1)
    
    H, x_edges, y_edges, _ = binned_statistic_2d(
        x, y, clean_df['Intensity'], statistic='mean', bins=[x_bins, y_bins]
    )
    
    # 5. Plotting
    if show_map:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(H.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       cmap='inferno', interpolation='nearest')
        plt.colorbar(im, label='Intensity')
        ax.invert_xaxis()
        ax.set_title(f"{title_type}\n{filename.split('/')[-1]}")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        plt.show()
    
    return H, x_edges, y_edges, clean_df

# ==========================================
# 3. ANALYSIS TOOLS (Updated to use new map)
# ==========================================
def plot_galactic_trace(clean_df):
    """
    Plots the telescope path in Galactic Coordinates (l, b).
    Required to see if the scan aligns with the Milky Way plane.
    """
    if clean_df is None or clean_df.empty: return

    ra_vals = clean_df['RA_Abs'].values * u.deg
    dec_vals = clean_df['Dec_Abs'].values * u.deg

    # Create SkyCoord object from RA/Dec
    c = SkyCoord(ra=ra_vals, dec=dec_vals, frame='icrs')
    gal = c.galactic

    plt.figure(figsize=(8, 6))
    
    # Use .deg accessor to plot raw values
    sc = plt.scatter(gal.l.deg, gal.b.deg, c=clean_df['Intensity'], cmap='inferno', s=10)
    plt.colorbar(sc, label='Intensity')
    
    plt.title("Telescope Trace in Galactic Coordinates")
    plt.xlabel("Galactic Longitude (l)")
    plt.ylabel("Galactic Latitude (b)")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis() # Astronomy convention (l increases to the left)
    plt.show()

def analyze_bbc_dependence(filepath, num_pixels=45):
    """
    Stand-alone analysis of Frequency Dependence.
    Reads FITS directly to ensure BBC columns are found for the diffraction check.
    """
    print(f"--- Frequency Analysis: {filepath.split('/')[-1]} ---")
    
    try:
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            t = Table(data)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Identify BBC Columns (Low Freq -> High Freq)
    bbc_cols = sorted([col for col in t.colnames if 'BBC' in col])
    
    if not bbc_cols:
        print("ERROR: No 'BBC' columns found in FITS file!")
        return None
        
    az, el = t['Azimuth'], t['Elevation']
    total_power = t['RIGHT_POL']
    peak_idx = np.argmax(total_power)
    
    # Calculate relative offsets centered on the peak for beam sizing
    d_az = (az - az[peak_idx]) * np.cos(np.radians(el[peak_idx]))
    d_el = (el - el[peak_idx])
    
    x_edges = np.linspace(d_az.min(), d_az.max(), num_pixels + 1)
    y_edges = np.linspace(d_el.min(), d_el.max(), num_pixels + 1)
    
    results = []
    print("-> Fitting Beams per BBC channel...", end=" ")
    for i, bbc in enumerate(bbc_cols):
        H, _, _, _ = binned_statistic_2d(d_az, d_el, t[bbc], statistic='mean', bins=[x_edges, y_edges])
        beam = fit_beam_profile(H, x_edges, y_edges) # fit_beam_profile handles the transpose internally
        
        if beam:
            avg_w = (beam['True_Beam_FWHM'][0] + beam['True_Beam_FWHM'][1]) / 2
            if 0.5 < avg_w < 10.0: # Filter out clearly failed fits
                results.append({
                    'BBC_Index': i, 'BBC_Name': bbc,
                    'Peak_Gain': beam['Peak_Gain'], 'Avg_Width': avg_w
                })
        if i % 4 == 0: print(".", end="")
            
    print(" Done.")
    return pd.DataFrame(results)

def plot_frequency_dependence(df_freq):
    """
    Visualizes Bandpass response and the Diffraction limit check (Width vs Channel).
    """
    if df_freq is None or df_freq.empty:
        print("No valid BBC data to plot.")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bandpass (Electronic sensitivity across channels)
    ax1.plot(df_freq['BBC_Index'], df_freq['Peak_Gain'], 'o-', color='orange')
    ax1.set_title("Bandpass Response (Gain vs Channel)")
    ax1.set_xlabel("Frequency Channel Index"); ax1.set_ylabel("Peak Signal (Counts)")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Diffraction Check (Physical beam width vs frequency)
    ax2.plot(df_freq['BBC_Index'], df_freq['Avg_Width'], 's-', color='purple')
    
    if len(df_freq) > 1:
        z = np.polyfit(df_freq['BBC_Index'], df_freq['Avg_Width'], 1)
        p = np.poly1d(z)
        ax2.plot(df_freq['BBC_Index'], p(df_freq['BBC_Index']), 'k--', label=f"Slope: {z[0]:.4f} deg/ch")
        ax2.legend()
        
    ax2.set_title("Beam Width vs. Channel (Diffraction Check)")
    ax2.set_xlabel("Channel Index (Low Freq -> High Freq)"); ax2.set_ylabel("Beam Width (Degrees)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    exponent = - (a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)
    return offset + amplitude * np.exp(exponent)

def fit_beam_profile(data_map, x_edges, y_edges):
    data = data_map.T 
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    
    x_valid = X.ravel()[~np.isnan(data.ravel())]
    y_valid = Y.ravel()[~np.isnan(data.ravel())]
    data_valid = data.ravel()[~np.isnan(data.ravel())]
    
    if len(data_valid) < 50: return None
    
    p0 = [np.max(data_valid), x_valid[np.argmax(data_valid)], y_valid[np.argmax(data_valid)], 0.5, 0.5, 0, 0]
    try:
        popt, _ = curve_fit(gaussian_2d, (x_valid, y_valid), data_valid, p0=p0, maxfev=5000)
    except: return None
    
    amp, x0, y0, sx, sy, theta, _ = popt
    fwhm_x, fwhm_y = abs(2.355 * sx), abs(2.355 * sy)
    
    return {
        'Peak_Gain': amp, 'Center': (x0, y0), 'Measured_FWHM': (fwhm_x, fwhm_y),
        'True_Beam_FWHM': (np.sqrt(max(0, fwhm_x**2-0.25)), np.sqrt(max(0, fwhm_y**2-0.25))),
        'Asymmetry': max(fwhm_x, fwhm_y)/min(fwhm_x, fwhm_y), 'Theta_deg': np.degrees(theta)%180
    }

def analyze_single_observation(filepath, lat, lon, alt):
    # SAFETY: Sun Only
    if 'SUN' not in filepath.split('/')[-1]: return None

    # UPDATED: Now uses generate_tod_map (with baseline sub enabled by default)
    map_data, x_bins, y_bins, clean_df = generate_tod_map(
        filepath, lat, lon, alt, 
        num_pixels=45, use_absolute=False, 
        show_cleaning_plots=False, show_map=False, do_baseline_sub=True
    )
    
    if map_data is None: return None

    beam = fit_beam_profile(map_data, x_bins, y_bins)
    if beam:
        return {
            'Filename': filepath.split('/')[-1], 'Elevation': clean_df['El_Abs'].mean(),
            'Peak_Gain': beam['Peak_Gain'], 'Beam_Width_X': beam['True_Beam_FWHM'][0],
            'Beam_Width_Y': beam['True_Beam_FWHM'][1], 'Asymmetry': beam['Asymmetry']
        }
    return None

def run_sun_elevation_batch(file_pattern, lat, lon, alt):
    file_list = sorted(glob.glob(file_pattern))
    results = []
    print(f"Batch Processing {len(file_list)} files...")
    for i, f in enumerate(file_list):
        if 'SUN' not in f: continue
        res = analyze_single_observation(f, lat, lon, alt)
        if res: results.append(res)
        print(f"[{i+1}] {f.split('/')[-1]} -> {'Done' if res else 'Skip'}")
    
    df = pd.DataFrame(results)
    if not df.empty: df = df.sort_values('Elevation').reset_index(drop=True)
    return df

def perform_residual_analysis(tpi_file, img_file, lat_str, lon_str, height_m):
    # UPDATED: Use the unified mapper with baseline subtraction
    _, _, _, clean_df = generate_tod_map(
        tpi_file, lat_str, lon_str, height_m, 
        num_pixels=45, show_cleaning_plots=False, use_absolute=True, show_map=False, do_baseline_sub=True
    )
    if clean_df is None: return

    with fits.open(img_file) as hdul:
        ref_map = np.squeeze(hdul[0].data if hdul[0].data is not None else hdul[1].data)
        hdr = hdul[0].header if hdul[0].data is not None else hdul[1].header

    # Reconstruct Reference Grid
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']
    ra_edges = np.linspace(hdr['CRVAL1'] + (0 - hdr['CRPIX1'] + 1)*hdr['CDELT1'],
                           hdr['CRVAL1'] + (nx - hdr['CRPIX1'] + 1)*hdr['CDELT1'], nx+1)
    dec_edges = np.linspace(hdr['CRVAL2'] + (0 - hdr['CRPIX2'] + 1)*hdr['CDELT2'],
                            hdr['CRVAL2'] + (ny - hdr['CRPIX2'] + 1)*hdr['CDELT2'], ny+1)
    
    if ra_edges[0] > ra_edges[-1]: ra_edges = ra_edges[::-1]
    if dec_edges[0] > dec_edges[-1]: dec_edges = dec_edges[::-1]

    user_map, _, _, _ = binned_statistic_2d(
        clean_df['RA_Abs'], clean_df['Dec_Abs'], clean_df['Intensity'], 
        statistic='mean', bins=[ra_edges, dec_edges]
    )
    user_map = user_map.T
    if hdr['CDELT1'] < 0: user_map = np.fliplr(user_map)
    
    # Normalize and Plot (Simplified)
    user_norm = (np.nan_to_num(user_map) - np.nanmin(user_map)) / (np.nanmax(user_map) - np.nanmin(user_map))
    ref_norm = (np.nan_to_num(ref_map) - np.nanmin(ref_map)) / (np.nanmax(ref_map) - np.nanmin(ref_map))
    
    # Shift Correction
    cy_u, cx_u = center_of_mass(user_norm)
    cy_r, cx_r = center_of_mass(ref_norm)
    if not np.isnan(cy_u):
        user_norm = shift(user_norm, shift=(cy_r-cy_u, cx_r-cx_u), mode='constant', cval=0)

    residue = user_norm - ref_norm
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: User
    ax[0].imshow(user_norm, origin='lower', cmap='inferno')
    ax[0].set_title("User (Cleaned)")
    
    # Plot 2: Reference
    ax[1].imshow(ref_norm, origin='lower', cmap='inferno')
    ax[1].set_title("Reference")
    
    # Plot 3: Residuals (with colorbar)
    im3 = ax[2].imshow(residue, origin='lower', cmap='seismic', vmin=-0.5, vmax=0.5)
    ax[2].set_title("Residuals")
    
    # Add the heat scale to the 3rd axis
    cbar = fig.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Difference")
    
    # Add arbitrary X and Y labels to all subplots
    for i in range(3):
        ax[i].set_xlabel("X (pixels)")
        ax[i].set_ylabel("Y (pixels)")
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. CALIBRATION & UTILS
# ==========================================

def calibrate_flux(raw_counts, target_elevation, gain_model_poly, solar_flux_jy=None, peak_gain_counts=None):
    gain = gain_model_poly(target_elevation)
    if gain <= 0: return None
    
    if peak_gain_counts is None:
        peak_gain_counts = np.max(gain_model_poly(np.linspace(0, 90, 100)))
        
    correction = peak_gain_counts / gain
    corrected_counts = raw_counts * correction
    
    flux_jy = (corrected_counts * (solar_flux_jy / peak_gain_counts)) if solar_flux_jy else None
    
    return {'Raw': raw_counts, 'Factor': correction, 'Corrected': corrected_counts, 'Flux_Jy': flux_jy}

def plot_gain_curves(df_results):
    if df_results.empty: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df_results['Elevation'], df_results['Peak_Gain'], 'o-', color='orange')
    if len(df_results) > 3:
        z = np.polyfit(df_results['Elevation'], df_results['Peak_Gain'], 2)
        p = np.poly1d(z)
        x = np.linspace(df_results['Elevation'].min(), df_results['Elevation'].max(), 100)
        ax1.plot(x, p(x), 'k--')
    ax1.set_title("Gain vs Elevation"); ax1.set_xlabel("Elevation"); ax1.set_ylabel("Gain")
    
    ax2.plot(df_results['Elevation'], df_results['Asymmetry'], 's-', color='green')
    ax2.axhline(1.0, color='k', linestyle='--')
    ax2.set_title("Asymmetry vs Elevation"); ax2.set_xlabel("Elevation")
    plt.show()

def check_observatory_coordinates(lat_str, lon_str, height_m):
    """
    Compares the provided GPS coordinates against Google Maps reference.
    """
    print("--- 0. Observatory Coordinate Check ---")
    # 1. GPS Coordinates (Variable)
    gps_loc = EarthLocation(lat=lat_str, lon=lon_str, height=height_m*u.m)
    
    # 2. Google Maps Coordinates (Fixed Reference from text)
    gmaps_loc = EarthLocation(lat=43.93294*u.deg, lon=5.71536*u.deg, height=654.8*u.m)
    
    # Calculate difference in meters
    diff = np.sqrt(
        (gps_loc.x - gmaps_loc.x)**2 + 
        (gps_loc.y - gmaps_loc.y)**2 + 
        (gps_loc.z - gmaps_loc.z)**2
    )
    
    print(f"GPS Position:    {gps_loc.lat.deg:.5f}, {gps_loc.lon.deg:.5f}")
    print(f"Google Position: {gmaps_loc.lat.deg:.5f}, {gmaps_loc.lon.deg:.5f}")
    print(f"Offset distance: {diff.to(u.m).value:.2f} meters")
    
    if diff < 100 * u.m:
        print("-> Positions are COMPATIBLE (within <100m).")
    else:
        print("-> Positions are SIGNIFICANTLY DIFFERENT.")

def verify_timestamps(filename):
    """
    Verifies that the Table JD timestamps match the Header DATE-OBS.
    """
    print(f"--- Time Verification for {filename.split('/')[-1]} ---")
    
    # Check binary table header first, then primary
    header = fits.getheader(filename, 1)
    if 'DATE-OBS' not in header:
        header = fits.getheader(filename, 0)
        
    t = Table.read(filename, hdu=1)
    
    # Get MJD from table (Start)
    t_start = Time(t['JD'][0], format='jd')
    
    print(f"Table Start (UTC): {t_start.iso}")
    print(f"Header DATE-OBS:   {header.get('DATE-OBS', 'NOT FOUND')}")
    
    # Logic check
    header_time = Time(header.get('DATE-OBS'))
    diff_sec = abs((t_start - header_time).sec)
    
    if diff_sec < 1.0:
        print("-> Timestamps MATCH (Difference < 1s).")
    else:
        print(f"-> Timestamps MISMATCH by {diff_sec:.2f} seconds.")

    
    # Add this to Section 4: CALIBRATION & UTILS

def calculate_calibration_factor(tpi_file, source_name, gain_model_poly, obs_gain_db, 
                                 lat_str, lon_str, height_m, freq_mhz=1420):
    """
    Calculates the System Calibration Factor (K_sys) in [Jy / Normalized_Count].
    
    Parameters:
    -----------
    tpi_file : str
        Path to the TPI file of the calibrator (e.g., Cas A).
    source_name : str
        Name of the source ('Cas A', 'Cyg A').
    gain_model_poly : np.poly1d
        Polynomial model of Gain vs Elevation (from Solar study).
    obs_gain_db : float
        The electronic gain setting used during observation (e.g., 30.0).
    freq_mhz : float
        Observation frequency (default 1420 MHz).
        
    Returns:
    --------
    K_sys : float
        The calibration factor. Multiply (Normalized Counts) by this to get Janskys.
    """
    from datetime import datetime
    
    print(f"=== CALIBRATION ROUTINE: {source_name} (Gain={obs_gain_db} dB) ===")

    # 1. Physics Model: Get True Flux
    # --------------------------------
    def get_standard_flux(name, date_obs):
        name = name.lower().replace(" ", "")
        if "cyg" in name:
            return 1590.0 * (freq_mhz / 1400.0)**(-1.25)
        elif "cas" in name:
            # Cas A Decay Model (Baars 1977)
            # Base: ~1080 Jy in 1980, decay 0.77% per year
            year = date_obs.jyear
            years_passed = year - 1980.0
            s_1400_epoch = 1080.0 * ((1 - 0.0077) ** years_passed)
            return s_1400_epoch * (freq_mhz / 1400.0)**(-0.77)
        return None

    # 2. Process File: Measure Raw Peak
    # ---------------------------------
    # We use generate_tod_map to get the clean data and map
    map_data, x_bins, y_bins, clean_df = generate_tod_map(
        tpi_file, lat_str, lon_str, height_m,
        num_pixels=45, show_map=False, use_absolute=False, do_baseline_sub=True
    )
    
    if map_data is None: 
        print("Error: File processing failed."); return None

    # Fit Beam to finding precise peak
    beam = fit_beam_profile(map_data, x_bins, y_bins)
    if not beam:
        print("Error: Beam fit failed (Source too weak?)."); return None
        
    P_raw = beam['Peak_Gain']
    El_obs = clean_df['El_Abs'].mean()
    t_obs = Time(clean_df['Time_Min'][0], format='mjd') # Approximate time reference
    
    # 3. Corrections: Normalize Measurement
    # -------------------------------------
    # A. Elevation (Geometric Efficiency)
    # Normalize to the "Best" elevation of the telescope
    x_test = np.linspace(0, 90, 100)
    max_geo_gain = np.max(gain_model_poly(x_test))
    current_geo_gain = gain_model_poly(El_obs)
    
    geo_factor = max_geo_gain / current_geo_gain
    
    # B. Electronic Gain (Amplifier)
    # Normalize to 0 dB reference
    linear_gain = 10**(obs_gain_db / 10.0)
    
    # P_norm = Counts if observed at Optimal Elevation with 0 dB Gain
    P_normalized = (P_raw * geo_factor) / linear_gain
    
    # 4. Result: Calculate K
    # ----------------------
    S_true = get_standard_flux(source_name, t_obs)
    K_sys = S_true / P_normalized
    
    print(f"   True Flux ({source_name}): {S_true:.1f} Jy")
    print(f"   Meas Peak: {P_raw:.1f} counts (El={El_obs:.1f}°)")
    print(f"   Norm Peak: {P_normalized:.4f} (Corrected)")
    print(f"-> SYSTEM K FACTOR: {K_sys:.4e} Jy/Count")
    
    return K_sys

# Add this to Section 4: CALIBRATION & UTILS
def produce_calibrated_map(filename, system_k, gain_model_poly, obs_gain_db, 
                           lat_str, lon_str, height_m, num_pixels=45,use_absolute=False):
    """
    Generates and plots a 2D map where pixel values are in Janskys (Jy).
    Includes Auto-Zoom (aspect='auto') and Contrast Clipping.
    """
    print(f"--- GENERATING CALIBRATED MAP: {filename.split('/')[-1]} ---")
    
    # 1. Get Raw Data (Counts)
    map_data, x_bins, y_bins, clean_df = generate_tod_map(
        filename, lat_str, lon_str, height_m,
        num_pixels=num_pixels, show_map=False, use_absolute=use_absolute, do_baseline_sub=True
    )
    
    if map_data is None: return None

    # 2. Calculate Global Correction Factor for this file
    mean_el = clean_df['El_Abs'].mean()
    
    # Geometric Correction (Safety check for low elevation)
    # If the model goes below zero (bad fit at edges), clip it to a small number
    try:
        current_geo_gain = gain_model_poly(mean_el)
        if current_geo_gain <= 0: current_geo_gain = 1e-5 # Avoid divide by zero
    except:
        current_geo_gain = 1.0

    x_test = np.linspace(0, 90, 100)
    max_geo_gain = np.max(gain_model_poly(x_test))
    
    geo_factor = max_geo_gain / current_geo_gain
    
    # Electronic Gain Correction
    linear_gain = 10**(obs_gain_db / 10.0)
    
    # Total Multiplier
    calibration_multiplier = system_k * geo_factor / linear_gain
    
    print(f"   Elevation: {mean_el:.1f}° | Gain: {obs_gain_db} dB")
    print(f"   Multiplier: {calibration_multiplier:.2e} (Jy per Raw Count)")
    
    # 3. Calibrate the Map
    calibrated_map_jy = map_data * calibration_multiplier
    
    # 4. Plotting (FIXED)
    img_to_plot = calibrated_map_jy.T # Transpose for plotting
    
    # EXTENT: Use actual data limits
    extent = [x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()]
    
    # CONTRAST: Clip the top/bottom 1% to ignore RFI spikes that wash out the image
    vmin = np.nanpercentile(img_to_plot, 1)
    vmax = np.nanpercentile(img_to_plot, 99)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ASPECT='AUTO' is the key fix for the "squashed" blank plot
    im = ax.imshow(img_to_plot, origin='lower', extent=extent, 
                   cmap='inferno', interpolation='nearest',
                   aspect='auto', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Flux Density (Janskys)', fontsize=12, rotation=270, labelpad=20)
    
    ax.set_title(f"Calibrated Image: {filename.split('/')[-1]}\nObserved at {obs_gain_db}dB")
    ax.set_xlabel("RA Offset (deg)")
    ax.set_ylabel("Dec Offset (deg)")
    ax.invert_xaxis() # Sky convention
    # ax.grid(False) # Optional
    
    plt.show()
    
    return calibrated_map_jy

# ==========================================
# 5. EXECUTION BLOCK (Safe to Import)
# ==========================================
if __name__ == "__main__":
    print("Library Loaded. Define your variables (lat, lon, file_path) to run functions.")
    # Example usage:
    # my_lat, my_lon, my_alt = '43d55.9800m', '5d42.9180m', 654.8
    # generate_tod_map("my_file.fits", my_lat, my_lon, my_alt)



