================================================================================
RADIO TPI ANALYSIS LIBRARY - README
================================================================================

OVERVIEW:
This library is a comprehensive Python pipeline for processing and calibrating 
radio telescope Time-Ordered Data (TOD). It is optimized for single-dish radio 
telescopes (e.g., GRAD-300) to handle signal cleaning, 2D mapping, beam 
characterization, and absolute flux calibration.

DEPENDENCIES:
- numpy, pandas, matplotlib
- scipy (stats, signal, optimize, ndimage)
- astropy (io.fits, table, time, units, coordinates)

================================================================================
CORE MODULES & FUNCTIONS
================================================================================

1. DATA CLEANING & TRAJECTORY ANALYSIS
   - analyze_trajectory_and_clean(): 
     The master cleaning function. 
     * Removes kinematic anomalies (speed spikes/turnarounds).
     * Trims "return-to-stow" segments.
     * Robustly handles truncated FITS files and empty data streams.
     * Visualizes the raw vs. clean scan path.

   - subtract_baseline(): 
     Removes instrumental drift (1/f noise) and background sky brightness by 
     fitting a polynomial to the off-source regions of the scan.

2. MAPPING ENGINE
   - generate_tod_map(): 
     The primary pipeline. Orchestrates cleaning, baseline subtraction, and 
     reprojects TOD into 2D maps. Supports:
     * Relative Coordinates (Offsets from Source Center).
     * Absolute Coordinates (RA/Dec J2000).

   - produce_calibrated_map():
     The scientific imaging tool. Converts a raw TPI map (Counts) into a 
     Flux Density map (Janskys).
     * Automatically corrects for Elevation Gain Loss (Dish deformation).
     * Normalizes for Electronic Gain settings (dB).
     * Applies the System K-Factor.
     * Features "Auto-Zoom" and contrast clipping for strip scans.

3. BEAM CHARACTERIZATION
   - fit_beam_profile(): 
     Fits a 2D elliptical Gaussian to the map to measure:
     * Peak Gain (Raw Counts).
     * Beam Width (FWHM) in X and Y.
     * Beam Asymmetry/Ellipticity.

   - analyze_bbc_dependence(): 
     Dissects the observation by frequency channel (BBCs) to verify the 
     diffraction limit (Theta ~ 1.22 * Lambda / D) and check bandpass shape.

4. BATCH PROCESSING & CALIBRATION
   - run_sun_elevation_batch(): 
     Automates analysis across multiple Solar observations to derive the 
     "Gain Curve" (Efficiency vs. Elevation) and Beam Width vs. Elevation.

   - calculate_calibration_factor(): 
     Derives the "System Constant" (K_sys) using a standard calibrator 
     (Cas A, Cyg A).
     * Includes time-dependent flux models (e.g., Baars 1977 decay for Cas A).
     * Returns K in units of [Jy / Count @ 0dB Gain].

   - plot_galactic_trace():
     Visualizes the telescope's scan path in Galactic Coordinates (l, b) 
     to check alignment with the Milky Way plane.

================================================================================
HOW TO USE (EXAMPLE WORKFLOW)
================================================================================

# 1. CHARACTERIZE THE TELESCOPE (Sun Study)
# ---------------------------------------------------------
# Process all sun files to find how gain changes with elevation
df_study = astro.run_sun_elevation_batch("data/*SUN*.fits", lat, lon, alt)
astro.plot_gain_curves(df_study)

# Create the Gain Model (Parabola)
z = np.polyfit(df_study['Elevation'], df_study['Peak_Gain'], 2)
solar_gain_model = np.poly1d(z)


# 2. CALIBRATE THE SYSTEM (Using Cassiopeia A)
# ---------------------------------------------------------
# Calculate the conversion factor (Jy/Count)
SYSTEM_K = astro.calculate_calibration_factor(
    tpi_file="data/cas_a_scan.fits",
    source_name="Cas A",
    gain_model_poly=solar_gain_model,
    obs_gain_db=30.0,  # Gain used for this file
    lat_str=lat, lon_str=lon, height_m=alt
)
print(f"System K: {SYSTEM_K:.2e} Jy/Count")


# 3. PRODUCE SCIENTIFIC IMAGES (Any Source)
# ---------------------------------------------------------
# Map a science target (e.g., Tau A or the Sun) in Janskys
calibrated_map = astro.produce_calibrated_map(
    filename="data/science_target.fits",
    system_k=SYSTEM_K,
    gain_model_poly=solar_gain_model,
    obs_gain_db=28.0,       # Gain used for this specific file
    lat_str=lat, lon_str=lon, height_m=alt,
    use_absolute=False      # False = Relative (Centered), True = RA/Dec
)

print(f"Peak Flux: {np.nanmax(calibrated_map):.2f} Jy")

================================================================================
SCIENTIFIC NOTES:
- Flux Calibration: Uses the Baars et al. (1977) scale with secular decay 
  correction for Cas A (-0.77%/yr).
- Gain Normalization: All flux measurements are normalized to the telescope's 
  "Peak Efficiency" elevation (derived from the Solar Model) and 0dB gain.
- Coordinate Systems: 
  * 'Relative' maps are best for point sources (Sun, Moon, Calibrators).
  * 'Absolute' maps are best for drift scans or identifying sky position.
================================================================================