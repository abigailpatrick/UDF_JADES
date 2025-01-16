# imports

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from numpy import ma
import numpy as np
from astropy.table import Table, Column, MaskedColumn, pprint
from astropy.io import fits
from astropy.wcs import WCS
from scipy import integrate
from astropy.coordinates import SkyCoord


from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import emcee
from scipy.interpolate import interp1d

import os
import requests
from astropy.table import Table
from scipy import interpolate
from mpdaf.obj import Cube
from astropy.coordinates import Angle



def mask_wavelength_region(wavelength, flux, flux_err):
    """
    Apply a mask to select the specified wavelength regions and exclude everything else.
    Wavelength regions based on Calzetti et al. 1994:
    1268-1284, 1309, 1316, 1342-1371, 1407, 1515, 1562-1583, 1677-1740,
    1760-1833, 1866-1890, 1930-1950, 2400-2580
    """
    """  


    # Define the wavelength regions to include, excluding major emission lines
        regions = [
            (1260, 1280),  # Avoiding Lyman-alpha
            (1320, 1340),  # Between O I and C II
            (1400, 1500),  # Avoiding Si IV at 1393-1402 Å
            (1520, 1570),  # Avoiding C IV at 1548-1550 Å
            (1600, 1670),  # Avoiding He II at 1640 Å
            (1700, 1850),  # Between N IV and Si III
            (2100, 2300),  # Avoiding C III at 1908 Å
            (2400, 2580)   # Between Fe II and Mg II
        ]


    # Define the three wavelength regions you want to include
        regions = [
            (1890, 2600),
        ]

        
        (1268, 1284),
            (1309, 1309),
            (1316, 1316),
            (1342, 1371),
     """ 

    # Define the wavelength regions to include
    regions = [
            
            (1407, 1407),
            (1515, 1515),
            (1562, 1583),
            (1677, 1740),
            (1760, 1833),
            (1866, 1890),
            (1930, 1950),
            (2400, 2580)
            ]


    
    # Initialize mask as False
    mask = np.zeros_like(wavelength, dtype=bool)

    # Apply each region to the mask
    for start, end in regions:
        mask |= (wavelength >= start) & (wavelength <= end)

    # Mask the arrays
    return wavelength[mask], flux[mask], flux_err[mask], regions


def log_transform(wavelength, flux):
    """
    Log transform the wavelength and flux data for linear fitting.
    """
    return np.log10(wavelength), np.log10(flux)


def uv_slope_model(log_wavelength, beta, log_FUV_1550):
    """
    UV slope model: log_flux = beta * (log_wavelength - log(1550)) + log_FUV_1550.
    """
    return beta * (log_wavelength - np.log10(1550)) + log_FUV_1550

def initial_fit(log_wavelength, log_flux, beta=-2.0, log_FUV_1550=np.log10(1e-15)):
    """
    Perform an initial fit using lmfit to get the UV slope and log FUV 1550 parameters.
    """
    model = Model(uv_slope_model)
    params = Parameters()
    params.add('beta', value=beta)  # Initial guess for beta
    params.add('log_FUV_1550', value=log_FUV_1550)  # Initial guess for log_FUV_1550

    result = model.fit(log_flux, params, log_wavelength=log_wavelength)
    print("Initial Fit Results:")
    print(result.fit_report())
    return result

def log_probability(param_array, log_wavelength, log_flux, flux_err):
    """
    Objective function for MCMC that calculates the log-probability.
    """
    beta, log_FUV_1550 = param_array
    model_flux = uv_slope_model(log_wavelength, beta, log_FUV_1550)
    sigma2 = flux_err**2  
    return -0.5 * np.sum((log_flux - model_flux)**2 / sigma2 + np.log(sigma2))

def run_mcmc(log_wavelength, log_flux, flux_err, result, nwalkers=50, steps=5000):
    """
    Perform MCMC sampling using emcee with the initial fit results as starting points.
    """
    ndim = 2
    initial_positions = [result.params['beta'].value, result.params['log_FUV_1550'].value]
    p0 = [initial_positions + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(log_wavelength, log_flux, flux_err))
    sampler.run_mcmc(p0, steps, progress=True)
    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    return samples

def get_mcmc_results(samples):
    """
    Calculate the median and credible intervals for beta from MCMC samples.
    """
    beta_samples, log_FUV_1550_samples = samples[:, 0], samples[:, 1]
    beta_median = np.median(beta_samples)
    beta_ci = np.percentile(beta_samples, [16, 84])
    log_FUV_1550_median = np.median(log_FUV_1550_samples)
    log_FUV_1550_ci = np.percentile(log_FUV_1550_samples, [16, 84])

    print(f"β_UV: {beta_median:.2f} with 16-84 percentile range: [{beta_ci[0]:.2f}, {beta_ci[1]:.2f}]")
    print(f"log_FUV_1550: {log_FUV_1550_median:.2e} with 16-84 percentile range: [{log_FUV_1550_ci[0]:.2e}, {log_FUV_1550_ci[1]:.2e}]")
    
    return beta_median, beta_ci, log_FUV_1550_median, log_FUV_1550_ci


def plot_spectrum(wavelength, flux, fitting_regions):
    """
    Plot the original spectrum and shaded regions for the fitting intervals.

    Parameters:
    - wavelength: Array of wavelength values (in Å).
    - flux: Array of flux values.
    - fitting_regions: List of tuples defining the fitting regions as (start, end).
    """
    # Plot the original spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, flux, label="Original Spectrum", color="gray", drawstyle='steps-mid')
    
    # Add shaded regions for the fitting intervals
    for start, end in fitting_regions:
        plt.fill_betweenx(y=[plt.ylim()[0], plt.ylim()[1]], 
                          x1=start, x2=end, 
                          color='green', alpha=0.3, label="Fitting Region" if start == fitting_regions[0][0] else "")
    
    # Labels and plot settings
    plt.xlabel("Rest Frame Wavelength (Å)")
    plt.ylabel(r'$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
    plt.xlim(1000, 3000)
    plt.ylim(-0.1e-20, 1.3e-20) 
    plt.legend()
    plt.show()

# Main analysis pipeline
def analyze_uv_spectrum(wavelength, flux, flux_err, beta=-2.0, log_FUV_1550=np.log10(1e-15)):
    """
    Analyze the UV spectrum of a galaxy to determine the UV slope beta and the FUV flux at 1550 Å.
    Interpolates NaN values in the flux using linear interpolation.

    Parameters:
    - wavelength: Array of wavelength values (in Å).
    - flux: Array of flux values.
    - flux_err: Array of flux error values.
    - beta: Initial guess for the UV slope beta (default: -2.0).
    - log_FUV_1550: Initial guess for the log FUV flux at 1550 Å (default: log(1e-15)).

    Returns:
    - wavelength_masked: Masked wavelength array used for fitting.
    - flux_masked: Masked flux array used for fitting.
    - regions: List of wavelength regions used for fitting.
    - beta_median: Median value of beta from MCMC results.
    - beta_ci: 16-84 percentile range of beta from MCMC results.
    - log_FUV_1550_median: Median value of log_FUV_1550 from MCMC results.
    - log_FUV_1550_ci: 16-84 percentile range of log_FUV_1550 from MCMC results.

    """

    # Mask to fitting region
    wavelength_masked, flux_masked, flux_err_masked, regions = mask_wavelength_region(wavelength, flux, flux_err)
    
    # Log transformation
    log_wavelength, log_flux = log_transform(wavelength_masked, flux_masked)

    print(np.isnan(log_flux).sum(), "NaNs in log_flux")
    #print(np.isnan(log_wavelength).sum(), "NaNs in log_wavelength")

    # Interpolate NaN values in the flux using linear interpolation (or another method if needed)
    if np.any(np.isnan(log_flux)):
        # Identify the indices where flux is NaN
        nan_indices = np.isnan(log_flux)

        # Create an interpolation function based on non-NaN values
        valid_indices = ~nan_indices
        interp_func = interpolate.interp1d(log_wavelength[valid_indices], log_flux[valid_indices], kind='linear', fill_value='extrapolate')

        # Interpolate the NaN values
        log_flux[nan_indices] = interp_func(log_wavelength[nan_indices])
      
    # Initial fit
    result = initial_fit(log_wavelength, log_flux, beta, log_FUV_1550)
    
    # MCMC sampling
    samples = run_mcmc(log_wavelength, log_flux, flux_err_masked, result)
    
    # MCMC results
    beta_median, beta_ci, log_FUV_1550_median, log_FUV_1550_ci  = get_mcmc_results(samples)
    
    # Plotting
    #plot_spectrum(wavelength, flux, regions)

    return wavelength_masked, flux_masked, regions ,beta_median, beta_ci, log_FUV_1550_median, log_FUV_1550_ci


def extrapolate_continuum(wavelength, beta_median, log_FUV_1550_median):
    """
    Extrapolate the full continuum across the wavelength range using the MCMC-derived slope (beta)
    and normalization (log_FUV_1550).
    
    Parameters:
    - wavelength: Array of wavelength values (in Å).
    - beta_median: Median value of beta from MCMC results.
    - log_FUV_1550_median: Median value of log_FUV_1550 from MCMC results.

    Returns:
    - f_cont: The extrapolated continuum flux values across the wavelength range.
    """
    log_wavelength = np.log10(wavelength)
    log_f_cont = beta_median * (log_wavelength - np.log10(1550)) + log_FUV_1550_median
    f_cont = 10**log_f_cont  # Convert back from log scale to linear scale
    return f_cont

def plot_spectrum_with_continuum(wavelength, flux, f_cont, fitting_regions):
    """
    Plot the original spectrum, masked region, and the extrapolated continuum.

    Parameters:
    - wavelength: Array of full wavelength values (in Å).
    - flux: Array of full flux values.
    - wavelength_masked: Masked wavelength array used for fitting.
    - flux_masked: Masked flux array used for fitting.
    - f_cont: Extrapolated continuum flux values across the full wavelength range.
    """

    # Plot the original spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, flux, label="Original Spectrum", color="gray", drawstyle='steps-mid')
    plt.plot(wavelength, f_cont, color="blue", linestyle="--", label=r'$F_{cont}$ extrapolated from $\beta_{UV}$')
    
    # Add shaded regions for the fitting intervals
    for start, end in fitting_regions:
        plt.fill_betweenx(y=[plt.ylim()[0], plt.ylim()[1]], 
                          x1=start, x2=end, 
                          color='green', alpha=0.3, label="Fitting Region" if start == fitting_regions[0][0] else "")
    
    # Labels and plot settings
    plt.xlabel("Rest Frame Wavelength (Å)")
    plt.ylabel(r'$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
    plt.xlim(1000, 3500)
    # find max flux that is not nan or inf
    max_flux = np.nanmax(flux[np.isfinite(flux)])
    plt.ylim(-0.1e-20, max_flux + 0.1e-20)
    plt.legend()


    return plt.show()

def calculate_damping_parameter(wavelength, flux, f_cont, zspec, lambda_LyA_low=1180, lambda_LyA_up=1350):
    """
    Calculate the LyA damping parameter (D_LyA) over the specified wavelength range.

    Parameters:
    - wavelength: Array of OBSERVED wavelengths (in Å).
    - flux: Array of observed flux values.
    - f_cont: Array of extrapolated continuum flux values.
    - zspec: Redshift of the source.
    - lambda_LyA_low: Lower limit of the LyA region in the rest-frame (default: 1180 Å).
    - lambda_LyA_up: Upper limit of the LyA region in the rest-frame (default: 1350 Å).

    Returns:
    - DLyA: The calculated LyA damping parameter.
    """
    # convert lambda_LyA_low and lambda_LyA_up to observed frame
    lambda_LyA_low = lambda_LyA_low * (1 + zspec)
    lambda_LyA_up = lambda_LyA_up * (1 + zspec)
    
    # Mask to Lyα region in the rest frame
    mask = (wavelength >= lambda_LyA_low) & (wavelength <= lambda_LyA_up )
    wavelength_LyA = wavelength[mask]
    flux_LyA = flux[mask]
    f_cont_LyA = f_cont[mask]
    
    # Calculate the (1 - Fλ / Fcont) term over the masked region
    integrand = (1 - flux_LyA / f_cont_LyA) / (1 + zspec)
    
    # Integrate over the wavelength range using Simpson's rule
    DLyA = simps(integrand, wavelength_LyA) 
    
    return DLyA


### downloadin the data


def download_fits_from_DJA(csv_path, folder_name='DJA_Mosaic'):
    """
    Downloads files specified in a CSV file from S3 URLs, with progress tracking.
    
    Parameters:
    - csv_path: str, path to the CSV file containing 'root' and 'file' columns
    - folder_name: str, the name of the folder where files will be saved (default: 'DJA_Mosaic')
    """
    # Read the CSV file
    csv = Table.read(csv_path, format='csv')
    total_files = len(csv)  # Total number of files

    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Loop through each row in the CSV table
    for index, row in enumerate(csv, start=1):
        # Get root and filename from the current row
        root = row['root']
        filename = row['file']
        target_url = f"https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{filename}"

        # Define the file path
        file_path = os.path.join(folder_name, filename)

        # Display progress
        print(f"Processing file {index}/{total_files}: {filename}")

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"{filename} already exists in {folder_name}. No need to download.")
        else:
            # File doesn't exist, download it
            try:
                response = requests.get(target_url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                    print(f"{filename} downloaded to {folder_name}.")
                else:
                    print(f"Failed to download {filename}. Status code: {response.status_code}")
            except Exception as e:
                print(f"An error occurred while downloading {filename}: {str(e)}")

    print(f"Download complete. {total_files} files processed.")
    
    return None


def filter_incomplete_spectra(csv_table, folder_name='DJA_Mosaic', min_wavelength=1.0, max_wavelength=4.5, target_wavelength= 1450, SNR_threshold=3.0):
    """
    Filters out entries with incomplete spectra for fitting and returns a list of filenames to exclude.
    
    Parameters:
    - csv_table: astropy Table, the table of sources and files
    - folder_name: str, the name of the folder containing downloaded FITS files
    - min_wavelength: int, minimum acceptable wavelength (in um)
    - max_wavelength: int, maximum acceptable wavelength (in um)
    - target_wavelength: int, the target wavelength for SNR calculation (in Å), default: 1450 Å
    - SNR_threshold: float, the minimum SNR required for a spectrum to be included (default: 3.0)
    
    Returns:
    - filtered_table: astropy Table, the filtered table with incomplete spectra removed
    """
    excluded_filenames = []

    # print the conditions for exclusion
    print(f"Excluding spectra with wavelength outside the range {min_wavelength} - {max_wavelength} Å")

    for row in csv_table:
        filename = row['file']
        file_path = os.path.join(folder_name, filename)
        z = row['z']
        
        print('---------------------------------------------------')
        print(file_path)
        try:
            with fits.open(file_path) as hdul:
                

                # Convert wavelength from microns to angstroms
                wavelengths_microns = hdul[1].data['wave'] * u.micron
                wavelengths_angstroms = wavelengths_microns.to(u.AA)
                
                # Convert flux to erg/s/cm^2/Å
                flux_microjy = hdul[1].data['flux'] * u.microjansky
                flux_err_microjy = hdul[1].data['err'] * u.microjansky
                
                flux_erg_per_s_cm2_A = flux_microjy.to(u.erg / (u.s * u.cm**2 * u.AA), equivalencies=u.spectral_density(wavelengths_angstroms))
                flux_err_erg_per_s_cm2_A = flux_err_microjy.to(u.erg / (u.s * u.cm**2 * u.AA), equivalencies=u.spectral_density(wavelengths_angstroms))

                # Add rest-frame wavelength
                rest_wave_angstroms = wavelengths_angstroms / (1 + z)

                
                wavelengths = wavelengths_microns.value
                # print the spectral range
                print(f"{filename} spectral range: {wavelengths.min()} - {wavelengths.max()}")

                
                # Check if the spectrum covers the desired range
                if wavelengths.min() > min_wavelength or wavelengths.max() < max_wavelength:
                    excluded_filenames.append(filename)
                    print(f"{filename} excluded due to incomplete spectral coverage.")
                    continue

                
                # Find the index of the closest wavelength point
                index = np.abs(rest_wave_angstroms.value - target_wavelength).argmin()

                # Calculate SNR at the closest point
                snr_at_target = flux_erg_per_s_cm2_A[index] / flux_err_erg_per_s_cm2_A[index]

                print(f"SNR at {rest_wave_angstroms[index]:.2f} Å: {snr_at_target:.2f}")

                # Exclude spectra with SNR < Threshold or if SNR is NaN
                if np.isnan(snr_at_target) or snr_at_target < SNR_threshold:
                    excluded_filenames.append(filename)
                    print(f"{filename} excluded due to SNR < {SNR_threshold} or NaN at {target_wavelength} Å.")
                    continue
                
                # Mask to fitting region and apply log transform to give NAN warnings
                wavelength_masked, flux_masked, flux_err_masked, regions = mask_wavelength_region(rest_wave_angstroms.value, flux_erg_per_s_cm2_A.value, flux_err_erg_per_s_cm2_A.value)
                log_wavelength, log_flux = log_transform(wavelength_masked, flux_masked)

                # Check for NaN values in log_flux, raise warning if found
                if np.isnan(log_flux).sum() > 0:
                    print("WARNING:", np.isnan(log_flux).sum(), "NaNs in log_flux. Increase SNR threshold before fitting.")
                else:
                    print(f"{filename} passed all checks.")
                    continue

                
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            excluded_filenames.append(filename)



        # Convert the list to a set
    excluded_filenames_set = set(excluded_filenames)

    # Use np.isin() to filter the table
    filtered_table = csv_table[~np.isin(csv_table['file'], list(excluded_filenames_set))]
    
    print (f"Excluded {len(excluded_filenames_set)} spectra.")
    print (f"Remaining {len(filtered_table)} spectra.")

    return filtered_table


def process_spectra(table, folder_name):

    # Constants
    c = 3e18  # Speed of light in Å/s
    # Output arrays
    file_names = []
    redshift_values = []
    damping_parameters = []
    b_uv_values = []

    # Loop through each row and process
    for row in table:
        filename = row['file']
        filepath = os.path.join(folder_name, filename)
        z = row['z']

        print(f"Processing {filepath} with redshift z = {z}")
        # Load the data
        tab = Table.read(filepath, format='fits')
        # Convert wavelength from microns to angstroms
        wavelengths_microns = tab['wave']
        wavelengths_angstroms = wavelengths_microns.to(u.AA)
        
        # Convert flux to erg/s/cm^2/Å
        flux_microjy = tab['flux']
        flux_err_microjy = tab['err']
        
        flux_erg_per_s_cm2_A = flux_microjy.to(u.erg / (u.s * u.cm**2 * u.AA), equivalencies=u.spectral_density(wavelengths_angstroms))
        flux_err_erg_per_s_cm2_A = flux_err_microjy.to(u.erg / (u.s * u.cm**2 * u.AA), equivalencies=u.spectral_density(wavelengths_angstroms))

        # Add rest-frame wavelength
        rest_wave_angstroms = wavelengths_angstroms / (1 + z)
        
        # Perform UV spectrum analysis
        wavelength_masked, flux_masked, regions, beta_median, beta_ci, log_FUV_1550_median, log_FUV_1550_ci = analyze_uv_spectrum(
            rest_wave_angstroms.value, 
            flux_erg_per_s_cm2_A.value, 
            flux_err_erg_per_s_cm2_A.value,
            beta=-1.5, 
            log_FUV_1550=np.log10(1e-15)
        )

        # Extrapolate continuum and plot
        f_cont = extrapolate_continuum(rest_wave_angstroms.value, beta_median, log_FUV_1550_median)
        plot_spectrum_with_continuum(rest_wave_angstroms.value, flux_erg_per_s_cm2_A.value, f_cont, regions)
        
        # Calculate D_LyA - note this has OBSEREVED wavelengths
        DLyA = calculate_damping_parameter(wavelengths_angstroms.value, flux_erg_per_s_cm2_A.value, f_cont, z)
        print(f"{filepath} | LyA Damping Parameter (D_LyA): {DLyA:.3f}")

        # Append results to the output arrays
        file_names.append(filename)
        redshift_values.append(z)
        damping_parameters.append(DLyA)
        b_uv_values.append(beta_median)

    
    # Create an Astropy Table from the results
    results_table = Table([file_names, redshift_values,b_uv_values, damping_parameters], 
                        names=('File Name', 'z_spec', 'B_UV','LyA Damping Parameter'))

    # Save the results table to a file
    results_table.write('spectrum_analysis_results.fits', format='fits', overwrite=True)

    # Display or save the results table
    print("\nSummary of Results as Table:")
    results_table.pprint()

    return results_table

def get_cube_ra_dec_bounds(cube):
    # Get spatial dimensions
    ny, nx = cube.shape[1], cube.shape[2]  
    wcs = cube.wcs

    # Define the four corners in pixel coordinates
    corners = [(0, 0), (nx-1, 0), (0, ny-1), (nx-1, ny-1)]

    # Convert each corner to RA and Dec
    ra_dec_corners = wcs.pix2sky(corners, unit='deg')
    
    # Separate RA and Dec values
    ras = [coord[1] for coord in ra_dec_corners]
    decs = [coord[0] for coord in ra_dec_corners]
    ra_min, ra_max = min(ras), max(ras)
    dec_min, dec_max = min(decs), max(decs)

    dec1, ra1 = ra_dec_corners[2]
    dec2, ra2 = ra_dec_corners[3]
    dec3, ra3 = ra_dec_corners[1]
    dec4, ra4 = ra_dec_corners[0]

    vertices_ra = [ra1, ra2, ra3, ra4]
    vertices_dec = [dec1, dec2, dec3, dec4] 


    return ra_min, ra_max, dec_min, dec_max, vertices_ra, vertices_dec

def filter_sources_in_moc(csv_data, moc, z_min=4.0, z_max=6.6):
    """
    Filters sources based on their location inside the MOC and redshift range.

    Parameters:
        csv_data (Table): Input CSV data with RA, Dec, and z columns.
        moc (MOC): MOC object for spatial filtering.
        z_min (float): Minimum redshift for sources (default: 4.0).
        z_max (float): Maximum redshift for sources (default: 6.6).

    Returns:
        Table: Filtered sources satisfying all criteria.
    """
    # Extract RA, Dec, and redshift from the input data
    ra_sources = csv_data['ra'] * u.deg
    dec_sources = csv_data['dec'] * u.deg
    redshift = csv_data['z']

    # Check which sources are inside the MOC
    inside_moc_mask = moc.contains_lonlat(lon=ra_sources, lat=dec_sources)

    # Filter sources inside the MOC
    sources_inside_moc = csv_data[inside_moc_mask]

    # Apply redshift range filtering
    #filtered_sources = sources_inside_moc
    redshift_mask = (sources_inside_moc['z'] > z_min) & (sources_inside_moc['z'] < z_max)
    filtered_sources = sources_inside_moc[redshift_mask]

    print (f"Number of sources in CSV: {len(csv_data)}")    
    print(f"Number of sources inside MOC: {inside_moc_mask.sum()}")
    print(f"Filtered sources RA: {filtered_sources['ra']}")
    print(f"Filtered sources Dec: {filtered_sources['dec']}")

   
    return filtered_sources

def visualize_moc_and_sources(moc, csv_data, filtered_sources):
    """
    Visualizes the MOC and the sources.

    Parameters:
        moc (MOC): MOC object representing the coverage area.
        csv_data (Table): Original source data with RA and Dec columns.
        filtered_sources (Table): Sources filtered by the MOC.
    """
    # Define WCS for the plot
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [53.155, -27.795]  # Central RA, Dec values of the region
    wcs.wcs.crpix = [150, 150]  # Reference pixel
    wcs.wcs.cdelt = [-0.0003, 0.0003]  # Pixel scale in degrees

    # Create the plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=wcs)

    # Plot the MOC
    moc.fill(ax=ax, wcs=wcs, alpha=0.2, label="MOC", color="red")

    # Plot all sources
    ax.scatter(
        csv_data["ra"], csv_data["dec"],
        s=5, alpha=0.2, color="blue", transform=ax.get_transform("world"), label="All Sources"
    )

    # Plot filtered sources
    ax.scatter(
        filtered_sources["ra"], filtered_sources["dec"],
        s=5, color="green", transform=ax.get_transform("world"), label="Filtered Sources"
    )

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.legend()
    plt.show()
