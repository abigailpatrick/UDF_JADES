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

from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from scipy.ndimage import gaussian_filter1d
from astropy.table import Table

def extract_source(filepath, ra, dec, redshift, rest_wavelength_range, spatial_extent, NIRSpec_ID, smoothing_sigma=3):
    """
    Extract a 2D stacked image and a 1D spectrum for a given source.

    Parameters:
    - filepath: Path to the FITS file (MUSE data cube)
    - ra: Right Ascension of the source (in degrees)
    - dec: Declination of the source (in degrees)
    - redshift: Redshift of the source
    - rest_wavelength_range: Tuple of (lambda_min, lambda_max) in Angstroms (rest frame)
    - spatial_extent: Half-size of the box in arcseconds around the source

    Returns:
    - stacked_image: 2D stacked image (summed over wavelengths)
    - spectrum_data: 1D flux spectrum of the source
    - spectrum_wave_rest: Rest-frame wavelength array for the spectrum
    """

    # Load the FITS file and extract WCS using astropy
    with fits.open(filepath) as hdul:
        wcs = AstropyWCS(hdul[1].header)

    # Convert RA/Dec/Wavelength to pixel coordinates
    wavelength = 4750  # Example wavelength (CRVAL3 from header or user-defined)
    pixel_coords = wcs.wcs_world2pix([[ra, dec, wavelength]], 0)

    # Extract pixel coordinates
    pixel_x, pixel_y, _ = pixel_coords[0]

    # Convert pixel coordinates to RA/Dec using pixel_x, pixel_y, and wavelength
    world_coords = wcs.wcs_pix2world(np.array([[pixel_x, pixel_y, wavelength]]), 0)

    # Load the cube using mpdaf
    cube = Cube(filepath)

    # Convert spatial extent to pixels using pixel scale from cube 
    spatial_extent_pix = int(spatial_extent / cube.wcs.get_step(unit='arcsec')[1])

    x_center, y_center = int(pixel_x), int(pixel_y)
    print(f"RA/Dec at center of image (wcs): {wcs.wcs_pix2world([[x_center, y_center, wavelength]], 0)[0][:2]}")
    print(f"Rest wavelength range: {rest_wavelength_range}")
    print(f"Redshift: {redshift}")

    print("Cube wavelength range:", cube.wave.coord().min(), cube.wave.coord().max())
   
    # Calculate the observed wavelength range
    lambda_min_obs = rest_wavelength_range[0] * (1 + redshift)
    lambda_max_obs = rest_wavelength_range[1] * (1 + redshift)

    print("Requested wavelength range:", lambda_min_obs, lambda_max_obs)

    # Find the indices corresponding to the wavelength range in the cube
    wave_start = np.argmin(np.abs(cube.wave.coord() - lambda_min_obs))
    wave_end = np.argmin(np.abs(cube.wave.coord() - lambda_max_obs))

    # Extract the subcube around the source in pixel coordinates
    x_slice = slice(max(0, x_center - spatial_extent_pix), min(cube.shape[2], x_center + spatial_extent_pix))
    y_slice = slice(max(0, y_center - spatial_extent_pix), min(cube.shape[1], y_center + spatial_extent_pix))

    subcube = cube[wave_start:wave_end, y_slice, x_slice]

    # Stack the wavelengths to create a 2D image
    stacked_image = subcube.data.sum(axis=0)
    spectrum_data = subcube.data.mean(axis=(1, 2))

    # Convert wavelength to rest frame
    spectrum_wave_rest = cube.wave.coord()[wave_start:wave_end] / (1 + redshift)
    
    return stacked_image, spectrum_data, spectrum_wave_rest


def process_sources(mosaic_file, filepath, smoothing_sigma=3):
    """
    Process sources from a mosaic catalog, extract 2D images and 1D spectra for each source.

    Parameters:
    - mosaic_file: Path to the mosaic catalog (CSV file)
    - filepath: Path to the MUSE data cube FITS file
    - smoothing_sigma: Standard deviation for Gaussian smoothing of the spectrum

    Returns:
    - stacked_images: Dictionary of stacked images
    - spectrum_data_dict: Dictionary of 1D spectrum data
    - spectrum_wave_rest_dict: Dictionary of rest-frame wavelength data
    """
    # Load the catalog
    mosaic = Table.read(mosaic_file, format='csv')

    # Initialize dictionaries to store the outputs
    stacked_images = {}
    spectrum_data_dict = {}
    spectrum_wave_rest_dict = {}

    # Process each source
    for i, source in enumerate(mosaic):
        # Load source details from the mosaic catalog
        NIRSpec_ID = source['uid']
        ra_center = source['ra']
        dec_center = source['dec']
        redshift = source['z']
        rest_wavelength_range = (1200, 1230)  # Example rest wavelength range in Angstroms
        spatial_extent = 3  # Spatial extent in arcseconds

        print(f"RA/Dec at center of image (input): {ra_center:.3f}, {dec_center:.3f}")
       

        # Extract the source for each source in source list
        stacked_image, spectrum_data, spectrum_wave_rest = extract_source(
            filepath, ra_center, dec_center, redshift, rest_wavelength_range, spatial_extent, NIRSpec_ID, smoothing_sigma
        )

        # Store the results in the dictionaries with unique keys
        stacked_images[f'image_{i+1}'] = stacked_image
        spectrum_data_dict[f'data_{i+1}'] = spectrum_data
        spectrum_wave_rest_dict[f'wave_rest_{i+1}'] = spectrum_wave_rest

        # Plot the results
        plot_results(stacked_image, spectrum_data, spectrum_wave_rest, NIRSpec_ID, redshift, smoothing_sigma)

    return stacked_images, spectrum_data_dict, spectrum_wave_rest_dict


def plot_results(stacked_image, spectrum_data, spectrum_wave_rest, NIRSpec_ID, redshift, smoothing_sigma):
    """
    Plot the stacked image and spectrum for a given source.

    Parameters:
    - stacked_image: 2D image of the source
    - spectrum_data: 1D flux spectrum of the source
    - spectrum_wave_rest: Rest-frame wavelength array for the spectrum
    - NIRSpec_ID: ID of the source
    - redshift: Redshift of the source
    - smoothing_sigma: Standard deviation for Gaussian smoothing of the spectrum
    """
    # Calculate the RMS of the stacked image
    rms = np.sqrt(np.mean(stacked_image**2))

    # Set the color scale to 3 times the RMS value
    vmin = -1 * rms
    vmax = 4 * rms

    # Print basic statistics of the stacked image
    print("Min value:", np.min(stacked_image))
    print("Max value:", np.max(stacked_image))
    print("Mean value:", np.mean(stacked_image))
    print("RMS:", rms)

    # Plot the stacked 2D image with RMS scaling
    plt.figure(figsize=(6, 6))
    im = plt.imshow(stacked_image, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, label=r'Flux (10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', shrink=0.7)
    plt.title(f'Stacked 2D Image of Source {NIRSpec_ID} at RA={spectrum_wave_rest[0]:.3f}, Dec={spectrum_wave_rest[0]:.3f}', fontsize=10, pad=12)
    plt.xlabel('Right Ascension (deg)')
    plt.ylabel('Declination (deg)')
    plt.tight_layout()
    plt.show()

    # Apply Gaussian smoothing to the spectrum
    smoothed_flux = gaussian_filter1d(spectrum_data, sigma=smoothing_sigma)

    # Plot the 1D spectrum
    plt.figure(figsize=(12, 4))
    plt.plot(spectrum_wave_rest, spectrum_data, color='green', alpha=0.4, label='Source Spectrum')
    plt.plot(spectrum_wave_rest, smoothed_flux, color='black', alpha=0.8, label=f'Smoothed Spectrum (σ={smoothing_sigma})')
    plt.xlim([spectrum_wave_rest[0], spectrum_wave_rest[-1]])  # Ensure the spectrum touches both ends
    plt.xlabel(r'Restframe Wavelength (Å)', fontsize=12)
    plt.ylabel(r'Flux (10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=12)
    plt.title(f'Spectrum of Source {NIRSpec_ID} at z_spec={redshift:.3f}', fontsize=12, pad=10)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def stack_2d_images(image_dict):
    """
    Stack multiple 2D images to create a single stacked image.

    Parameters:
    - image_dict: Dictionary of 2D images (e.g., output from process_sources)

    Returns:
    - stacked_image: Final stacked 2D image
    """
    images = list(image_dict.values())
    images = np.array(images)  # Shape: (n_images, height, width)

    # Calculate the weighted average across all images
    stacked_image = np.nanmedian(images, axis=0)  # Use nanmean to ignore NaN values if any

    return stacked_image

def stack_1d_spectra(spectrum_dict, wave_dict):
    """

    Stack multiple 1D spectra to create a single stacked spectrum.

    Parameters:
    - spectrum_dict: Dictionary of 1D spectra (e.g., output from process_sources)
    - wave_dict: Dictionary of wavelength arrays corresponding to each spectrum

    Returns:
    - stacked_flux: Final stacked 1D spectrum
    - common_wave: Common wavelength grid
    """
    
    # Find the overall wavelength range
    min_wave = min([np.min(wave) for wave in wave_dict.values()])
    max_wave = max([np.max(wave) for wave in wave_dict.values()])

    # Define a common wavelength grid
    common_wave = np.linspace(min_wave, max_wave, 5000)

    # Stack spectra
    stacked_flux = []
    for key in spectrum_dict:
        # Derive the corresponding key in wave_dict
        wave_key = key.replace('data_', 'wave_rest_')
        if wave_key not in wave_dict:
            print(f"Missing wavelength data for {key}, skipping.")
            continue

        flux = spectrum_dict[key]
        wave = wave_dict[wave_key]

        # Interpolate flux to the common wavelength grid
        interpolated_flux = np.interp(common_wave, wave, flux, left=np.nan, right=np.nan)
        stacked_flux.append(interpolated_flux)

    # Compute the average flux while ignoring NaNs
    stacked_flux = np.nanmedian(stacked_flux, axis=0)

    return stacked_flux, common_wave
