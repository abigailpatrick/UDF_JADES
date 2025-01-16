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


def mask_wavelength_region(wavelength, flux, flux_err):
    """
    Apply a mask to select the specified wavelength regions and exclude everything else.
    Wavelength regions based on Calzetti et al. 1994:
    1268-1284, 1309, 1316, 1342-1371, 1407, 1515, 1562-1583, 1677-1740,
    1760-1833, 1866-1890, 1930-1950, 2400-2580

    Parameters:
    - wavelength: Array of wavelength values.
    - flux: Array of flux values.
    - flux_err: Array of flux error values.

    Returns:
    - wavelength_masked: Masked wavelength values.
    - flux_masked: Masked flux values.
    - flux_err_masked: Masked flux error values.

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

     """ 

    # Define the wavelength regions to include
    regions = [
            (1268, 1284),
            (1309, 1309),
            (1316, 1316),
            (1342, 1371),
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

def interpolate_flux_errors(wavelength_masked, flux_masked, flux_err_masked):
    """
    Interpolate the flux and flux errors over the masked wavelength regions.

    Parameters:
    - wavelength_masked: Array of masked wavelength values.
    - flux_masked: Array of masked flux values.
    - flux_err_masked: Array of masked flux error values.

    Returns:
    - wavelength_masked: Array of masked wavelength values.
    - interpolated_flux: Interpolated flux values over the masked regions.
    - interpolated_flux_err: Interpolated flux error values over the masked regions.
    """
    # Linear interpolation over masked wavelength regions
    interp_flux = interp1d(wavelength_masked, flux_masked, kind='linear', fill_value="extrapolate")
    interp_flux_err = interp1d(wavelength_masked, flux_err_masked, kind='linear', fill_value="extrapolate")
    
    # Interpolated flux and error for all masked wavelengths
    interpolated_flux = interp_flux(wavelength_masked)
    interpolated_flux_err = interp_flux_err(wavelength_masked)
    
    return wavelength_masked, interpolated_flux, interpolated_flux_err

def weight_interpolated_errors(wavelength_masked, flux_interpolated, flux_err_interpolated):
    """
    Calculate the weights for the interpolated flux and flux errors.
    Weight the flux based on data density in each masked region.

    Parameters:
    - wavelength_masked: Array of masked wavelength values.
    - flux_interpolated: Array of interpolated flux values.
    - flux_err_interpolated: Array of interpolated flux error values.

    Returns:
    - wavelength_masked: Array of masked wavelength values.
    - weighted_flux: Weighted flux values.
    - weighted_flux_err: Weighted flux error values.
    """
    weights = 1 / (flux_err_interpolated + 1e-6)  # Prevent division by zero
    weighted_flux = flux_interpolated * weights
    weighted_flux_err = flux_err_interpolated / weights
    
    return wavelength_masked, weighted_flux, weighted_flux_err



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

    Parameters:
    - log_wavelength: Array of log wavelength values.
    - log_flux: Array of log flux values.
    - beta: Initial guess for the UV slope (default: -2.0).
    - log_FUV_1550: Initial guess for the log FUV 1550 flux (default: log10(1e-15)).


    Returns:
    - result: The result of the initial fit.
    
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

    Parameters:
    - param_array: Array of parameters (beta, log_FUV_1550).
    - log_wavelength: Array of log wavelength values.
    - log_flux: Array of log flux values.
    - flux_err: Array of flux error values.

    Returns:
    - log_probability: The log-probability of the model given the data.

    """
    beta, log_FUV_1550 = param_array
    model_flux = uv_slope_model(log_wavelength, beta, log_FUV_1550)
    sigma2 = flux_err**2 + model_flux**2 * np.exp(2 * log_FUV_1550)
    return -0.5 * np.sum((log_flux - model_flux)**2 / sigma2 + np.log(sigma2))

def run_mcmc(log_wavelength, log_flux, flux_err, result, nwalkers=50, steps=5000):
    """
    Perform MCMC sampling using emcee with the initial fit results as starting points.

    Parameters:
    - log_wavelength: Array of log wavelength values.
    - log_flux: Array of log flux values.
    - flux_err: Array of flux error values.
    - result: The result of the initial fit.
    - nwalkers: Number of walkers for the MCMC sampler (default: 50).
    - steps: Number of steps for the MCMC sampler (default: 5000).

    Returns:
    - samples: The MCMC samples for beta and log_FUV_1550.

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

    Parameters:
    - samples: Array of MCMC samples.

    Returns:
    - beta_median: Median value of beta.
    - beta_ci: 16-84 percentile range of beta.
    - log_FUV_1550_median: Median value of log_FUV_1550.
    - log_FUV_1550_ci: 16-84 percentile range of log_FUV_1550.

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
    Analyze the UV spectrum to determine the UV slope (beta) and the FUV 1550 flux.

    Parameters:
    - wavelength: Array of wavelength values (in Å).
    - flux: Array of flux values.
    - flux_err: Array of flux error values.
    - beta: Initial guess for the UV slope (default: -2.0).
    - log_FUV_1550: Initial guess for the log FUV 1550 flux (default: log10(1e-15)).

    Returns:
    - wavelength_masked: Masked wavelength values used for fitting.
    - flux_masked: Masked flux values used for fitting.
    - beta_median: Median value of beta from MCMC results.
    - beta_ci: 16-84 percentile range of beta from MCMC results.
    - log_FUV_1550_median: Median value of log_FUV_1550 from MCMC results.
    - log_FUV_1550_ci: 16-84 percentile range of log_FUV_1550 from MCMC results.
    """
    # Mask to fitting region
    wavelength_masked, flux_masked, flux_err_masked, regions  = mask_wavelength_region(wavelength, flux, flux_err)
    
    # Interpolation within masked regions
    wavelength_masked, interpolated_flux, interpolated_flux_err = interpolate_flux_errors(wavelength_masked, flux_masked, flux_err_masked)
    
    # Error weighting in masked regions
    wavelength_masked, weighted_flux, weighted_flux_err = weight_interpolated_errors(wavelength_masked, interpolated_flux, interpolated_flux_err)
    
    # Log transformation
    log_wavelength, log_flux = log_transform(wavelength_masked, weighted_flux)
    
    # Initial fit
    result = initial_fit(log_wavelength, log_flux, beta, log_FUV_1550)
    
    # MCMC sampling
    samples = run_mcmc(log_wavelength, log_flux, weighted_flux_err, result)
    
    # MCMC results
    beta_median, beta_ci, log_FUV_1550_median, log_FUV_1550_ci = get_mcmc_results(samples)
    
    # Plotting
    plot_spectrum(wavelength, flux, regions)

    return wavelength_masked, regions ,weighted_flux, beta_median, beta_ci, log_FUV_1550_median, log_FUV_1550_ci



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

def plot_spectrum_with_continuum(wavelength, flux, wavelength_masked, weighted_flux, f_cont, fitting_regions):
    """
    Plot the original spectrum, masked region, and the extrapolated continuum.

    Parameters:
    - wavelength: Array of full wavelength values (in Å).
    - flux: Array of full flux values.
    - wavelength_masked: Masked wavelength array used for fitting.
    - flux_masked: Masked flux array used for fitting.
    - f_cont: Extrapolated continuum flux values across the full wavelength range.
    """

    # Normalize flux and f_cont to the same range
    flux_normalized = normalize_to_range(flux)
    f_cont_normalized = normalize_to_range(f_cont) 

    # Plot the original spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, flux_normalized, label="Original Spectrum", color="gray", drawstyle='steps-mid')
    plt.plot(wavelength, f_cont_normalized, color="blue", linestyle="--", label=r'$F_{cont}$ extrapolated from $\beta_{UV}$')
    
    # Add shaded regions for the fitting intervals
    for start, end in fitting_regions:
        plt.fill_betweenx(y=[plt.ylim()[0], plt.ylim()[1]], 
                          x1=start, x2=end, 
                          color='green', alpha=0.3, label="Fitting Region" if start == fitting_regions[0][0] else "")
    
    # Labels and plot settings
    plt.xlabel("Rest Frame Wavelength (Å)")
    plt.ylabel(r'Normalised $F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
    plt.xlim(1000, 3000)
    plt.ylim(-0.1, 1.3) 
    plt.legend()


    return plt.show()


def normalize_to_range(values, new_min=0, new_max=1):
    """
    Normalize the values to a specified range [new_min, new_max].

    Parameters:
    - values: Array of values to normalize.
    - new_min: Minimum value of the new range.
    - new_max: Maximum value of the new range.

    Returns:
    - normalized_values: The normalized values within the specified range.
    """
    min_val = np.nanmin(values)  # Minimum value in the values array
    max_val = np.nanmax(values)  # Maximum value in the values array

    # Avoid division by zero
    if max_val == min_val:
        return np.full_like(values, new_min)

    # Normalize to range
    normalized_values = (values - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return normalized_values


def calculate_damping_parameter(wavelength, flux, f_cont, zspec, lambda_LyA_low=1180, lambda_LyA_up=1350):
    """
    Calculate the LyA damping parameter (D_LyA) over the specified wavelength range.

    Parameters:
    - wavelength: Array of observed wavelengths (in Å).
    - flux: Array of observed flux values.
    - f_cont: Array of extrapolated continuum flux values.
    - zspec: Redshift of the source.
    - lambda_LyA_low: Lower limit of the LyA region in the rest-frame (default: 1180 Å).
    - lambda_LyA_up: Upper limit of the LyA region in the rest-frame (default: 1350 Å).

    Returns:
    - DLyA: The calculated LyA damping parameter.
    """
    
    # Normalize flux and f_cont to the same range
    flux_normalized = normalize_to_range(flux)
    f_cont_normalized = normalize_to_range(f_cont)

    # Mask to Lyα region in the rest frame
    mask = (wavelength >= lambda_LyA_low) & (wavelength <= lambda_LyA_up)
    wavelength_LyA = wavelength[mask]
    flux_LyA = flux_normalized[mask]
    f_cont_LyA = f_cont_normalized[mask]
    
    # Calculate the (1 - Fλ / Fcont) term over the masked region
    integrand = (1 - flux_LyA / f_cont_LyA) / (1 + zspec)
    
    # Integrate over the wavelength range using Simpson's rule
    DLyA = simps(integrand, wavelength_LyA) 
    
    return DLyA




