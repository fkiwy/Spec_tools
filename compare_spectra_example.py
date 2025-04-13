import os
import warnings
import astropy.units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from spec_tools.core import create_object_name
from flux_comp.core import SED, WaveFlux, TemplateProvider

warnings.simplefilter("ignore", category=AstropyWarning)


# ----------------------------
# Compare spectra to templates
# ----------------------------

# Select template(s) for comparison (Kesseli+2017)
provider = TemplateProvider()
template_name = "Kesseli+2017"
templates = provider.get_Kesseli_2017_templates()

# Define the directory path
directory_path = "results/spectra"

# List all FITS files in the directory
fits_files = [f for f in os.listdir(directory_path) if f.endswith(".fits")]

# Read and process each FITS file
for fits_file in fits_files:
    file_path = os.path.join(directory_path, fits_file)
    with fits.open(file_path) as hdul:
        hdu = hdul[1]  # BinTableHDU is usually the second HDU, hence index 1

        # Extract the header
        header = hdu.header

        # Extract the table data (it is an astropy Table)
        data = hdu.data

        # Convert the string to an astropy unit
        wavelength_unit = u.Unit(header["WUNIT"])
        flux_unit = u.Unit(header["FUNIT"])
        flux_scale = header["FSCALE"]

        # Extract wavelength and flux data from the input
        wavelength = data["WAVE"] * wavelength_unit
        flux = data["FLUX"] * flux_scale * flux_unit

        # Create a WaveFlux object for comparison
        spectrum = WaveFlux(label=header["DR"], wavelength=wavelength, flux=flux)

        # Generate a unique object name using RA and DEC, formatted as 'J{ra}{dec}' with 2 decimal precision
        object_name = create_object_name(
            header["RA"], header["DEC"], precision=2, shortform=False, prefix="J", decimal=False
        )

        # Set up the SED object and compare the spectrum to the templates
        sed = SED(object_name + " vs. " + template_name, directory="results/comparison")
        sed.compare(
            spectrum,
            templates,
            trim_wave=True,
            number_of_matches=1,
            metric="reduced-chi2",
            add_stat_to_template_label=False,
        )

        # Convert to flux lambda and plot the results
        sed.to_flux_lambda()
        sed.plot(
            reference_on_top=False,
            spec_uncertainty=False,
            plot_format="pdf",
            open_plot=False,
        )
