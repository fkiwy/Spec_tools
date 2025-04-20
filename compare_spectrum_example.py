import warnings
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning

from spec_tools.core import retrieve_objects, retrieve_spectrum, create_object_name
from flux_comp.core import SED, WaveFlux, TemplateProvider

warnings.simplefilter("ignore", category=AstropyWarning)


# Define the RA and Dec coordinates of the target area (in degrees)
ra, dec = 47.8149931, 1.1085024

# Define the search radius around the target coordinates (in arcseconds)
radius = 5

# Retrieve objects based on coordinates
results = retrieve_objects(ra, dec, radius)

if results:
    result = results[0]

    # Select template(s) for comparison (Kesseli+2017)
    provider = TemplateProvider()
    template_name = "Kesseli+2017"
    templates = provider.get_Kesseli_2017_templates()

    # Retrieve the spectrum for the object
    hdu = retrieve_spectrum(dict(result))

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
    sed = SED(object_name + " vs. " + template_name)
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
        figure_size=(10, 3),
        plot_format="png",
        open_plot=True,
    )
else:
    print("No object found for the given coordinates and search radius")
