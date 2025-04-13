import warnings
from astropy.utils.exceptions import AstropyWarning

from spec_tools.core import retrieve_objects, retrieve_spectrum, create_object_name
from flux_comp.core import SED, WaveFlux, TemplateProvider

warnings.simplefilter("ignore", category=AstropyWarning)


# ------------------------------
# Compare spectrum to templates
# ------------------------------

# Coordinates for the object of interest
ra, dec = 209.2891781, 55.7474398
search_radius = 5  # arcsec

# Retrieve objects based on coordinates
results = retrieve_objects(ra, dec, search_radius)

if results:
    result = results[0]
    object_id = result["specid"]
    ra = result["ra"]
    dec = result["dec"]

    print(f"Object found at RA: {round(ra, 7)}, Dec: {round(dec, 7)}")

    # Retrieve the spectrum for the object
    data, data_release = retrieve_spectrum(object_id)

    if data and len(data) > 0:
        # Select template(s) for comparison (Kesseli+2017)
        provider = TemplateProvider()
        template_name = "Kesseli+2017"
        templates = provider.get_Kesseli_2017_templates()

        # Create a WaveFlux object for comparison
        spectrum = WaveFlux(label=data_release, wavelength=data["WAVELENGTH"], flux=data["FLUX"])

        # Create object name for plotting
        object_name = create_object_name(ra, dec, precision=2, shortform=False, prefix="J", decimal=False)

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
            plot_format="pdf",
        )
    else:
        print("No spectrum found for the given object ID")
else:
    print("No object found for the given coordinates and search radius")
