import os
import sys
import tempfile
import requests
import subprocess
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from sparcl.client import SparclClient


def retrieve_objects(ra: float, dec: float, radius: float) -> Table:
    """
    Retrieves astronomical objects from the SPARCL database within a specified radius
    from a given right ascension (RA) and declination (DEC). The function sends a
    query to the SPARCL database using ADQL (Astronomical Data Query Language) and
    returns the results as an Astropy Table with additional calculated separation
    (distance) from the input RA and DEC.

    Parameters:
    -----------
    ra : float
        Right Ascension (RA) in degrees of the central point to search around.

    dec : float
        Declination (DEC) in degrees of the central point to search around.

    radius : float
        Search radius in arcseconds to define the search area around the given RA and DEC.

    Returns:
    --------
    Table
        An Astropy Table containing the queried objects with their properties
        such as `sparcl_id`, `specid`, `ra`, `dec`, `spectype`, etc., along with
        the calculated separation (distance) from the input coordinates, sorted by
        the closest distance. If no objects are found, `None` is returned.
    """

    # Convert radius from arcseconds to degrees (astropy units)
    radius = u.Quantity(radius, u.arcsec).to(u.deg).value

    # SQL query to fetch relevant astronomical objects from the SPARCL database
    query = f"""
            SELECT sparcl_id, specid, ra, dec, spectype, redshift, redshift_err, data_release,
                   dateobs_center, site, telescope, instrument
              FROM sparcl.main
             WHERE 't'=q3c_radial_query(ra, dec, {ra}, {dec}, {radius})
               AND specprimary = 1
            """

    # Define the payload for the HTTP request to execute the ADQL query
    payload = {
        "request": "doQuery",  # Indicates a query request
        "lang": "ADQL",  # Specifies the query language (ADQL)
        "format": "csv",  # The expected response format is CSV
        "query": query,  # The actual ADQL query string
    }

    # Define the URL endpoint for the query execution
    query_url = "https://datalab.noirlab.edu/tap/sync"

    # Send the GET request to the server to execute the query
    response = requests.get(query_url, params=payload, timeout=300)

    # Read the response into an Astropy Table from CSV format
    table = Table.read(response.text, format="csv")

    # If no objects were returned, return None
    if len(table) == 0:
        return None

    # Create a SkyCoord object for the target
    target_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs")

    # Create a SkyCoord object for the objects in the result table
    object_coords = SkyCoord(table["ra"], table["dec"], unit=(u.deg, u.deg), frame="icrs")

    # Calculate the angular separation between the target and each object in the result table
    separations = target_coord.separation(object_coords).value

    # Convert speparations from degree to arcsec
    sparations = (separations * u.deg).to(u.arcsec).value

    # Add the separations to the result table
    table["separation"] = np.round(sparations, 3)

    # Sort the result table on separation
    table.sort("separation")

    # Return the sorted table with the separation column added
    return table


def retrieve_spectrum(
    data: dict,
    data_releases: list = ["SDSS-DR16", "BOSS-DR16", "DESI-EDR", "DESI-DR1"],
    save_spectrum=False,
    output_dir=tempfile.gettempdir(),
) -> (QTable, str):
    """
    Retrieves the spectrum of an astronomical object from the SPARCL database.

    Parameters:
    -----------
    object_id : str
        The unique identifier of the astronomical object for which the spectrum is to be retrieved.

    data_releases : list, optional
        A list of data release versions to query for the spectrum. The default list includes popular datasets:
        "SDSS-DR16", "BOSS-DR16", "DESI-EDR", and "DESI-DR1".

    Returns:
    --------
    QTable
        A QTable containing the retrieved spectrum's wavelength and flux.

    str
        The data release from which the spectrum was obtained. If no spectrum is found, both values will be None.
    """

    # Instantiate the SPARCL Client to interact with the database
    client = SparclClient()

    object_id = data["specid"]

    # Retrieve spectra using the SPARCL client's method for the given object ID and data releases
    result = client.retrieve_by_specid(specid_list=[int(object_id)], dataset_list=data_releases)
    records = result.records  # Extract the records from the result object

    # If no records are found, notify the user and return None
    if len(records) == 0:
        print("No spectrum found with object ID:", object_id)
        return None, None

    # Extract the first record (assuming only one spectrum is needed)
    spectrum = records[0]

    data["wavelength"] = spectrum.wavelength
    data["flux"] = spectrum.flux

    hdu = createHdu(data)

    if save_spectrum:
        # Generate a unique object name using RA and DEC, formatted as 'J{ra}{dec}' with 2 decimal precision
        object_name = create_object_name(
            data["ra"], data["dec"], precision=2, shortform=False, prefix="J", decimal=False
        )

        # Create the full file path for the plot image
        output_filename = os.path.join(output_dir, object_name + "_spectrum.fits")
        hdu.writeto(output_filename, overwrite=True)

    return hdu


def createHdu(data):
    # Create the FITS Header
    header = fits.Header()
    header["SPARCLID"] = data["sparcl_id"]
    header["SPECID"] = data["specid"]
    header["RA"] = data["ra"]
    header["DEC"] = data["dec"]
    header["WUNIT"] = "Angstrom"
    header["FUNIT"] = "erg / (s cm^2 Angstrom)"
    header["FSCALE"] = 1e-17
    header["SPECTYPE"] = data["spectype"]
    header["REDSHIFT"] = data["redshift"]
    header["REDSHERR"] = data["redshift_err"]
    header["DR"] = data["data_release"]
    header["DATEOBS"] = data["dateobs_center"]
    header["SITE"] = data["site"]
    header["TELESCOP"] = data["telescope"]
    header["INSTRUMT"] = data["instrument"]

    # Create a table with the wavelength and flux data
    columns = [
        fits.Column(name="WAVE", format="D", array=data["wavelength"]),  # 'D' for double precision (float64)
        fits.Column(name="FLUX", format="D", array=data["flux"]),  # 'D' for double precision (float64)
    ]

    # Create the BinTableHDU with the data and header
    table_hdu = fits.BinTableHDU.from_columns(columns, header=header)

    return table_hdu

    # Write the data to a FITS file
    # output_filename = "output_spectrum_with_units.fits"
    # table_hdu.writeto(output_filename, overwrite=True)


def plot_spectrum(hdu, output_dir=tempfile.gettempdir(), open_plot=True, plot_format="pdf"):
    """
    Plots the spectrum of an astronomical object (wavelength vs. flux) and saves the plot to a specified directory.

    Parameters:
    -----------
    data : `Table`
        The data containing the spectrum with columns `WAVELENGTH` and `FLUX`.

    data_release : str
        The data release version (e.g., "SDSS-DR16") to label the plot.

    ra : float
        The right ascension of the object, used to create the plot's file name.

    dec : float
        The declination of the object, used to create the plot's file name.

    output_dir : str, optional
        Directory where the plot will be saved. Defaults to the system's temporary directory.

    open_plot : bool, optional
        If True, the plot will be opened automatically after saving. Defaults to True.

    plot_format : str, optional
        The format to save the plot as (e.g., "pdf", "png"). Defaults to "pdf".

    Returns:
    --------
    None
        The function saves the plot as a file and optionally opens it.
    """

    # Extract the header
    header = hdu.header
    # print(header)

    # Extract the table data (it is an astropy Table)
    data = hdu.data
    # print(table_data)

    # Generate a unique object name using RA and DEC, formatted as 'J{ra}{dec}' with 2 decimal precision
    object_name = create_object_name(
        header["RA"], header["DEC"], precision=2, shortform=False, prefix="J", decimal=False
    )

    # Create the full file path for the plot image
    output_filename = os.path.join(output_dir, object_name + "_spectrum." + plot_format)

    # Convert the string to an astropy unit
    wavelength_unit = u.Unit(header["WUNIT"])
    flux_unit = u.Unit(header["FUNIT"])
    flux_scale = header["FSCALE"]

    # Extract wavelength and flux data from the input
    wavelength = data["WAVE"] * wavelength_unit
    flux = data["FLUX"] * flux_scale * flux_unit

    # Define the matplotlib style settings for the plot for better aesthetics
    settings = {
        "font.family": "Arial",  # Set the font family
        "font.size": 16,  # Set the font size for labels and title
        "axes.linewidth": 2.0,  # Make axis lines thicker
        "xtick.major.size": 6.0,  # Set major x-tick size
        "xtick.minor.size": 4.0,  # Set minor x-tick size
        "xtick.major.width": 2.0,  # Set major x-tick line width
        "xtick.minor.width": 1.5,  # Set minor x-tick line width
        "xtick.direction": "in",  # Set x-ticks to point inward
        "xtick.minor.visible": True,  # Make minor ticks visible
        "xtick.top": True,  # Make top x-ticks visible
        "ytick.major.size": 6.0,  # Set major y-tick size
        "ytick.minor.size": 4.0,  # Set minor y-tick size
        "ytick.major.width": 2.0,  # Set major y-tick line width
        "ytick.minor.width": 1.5,  # Set minor y-tick line width
        "ytick.direction": "in",  # Set y-ticks to point inward
        "ytick.minor.visible": True,  # Make minor ticks visible
        "ytick.right": True,  # Make right y-ticks visible
    }

    # Apply the style settings to matplotlib
    plt.rcParams.update(**settings)

    # Create a new figure with a specified size (20x6 inches)
    plt.figure(figsize=(20, 6))

    # Plot the spectrum (wavelength vs flux) with some transparency and label the data release
    plt.plot(wavelength, flux, color="k", alpha=0.5, label=header["DR"])

    def to_latex(unit):
        return unit.to_string("latex_inline")

    # Label the x and y axes with appropriate units
    plt.xlabel(f"Wavelength [{to_latex(wavelength.unit)}]")
    plt.ylabel(f"Flux [{to_latex(flux.unit)}]")

    # Set the title to the object name (RA/DEC formatted)
    plt.title(object_name)

    # Add a legend at the best location on the plot
    legend = plt.legend(loc="best")
    legend.get_frame().set_boxstyle("Square")

    # Save the plot as an image file (PDF or other format)
    plt.savefig(output_filename, dpi=300, bbox_inches="tight", format=plot_format)

    # Close the plot to free up memory
    plt.close()

    # Optionally open the plot after saving (depending on the open_plot flag)
    if open_plot:
        open_file(output_filename)


def redshift_to_velocity(z, sigma_z):
    """
    Calculate the radial velocity and its uncertainty from the redshift for stars.

    This function uses the small redshift approximation to convert redshift to
    radial velocity for stars. It also propagates the uncertainty in the redshift
    to the radial velocity using the error propagation formula:

    v_r = z * c
    sigma_v_r = c * sigma_z

    Parameters
    ----------
    z : float
        The redshift (dimensionless).

    sigma_z : float
        The uncertainty in the redshift (dimensionless).

    Returns
    -------
    tuple
        A tuple containing the radial velocity (in km/s) and its uncertainty (in km/s).
    """
    # Speed of light in km/s
    c = 3.0e5  # km/s

    # Radial velocity calculation
    v_r = z * c

    # Uncertainty in the radial velocity
    sigma_v_r = c * sigma_z

    return round(v_r), round(sigma_v_r)


def create_object_name(ra, dec, precision=0, sep="", prefix=None, shortform=False, decimal=True):
    """
    Generate a string-based object name from celestial coordinates.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    precision : int, optional
        Number of decimal places for coordinates (default is 0).
    sep : str, optional
        Separator to use in formatted output (default is "").
    prefix : str, optional
        String to prepend to the object name.
    shortform : bool, optional
        If True, returns a short HMS/DMS form like '1234+5678'.
    decimal : bool, optional
        If True, returns decimal-formatted coordinates.

    Returns
    -------
    str
        A formatted object name string.
    """

    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

    if shortform:
        coords_str = coords.to_string("hmsdms", decimal=False, sep=sep, precision=0)
        object_name = coords_str[0:4] + coords_str[7:12]
    else:
        if decimal:
            object_name = coords.to_string(decimal=True, precision=precision)
        else:
            object_name = coords.to_string("hmsdms", decimal=False, sep=sep, precision=precision).replace(" ", "")

    if prefix:
        object_name = prefix + object_name

    return str(object_name)


def open_file(filename):
    """
    Opens a file using the default application based on the operating system.

    Parameters:
    -----------
    filename : str
        The path to the file to be opened.

    Behavior:
    ---------
    - On Windows (win32), the file will be opened using the default associated application.
    - On macOS (darwin), the file will be opened using the `open` command.
    - On Linux and other Unix-like systems, the file will be opened using the `evince` viewer (can be changed to another viewer).
    """

    # Check if the operating system is Windows
    if sys.platform == "win32":
        # On Windows, use the default application associated with the file type
        os.startfile(filename)

    # Check if the operating system is macOS (darwin)
    elif sys.platform == "darwin":
        # On macOS, use the 'open' command to open the file
        subprocess.call(["open", filename])

    # For Linux or other Unix-like systems (including Linux and some BSDs)
    else:
        # For Linux, use 'evince' as the default viewer for PDFs and other files.
        # Change this to another viewer (e.g., "xdg-open") depending on your preference
        subprocess.call(["evince", filename])
