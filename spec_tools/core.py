import os
import sys
import tempfile
import requests
import subprocess
import astropy.units as u
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
        such as `sparcl_id`, `specid`, `ra`, `dec`, `redshift`, etc., along with
        the calculated separation (distance) from the input coordinates, sorted by
        the closest distance. If no objects are found, `None` is returned.
    """

    # Convert radius from arcseconds to degrees (astropy units)
    radius = u.Quantity(radius, u.arcsec).to(u.deg).value

    # SQL query to fetch relevant astronomical objects from the SPARCL database
    query = f"""
            SELECT sparcl_id, specid, ra, dec, redshift, spectype, data_release, redshift_err
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

    # Calculate the separation (distance) between the input coordinates (ra, dec)
    # and the coordinates of the objects returned in the query
    table.add_column(calculate_separation(ra, dec, table["ra"], table["dec"]), name="distance")

    # Sort the table by the calculated separation (distance) in ascending order
    table.sort("distance")

    # Return the sorted table with the distance column added
    return table


def retrieve_spectrum(
    object_id: str, data_releases: list = ["SDSS-DR16", "BOSS-DR16", "DESI-EDR", "DESI-DR1"]
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

    # Retrieve spectra using the SPARCL client's method for the given object ID and data releases
    result = client.retrieve_by_specid(specid_list=[int(object_id)], dataset_list=data_releases)
    records = result.records  # Extract the records from the result object

    # If no records are found, notify the user and return None
    if len(records) == 0:
        print("No spectrum found with object ID:", object_id)
        return None, None

    # Extract the first record (assuming only one spectrum is needed)
    spectrum = records[0]

    # Retrieve the data release associated with the spectrum
    data_release = spectrum._dr
    print("Data release:", data_release)

    # Convert wavelength to Angstroms and flux to erg/s/cm^2/Å
    wavelength = spectrum.wavelength * u.AA
    flux = spectrum.flux * 1e-17 * u.erg / u.s / u.cm**2 / u.AA

    # Create a QTable from the wavelength and flux arrays
    result = QTable([wavelength, flux], names=("WAVELENGTH", "FLUX"))

    # Return the result table and the data release
    return result, data_release


def plot_spectrum(data, data_release, ra, dec, output_dir=tempfile.gettempdir(), open_plot=True, plot_format="pdf"):
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

    # Generate a unique object name using RA and DEC, formatted as 'J{ra}{dec}' with 2 decimal precision
    object_name = create_object_name(ra, dec, precision=2, shortform=False, prefix="J", decimal=False)

    # Create the full file path for the plot image
    filename = os.path.join(output_dir, object_name + "_spectrum." + plot_format)

    # Extract wavelength and flux data from the input
    wavelength = data["WAVELENGTH"]
    flux = data["FLUX"]

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
    plt.plot(wavelength, flux, color="k", alpha=0.5, label=data_release)

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
    plt.savefig(filename, dpi=300, bbox_inches="tight", format=plot_format)

    # Close the plot to free up memory
    plt.close()

    # Optionally open the plot after saving (depending on the open_plot flag)
    if open_plot:
        open_file(filename)


def calculate_separation(target_ra, target_dec, catalog_ra, catalog_dec):
    """
    Calculates the angular separation between a target coordinate and a catalog of coordinates.

    Parameters:
    -----------
    target_ra : float
        Right Ascension (RA) of the target in degrees.

    target_dec : float
        Declination (DEC) of the target in degrees.

    catalog_ra : array-like
        Array or list of Right Ascensions (RA) of catalog objects in degrees.

    catalog_dec : array-like
        Array or list of Declinations (DEC) of catalog objects in degrees.

    Returns:
    --------
    separation : `Quantity`
        Angular separation(s) between the target and catalog coordinates in arcseconds.
    """

    # Create SkyCoord object for the target coordinates
    target_coords = SkyCoord([target_ra * u.deg], [target_dec * u.deg])

    # Create SkyCoord object for the catalog coordinates (this can handle multiple entries)
    catalog_coords = SkyCoord(catalog_ra, catalog_dec, unit="deg")

    # Calculate the angular separation between the target and each object in the catalog
    separation = target_coords.separation(catalog_coords)

    # Return the separation in arcseconds
    return separation.arcsec


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
