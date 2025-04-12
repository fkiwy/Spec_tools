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
    radius = u.Quantity(radius, u.arcsec).to(u.deg).value

    query = f"""
            SELECT sparcl_id, specid, ra, dec, redshift, spectype, data_release, redshift_err
              FROM sparcl.main
             WHERE 't'=q3c_radial_query(ra, dec, {ra}, {dec}, {radius})
               AND specprimary = 1
            """

    payload = {"request": "doQuery", "lang": "ADQL", "format": "csv", "query": query}
    query_url = "https://datalab.noirlab.edu/tap/sync"
    response = requests.get(query_url, params=payload, timeout=300)
    table = Table.read(response.text, format="csv")

    if len(table) == 0:
        return None

    table.add_column(calculate_separation(ra, dec, table["ra"], table["dec"]), name="distance")
    table.sort("distance")

    return table


def retrieve_spectrum(
    object_id: str, data_releases: list = ["SDSS-DR16", "BOSS-DR16", "DESI-EDR", "DESI-DR1"]
) -> (QTable, str):

    ## Instantiate SPARCL Client
    client = SparclClient()

    ## Retrieve Spectra
    result = client.retrieve_by_specid(specid_list=[int(object_id)], dataset_list=data_releases)
    records = result.records

    if len(records) == 0:
        print("No spectrum found with object ID:", object_id)
        return None

    spectrum = records[0]
    data_release = spectrum._dr
    print("Data release:", data_release)

    wavelength = spectrum.wavelength * u.AA
    flux = spectrum.flux * u.erg / u.s / u.cm**2 / u.AA

    # Create a new QTable for the result
    result = QTable([wavelength, flux], names=("WAVELENGTH", "FLUX"))

    return result, data_release


def plot_spectrum(data, data_release, ra, dec, output_dir=tempfile.gettempdir(), open_plot=True, plot_format="pdf"):
    object_name = create_object_name(ra, dec, precision=2, shortform=False, prefix="J", decimal=False)
    filename = os.path.join(output_dir, object_name + "_spectrum." + plot_format)

    wavelength = data["WAVELENGTH"]
    flux = data["FLUX"]

    # Making the matplotlib plots look nicer
    settings = {
        "font.family": "Arial",
        "font.size": 16,
        "axes.linewidth": 2.0,
        "xtick.major.size": 6.0,
        "xtick.minor.size": 4.0,
        "xtick.major.width": 2.0,
        "xtick.minor.width": 1.5,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.top": True,
        "ytick.major.size": 6.0,
        "ytick.minor.size": 4.0,
        "ytick.major.width": 2.0,
        "ytick.minor.width": 1.5,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.right": True,
    }

    plt.rcParams.update(**settings)
    plt.figure(figsize=(20, 6))

    plt.plot(wavelength, flux, color="k", alpha=0.5, label=data_release)
    plt.xlabel(f"Wavelength [{to_latex(wavelength.unit)}]")
    plt.ylabel(f"Flux [{to_latex(flux.unit)}]")
    plt.title(object_name)
    legend = plt.legend(loc="best")
    legend.get_frame().set_boxstyle("Square")

    plt.savefig(filename, dpi=300, bbox_inches="tight", format=plot_format)
    plt.close()
    open_file(filename)


def calculate_separation(target_ra, target_dec, catalog_ra, catalog_dec):
    target_coords = SkyCoord([target_ra * u.deg], [target_dec * u.deg])
    catalog_coords = SkyCoord(catalog_ra, catalog_dec, unit="deg")
    return target_coords.separation(catalog_coords).arcsec


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


def to_latex(unit):
    return unit.to_string("latex_inline")


def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "evince"
        subprocess.call([opener, filename])
