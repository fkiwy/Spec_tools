import requests
from io import BytesIO
from astropy.io import fits
from astropy.table import Table, TableColumns
from bs4 import BeautifulSoup


def download_html(ra, dec, radius, database):
    """
    Downloads HTML content from the WSA database for a given region.

    Parameters:
        ra (str): Right Ascension coordinate in degrees.
        dec (str): Declination coordinate in degrees.
        radius (int): Radius of the region in arcminutes.
        database (str): Name of the database (e.g., 'UHSDR2').

    Returns:
        str: HTML content if the download is successful, None otherwise.
    """
    base_url = "http://wsa.roe.ac.uk:8080/wsa/WSASQL"
    params = {
        "database": database,
        "programmeID": 107,
        "from": "source",
        "formaction": "region",
        "ra": ra,
        "dec": dec,
        "sys": "J",
        "radius": radius / 60,
        "xSize": "",
        "ySize": "",
        "boxAlignment": "RADec",
        "format": "FITS",
        "select": "default",
        "where": "",
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to download HTML. Status code: {response.status_code}")
        print(response.text)
        return None


def extract_fits_url(html_content):
    """
    Extracts the FITS file URL from the HTML content.

    Parameters:
        html_content (str): HTML content obtained from the WSA database.

    Returns:
        str: FITS file URL if found, None otherwise.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    download_link = soup.find("a", class_="dl")
    if download_link:
        return download_link["href"]
    else:
        return None


def download_fits_file(fits_url):
    """
    Downloads and converts the FITS file to an Astropy table.

    Parameters:
        fits_url (str): URL of the FITS file.

    Returns:
        astropy.table.Table: Astropy table if successful, None otherwise.
    """
    response = requests.get(fits_url)
    if response.status_code == 200:
        fits_data = fits.open(BytesIO(response.content))
        table = Table(fits_data[1].data)
        if table:
            # Flatten multidimensional columns
            flattened_table = TableColumns()
            for col_name in table.colnames:
                if table[col_name].ndim > 1:
                    flattened_columns = [f"{col_name}" for i in range(table[col_name].shape[1])]
                    for i, flattened_col_name in enumerate(flattened_columns):
                        flattened_table[flattened_col_name] = table[col_name][:, i]
                else:
                    flattened_table[col_name] = table[col_name]
            return Table(flattened_table)
        else:
            return None
    else:
        print(f"Failed to download FITS file. Status code: {response.status_code}")
        print(response.text)
        return None


def query_region(ra, dec, radius=5, database="UHSDR2"):
    """
    Queries the WSA database for a specific region and returns the results as an Astropy table.

    Parameters:
        ra (str): Right Ascension coordinate in degrees.
        dec (str): Declination coordinate in degrees.
        radius (int): Radius of the region in arcminutes (default is 5 arcminutes).
        database (str): Name of the database (default is 'UHSDR2').

    Returns:
        astropy.table.Table: Astropy table containing the query results, or None if unsuccessful.
    """
    table = None
    html_content = download_html(ra, dec, radius, database)
    if html_content:
        fits_url = extract_fits_url(html_content)
        if fits_url:
            table = download_fits_file(fits_url)
            if table:
                table.sort("DISTANCE")
    return table
