# SPARCL Spectra Tools

This repository provides a suite of Python tools to **download, inspect, and compare astronomical spectra** from the SDSS, BOSS, and DESI surveys, specifically those made accessible via the [SPARCL interface](https://astrosparcl.datalab.noirlab.edu/). These tools simplify access to spectral data and allow comparisons with empirical templates from the literature such as those from Kesseli+2017.

## Example Scripts

This repository includes several example scripts demonstrating how to use the `spec_tools` and `flux_comp` modules.

### 1. `download_spectrum_example.py`

This script downloads a single spectrum for an astronomical object located at a specified right ascension (RA) and declination (Dec), with a given search radius:

- Uses `retrieve_objects()` to find targets from SPARCL.
- Fetches the spectrum with `retrieve_spectrum()`.
- Saves the spectrum to a FITS file for further analysis.

### 2. `compare_spectrum_example.py`

Expands on the download example by:

- Downloading a single spectrum.
- Comparing it against a set of spectral templates (e.g., Kesseli+2017) using the `flux_comp` module.
- Uses the `SED` class to:
  - Compare spectrum and templates.
  - Convert to flux lambda.
  - Plot and display the comparison.

### 3. `bulk_download_example.py`

A batch version that processes a list of object IDs:

- Reads a CSV file (e.g., `ucsheet_x_sparcl.csv`) listing object identifiers.
- Downloads spectra for all listed objects using `retrieve_spectra()`.
- Plots each spectrum with `plot_spectrum()` and saves the plots.

### 4. `bulk_compare_example.py`

Automates comparison of all spectra in a folder:

- Loads all `.fits` files in a given directory.
- For each file:
  - Parses header and data to extract spectral flux and wavelength.
  - Compares the spectrum with template SEDs using `SED.compare()`.
  - Saves a plot if a good match is found (e.g., `chi² < 500`).

## Core Functionality — `spec_tools/core.py`

The `core.py` module is the main interface for retrieving and analyzing spectra. Below is a summary of the available functions:

### `retrieve_objects(ra, dec, radius_arcsec)`

Query SPARCL to find spectra near the specified RA/Dec coordinates.

- **Parameters:**
  - `ra` (float): Right ascension in degrees.
  - `dec` (float): Declination in degrees.
  - `radius_arcsec` (float): Search radius in arcseconds.

- **Returns:** List of result dictionaries containing metadata about nearby spectra.

### `retrieve_spectrum(row, output_dir=None, filename=None)`

Download a single spectrum based on a SPARCL metadata row.

- **Parameters:**
  - `row` (dict): Metadata describing the spectrum (from `retrieve_objects`).
  - `output_dir` (str, optional): Directory to save the spectrum.
  - `filename` (str, optional): Custom filename (defaults to derived object name).

- **Returns:** A `BinTableHDU` object containing the spectrum.

### `retrieve_spectra(specids, output_dir=None)`

Batch download spectra given a list of object identifiers.

- **Parameters:**
  - `specids` (list[str]): List of SPARCL object IDs.
  - `output_dir` (str, optional): Where to save the downloaded FITS files.

- **Returns:** List of filenames corresponding to the saved FITS files.

### `plot_spectrum(hdu, output_dir=None, open_plot=True)`

Plot a spectrum from a `BinTableHDU` and optionally save the figure.

- **Parameters:**
  - `hdu` (`BinTableHDU`): Spectral data container.
  - `output_dir` (str, optional): Where to save the plot (default is no save).
  - `open_plot` (bool): Whether to show the plot window (default: True).

### `create_object_name(ra, dec, precision=2, shortform=False, prefix="J", decimal=False)`

Utility function to generate a standard astronomical name from RA and Dec.

- **Parameters:**
  - `ra` (float): Right ascension in degrees.
  - `dec` (float): Declination in degrees.
  - `precision` (int): Decimal precision in coordinates.
  - `shortform` (bool): Whether to abbreviate the name.
  - `prefix` (str): Prefix for the name (default: 'J').
  - `decimal` (bool): Use decimal formatting instead of HMS/DMS.

- **Returns:** A string representing the formatted object name (e.g., `J1234+5678`).

## Dependencies

This repository requires the following Python packages:

- [`aiohttp`](https://pypi.org/project/aiohttp/)
- [`astropy`](https://www.astropy.org/)
- [`astroquery`](https://astroquery.readthedocs.io/)
- [`matplotlib`](https://matplotlib.org/)
- [`numpy`](https://numpy.org/)
- [`requests`](https://pypi.org/project/requests/)
- [`sparclclient`](https://astrosparcl.datalab.noirlab.edu/)
- [`scipy`](https://scipy.org/)

## Installation

To install the Spec Tools package, clone the repository and install the dependencies:

```bash
git clone https://github.com/fkiwy/Spec_tools.git
cd Spec_tools
pip install .
```