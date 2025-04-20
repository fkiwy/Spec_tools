from astropy.io import fits
from astropy.table import Table

from spec_tools.core import retrieve_spectra, plot_spectrum


# Read the table of objects from a CSV file into an Astropy Table
# The 'ucsheet_x_sparcl.csv' file contains the list of object IDs whose spectra we will retrieve
table = Table.read("ucsheet_x_sparcl.csv", format="csv")

# Check if the table has any objects to process
if len(table) > 0:
    # Print the number of objects (spectra) to retrieve
    print("Number of spectra to retrieve:", len(table))

    # Call the 'retrieve_spectra' function, passing the 'specid' column from the table as input
    # The spectra will be saved in the "results/spectra" directory
    filenames = retrieve_spectra(table["specid"], output_dir="results/spectra")

    # Iterate over the retrieved FITS files (one for each object)
    for filename in filenames:
        # Open the FITS file
        with fits.open(filename) as hdul:
            hdu = hdul[1]  # BinTableHDU is usually the second HDU, hence index 1

            # Call the 'plot_spectrum' function to generate and save a plot of the spectrum data
            # The 'open_plot' argument is set to False to avoid automatically opening the plot viewer
            plot_spectrum(hdu, output_dir="results/plots", open_plot=False)
