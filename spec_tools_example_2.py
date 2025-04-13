from astropy.io import fits
from astropy.table import Table

# Import necessary functions from the spec_tools.core module
from spec_tools.core import retrieve_spectra, plot_spectrum


table = Table.read("ucsheet_x_sparcl.csv", format="csv")

# Check if any objects were found in the table
if len(table) > 0:
    print("Number of spectra:", len(table))

    # Retrieve the spectrum data for the object
    filenames = retrieve_spectra(table["specid"], output_dir="results/spectra")

    for filename in filenames:
        with fits.open(filename) as hdul:
            # Extract the first HDU (since we only have one HDU in the example)
            hdu = hdul[1]  # BinTableHDU is usually the second HDU, hence index 1

            # Plot the spectrum data
            plot_spectrum(hdu, output_dir="results/plots", open_plot=False)
