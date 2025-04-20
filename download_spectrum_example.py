from spec_tools.core import retrieve_objects, retrieve_spectrum, plot_spectrum, redshift_to_velocity


# Define the RA and Dec coordinates of the target area (in degrees)
ra, dec = 47.8149931, 1.1085024

# Define the search radius around the target coordinates (in arcseconds)
radius = 5

# Retrieve objects within the specified radius around (ra, dec)
table = retrieve_objects(ra, dec, radius)

# Check if any objects were found in the table
if len(table) > 0:
    # Pretty-print all entries in the retrieved table to inspect the data
    table.pprint_all()

    # Select the first object from the table
    row = table[0]

    # If this object is a star, calculate its radial velocity
    if row["spectype"] == "STAR":
        # Extract the redshift (z) and its uncertainty (z_err) for the object
        z = row["redshift"]
        z_err = row["redshift_err"]

        # Convert the redshift (z) and its uncertainty (z_err) into radial velocity (v) and its error (v_err)
        v, v_err = redshift_to_velocity(z, z_err)

        # Print the radial velocity and its uncertainty in km/s
        print(f"Radial velocity = {v} ± {v_err} km/s")

    # Retrieve the spectrum data for the object
    hdu = retrieve_spectrum(dict(row), save_spectrum=False)

    # Plot the spectrum data, passing the object coordinates (ra, dec) for labeling/plotting purposes
    plot_spectrum(hdu, plot_format="png")
