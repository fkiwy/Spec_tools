from spec_tools.core import retrieve_objects, retrieve_spectrum, plot_spectrum


ra, dec = 279.3540369, 48.1993358
# ra, dec = 209.2891781, 55.7474398
radius = 5  # arcsec

table = retrieve_objects(ra, dec, radius)

table.pprint_all()

if len(table) > 0:
    row = table[0]
    object_id = row["specid"]
    ra = row["ra"]
    dec = row["dec"]

    data, data_release = retrieve_spectrum(object_id)

    plot_spectrum(data, data_release, ra, dec)
