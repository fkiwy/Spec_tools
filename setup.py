from setuptools import setup

setup(
    name="Spec_tools",
    version="0.1.0",
    description="A package of tools related to SDSS, BOSS, and DESI spectral surveys",
    url="https://github.com/fkiwy/Spec_tools",
    author="Frank Kiwy",
    author_email="frank.kiwy@outlook.com",
    license="MIT",
    install_requires=[
        "aiohttp",
        "astropy",
        "astroquery",
        "matplotlib",
        "numpy",
        "requests",
        "sparclclient",
        "scipy",
    ],
    packages=["spec_tools", "flux_comp"],
    package_dir={"spec_tools": "spec_tools", "flux_comp": "flux_comp"},
    package_data={
        "flux_comp": [
            "flux_comp/templates/Burgasser+2017/*.fits",
            "flux_comp/templates/Kesseli+2017/*.fits",
            "flux_comp/templates/Theissen+2022/*.fits",
        ]
    },
    zip_safe=False,
    include_package_data=True,
)
