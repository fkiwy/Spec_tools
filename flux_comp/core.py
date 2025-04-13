import re
import os
import sys
import math
import json
import time
import inspect
import warnings
import tempfile
import requests
import itertools
import subprocess
import multiprocessing
import numpy as np
from io import BytesIO
from os.path import isfile, join
import astropy.units as u
from astropy.io import ascii, fits
from astropy.io.votable import parse, parse_single_table
from astropy.table import Table
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.ukidss import Ukidss
from astroquery.vsa import Vsa
from astroquery.sdss import SDSS
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

import flux_comp.uhs as uhs


SVO_URL = "http://svo2.cab.inta-csic.es/theory/fps3/fps.php"


class Bucket:

    def __init__(self, x):
        self.x = x


class SED:

    def __init__(self, target_name=None, directory=tempfile.gettempdir()):
        warnings.simplefilter("ignore", category=Warning)
        self.target_name = target_name
        self.directory = directory
        self.flux_unit = u.Jy
        self.data = []
        self.results = []

    def add(self, waveflux):
        waveflux.data.sort("Wavelength")
        self.data.append(waveflux)

    def compare_with_uncertainties(
        self,
        reference,
        templates,
        number_of_bands_to_omit=0,
        trim_wave=False,
        verbose=True,
        multi_processing=True,
        iterations=100,
        parameter_keys=[],
    ):
        """
        Compares the reference data with the provided templates, estimating parameter errors by introducing
        random noise based on uncertainties. This method runs multiple iterations to compute the uncertainty
        in the model parameters, using either multi-processing or sequential execution.

        Parameters:
        -----------
        reference : WaveFlux
            The reference data to be compared with the templates. The reference must be an instance of
            the `WaveFlux` class, containing wavelength, flux, and uncertainty data.

        templates : list of WaveFlux
            A list of `WaveFlux` templates that are compared to the reference data. Each template should
            be an instance of `WaveFlux` containing the data to fit to the reference.

        number_of_bands_to_omit : int, optional, default=0
            The number of photometric bands to omit during the comparison process. Useful for handling
            bands that might not be reliable or need to be excluded.

        trim_wave : bool, optional, default=False
            If True, trims the wavelength range to only include the region where both the reference and
            templates overlap.

        verbose : bool, optional, default=True
            If True, prints detailed progress messages, including the number of iterations, execution time,
            and estimated parameter errors.

        multi_processing : bool, optional, default=True
            If True, uses multiple processing (parallelization) to speed up the computation by distributing
            the iterations across multiple CPU cores. If False, runs the iterations sequentially.

        iterations : int, optional, default=100
            The number of iterations to run for estimating the parameter errors. Each iteration involves
            adding random noise to the reference flux based on the uncertainty and comparing it with the templates.

        parameter_keys : list of str, optional, default=[]
            A list of keys corresponding to model parameters that should be extracted from the template
            parameters for error estimation. These parameters are printed with their mean value and standard
            error after the comparison.

        Returns:
        --------
        template : WaveFlux
            The template with the updated flux, uncertainty, and estimated model parameters. The flux is
            computed as the mean flux from all iterations, and the uncertainty is the standard deviation of
            the flux. The model parameters are stored in the `model_params` attribute of the template, along
            with their errors.

        Notes:
        ------
        - This method performs a Monte Carlo simulation by adding random noise to the reference flux in each
          iteration based on the uncertainty in the reference data. It then compares the noisy reference to
          the provided templates.
        - The method uses either parallel processing (multi-processing) or sequential execution to run the
          iterations. Parallel processing speeds up the estimation of parameter errors by utilizing multiple
          CPU cores.
        - After all iterations, the method computes the mean flux and its uncertainty, and also extracts the
          model parameters and their errors (mean and standard deviation).
        - The method returns the template that has the updated flux, uncertainty, and model parameters.
        """
        if verbose:
            start_time = time.time()
            print("Estimating parameter errors ...")
            print("Number of iterations:", iterations)

        tasks = []
        for i in range(iterations):
            # Clone the reference
            data = reference.data
            clone = WaveFlux()
            clone.create_table(data["Wavelength"], data["Flux"], data["Uncertainty"])
            clone.label = reference.label
            clone.used_bands = reference.used_bands
            clone.used_photometry = reference.used_photometry
            clone.reference = reference.reference
            clone.photometry = reference.photometry
            clone.connect_markers = reference.connect_markers

            data = clone.data
            flux = data["Flux"]
            uncertainty = data["Uncertainty"]

            # Add some random noise to the flux
            data["Flux"] = flux + np.random.normal(0, uncertainty)
            tasks.append((clone, templates, number_of_bands_to_omit, trim_wave))

        # Multi-processing
        if multi_processing:
            if verbose:
                print("  Multi-processing is enabled")
                try:
                    print("  Number of CPUs available:", len(os.sched_getaffinity(0)))
                except:
                    print("  Number of CPUs in the system:", multiprocessing.cpu_count())

            pool = multiprocessing.Pool()
            results = pool.starmap(self.call_compare_method, tasks)
            pool.close()
            pool.join()
        else:
            results = []
            for i, task in enumerate(tasks):
                reference, templates, number_of_bands_to_omit, trim_wave = task
                result = self.call_compare_method(reference, templates, number_of_bands_to_omit, trim_wave)
                results.append(result)
                if verbose:
                    print(i, result.model_params)

        # Collect template parameters
        template = None
        fluxes = []
        statistics = []
        parameters_list = []
        for result in results:
            if result:
                template = result
                fluxes.append(result.data["Flux"])
                statistics.append(result.statistic)
                parameters_list.append(result.model_params)

        flux = np.mean(fluxes, axis=0)
        uncertainty = np.std(fluxes, axis=0)  # / np.sqrt(np.size(fluxes, axis=0))
        statistic = np.mean(statistics)
        template.data["Flux"] = flux
        template.data["Uncertainty"] = uncertainty
        template.statistic = statistic

        # Print mean template parameters with their error
        mean_parameters = {}
        for parameter_key in parameter_keys:
            values = []
            for parameters in parameters_list:
                values.append(parameters[parameter_key])
            param_mean = np.mean(values)
            param_sem = np.std(values)  # / np.sqrt(np.size(values))
            print(f"{parameter_key} = {param_mean} ± {param_sem}")
            mean_parameters[parameter_key] = param_mean
            mean_parameters[parameter_key + "_err"] = param_sem
        template.model_params = mean_parameters

        self.add(reference)
        self.add(template)

        if verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Execution time: {:.2f} seconds".format(elapsed_time))

        return template

    def call_compare_method(self, reference, templates, number_of_bands_to_omit, trim_wave):
        self.data = []
        return self.compare(reference, templates, number_of_bands_to_omit=number_of_bands_to_omit, trim_wave=trim_wave)

    def compare(
        self,
        reference,
        templates,
        number_of_matches=1,
        number_of_bands_to_omit=0,
        metric="reduced-chi2",
        trim_wave=False,
        verbose=False,
        plot=False,
        print_observed_photometry=False,
        print_template_photometry=False,
        add_stat_to_template_label=False,
        normalize=True,
    ):
        """
        Compares the flux data of a reference object (WaveFlux) with a list of template objects (WaveFlux)
        using a specified comparison metric. The method computes a statistic (e.g., reduced-chi2, delta, or chi2)
        to quantify the similarity between the reference and each template. It provides options to omit certain bands,
        normalize flux values, trim wavelength ranges, and display or print relevant information.

        Parameters:
        -----------
        reference : WaveFlux
            The reference object (WaveFlux) containing observed flux data to be compared with the templates.

        templates : list of WaveFlux
            A list of template objects (WaveFlux) to be compared with the reference.

        number_of_matches : int, optional, default=1
            The number of best matches to return based on the computed statistic.

        number_of_bands_to_omit : int, optional, default=0
            The number of photometric bands to omit during the comparison (applies if reference has photometric data).

        metric : str, optional, default="reduced-chi2"
            The metric used to compare the reference flux with the template flux. Options include "reduced-chi2",
            "chi2", and "delta".

        trim_wave : bool, optional, default=False
            If True, trims the reference and template flux data to the overlapping wavelength range.

        verbose : bool, optional, default=False
            If True, prints detailed information about the comparison process.

        plot : bool, optional, default=False
            If True, generates a plot of the comparison results.

        print_observed_photometry : bool, optional, default=False
            If True, prints the photometric data of the reference object.

        print_template_photometry : bool, optional, default=False
            If True, prints the photometric data of the template objects.

        add_stat_to_template_label : bool, optional, default=False
            If True, appends the comparison statistic to the label of each template.

        normalize : bool, optional, default=True
            If True, normalizes the template flux to match the reference flux.

        Returns:
        --------
        best_match : WaveFlux or None
            The template with the best match to the reference based on the comparison metric. If no match is found,
            returns None.

        Notes:
        ------
        - The method creates copies of both the reference and template data to ensure the original objects remain unmodified.
        - The statistic is computed based on the specified metric, and the results are sorted to find the best matches.
        - The comparison includes an option to omit bands, normalize flux, trim the wavelength range, and print relevant information.
        """
        if verbose:
            print("Comparing observed flux with templates ...")

        if len(reference.data["Flux"]) == 0:
            print("Flux array is empty. Nothing to compare with.")
            return None

        references = []
        if reference.photometry and number_of_bands_to_omit > 0:
            data = reference.data
            ref_wavelength = data["Wavelength"]
            ref_flux = data["Flux"]
            ref_uncertainty = data["Uncertainty"]
            used_photometry = list(reference.used_photometry.items())
            used_photometry = [Bucket(x) for x in used_photometry]
            phot_combinations = self.calculate_combinations(used_photometry, number_of_bands_to_omit)
            band_combinations = self.calculate_combinations(reference.used_bands, number_of_bands_to_omit)
            wavelength_combinations = self.calculate_combinations(ref_wavelength, number_of_bands_to_omit)
            flux_combinations = self.calculate_combinations(ref_flux, number_of_bands_to_omit)
            uncertainty_combinations = self.calculate_combinations(ref_uncertainty, number_of_bands_to_omit)
            for _, (phot, bands, wavelength, flux, uncertainty) in enumerate(
                zip(
                    phot_combinations,
                    band_combinations,
                    wavelength_combinations,
                    flux_combinations,
                    uncertainty_combinations,
                )
            ):
                wf = WaveFlux()
                wf.create_table(wavelength, flux, uncertainty)
                wf.label = reference.label
                wf.used_bands = bands.tolist()
                used_photometry = {}
                for bucket in phot:
                    used_photometry[bucket.x[0]] = bucket.x[1]
                wf.used_photometry = used_photometry
                wf.reference = True
                wf.photometry = reference.photometry
                wf.connect_markers = reference.connect_markers
                references.append(wf)
        else:
            reference.reference = True
            references.append(reference)

        results = []
        for reference0 in references:
            for template0 in templates:
                # Clone the reference (the clone will be modified here after, the original must remain the same)
                data = reference0.data
                reference = WaveFlux()
                reference.create_table(data["Wavelength"], data["Flux"], data["Uncertainty"])
                reference.label = reference0.label
                reference.used_bands = reference0.used_bands
                reference.used_photometry = reference0.used_photometry
                reference.reference = reference0.reference
                reference.photometry = reference0.photometry
                reference.connect_markers = reference0.connect_markers

                # Sort the reference data by wavelength
                data = reference.data
                data.sort("Wavelength")
                ref_wavelength = data["Wavelength"]

                # Clone the template (the clone will be modified here after, the original must remain the same)
                data = template0.data
                template = WaveFlux()
                template.create_table(data["Wavelength"], data["Flux"], data["Uncertainty"])
                template.label = template0.label
                template.used_bands = template0.used_bands
                template.used_photometry = template0.used_photometry
                template.template = template0.reference
                template.photometry = template0.photometry
                template.connect_markers = template0.connect_markers
                template.model_params = template0.model_params

                # Sort the template data by wavelength
                data = template.data
                data.sort("Wavelength")
                tpl_wavelength = data["Wavelength"]

                if trim_wave:
                    # Trim the reference and template based on the determined wavelength range
                    min_wave, max_wave = determine_overlapping_wavelength_range(ref_wavelength, tpl_wavelength)
                    reference.trim(min_wave, max_wave)
                    template.trim(min_wave, max_wave)

                # Normalize the template flux to the reference flux
                if normalize:
                    template.normalize(reference)

                # Extract the reference data
                data = reference.data
                ref_wavelength = data["Wavelength"]
                ref_flux = data["Flux"]
                ref_uncertainty = data["Uncertainty"]

                # Extract the template data
                data = template.data
                tpl_wavelength = data["Wavelength"]
                tpl_flux = data["Flux"]

                # Trim reference and template data to their overlapping wavelength range
                min_wave, max_wave = determine_overlapping_wavelength_range(tpl_wavelength, ref_wavelength)
                tpl_wavelength, tpl_flux = trim(tpl_wavelength, tpl_flux, min_wave, max_wave)
                _, ref_flux = trim(ref_wavelength, ref_flux, min_wave, max_wave)
                ref_wavelength, ref_uncertainty = trim(ref_wavelength, ref_uncertainty, min_wave, max_wave)

                # Interpolate the template flux onto the same wavelength grid as that of the reference
                tpl_flux = np.interp(ref_wavelength, tpl_wavelength, tpl_flux)

                # Compare the reference flux to the template flux
                statistic = self.compare_fluxes(ref_flux, ref_uncertainty, tpl_flux, metric)

                results.append([statistic, reference, template])

                if verbose:
                    print(f"Compared {reference.label} with {template.label}, statistic = {statistic:e}")

        results.sort(key=lambda x: x[0])
        self.results = results
        best_match = None
        number_of_results = 0
        for result in results:
            statistic = result[0]
            if np.isnan(statistic) or statistic == 0:
                continue
            if number_of_results == 0:
                reference = result[1]
                self.add(reference)
                best_match = result[2]
                if print_observed_photometry:
                    print("Observed photometry:")
                    [
                        print(f'{key}: {value[0]:.3f}{"" if np.isnan(value[1]) else f" ± {value[1]:.3f}"} mag')
                        for key, value in reference.used_photometry.items()
                    ]
            template = result[2]
            template.statistic = statistic
            if add_stat_to_template_label:
                if metric == "delta":
                    stat_symbol = r"$\overline{\delta}$"
                    # stat_value = '{:.3e}'.format(statistic)
                    stat_value = self.as_si(statistic, 3)
                elif metric == "chi2":
                    stat_symbol = r"$\chi^{2}$"
                    # stat_value = '{:.3e}'.format(statistic)
                    stat_value = self.as_si(statistic, 3)
                elif metric == "reduced-chi2":
                    stat_symbol = r"$\chi_{\nu}^{2}$"
                    stat_value = "{:.3f}".format(statistic)
                template.label = f"{template.label}\n{stat_symbol} = {stat_value}"
            self.add(template)
            if verbose:
                print(f"Best match : {template.label}, statistic = {statistic:e}")
            if print_template_photometry:
                print("Template photometry:")
                [
                    print(f'{key}: {value[0]:.3f}{"" if np.isnan(value[1]) else f" ± {value[1]:.3f}"} mag')
                    for key, value in template.used_photometry.items()
                ]
            number_of_results += 1
            if number_of_results == number_of_matches:
                break

        if plot:
            self.plot()

        return best_match

    def as_si(self, x, ndp):
        s = "{x:0.{ndp:d}e}".format(x=x, ndp=ndp)
        m, e = s.split("e")
        return r"{m:s}$\times 10^{{{e:d}}}$".format(m=m, e=int(e))

    def compare_fluxes(self, ref_flux, ref_uncertainty, tpl_flux, metric):
        if metric == "delta":
            return np.nanmean((ref_flux - tpl_flux) ** 2)
        elif metric == "chi2":
            return np.sum((ref_flux - tpl_flux) ** 2 / tpl_flux)
        elif metric == "reduced-chi2":
            if np.isnan(ref_uncertainty).all():
                ref_uncertainty[:] = np.nanmedian(ref_flux) * 0.1  # Default uncertainty
            dof = np.count_nonzero(~np.isnan(ref_flux)) - 1  # Degrees of freedom
            return np.sum((ref_flux - tpl_flux) ** 2 / ref_uncertainty**2) / dof  # Reduced chi-square
        else:
            raise Exception(f"No such metric: {metric}")

    def calculate_combinations(self, data, number_of_bands_to_omit):
        all_combinations = []
        number_of_bands = len(data)
        for i in range(1, number_of_bands_to_omit + 1, 1):
            for combination in itertools.combinations(range(number_of_bands), number_of_bands - i):
                combined_data = np.array(data)[list(combination)]
                all_combinations.append(combined_data)
        return all_combinations

    def plot(
        self,
        spec_uncertainty=False,
        phot_uncertainty=False,
        show_grid=False,
        xscale="linear",
        yscale="linear",
        figure_size=None,
        margins=(0.02, 0.02),
        line_width=1.0,
        open_plot=True,
        plot_format="png",
        offset=0,
        distinct_colors=None,
        colors=None,
        label_bands=False,
        legend_below_plot=False,
        legend_beside_plot=False,
        legend_anchor=None,
        flat_legend=False,
        legend_no_border=False,
        spectral_features=None,
        feature_color="black",
        legend_location="best",
        reference_on_top=True,
        relative_flux=False,
        x_limits=None,
        y_limits=None,
        label_aside_curve=False,
        label_position="left",
        additional_legend_text=None,
    ):
        """
        Plots the spectral energy distribution (SED) of the waveflux data, with various customization options
        for visualizing fluxes, uncertainties, and annotations such as bands, spectral features, and legends.
        The plot can include multiple curves corresponding to different waveflux data, each with options for
        error bars, scaling, and various plot adjustments.

        Parameters:
        -----------
        spec_uncertainty : bool, optional, default=False
            If True, includes the spectral uncertainty (error bars) in the plot.

        phot_uncertainty : bool, optional, default=False
            If True, includes the photometric uncertainty (error bars) in the plot.

        show_grid : bool, optional, default=False
            If True, displays gridlines on the plot.

        xscale : str, optional, default="linear"
            The scale for the x-axis. Options are "linear" and "log".

        yscale : str, optional, default="linear"
            The scale for the y-axis. Options are "linear" and "log".

        figure_size : tuple of float, optional, default=None
            The size of the figure (width, height) in inches.

        margins : tuple of float, optional, default=(0.02, 0.02)
            The margins around the plot as a fraction of the axis range.

        line_width : float, optional, default=1.0
            The line width for the plotted curves.

        open_plot : bool, optional, default=True
            If True, the plot will be opened after being saved.

        plot_format : str, optional, default="png"
            The format of the saved plot file. Common options are "png", "jpg", or "pdf".

        offset : float, optional, default=0
            The vertical offset between the curves in the plot.

        distinct_colors : int or None, optional, default=None
            The number of distinct colors for the curves. If None, it is determined based on the number of waveflux data.

        colors : list of str or None, optional, default=None
            A list of specific colors to use for the curves.

        label_bands : bool, optional, default=False
            If True, labels the photometric bands used by the reference or template.

        legend_below_plot : bool, optional, default=False
            If True, the legend will be placed below the plot.

        legend_beside_plot : bool, optional, default=False
            If True, the legend will be placed beside the plot.

        legend_anchor : tuple of float, optional, default=None
            The anchor point for the legend's placement.

        flat_legend : bool, optional, default=False
            If True, places the legend items in a single row without wrapping.

        legend_no_border : bool, optional, default=False
            If True, removes the border from the legend.

        spectral_features : list of dict, optional, default=None
            A list of spectral feature dictionaries to annotate on the plot. Each dictionary should include:
            - "label": The label for the feature (str)
            - "type": The type of feature ("band" for range or "line" for single wavelength)
            - "wavelengths": List of wavelengths defining the feature (for "band" type, a range of wavelengths; for "line" type, individual wavelengths)
            - "offset": The vertical offset for the feature label.

        feature_color : str, optional, default="black"
            The color of the spectral feature annotations.

        legend_location : str, optional, default="best"
            The location of the legend on the plot. Options include "best", "upper right", "lower left", etc.

        reference_on_top : bool, optional, default=True
            If True, the reference curve is placed on top of other curves in the plot.

        relative_flux : bool, optional, default=False
            If True, the flux is normalized to the reference flux and plotted as relative flux.

        x_limits : tuple of float, optional, default=None
            The x-axis limits as (xmin, xmax).

        y_limits : tuple of float, optional, default=None
            The y-axis limits as (ymin, ymax).

        label_aside_curve : bool, optional, default=False
            If True, labels the curves at the side of the plot (left or right).

        label_position : str, optional, default="left"
            The position of the curve labels when `label_aside_curve` is True. Options are "left", "right", or "both".

        additional_legend_text : tuple of (int, str), optional, default=None
            Adds additional text to the legend at the specified index.

        Returns:
        --------
        None

        Notes:
        ------
        - This method generates a plot for the spectral energy distribution (SED) from the waveflux data, optionally
          including uncertainties, labels, and spectral feature annotations.
        - The plot is saved to a file, and if `open_plot` is True, it is opened for viewing.
        - The plotting behavior can be customized with various optional parameters for axes scaling, figure size,
          color schemes, and more.
        - Spectral features can be annotated by providing a list of feature dictionaries. Each feature can be a band
          (range of wavelengths) or a line (individual wavelength).
        """
        number_of_waveflux = len(self.data)
        if figure_size:
            plt.figure(figsize=(figure_size))
        plt.rcParams.update({"font.size": 8, "font.family": "Arial", "axes.linewidth": 0.7, "axes.edgecolor": "silver"})
        plt.rcParams["axes.formatter.use_mathtext"] = True
        if colors:
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)
        else:
            if not distinct_colors:
                distinct_colors = number_of_waveflux if number_of_waveflux > 5 else 5
            distinct_colors = distinct_colors if distinct_colors else number_of_waveflux
            plt.rcParams["axes.prop_cycle"] = plt.cycler(
                "color", plt.cm.nipy_spectral(np.linspace(0, 1, distinct_colors))
            )
        i = 0
        scale = 1
        normalized = False
        for waveflux in self.data:
            is_reference = waveflux.reference
            normalized = waveflux.normalized
            label = waveflux.label
            data = waveflux.data
            if len(data) == 0:
                continue
            i += 1
            delta = offset * i
            if reference_on_top:
                zorder = 1 if is_reference else 0
            else:
                zorder = 0 if is_reference else 1
            data.sort("Wavelength")
            wavelength = data["Wavelength"]
            flux = data["Flux"]
            uncertainty = data["Uncertainty"]
            if relative_flux:
                if is_reference:
                    exp = self.get_exponent(np.nanmean(flux))
                    scale = 10**exp
                flux = (flux / scale) + 1
                ratio = np.nanmean(flux) / np.nanmean(data["Flux"])
                uncertainty = ratio * uncertainty
            if is_reference or number_of_waveflux == 1:
                ref_wavelength = wavelength
                ref_flux = flux
            if waveflux.photometry:
                used_bands = waveflux.used_bands
                if is_reference and label_bands and used_bands:
                    if flat_legend:
                        label = label + ": " if len(label) > 0 else ""
                        label += ", ".join(used_bands)
                    else:
                        label = label + ":\n" if len(label) > 0 else ""
                        for i, used_band in enumerate(used_bands):
                            sep = ",\n" if i % 3 == 0 else ", "
                            sep = "" if i == 0 else sep
                            label += sep + used_band
                width = line_width if waveflux.connect_markers else 0
                if phot_uncertainty:
                    uncertainties = np.stack((uncertainty, uncertainty))
                    curves = plt.errorbar(
                        wavelength,
                        flux + delta,
                        marker="o",
                        ms=2,
                        lw=width,
                        yerr=np.abs(uncertainties + delta),
                        capsize=1.5,
                        capthick=0.3,
                        elinewidth=0.3,
                        label=label,
                        zorder=zorder,
                    )
                else:
                    curves = plt.errorbar(
                        wavelength,
                        flux + delta,
                        marker="o",
                        ms=2,
                        lw=width,
                        capsize=1.5,
                        capthick=0.3,
                        elinewidth=0.3,
                        label=label,
                        zorder=zorder,
                    )
            else:
                curves = plt.plot(wavelength, flux + delta, lw=line_width, label=label, zorder=zorder)
                if spec_uncertainty:
                    color = curves[-1].get_color()
                    flux_lower_bound = flux - uncertainty
                    flux_upper_bound = flux + uncertainty
                    plt.fill_between(
                        wavelength,
                        flux_lower_bound + delta,
                        flux_upper_bound + delta,
                        color=color,
                        alpha=0.2,
                        label="Uncertainty",
                    )
            if label_aside_curve:
                color = curves[-1].get_color()
                if label_position == "left" or label_position == "both":
                    plt.text(
                        wavelength[0],
                        (flux + delta)[0],
                        waveflux.label,
                        fontsize=7,
                        color=color,
                        ha="right",
                        va="bottom",
                    )
                if label_position == "right" or label_position == "both":
                    plt.text(
                        wavelength[-1],
                        (flux + delta)[0],
                        waveflux.label,
                        fontsize=7,
                        color=color,
                        ha="left",
                        va="bottom",
                    )

        plt.xscale(xscale)
        plt.yscale(yscale)

        plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        plt.gca().xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

        if yscale == "log":
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            plt.gca().yaxis.set_major_formatter(formatter)
            # plt.gca().yaxis.set_minor_formatter(formatter)
            # plt.gca().yaxis.set_minor_locator(plt.MaxNLocator(1))

        if show_grid:
            plt.grid(color="grey", alpha=0.3, linestyle="-.", linewidth=0.2, axis="both", which="both")
            plt.gca().tick_params(axis="x", which="both", length=0)
            plt.gca().tick_params(axis="y", which="both", length=0)
        else:
            plt.gca().tick_params(axis="x", which="both", length=2, width=0.2)
            plt.gca().tick_params(axis="y", which="both", length=2, width=0.2)

        plt.margins(x=margins[0], y=margins[1])

        plt.xlabel(f"Wavelength [{u.um:latex}]")
        if relative_flux:
            plt.ylabel("Relative Flux")
        elif normalized:
            plt.ylabel("Normalized Flux")
        else:
            plt.ylabel(f"Flux [{self.flux_unit:latex_inline}]")

        if x_limits:
            xmin, xmax = x_limits
            if xmin:
                plt.xlim(left=xmin)
            if xmax:
                plt.xlim(right=xmax)

        if y_limits:
            ymin, ymax = y_limits
            if ymin:
                plt.ylim(bottom=ymin)
            if ymax:
                plt.ylim(top=ymax)

        legend_fontsize = 7.5

        if not label_aside_curve:
            if additional_legend_text:
                handles, labels = plt.gca().get_legend_handles_labels()
                index = additional_legend_text[0]
                labels.insert(index, additional_legend_text[1])
                handles.insert(index, plt.Line2D([0], [0], color="None"))
            else:
                handles, labels = None, None
            if legend_below_plot:
                anchor = legend_anchor if legend_anchor else (0.5, -0.2)
                if handles and labels:
                    legend = plt.legend(handles, labels, bbox_to_anchor=anchor, loc="center", fontsize=legend_fontsize)
                else:
                    legend = plt.legend(bbox_to_anchor=anchor, loc="center", fontsize=legend_fontsize)
                legend.get_frame().set_facecolor("none")
                legend.get_frame().set_linewidth(0)
            elif legend_beside_plot:
                anchor = legend_anchor if legend_anchor else (1.1, 0.5)
                if handles and labels:
                    legend = plt.legend(handles, labels, bbox_to_anchor=anchor, loc="center", fontsize=legend_fontsize)
                else:
                    legend = plt.legend(bbox_to_anchor=anchor, loc="center", fontsize=legend_fontsize)
                legend.get_frame().set_facecolor("none")
                legend.get_frame().set_linewidth(0)
            else:
                if handles and labels:
                    legend = plt.legend(handles, labels, loc=legend_location, fontsize=legend_fontsize)
                else:
                    legend = plt.legend(loc=legend_location, fontsize=legend_fontsize)
                if legend_no_border:
                    legend.get_frame().set_facecolor("none")
                    legend.get_frame().set_linewidth(0)
                else:
                    legend.get_frame().set_facecolor("white")
                    legend.get_frame().set_boxstyle("Square")
                    legend.get_frame().set_linewidth(0.5)

        if self.target_name:
            plt.title(self.target_name)

        # Plot spectral features
        if spectral_features:
            # Check if feature range overlaps spectrum wavelength range
            features_matching_range = []
            min_wave, max_wave = np.nanmin(ref_wavelength), np.nanmax(ref_wavelength)
            for feature in spectral_features:
                min_wave_feature, max_wave_feature = feature["wavelengths"][0], feature["wavelengths"][-1]
                if min_wave_feature > max_wave or max_wave_feature < min_wave:
                    continue
                if min_wave_feature < min_wave and max_wave_feature > min_wave:
                    if feature["type"] == "band":
                        feature["wavelengths"][0] = min_wave
                    else:
                        feature["wavelengths"] = [w for w in feature["wavelengths"] if w >= min_wave]
                if max_wave_feature > max_wave and min_wave_feature < max_wave:
                    if feature["type"] == "band":
                        feature["wavelengths"][-1] = max_wave
                    else:
                        feature["wavelengths"] = [w for w in feature["wavelengths"] if w <= max_wave]
                features_matching_range.append(feature)

            # Find the highest flux value within each feature's wavelength range
            feature_flux_values = []
            for feature in features_matching_range:
                min_wave_feature, max_wave_feature = feature["wavelengths"][0], feature["wavelengths"][-1]
                indices = np.where((ref_wavelength >= min_wave_feature) & (ref_wavelength <= max_wave_feature))
                if len(indices[0]) == 0:
                    indices = self.find_nearest(ref_wavelength, (min_wave_feature + max_wave_feature) / 2)
                max_flux_value = np.nanmax(ref_flux[indices])
                feature_flux_values.append(max_flux_value)

            mean_flux = np.nanmean(ref_flux)
            exp = self.get_exponent(mean_flux)
            flux_offset = 10**exp
            for index, item in enumerate(feature_flux_values[1:], 1):
                prev_item = feature_flux_values[index - 1]
                delta = abs(prev_item - item)
                if delta < flux_offset / 2:
                    feature_flux_values[index] = item + flux_offset

            # Plot lines for spectral features at the highest flux values
            y_offset = 10**exp
            for i, feature in enumerate(features_matching_range):
                feature_label, feature_type, wavelengths = feature["label"], feature["type"], feature["wavelengths"]
                offset = feature["offset"] if "offset" in feature else 1
                max_flux_value = feature_flux_values[i]
                if feature_type == "band":
                    min_wave, max_wave = feature["wavelengths"][0], feature["wavelengths"][-1]
                    y = max_flux_value + y_offset * offset
                    plt.hlines(xmin=min_wave, xmax=max_wave, y=y, lw=0.5, color=feature_color, linestyle="-")
                    plt.text(
                        (min_wave + max_wave) / 2,
                        y,
                        f"{feature_label}",
                        fontsize=7,
                        color=feature_color,
                        ha="center",
                        va="bottom",
                    )
                else:
                    for wavelength in wavelengths:
                        line_offset = y_offset / 5
                        line_length = y_offset / 2
                        ymin = max_flux_value + line_offset * offset
                        plt.vlines(
                            x=wavelength, ymin=ymin, ymax=ymin + line_length, lw=0.5, color=feature_color, linestyle="-"
                        )
                    plt.text(
                        np.nanmean(wavelengths),
                        ymin + line_length,
                        f"{feature_label}",
                        fontsize=7,
                        color=feature_color,
                        ha="center",
                        va="bottom",
                    )

        # Save and open the plot
        target_name = self.target_name if self.target_name else "SED"
        target_name = target_name.replace(" ", "_")
        filename = join(self.directory, target_name + "." + plot_format)
        plt.savefig(filename, dpi=300, bbox_inches="tight", format=plot_format)
        plt.close()

        # Show plot
        if open_plot:
            self.start_file(filename)

    def to_flux_lambda(self, density_unit=u.AA, normalize_at_wavelength=None, subtract_continuum=None):
        for waveflux in self.data:
            data = waveflux.data
            wavelength = data["Wavelength"].value * u.um
            data["Flux"] = flux_to_flux_lambda(wavelength, data["Flux"].value * u.Jy, density_unit)
            data["Uncertainty"] = flux_to_flux_lambda(wavelength, data["Uncertainty"].value * u.Jy, density_unit)
            if normalize_at_wavelength:
                waveflux.normalize_at_wavelength(normalize_at_wavelength)
            if subtract_continuum:
                waveflux.subtract_continuum(subtract_continuum)
            self.flux_unit = u.erg / u.s / u.cm**2 / density_unit

    def get_reference(self):
        return self.data[0]

    def get_best_match(self):
        return self.data[1]

    def find_nearest(self, array, value):
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()

    def get_exponent(self, number):
        base10 = math.log10(abs(number))
        return math.floor(base10)

    def start_file(self, filename):
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "evince"
            subprocess.call([opener, filename])


class WaveFlux:

    VSA_BASE_URL = "http://vsa.roe.ac.uk:8080/vdfs/"

    def __init__(self, label="", wavelength=None, flux=None, uncertainty=None, spectrum=None, model_params=None):
        """
        Initialize a WaveFlux instance with spectral data.

        Parameters:
        - label (str, optional): A label or identifier for the spectrum. Default is an empty string.
        - wavelength (Quantity, optional): The wavelength values in any valid astropy wavelength unit.
        - flux (Quantity, optional): The flux values in any valid astropy flux unit.
        - uncertainty (Quantity, optional): The uncertainty (error) in flux values in the same unit as flux.
        - spectrum (Spectrum1D, optional): An astropy Spectrum1D instance representing the spectrum.

        Note:
        - If wavelength and flux are provided, the method creates a WaveFlux instance (with associated uncertainty, if provided).
        - If a spectrum is provided, the method extracts wavelength, flux, and uncertainty from the spectrum and creates a WaveFlux instance.
        - If no data is provided, an empty WaveFlux instance is created. This instance then can be used to add photometry to create an SED.
        - Units are converted as needed to ensure consistent representation.

        Example:
        ```
        # Initialize a WaveFlux instance with wavelength and flux data
        wave_flux = WaveFlux(label='Spectrum A', wavelength=wavelength, flux=flux, uncertainty=uncertainty)

        # Initialize a WaveFlux instance from a Spectrum1D instance
        wave_flux = WaveFlux(label='Spectrum B', spectrum=spectrum)

        # Initialize an empty WaveFlux instance
        wave_flux = WaveFlux(label='Photometry')
        ```
        """
        self.filter_service_url = "http://svo2.cab.inta-csic.es/theory/fps3/fps.php"
        self.label = label
        self.used_bands = []
        self.used_photometry = {}
        self.filter_info = {}
        self.reference = False
        self.photometry = False
        self.normalized = False
        self.connect_markers = True
        self.statistic = np.nan
        self.model_params = model_params

        # Override deprecated VSA access URLs
        Vsa.LOGIN_URL = self.VSA_BASE_URL + "DBLogin"
        Vsa.IMAGE_URL = self.VSA_BASE_URL + "GetImage"
        Vsa.ARCHIVE_URL = self.VSA_BASE_URL + "ImageList"
        Vsa.REGION_URL = self.VSA_BASE_URL + "WSASQL"

        if wavelength is not None and flux is not None:
            # Convert flux and uncertainty to Jansky
            flux = flux.to(u.Jy, equivalencies=u.spectral_density(wavelength))
            if uncertainty is None:
                uncertainty = np.empty(len(flux))
                uncertainty[:] = np.nan
            else:
                uncertainty = uncertainty.to(u.Jy, equivalencies=u.spectral_density(wavelength))

            # Convert wavelength to micron
            wavelength = wavelength.to(u.um, equivalencies=u.spectral())

            self.create_table(wavelength, flux, uncertainty)
        elif spectrum:
            wavelength = spectrum.spectral_axis

            # Convert flux and uncertainty to Jansky
            flux = spectrum.flux.to(u.Jy, equivalencies=u.spectral_density(wavelength))
            uncertainty = spectrum.uncertainty
            if uncertainty is None:
                uncertainty = np.empty(len(flux)) * u.Jy
                uncertainty[:] = np.nan
            else:
                uncertainty = uncertainty.array * u.Jy
                uncertainty = uncertainty.to(u.Jy, equivalencies=u.spectral_density(wavelength))

            # Convert wavelength to micron
            wavelength = wavelength.to(u.um, equivalencies=u.spectral())

            self.create_table(wavelength, flux, uncertainty)
        else:
            self.data = Table(names=["Wavelength", "Flux", "Uncertainty"], dtype=["f", "f", "f"])
            self.photometry = True

    def add(self, filter_id, magnitude, error=np.nan, band=None):
        """
        Add photometric data to the WaveFlux instance.

        Parameters:
        - filter_id (str): Identifier or name of the filter or photometric band suffixed with the magnitude system ('/AB' or '/Vega').
        - magnitude (float): Magnitude value for the given filter or band.
        - error (float, optional): Magnitude measurement error. Default is NaN.
        - band (str, optional): Identifier for the band corresponding to the filter, if applicable.

        Note:
        - The method calculates flux and its uncertainty based on the provided magnitude and error.
        - If the filter information is available in the internal cache, it uses cached values.
        - If not cached, the method retrieves filter information from an external source and caches it.
        - The wavelength and zeropoint for the filter are used to calculate flux.
        - The calculated flux and its upper and lower bounds are added to the WaveFlux instance.

        Example:
        ```
        # Add photometric data for the Gaia BP filter
        wave_flux.add('GAIA/GAIA3.Gbp/Vega', 12.0, 0.001, 'Gaia BP')
        ```
        """
        if np.isnan(magnitude):
            print(f"{filter_id} magnitude is NaN and has been skipped")
            return

        file_path = join(tempfile.gettempdir(), "filter_info.json")
        if not self.filter_info:
            try:
                self.filter_info = json.load(open(file_path, "r"))
            except:
                pass

        if filter_id in self.filter_info:
            zeropoint, wavelength = self.filter_info[filter_id]
        else:
            zeropoint, wavelength = retrieve_filter_info_by_name(filter_id)
            self.filter_info[filter_id] = (zeropoint, wavelength)
            try:
                json.dump(self.filter_info, open(file_path, "w"))
            except:
                pass

        wavelength *= u.um
        flux = magnitude_to_flux(zeropoint, magnitude)
        uncertainty = flux - magnitude_to_flux(zeropoint, magnitude + error)

        self.data.add_row([wavelength, flux, uncertainty])

        self.used_photometry[band] = [magnitude, error]

        if band:
            self.used_bands.append(band)

    def create_table(self, wavelength, flux, uncertainty):
        columns = [wavelength, flux, uncertainty]
        self.data = Table(columns, names=["Wavelength", "Flux", "Uncertainty"], dtype=["f", "f", "f"])

    def split(self, wave_ranges):
        """
        Split a WaveFlux instance into multiple parts based on specified wavelength ranges.

        This method splits the WaveFlux instance into multiple parts, each corresponding to
        a specified wavelength range defined by a list of (min_wave, max_wave) tuples. It uses
        the 'trim' method of the WaveFlux instance to extract each part.

        Parameters:
        wave_ranges (list): A list of (min_wave, max_wave) tuples, defining the wavelength
            ranges for splitting the WaveFlux instance.

        Returns:
        list: A list of spectrum parts, each represented as a tuple of wavelength and flux arrays.
        """
        parts = []
        for wave_range in wave_ranges:
            min_wave, max_wave = wave_range
            parts.append(self.trim(min_wave, max_wave))
        return parts

    def align_flux(self):
        wavelength = np.array(self.data["Wavelength"])
        flux = np.array(self.data["Flux"])
        self.data["Flux"] = align_flux(wavelength, flux)

    def merge(self, others):
        """
        Merge multiple WaveFlux instances into a single instance.

        This method merges the current WaveFlux instance with a list of other WaveFlux
        instances to create a single merged instance. It combines the wavelength and flux
        data of all instances and creates a new WaveFlux instance with the merged data.

        Parameters:
        others (list): A list of WaveFlux instances to be merged with the current instance.

        Returns:
        None
        """
        parts = []
        parts.append((self.data["Wavelength"], self.data["Flux"]))
        for other in others:
            parts.append((other.data["Wavelength"], other.data["Flux"]))
        wavelength, flux = merge(parts)
        parts = []
        parts.append((self.data["Wavelength"], self.data["Uncertainty"]))
        for other in others:
            parts.append((other.data["Wavelength"], other.data["Uncertainty"]))
        _, uncertainty = merge(parts)
        self.create_table(wavelength, flux, uncertainty)

    def trim(self, min_wave=None, max_wave=None):
        """
        Trim the data within the WaveFlux instance to a specified wavelength range.

        This method trims the wavelength and flux data within the current WaveFlux instance
        to a specified wavelength range defined by 'min_wave' and 'max_wave'. It also trims
        the lower and upper uncertainty data accordingly and updates the WaveFlux instance
        with the trimmed data.

        Parameters:
        min_wave (float): The minimum wavelength value of the trimming range.
        max_wave (float): The maximum wavelength value of the trimming range.

        Returns:
        None
        """
        wavelength = self.data["Wavelength"]
        if min_wave is None:
            min_wave = np.nanmin(wavelength)
        if max_wave is None:
            max_wave = np.nanmax(wavelength)
        wavelength, flux = trim(wavelength, self.data["Flux"], min_wave, max_wave)
        _, uncertainty = trim(wavelength, self.data["Uncertainty"], min_wave, max_wave)
        self.create_table(wavelength, flux, uncertainty)

    def smooth(self, width):
        """
        Smooth the flux and uncertainty data within the WaveFlux instance.

        This method applies a smoothing operation to the flux and uncertainty data within
        the current WaveFlux instance using a specified 'width'. It updates the instance
        with the smoothed data.

        Parameters:
        width (float): The width of the smoothing window.

        Returns:
        None
        """
        wavelength = self.data["Wavelength"].quantity
        flux = smooth(self.data["Flux"].quantity, width)
        uncertainty = smooth(self.data["Uncertainty"].quantity, width)
        self.create_table(wavelength, flux, uncertainty)

    def normalize(self, reference):
        """
        Normalize the flux and uncertainty data within the WaveFlux instance.

        This method normalizes the flux and uncertainty data within the current WaveFlux instance
        based on a reference WaveFlux instance. The method updates the instance with the normalized data.

        Parameters:
        reference (WaveFlux instance): A reference WaveFlux instance to normalize against.

        Returns:
        None
        """
        ref_wavelength = reference.data["Wavelength"]
        ref_flux = reference.data["Flux"]
        wavelength = self.data["Wavelength"]
        flux = self.data["Flux"]
        uncertainty = self.data["Uncertainty"]
        flux, uncertainty = normalize(ref_wavelength, ref_flux, wavelength, flux, uncertainty)
        self.create_table(wavelength, flux, uncertainty)

    def normalize_at_wavelength(self, value):
        """
        Normalize the flux at a specific wavelength value within the WaveFlux instance.

        This method normalizes the flux and uncertainty data within the current WaveFlux instance
        by dividing it by the flux value at the specified wavelength value ('value'). It updates
        the instance with the normalized data.

        Parameters:
        value (float): The wavelength value at which the flux normalization is performed.

        Returns:
        None
        """
        self.normalized = True
        wavelength = self.data["Wavelength"]
        flux = normalize_at_wavelength(wavelength, self.data["Flux"], value)
        ratio = np.nanmean(flux) / np.nanmean(self.data["Flux"])
        uncertainty = ratio * self.data["Uncertainty"]
        self.create_table(wavelength, flux, uncertainty)

    def subtract_continuum(self, degree=3):
        """
        Subtract the continuum from the spectrum.

        Parameters:
            wavelengths (array-like): Array of wavelengths.
            spectrum (array-like): Array of spectral intensities.
            continuum (array-like): Array of continuum values.

        Returns:
            array-like: Resulting spectrum after subtracting the continuum.
        """
        wavelength = self.data["Wavelength"]
        flux = self.data["Flux"]
        uncertainty = self.data["Uncertainty"]
        continuum = estimate_continuum(wavelength, flux, degree)
        self.create_table(wavelength, flux - continuum, uncertainty)

    def shift(self, reference):
        """
        Shift the flux of the WaveFlux instance to match a reference WaveFlux instance.

        This method shifts the flux data within the current WaveFlux instance to match the
        flux of a reference WaveFlux instance. It updates the instance with the shifted data.

        Parameters:
        reference (WaveFlux instance): A reference WaveFlux instance whose flux will be used
            for shifting.

        Returns:
        None
        """
        data = reference.data
        wavelength = self.data["Wavelength"]
        flux = shift(data["Wavelength"], data["Flux"], self.data["Wavelength"], self.data["Flux"])
        ratio = np.nanmean(flux) / np.nanmean(self.data["Flux"])
        uncertainty = ratio * self.data["Uncertainty"]
        self.create_table(wavelength, flux, uncertainty)

    def scale(self, reference):
        """
        Scale the flux values of the current WaveFlux instance to match a reference WaveFlux instance.

        This method scales the flux values of the current WaveFlux instance to match the reference WaveFlux instance.
        It calculates a scaling factor based on the mean flux values of both spectra and applies
        this scaling factor to the current spectrum. It updates the instance with the scaled data.

        Parameters:
        reference (WaveFlux): The reference WaveFlux instance to which the current spectrum will be scaled.

        Returns:
        None
        """
        data = reference.data
        wavelength = self.data["Wavelength"]
        flux = scale(data["Flux"], self.data["Flux"])
        ratio = np.nanmean(flux) / np.nanmean(self.data["Flux"])
        uncertainty = ratio * self.data["Uncertainty"]
        self.create_table(wavelength, flux, uncertainty)

    def scale_to_distance(self, value, error, new_distance=10, parallax=True):
        """
        Scale the flux values of the current WaveFlux instance to a new distance.

        This method scales the flux values and associated uncertainties of the current WaveFlux instance to a new distance
        using the inverse square law, taking into account the original flux, its error, the original distance, its error,
        and the desired new distance. It updates the instance with the scaled data.

        Parameters:
        -----------
        distance : float
            The original distance in parsecs at which flux measurements were taken.
        distance_error : float
            The error associated with the original distance measurements (parsecs).
        new_distance : float, optional
            The desired new distance in parsecs to which flux measurements should be scaled.
            Default is 10 parsecs.

        Returns:
        --------
        None
        """
        if parallax:
            distance, distance_error = parallax_to_distance(value, error)
        else:
            distance, distance_error = value, error

        wavelength = self.data["Wavelength"]
        flux = self.data["Flux"]
        flux_error = self.data["Uncertainty"]
        flux, uncertainty = scale_to_distance(flux, flux_error, distance, distance_error, new_distance)
        self.create_table(wavelength, flux, uncertainty)

    def add_featured_photometry(self, ra, dec, radius=5):
        """
        Add photometric data from various surveys to the WaveFlux instance.

        This method attempts to add photometric data from multiple surveys,
        including Pan-STARRS, NSC, DES, 2MASS, UKIDSS, VHS, and AllWISE, to the
        WaveFlux instance based on the specified celestial coordinates.

        Parameters:
        - ra (float): The right ascension (RA) coordinate in degrees.
        - dec (float): The declination (Dec) coordinate in degrees.
        - radius (float, optional): The search radius in arcseconds. Default is 5.

        Note:
        - Photometric data is added if available for each survey, and the method
          attempts to add data from one survey if another does not provide results.

        Example:
        ```
        wave_flux = WaveFlux()
        wave_flux.add_featured_photometry(120.0, 45.0, radius=10)
        ```
        """
        if not self.add_panstarrs_photometry(ra, dec, radius):
            if not self.add_nsc_photometry(ra, dec, radius):
                self.add_des_photometry(ra, dec, radius)

        if not self.add_2mass_photometry(ra, dec, radius):
            if not self.add_ukidss_photometry(ra, dec, radius):
                self.add_vhs_photometry(ra, dec, radius)

        self.add_allwise_photometry(ra, dec, radius)

    def add_gaia_photometry(self, ra, dec, radius=5, omit_bands=[]):
        """
        Add Gaia photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the Gaia DR3 catalog for photometric data (BP, G, RP) based on the
        given celestial coordinates (RA, Dec) and a specified search radius. If the query returns
        valid photometric data, it adds the Gaia photometry to the WaveFlux instance. The method
        returns True if Gaia photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).

        Returns:
        bool: True if Gaia photometry was added successfully, False otherwise.
        """
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        v = Vizier(columns=["+_r", "Source", "BPmag", "e_BPmag", "Gmag", "e_Gmag", "RPmag", "e_RPmag"])
        tables = v.query_region(coords, radius=radius, catalog="I/355/gaiadr3")
        if tables:
            table = tables[0]
            row = table[0]
            bp = row["BPmag"]
            g = row["Gmag"]
            rp = row["RPmag"]
            e_bp = row["e_BPmag"]
            e_g = row["e_Gmag"]
            e_rp = row["e_RPmag"]
            len1 = len(self.data)
            if ~np.isnan(e_bp) and "BP" not in omit_bands:
                self.add("GAIA/GAIA3.Gbp/Vega", bp, e_bp, "Gaia BP")
            if ~np.isnan(e_g) and "G" not in omit_bands:
                self.add("GAIA/GAIA3.G/Vega", g, e_g, "Gaia G")
            if ~np.isnan(e_rp) and "RP" not in omit_bands:
                self.add("GAIA/GAIA3.Grp/Vega", rp, e_rp, "Gaia RP")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_panstarrs_photometry(self, ra, dec, radius=5, omit_bands=[]):
        """
        Add Pan-STARRS photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the Pan-STARRS DR2 catalog for photometric data (g, r, i, z, y) based
        on the given celestial coordinates (RA, Dec) and a specified search radius. If the query
        returns valid photometric data, it adds the Pan-STARRS photometry to the WaveFlux instance
        for the specified bands. The method returns True if Pan-STARRS photometry was successfully
        added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve
            ('grizy' for all bands, or a subset of them, e.g., 'gr' for g and r bands).

        Returns:
        bool: True if Pan-STARRS photometry was added successfully, False otherwise.
        """
        if dec < -30:
            return
        radius *= u.arcsec
        query_url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv"
        payload = {
            "ra": ra,
            "dec": dec,
            "radius": radius.to(u.deg).value,
            "nStackDetections.gte": 3,
            "sort_by": [("distance", "desc")],
        }
        response = requests.get(query_url, params=payload, timeout=300)

        try:
            table = ascii.read(response.text, format="csv")
        except:
            return False

        if len(table) > 0:
            row = table[0]
            g = row["gMeanPSFMag"]
            r = row["rMeanPSFMag"]
            i = row["iMeanPSFMag"]
            z = row["zMeanPSFMag"]
            y = row["yMeanPSFMag"]
            e_g = row["gMeanPSFMagErr"]
            e_r = row["rMeanPSFMagErr"]
            e_i = row["iMeanPSFMagErr"]
            e_z = row["zMeanPSFMagErr"]
            e_y = row["yMeanPSFMagErr"]
            len1 = len(self.data)
            if e_g > -999 and "g" not in omit_bands:
                self.add("PAN-STARRS/PS1.g/AB", g, e_g, "PS1 g")
            if e_r > -999 and "r" not in omit_bands:
                self.add("PAN-STARRS/PS1.r/AB", r, e_r, "PS1 r")
            if e_i > -999 and "i" not in omit_bands:
                self.add("PAN-STARRS/PS1.i/AB", i, e_i, "PS1 i")
            if e_z > -999 and "z" not in omit_bands:
                self.add("PAN-STARRS/PS1.z/AB", z, e_z, "PS1 z")
            if e_y > -999 and "y" not in omit_bands:
                self.add("PAN-STARRS/PS1.y/AB", y, e_y, "PS1 y")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_nsc_photometry(self, ra, dec, radius=5, omit_bands=[]):
        """
        Add NOIRLab Source Catalog (NSC) photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the NOIRLab Source Catalog (NSC) DR2 for photometric data (g, r, i, z, y) based on
        the given celestial coordinates (RA, Dec) and a specified search radius. If the query returns valid
        photometric data, it adds the NSC photometry to the WaveFlux instance for the specified bands. The
        method returns True if NSC photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('grizy' for
            all bands, or a subset of them, e.g., 'gr' for g and r bands).

        Returns:
        bool: True if NSC photometry was added successfully, False otherwise.
        """
        radius *= u.arcsec
        query_url = "https://datalab.noirlab.edu/tap/sync"
        adql = f"""
            SELECT id, ra, dec, gmag, rmag, imag, zmag, ymag, gerr, rerr, ierr, zerr, yerr
            FROM   nsc_dr2.object
            WHERE  't'=q3c_radial_query(ra, dec, {ra}, {dec}, {radius.to(u.deg).value})
            """
        payload = {"request": "doQuery", "lang": "ADQL", "format": "csv", "query": adql}
        response = requests.get(query_url, params=payload, timeout=300)
        table = ascii.read(response.text, format="csv")
        if table and len(table) > 0:
            table.add_column(calculate_separation(ra, dec, table["ra"], table["dec"]), name="distance")
            table.sort("distance")
            row = table[0]
            g = row["gmag"]
            r = row["rmag"]
            i = row["imag"]
            z = row["zmag"]
            y = row["ymag"]
            e_g = row["gerr"]
            e_r = row["rerr"]
            e_i = row["ierr"]
            e_z = row["zerr"]
            e_y = row["yerr"]
            len1 = len(self.data)
            if e_g < 9.9 and "g" not in omit_bands:
                self.add("CTIO/DECam.g/AB", g, e_g, "NSC g")
            if e_r < 9.9 and "r" not in omit_bands:
                self.add("CTIO/DECam.r/AB", r, e_r, "NSC r")
            if e_i < 9.9 and "i" not in omit_bands:
                self.add("CTIO/DECam.i/AB", i, e_i, "NSC i")
            if e_z < 9.9 and "z" not in omit_bands:
                self.add("CTIO/DECam.z/AB", z, e_z, "NSC z")
            if e_y < 9.9 and "y" not in omit_bands:
                self.add("CTIO/DECam.Y/AB", y, e_y, "NSC y")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_des_photometry(self, ra, dec, radius=5, omit_bands=[], dereddened=False):
        """
        Add Dark Energy Survey (DES) photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the Dark Energy Survey (DES) DR2 catalog for photometric data (g, r, i, z, Y) based
        on the given celestial coordinates (RA, Dec) and a specified search radius. If the query returns
        valid photometric data, it adds the DES photometry to the WaveFlux instance for the specified bands.
        The method returns True if DES photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('grizy' for
            all bands, or a subset of them, e.g., 'gr' for g and r bands).

        Returns:
        bool: True if DES photometry was added successfully, False otherwise.
        """
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        v = Vizier(
            columns=[
                "+_r",
                "DES",
                "gmag",
                "e_gmag",
                "rmag",
                "e_rmag",
                "imag",
                "e_imag",
                "zmag",
                "e_zmag",
                "Ymag",
                "e_Ymag",
                "gFlag",
                "rFlag",
                "iFlag",
                "zFlag",
                "yFlag",
                "gmag0",
                "rmag0",
                "imag0",
                "zmag0",
                "Ymag0",
            ]
        )
        tables = v.query_region(coords, radius=radius, catalog="II/371/des_dr2")
        if tables:
            table = tables[0]
            row = table[0]
            g = row["gmag0"] if dereddened else row["gmag"]
            r = row["rmag0"] if dereddened else row["rmag"]
            i = row["imag0"] if dereddened else row["imag"]
            z = row["zmag0"] if dereddened else row["zmag"]
            y = row["Ymag0"] if dereddened else row["Ymag"]
            e_g = row["e_gmag"]
            e_r = row["e_rmag"]
            e_i = row["e_imag"]
            e_z = row["e_zmag"]
            e_y = row["e_Ymag"]
            len1 = len(self.data)
            if row["gFlag"] <= 3 and "g" not in omit_bands:
                self.add("CTIO/DECam.g/AB", g, e_g, "DES g")
            if row["rFlag"] <= 3 and "r" not in omit_bands:
                self.add("CTIO/DECam.r/AB", r, e_r, "DES r")
            if row["iFlag"] <= 3 and "i" not in omit_bands:
                self.add("CTIO/DECam.i/AB", i, e_i, "DES i")
            if row["zFlag"] <= 3 and "z" not in omit_bands:
                self.add("CTIO/DECam.z/AB", z, e_z, "DES z")
            if row["yFlag"] <= 3 and "y" not in omit_bands:
                self.add("CTIO/DECam.Y/AB", y, e_y, "DES Y")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_sdss_photometry(self, ra, dec, radius=5, omit_bands=[]):
        """
        Add Sloan Digital Sky Survey (SDSS) photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the Sloan Digital Sky Survey (SDSS) DR17 for photometric data (u, g, r, i, z) based
        on the given celestial coordinates (RA, Dec) and a specified search radius. If the query returns
        valid photometric data, it adds the SDSS photometry to the WaveFlux instance for the specified bands.
        The method returns True if SDSS photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('ugriz' for
            all bands, or a subset of them, e.g., 'ug' for u and g bands).

        Returns:
        bool: True if SDSS photometry was added successfully, False otherwise.
        """
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        table = SDSS.query_region(
            coords,
            radius=radius,
            data_release=17,
            fields=["ra", "dec", "u", "g", "r", "i", "z", "Err_u", "Err_g", "Err_r", "Err_i", "Err_z"],
        )
        if table and len(table) > 0:
            table.add_column(calculate_separation(ra, dec, table["ra"], table["dec"]), name="distance")
            table.sort("distance")
            row = table[0]
            U = row["u"]
            g = row["g"]
            r = row["r"]
            i = row["i"]
            z = row["z"]
            e_u = row["Err_u"]
            e_g = row["Err_g"]
            e_r = row["Err_r"]
            e_i = row["Err_i"]
            e_z = row["Err_z"]
            len1 = len(self.data)
            if "u" not in omit_bands:
                self.add("SLOAN/SDSS.u/AB", U, e_u, "SDSS u")
            if "g" not in omit_bands:
                self.add("SLOAN/SDSS.g/AB", g, e_g, "SDSS g")
            if "r" not in omit_bands:
                self.add("SLOAN/SDSS.r/AB", r, e_r, "SDSS r")
            if "i" not in omit_bands:
                self.add("SLOAN/SDSS.i/AB", i, e_i, "SDSS i")
            if "z" not in omit_bands:
                self.add("SLOAN/SDSS.z/AB", z, e_z, "SDSS z")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_2mass_photometry(self, ra, dec, radius=5, omit_bands=[]):
        """
        Add Two Micron All-Sky Survey (2MASS) photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the Two Micron All-Sky Survey (2MASS) catalog for photometric data (J, H, K) based
        on the given celestial coordinates (RA, Dec) and a specified search radius. If the query returns
        valid photometric data, it adds the 2MASS photometry to the WaveFlux instance for the specified bands.
        The method returns True if 2MASS photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('JHK' for
            all bands, or a subset of them, e.g., 'JK' for J and K bands).

        Returns:
        bool: True if 2MASS photometry was added successfully, False otherwise.
        """
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        v = Vizier(columns=["+_r", "2MASS", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag"])
        tables = v.query_region(coords, radius=radius, catalog="II/246/out")
        if tables:
            table = tables[0].filled(np.nan)
            row = table[0]
            j = row["Jmag"]
            h = row["Hmag"]
            k = row["Kmag"]
            e_j = row["e_Jmag"]
            e_h = row["e_Hmag"]
            e_k = row["e_Kmag"]
            len1 = len(self.data)
            if ~np.isnan(e_j) and "J" not in omit_bands:
                self.add("2MASS/2MASS.J/Vega", j, e_j, "2MASS J")
            if ~np.isnan(e_h) and "H" not in omit_bands:
                self.add("2MASS/2MASS.H/Vega", h, e_h, "2MASS H")
            if ~np.isnan(e_k) and "K" not in omit_bands:
                self.add("2MASS/2MASS.Ks/Vega", k, e_k, "2MASS Ks")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_ukidss_photometry(self, ra, dec, radius=5, omit_bands=[], programme_id="LAS", dereddened=False):
        """
        Add United Kingdom Infrared Telescope (UKIRT) Infrared Deep Sky Survey (UKIDSS) photometry to the WaveFlux instance
        based on the specified coordinates.

        This method queries the UKIDSS DR11 catalog for infrared photometric data (Y, J, H, K) based on the given celestial
        coordinates (RA, Dec) and a specified search radius. If the query returns valid photometric data, it adds the
        UKIDSS photometry to the WaveFlux instance for the specified bands. The method returns True if UKIDSS photometry
        was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('JHK' for all bands, or
            a subset of them, e.g., 'JK' for J and K bands).
        programme_id (str, optional): Must be one of the following: 'LAS', 'GPS', 'GCS', 'DXS', 'UDS'

        Returns:
        bool: True if UKIDSS photometry was added successfully, False otherwise.
        """
        if dec < -5:
            return
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        table = Ukidss.query_region(coords, radius, database="UKIDSSDR11PLUS", programme_id=programme_id)
        if table:
            table.sort("distance")
            row = table[0]
            y = row["yAperMag3"] - row["aY"] if dereddened else row["yAperMag3"]
            j = row["jAperMag3"] - row["aJ"] if dereddened else row["jAperMag3"]
            h = row["hAperMag3"] - row["aH"] if dereddened else row["hAperMag3"]
            k = row["kAperMag3"] - row["aK"] if dereddened else row["kAperMag3"]
            e_y = row["yAperMag3Err"]
            e_h = row["hAperMag3Err"]
            e_j = row["jAperMag3Err"]
            e_h = row["hAperMag3Err"]
            e_k = row["kAperMag3Err"]
            len1 = len(self.data)
            if e_y > -999 and "Y" not in omit_bands:
                self.add("UKIRT/UKIDSS.Y/Vega", y, e_y, "UKIDSS Y")
            if e_j > -999 and "J" not in omit_bands:
                self.add("UKIRT/UKIDSS.J/Vega", j, e_j, "UKIDSS J")
            if e_h > -999 and "H" not in omit_bands:
                self.add("UKIRT/UKIDSS.H/Vega", h, e_h, "UKIDSS H")
            if e_k > -999 and "K" not in omit_bands:
                self.add("UKIRT/UKIDSS.K/Vega", k, e_k, "UKIDSS K")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_uhs_photometry(self, ra, dec, radius=5, omit_bands=[], dereddened=False):
        """
        Add United Kingdom Infrared Telescope (UKIRT) Hemisphere Survey (UHS) photometry to the WaveFlux instance
        based on the specified coordinates.

        This method queries the UHS DR2 catalog for infrared photometric data (J, K) based on the given celestial
        coordinates (RA, Dec) and a specified search radius. If the query returns valid photometric data, it adds the
        UKIDSS photometry to the WaveFlux instance for the specified bands. The method returns True if UHS photometry
        was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('JK' for all bands, or
            a subset of them, e.g., 'J' for J band).

        Returns:
        bool: True if UHS photometry was added successfully, False otherwise.
        """
        if dec < -5:
            return
        table = uhs.query_region(ra, dec, radius, database="UHSDR2")
        if table:
            row = table[0]
            j = row["JAPERMAG3"] - row["AJ"] if dereddened else row["JAPERMAG3"]
            k = row["KAPERMAG3"] - row["AK"] if dereddened else row["KAPERMAG3"]
            e_j = row["JAPERMAG3ERR"]
            e_k = row["KAPERMAG3ERR"]
            len1 = len(self.data)
            if e_j > -999 and "J" not in omit_bands:
                self.add("UKIRT/UKIDSS.J/Vega", j, e_j, "UHS J")
            if e_k > -999 and "K" not in omit_bands:
                self.add("UKIRT/UKIDSS.K/Vega", k, e_k, "UHS K")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_vhs_photometry(self, ra, dec, radius=5, omit_bands=[], dereddened=False):
        """
        Add VISTA Hemisphere Survey (VHS) photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the VHS DR6 catalog for infrared photometric data (Y, J, H, Ks) based on the given celestial
        coordinates (RA, Dec) and a specified search radius. If the query returns valid photometric data, it adds
        the VHS photometry to the WaveFlux instance for the specified bands. The method returns True if VHS
        photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('JHK' for all bands,
            or a subset of them, e.g., 'JK' for J and Ks bands).

        Returns:
        bool: True if VHS photometry was added successfully, False otherwise.
        """
        if dec > 5:
            return
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        Vsa.TIMEOUT = 3000
        table = Vsa.query_region(coords, radius, database="VHSDR6", programme_id="VHS")
        if table:
            table.sort("distance")
            row = table[0]
            y = row["yAperMag3"] - row["aY"] if dereddened else row["yAperMag3"]
            j = row["jAperMag3"] - row["aJ"] if dereddened else row["jAperMag3"]
            h = row["hAperMag3"] - row["aH"] if dereddened else row["hAperMag3"]
            k = row["ksAperMag3"] - row["aKs"] if dereddened else row["ksAperMag3"]
            e_y = row["yAperMag3Err"]
            e_j = row["jAperMag3Err"]
            e_h = row["hAperMag3Err"]
            e_k = row["ksAperMag3Err"]
            len1 = len(self.data)
            if e_y > -999 and "Y" not in omit_bands:
                self.add("Paranal/VISTA.Y/Vega", y, e_y, "VHS Y")
            if e_j > -999 and "J" not in omit_bands:
                self.add("Paranal/VISTA.J/Vega", j, e_j, "VHS J")
            if e_h > -999 and "H" not in omit_bands:
                self.add("Paranal/VISTA.H/Vega", h, e_h, "VHS H")
            if e_k > -999 and "K" not in omit_bands:
                self.add("Paranal/VISTA.Ks/Vega", k, e_k, "VHS Ks")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_viking_photometry(self, ra, dec, radius=5, omit_bands=[], dereddened=False):
        """
        Add VISTA Kilo-Degree Infrared Galaxy Survey (VIKING) photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the VIKING catalog for infrared photometric data (Y, J, H, Ks) based on the given celestial
        coordinates (RA, Dec) and a specified search radius. If the query returns valid photometric data, it adds
        the VIKING photometry to the WaveFlux instance for the specified bands. The method returns True if VIKING
        photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('JHK' for all bands,
            or a subset of them, e.g., 'JK' for J and Ks bands).

        Returns:
        bool: True if VIKING photometry was added successfully, False otherwise.
        """
        if dec > 5:
            return
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        Vsa.TIMEOUT = 3000
        table = Vsa.query_region(coords, radius, database="VIKINGDR5", programme_id="VIKING")
        if table:
            table.sort("distance")
            row = table[0]
            y = row["yAperMag3"] - row["aY"] if dereddened else row["yAperMag3"]
            j = row["jAperMag3"] - row["aJ"] if dereddened else row["jAperMag3"]
            h = row["hAperMag3"] - row["aH"] if dereddened else row["hAperMag3"]
            k = row["ksAperMag3"] - row["aKs"] if dereddened else row["ksAperMag3"]
            e_y = row["yAperMag3Err"]
            e_j = row["jAperMag3Err"]
            e_h = row["hAperMag3Err"]
            e_k = row["ksAperMag3Err"]
            len1 = len(self.data)
            if e_y > -999 and "Y" not in omit_bands:
                self.add("Paranal/VISTA.Y/Vega", y, e_y, "VIKING Y")
            if e_j > -999 and "J" not in omit_bands:
                self.add("Paranal/VISTA.J/Vega", j, e_j, "VIKING J")
            if e_h > -999 and "H" not in omit_bands:
                self.add("Paranal/VISTA.H/Vega", h, e_h, "VIKING H")
            if e_k > -999 and "K" not in omit_bands:
                self.add("Paranal/VISTA.Ks/Vega", k, e_k, "VIKING Ks")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_vvv_photometry(self, ra, dec, radius=5, omit_bands=[], dereddened=False):
        """
        Add VISTA Variables in the Via Lactea Survey (VVV) photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the VVV catalog for infrared photometric data (Y, J, H, Ks) based on the given celestial
        coordinates (RA, Dec) and a specified search radius. If the query returns valid photometric data, it adds
        the VVV photometry to the WaveFlux instance for the specified bands. The method returns True if VVV
        photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('JHK' for all bands,
            or a subset of them, e.g., 'JK' for J and Ks bands).

        Returns:
        bool: True if VVV photometry was added successfully, False otherwise.
        """
        if dec > 5:
            return
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        Vsa.TIMEOUT = 3000
        table = Vsa.query_region(coords, radius, database="VVVDR5", programme_id="VVV")
        if table:
            table.sort("distance")
            row = table[0]
            y = row["yAperMag3"] - row["aY"] if dereddened else row["yAperMag3"]
            j = row["jAperMag3"] - row["aJ"] if dereddened else row["jAperMag3"]
            h = row["hAperMag3"] - row["aH"] if dereddened else row["hAperMag3"]
            k = row["ksAperMag3"] - row["aKs"] if dereddened else row["ksAperMag3"]
            e_y = row["yAperMag3Err"]
            e_j = row["jAperMag3Err"]
            e_h = row["hAperMag3Err"]
            e_k = row["ksAperMag3Err"]
            len1 = len(self.data)
            if e_y > -999 and "Y" not in omit_bands:
                self.add("Paranal/VISTA.Y/Vega", y, e_y, "VVV Y")
            if e_j > -999 and "J" not in omit_bands:
                self.add("Paranal/VISTA.J/Vega", j, e_j, "VVV J")
            if e_h > -999 and "H" not in omit_bands:
                self.add("Paranal/VISTA.H/Vega", h, e_h, "VVV H")
            if e_k > -999 and "K" not in omit_bands:
                self.add("Paranal/VISTA.Ks/Vega", k, e_k, "VVV Ks")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_allwise_photometry(self, ra, dec, radius=5, omit_bands=[]):
        """
        Add AllWISE photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the AllWISE catalog for photometric data (W1, W2, W3) based on the given celestial
        coordinates (RA, Dec) and a specified search radius. If the query returns valid photometric data, it adds
        the AllWISE photometry to the WaveFlux instance for the specified bands. The method returns True if AllWISE
        photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('W1W2W3' for all bands,
            or a subset of them, e.g., 'W1W2' for W1 and W2 bands).

        Returns:
        bool: True if AllWISE photometry was added successfully, False otherwise.
        """
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        v = Vizier(
            columns=[
                "+_r",
                "AllWISE",
                "W1mag",
                "e_W1mag",
                "W2mag",
                "e_W2mag",
                "W3mag",
                "e_W3mag",
                "W4mag",
                "e_W4mag",
                "Jmag",
                "e_Jmag",
                "Hmag",
                "e_Hmag",
                "Kmag",
                "e_Kmag",
            ]
        )
        tables = v.query_region(coords, radius=radius, catalog="II/328/allwise")
        if tables:
            table = tables[0].filled(np.nan)
            row = table[0]
            w1 = row["W1mag"]
            w2 = row["W2mag"]
            w3 = row["W3mag"]
            w4 = row["W4mag"]
            e_w1 = row["e_W1mag"]
            e_w2 = row["e_W2mag"]
            e_w3 = row["e_W3mag"]
            e_w4 = row["e_W4mag"]
            len1 = len(self.data)
            if ~np.isnan(e_w1) and "W1" not in omit_bands:
                self.add("WISE/WISE.W1/Vega", w1, e_w1, "W1")
            if ~np.isnan(e_w2) and "W2" not in omit_bands:
                self.add("WISE/WISE.W2/Vega", w2, e_w2, "W2")
            if ~np.isnan(e_w3) and "W3" not in omit_bands:
                self.add("WISE/WISE.W3/Vega", w3, e_w3, "W3")
            if ~np.isnan(e_w4) and "W4" not in omit_bands:
                self.add("WISE/WISE.W4/Vega", w4, e_w4, "W4")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False

    def add_catwise_photometry(self, ra, dec, radius=5, omit_bands=[]):
        """
        Add CatWISE photometry to the WaveFlux instance based on the specified coordinates.

        This method queries the CatWISE catalog for photometric data (W1, W2) based on the given celestial
        coordinates (RA, Dec) and a specified search radius. If the query returns valid photometric data, it adds
        the CatWISE photometry to the WaveFlux instance for the specified bands. The method returns True if CatWISE
        photometry was successfully added, and False otherwise.

        Parameters:
        ra (float): The right ascension (RA) coordinate in degrees.
        dec (float): The declination (Dec) coordinate in degrees.
        radius (float, optional): The search radius in arcseconds (default is 5 arcseconds).
        bands (str, optional): A string containing the desired photometric bands to retrieve ('W1W2' for both bands,
            or a subset of them, e.g., 'W1' for W1 band).

        Returns:
        bool: True if CatWISE photometry was added successfully, False otherwise.
        """
        coords = SkyCoord(ra, dec, unit=u.deg)
        radius *= u.arcsec
        v = Vizier(columns=["+_r", "Name", "W1mproPM", "e_W1mproPM", "W2mproPM", "e_W2mproPM"])
        tables = v.query_region(coords, radius=radius, catalog="II/365/catwise")
        if tables:
            table = tables[0].filled(np.nan)
            row = table[0]
            w1 = row["W1mproPM"]
            w2 = row["W2mproPM"]
            e_w1 = row["e_W1mproPM"]
            e_w2 = row["e_W2mproPM"]
            len1 = len(self.data)
            if ~np.isnan(e_w1) and "W1" not in omit_bands:
                self.add("WISE/WISE.W1/Vega", w1, e_w1, "CW1")
            if ~np.isnan(e_w2) and "W2" not in omit_bands:
                self.add("WISE/WISE.W2/Vega", w2, e_w2, "CW2")
            len2 = len(self.data)
            if len2 > len1:
                return True
        return False


def create_obj_name(ra_deg, dec_deg):
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs")
    ra_str = coord.ra.to_string(unit="hour", sep="", precision=2, pad=True)[:4]
    dec_str = coord.dec.to_string(unit="deg", sep="", precision=2, alwayssign=True, pad=True)[:5]
    return f"J{ra_str}{dec_str}"


def flux_to_flux_lambda(wavelength, flux, density_unit):
    """
    Convert flux to flux per unit wavelength at a given wavelength.

    This function converts a flux with arbitrary units to flux density per unit wavelength
    using the provided wavelength. Both the input wavelength and flux can be in any valid
    astropy units.

    Parameters:
    wavelength (astropy.units.quantity.Quantity): The wavelength at which the flux is measured,
        specified in astropy-compatible units.
    flux (astropy.units.quantity.Quantity): The flux with arbitrary astropy-compatible units.

    Returns:
    astropy.units.quantity.Quantity: The flux density per unit wavelength in erg/s/cm^2/Ångstrom.
    """
    return flux.to(u.erg / u.s / u.cm**2 / density_unit, equivalencies=u.spectral_density(wavelength))


def magnitude_to_flux_lambda(zeropoint, wavelength, magnitude, density_unit=u.AA):
    """
    Convert magnitude to flux per unit wavelength at a given wavelength.

    This function converts a magnitude to flux density per unit wavelength in erg/s/cm^2/Ångstrom
    using the provided zero point magnitude and wavelength.

    Parameters:
    zeropoint (float): The zero point magnitude for the filter.
    wavelength (astropy.units.quantity.Quantity): The wavelength at which the magnitude is measured,
        specified in astropy-compatible units.
    magnitude (float): The magnitude to be converted to flux per unit wavelength.

    Returns:
    astropy.units.quantity.Quantity: The flux density per unit wavelength in erg/s/cm^2/Ångstrom.
    """
    flux = magnitude_to_flux(zeropoint, magnitude)
    return flux_to_flux_lambda(wavelength, flux, density_unit)


def magnitude_to_flux(zeropoint, magnitude):
    """
    Convert magnitude to flux using the zero point and wavelength.

    This function converts a magnitude to flux in units of Jansky (Jy) using the
    provided zero point and wavelength.

    Parameters:
    zeropoint (float): The zero point magnitude for the filter.
    magnitude (float): The magnitude to be converted to flux.

    Returns:
    astropy.units.quantity.Quantity: The flux in Jansky (Jy) as an Astropy Quantity.
    """
    return 10 ** (-0.4 * magnitude) * zeropoint * u.Jy


def retrieve_filter_info_by_facility(facility_name):
    """
    Retrieve filter information for a specific facility from the SVO Filter Profile Service.

    This function sends a request to the SVO Filter Profile Service to retrieve filter
    information for a specific astronomical facility or telescope. It parses the response
    and returns the filter information as a table.

    Parameters:
    facility_name (str): The name of the facility or telescope for which filter information is requested.

    Returns:
    astropy.table.Table: A table containing filter information for the specified facility.
    """
    params = {"Facility": facility_name}
    response = requests.get(SVO_URL, params=params, timeout=300)
    votable = parse_single_table(BytesIO(response.content))

    return Table(votable.array)


def retrieve_filter_info_by_instrument(instrument_name):
    """
    Retrieve filter information for a specific instrument from the SVO Filter Profile Service.

    This function sends a request to the SVO Filter Profile Service to retrieve filter
    information for a specific instrument. It parses the response and returns the filter
    information as a table.

    Parameters:
    instrument_name (str): The name of the instrument for which filter information is requested.

    Returns:
    astropy.table.Table: A table containing filter information for the specified instrument.
    """
    params = {"Instrument": instrument_name}
    response = requests.get(SVO_URL, params=params, timeout=300)
    votable = parse_single_table(BytesIO(response.content))

    return Table(votable.array)


def retrieve_filter_info_by_name(filter_id):
    """
    Retrieve filter information by its identifier from the SVO Filter Profile Service.

    This function sends a request to the SVO Filter Profile Service to retrieve
    information about a filter based on its identifier ('PhotCalID'). It parses
    the response and extracts the zero point and effective wavelength of the filter.

    Parameters:
    filter_id (str): The identifier of the filter to retrieve information for.

    Returns:
    tuple: A tuple containing two values:
        - zero_point (float): The zero point magnitude of the filter.
        - wavelength (float): The effective wavelength of the filter in microns.
    """
    params = {"PhotCalID": filter_id, "VERB": 0}
    response = requests.get(SVO_URL, params=params, timeout=300)
    votable = parse(BytesIO(response.content))

    zero_point = np.nan
    wavelength = np.nan
    for param in votable.iter_fields_and_params():
        if param.name == "ZeroPoint":
            zero_point = param.value
        if param.name == "WavelengthEff":
            wavelength = param.value

    return zero_point, wavelength / 10000  # micron


def determine_overlapping_wavelength_range(ref_wavelength, wavelength):
    """
    Determine the overlapping wavelength range between two arrays of wavelengths.

    Parameters:
    ref_wavelength (array-like): Array containing the wavelengths of the reference spectrum.
    wavelength (array-like): Array containing the wavelengths of another spectrum.

    Returns:
    tuple: A tuple containing the minimum and maximum wavelengths within the overlapping range.
    """
    ref_min_wave, ref_max_wave = np.nanmin(ref_wavelength), np.nanmax(ref_wavelength)
    tpl_min_wave, tpl_max_wave = np.nanmin(wavelength), np.nanmax(wavelength)
    min_wave = max(ref_min_wave, tpl_min_wave)
    max_wave = min(ref_max_wave, tpl_max_wave)
    return min_wave, max_wave


def align_flux(wavelength, flux):
    """
    Normalizes the flux of a spectrum based on a fitting model.

    This function aligns the flux of a spectrum by normalizing it to a model fit.
    The model is fit to the spectrum using a Gaussian-like function to estimate
    the scaling factor, which is then used to normalize the flux.

    The fitting process assumes that the spectrum can be approximated by a
    Gaussian-like profile and uses the `curve_fit` function from `scipy.optimize`
    to perform the fit. The scaling factor for the flux is determined from the
    amplitude of the fit.

    Parameters
    ----------
    wavelength : array-like
        The array of wavelength values (e.g., in nanometers or angstroms) corresponding
        to the flux data.

    flux : array-like
        The array of flux values that correspond to the wavelength array. These could
        represent spectral intensities, observed fluxes, or other similar quantities.

    Returns
    -------
    normalized_flux : numpy.ndarray
        The flux values normalized by the scaling factor obtained from the model fit.
        The result is a spectrum where the peak flux is normalized to 1.

    Notes
    -----
    - The initial guess for the fitting model is based on the maximum flux and
      the center of the spectrum. The width of the model is assumed to be a
      quarter of the total wavelength range.
    - The function uses the `curve_fit` method from `scipy.optimize` to fit the model.
    - The `curve_model` function used for fitting should be defined externally and
      should represent the functional form of the spectral peak (e.g., a Gaussian).
    """

    # Initial guesses for the parameters based on the data
    initial_guess = [np.max(flux), wavelength[np.argmax(flux)], (np.max(wavelength) - np.min(wavelength)) / 4]

    # Fit the model to each spectrum
    popt, _ = curve_fit(curve_model, wavelength, flux, p0=initial_guess)

    # Determine scaling factors based on model fits
    scaling_factor = popt[0]

    # Normalize the spectra
    normalized_flux = flux / scaling_factor

    return normalized_flux


def curve_model(wavelength, a, b, c):
    """
    Gaussian model function for fitting spectral data.

    This function models a Gaussian curve, which is commonly used to represent
    spectral peaks, absorption lines, or emission lines in astronomy and physics.

    The model is given by the equation:
    f(lambda) = a * exp(-(lambda - b)^2 / (2 * c^2))
    where:
    - a is the amplitude (peak value),
    - b is the center of the peak (mean of the Gaussian),
    - c is the standard deviation, which controls the width of the peak.

    Parameters
    ----------
    wavelength : array-like
        The independent variable, typically the wavelength values (e.g., in nanometers or angstroms).

    a : float
        The amplitude of the Gaussian function, representing the height of the peak.

    b : float
        The mean or center of the Gaussian, representing the position of the peak in the wavelength domain.

    c : float
        The standard deviation (width) of the Gaussian, which determines the spread of the peak.

    Returns
    -------
    numpy.ndarray
        The computed values of the Gaussian function for each wavelength, representing the model's output.

    Notes
    -----
    - This model is typically used to fit spectral peaks or profiles in data.
    - The Gaussian function is symmetric around the center b, and the width of the peak is determined by c.
    - This function can be used in fitting routines like `curve_fit` from `scipy.optimize` to estimate the parameters a, b, and c from data.
    """
    return a * np.exp(-((wavelength - b) ** 2) / (2 * c**2))


def merge(parts):
    """
    Merge a list of wavelength and flux parts into a single spectrum.

    This function takes a list of parts, where each part is a tuple containing
    wavelength and flux arrays. It concatenates these arrays to create a single
    spectrum.

    Parameters:
    parts (list): A list of tuples, each containing a wavelength array and a flux array.

    Returns:
    tuple: A tuple containing two numpy arrays: the merged wavelengths and merged fluxes.
    """
    wavelengths = []
    fluxes = []
    for part in parts:
        wavelength, flux = part
        wavelengths.append(wavelength)
        fluxes.append(flux)
    return np.concatenate(wavelengths), np.concatenate(fluxes)


def trim(wavelength, flux, min_wave, max_wave):
    """
    Trim a spectrum to a specified wavelength range.

    This function takes a wavelength array, a flux array, and minimum and maximum
    wavelength values. It trims the input spectrum to the specified wavelength range.

    Parameters:
    wavelength (numpy.ndarray): Array of wavelength values for the spectrum.
    flux (numpy.ndarray): Array of flux values corresponding to the wavelength array.
    min_wave (float): The minimum wavelength value for the trimmed range.
    max_wave (float): The maximum wavelength value for the trimmed range.

    Returns:
    tuple: A tuple containing two numpy arrays: the trimmed wavelength and flux arrays.
    """
    index_1 = np.nanargmin(np.abs(wavelength - min_wave))
    index_2 = np.nanargmin(np.abs(wavelength - max_wave))
    index_min = min(index_1, index_2)
    index_max = max(index_1, index_2)
    wave_slice = wavelength[index_min : index_max + 1]
    flux_slice = flux[index_min : index_max + 1]
    return wave_slice, flux_slice


def smooth(flux, window_size):
    """
    Smooth a given flux array using a moving average.

    This function applies a moving average smoothing to the input flux array
    using a specified window size.

    Parameters:
    flux (numpy.ndarray): The input flux array to be smoothed.
    window_size (int): The size of the moving average window.

    Returns:
    numpy.ndarray: A smoothed flux array of the same length as the input.
    """
    return np.convolve(flux, np.ones(window_size) / window_size, mode="same")


def normalize(ref_wavelength, ref_flux, wavelength, flux, uncertainty):
    """
    Normalize the flux and uncertainty of a spectrum to match a reference spectrum within an overlapping wavelength range.

    The normalization is performed by scaling the flux and uncertainty of the input spectrum (`flux`, `uncertainty`) to match
    the flux level of the reference spectrum (`ref_flux`) within the overlapping wavelength range.

    Parameters:
    ref_wavelength (array-like): Array containing the wavelengths of the reference spectrum.
    ref_flux (array-like): Array containing the flux values of the reference spectrum.
    wavelength (array-like): Array containing the wavelengths of the spectrum to be normalized.
    flux (array-like): Array containing the flux values of the spectrum to be normalized.
    uncertainty (array-like): Array containing the uncertainties (errors) associated with the flux values of the spectrum to be normalized.

    Returns:
    tuple: A tuple containing the normalized flux values and uncertainties.
    """
    min_wave, max_wave = determine_overlapping_wavelength_range(wavelength, ref_wavelength)
    this_wave, this_flux = trim(wavelength, flux, min_wave, max_wave)
    other_wave, other_flux = trim(ref_wavelength, ref_flux, min_wave, max_wave)
    interp_flux = np.interp(other_wave, this_wave, this_flux)
    ratio = np.nanmean(other_flux) / np.nanmean(interp_flux)
    flux *= ratio
    uncertainty *= ratio
    return flux, uncertainty


def normalize_at_wavelength(wavelength, flux, value):
    """
    Normalize a spectrum by dividing it by the intensity at a specified wavelength.

    This function normalizes a given spectrum by dividing it by the intensity at the
    specified 'value' wavelength. It performs linear interpolation between the closest
    lower and upper wavelengths to obtain the intensity at the desired wavelength.

    Parameters:
    wavelength (numpy.ndarray): Array of wavelength values for the spectrum.
    flux (numpy.ndarray): Array of flux values corresponding to the wavelength array.
    value (float): The wavelength at which to normalize the spectrum.

    Returns:
    numpy.ndarray: A normalized flux array, where each flux value is divided by
    the intensity at the specified 'value' wavelength.
    """
    # Find indices of closest lower and upper wavelengths
    lower_idx = np.searchsorted(wavelength, value, side="right") - 1
    upper_idx = lower_idx + 1

    # Calculate interpolation weights
    lambda_lower = wavelength[lower_idx]
    lambda_upper = wavelength[upper_idx]
    weight_upper = (value - lambda_lower) / (lambda_upper - lambda_lower)
    weight_lower = 1 - weight_upper

    # Interpolate intensities to get the intensity at the desired wavelength
    interpolated_flux = weight_lower * flux[lower_idx] + weight_upper * flux[upper_idx]

    # Normalize the spectrum
    return flux / interpolated_flux


def shift(ref_wavelength, ref_flux, wavelength, flux):
    """
    Shift the flux values of a spectrum to align with a reference spectrum.

    This function aligns the flux values of an input spectrum with a reference spectrum
    by calculating the mean difference between the reference flux and the interpolated
    flux values of the input spectrum. It then shifts the input spectrum by adding this
    mean difference to its flux values.

    Parameters:
    ref_wavelength (numpy.ndarray): Array of wavelength values for the reference spectrum.
    ref_flux (numpy.ndarray): Array of flux values corresponding to the reference wavelength array.
    wavelength (numpy.ndarray): Array of wavelength values for the input spectrum to be shifted.
    flux (numpy.ndarray): Array of flux values corresponding to the input wavelength array.

    Returns:
    numpy.ndarray: A shifted flux array, aligned with the reference spectrum.
    """
    int_flux = np.interp(ref_wavelength, wavelength, flux)
    mean_distance = np.nanmean(ref_flux - int_flux)
    return flux + mean_distance


def scale(ref_flux, flux):
    """
    Scale the flux values of a spectrum based on a reference spectrum.

    This function scales the input flux values to match the reference flux values. The scaling factor is determined by
    calculating the ratio of the mean of the reference flux to the mean of the input flux. The flux values are scaled
    proportionally, preserving their relative differences.
    If the scaling ratio is zero, infinity, or contains NaN values, the function returns the input flux unchanged.

    Parameters:
    ref_flux (array-like): Reference flux values.
    flux (array-like): Flux values to be scaled.

    Returns:
    array-like: Scaled flux values.
    """
    ratio = np.nanmean(ref_flux) / np.nanmean(flux)
    if np.isinf(ratio) or ratio == 0:
        return flux

    base10 = math.log10(abs(ratio))
    if np.isnan(base10):
        return flux

    exp = math.floor(base10)
    if exp == 0:
        return flux

    scale = 10 ** (exp)
    return flux * scale


def scale_to_distance(flux, flux_error, distance, distance_error, new_distance=10):
    """
    Scale flux values of a spectrum to a new distance.

    This function scales input flux values to a new distance using the inverse square law,
    taking into account the original flux, its error, the original distance, its error, and the
    desired new distance.

    Parameters:
    -----------
    flux : float or array-like
        Original flux measurements.

    flux_error : float or array-like
        Error associated with the original flux measurements.

    distance : float or array-like
        Original distance in parsecs at which flux measurements were taken.

    distance_error : float or array-like
        Error associated with the original distance measurements (parsecs).

    new_distance : float, optional
        Desired new distance in parsecs to which flux measurements should be scaled.
        Default is 10 parsecs.

    Returns:
    --------
    scaled_flux : float or array-like
        Scaled flux measurements at the new distance.

    scaled_flux_error : float or array-like
        Error associated with the scaled flux measurements at the new distance.
    """
    scaled_flux = flux * (distance / new_distance) ** 2
    scaled_flux_error = np.sqrt(
        ((distance / new_distance) ** 2 * flux_error) ** 2
        + (2 * (1 / new_distance) ** 2 * flux * distance * distance_error) ** 2
    )
    return scaled_flux, scaled_flux_error


def estimate_continuum(wavelength, flux, degree=3):
    """
    Estimate the continuum of a flux using polynomial fitting.

    Parameters:
        wavelength (array-like): Array of wavelengths.
        flux (array-like): Array of spectral intensities.
        degree (int): Degree of the polynomial to fit.

    Returns:
        array-like: Array of continuum values.
    """
    # Fit a polynomial to the flux
    poly_coefficients = np.polyfit(wavelength, flux, degree)
    continuum = np.polyval(poly_coefficients, wavelength)

    return continuum


def calculate_separation(target_ra, target_dec, catalog_ra, catalog_dec):
    """
    Calculate the angular separation between a target position and catalog positions.

    This function computes the angular separation, in arcseconds, between a target
    position defined by its right ascension (RA) and declination (Dec) and a set
    of catalog positions defined by their RA and Dec coordinates.

    Parameters:
    target_ra (float): The right ascension (RA) of the target position in degrees.
    target_dec (float): The declination (Dec) of the target position in degrees.
    catalog_ra (numpy.ndarray): Array of RA coordinates for the catalog positions in degrees.
    catalog_dec (numpy.ndarray): Array of Dec coordinates for the catalog positions in degrees.

    Returns:
    numpy.ndarray: An array of angular separations in arcseconds between the target
    position and each catalog position.
    """
    target_coords = SkyCoord([target_ra * u.deg], [target_dec * u.deg])
    catalog_coords = SkyCoord(catalog_ra, catalog_dec, unit="deg")
    return target_coords.separation(catalog_coords).arcsec


def parallax_to_distance(plx, e_plx):
    """
    Converts parallax to distance and calculates the corresponding uncertainty in distance.

    This function computes the distance (in parsecs) from the parallax (in milliarcseconds)
    using the formula:
    d = 1000 / p
    where d is the distance in parsecs and p is the parallax in milliarcseconds.
    It also computes the uncertainty in distance using error propagation:
    e_d = sqrt((1000 / p^2 * e_p)^2)
    where e_p is the uncertainty in the parallax.

    Parameters
    ----------
    plx : float
        The parallax in milliarcseconds (mas).

    e_plx : float
        The uncertainty in the parallax (in milliarcseconds).

    Returns
    -------
    dist : float
        The calculated distance in parsecs.

    e_dist : float
        The calculated uncertainty in the distance (in parsecs).

    Notes
    -----
    - The input parallax `plx` should be given in milliarcseconds (mas), and the output
      distance `dist` will be in parsecs.
    - The uncertainty in distance `e_dist` is computed using standard error propagation
      based on the parallax uncertainty `e_plx`.
    """
    dist = 1000 / plx
    e_dist = math.sqrt(pow((1000 / plx**2) * e_plx, 2))
    return dist, e_dist


class TemplateProvider:

    def __init__(self):
        self.module_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

    def get_Kesseli_2017_templates(self, wave_range=None, spt=None, smooth_window=None):
        print("Loading Kesseli+2017 templates ...")
        template_dir = join(self.module_path, "templates/Kesseli+2017/")
        template_paths = [
            template_dir + f for f in os.listdir(template_dir) if f.endswith(".fits") and isfile(join(template_dir, f))
        ]
        templates = []
        for template_path in template_paths:
            template_info = self.extract_Kesseli_2017_template_info(template_path)
            if spt and not template_info.startswith(spt):
                continue
            template = fits.open(template_path)
            data = template[1].data
            wavelength = (10 ** data["LogLam"]) * u.AA
            flux = data["Flux"] * u.erg / u.s / u.cm**2 / u.AA
            uncertainty = data["PropErr"] * u.erg / u.s / u.cm**2 / u.AA
            wf = WaveFlux(template_info, wavelength, flux, uncertainty)
            if wave_range:
                wf.trim(wave_range[0], wave_range[1])
            if smooth_window:
                wf.smooth(smooth_window)
            templates.append(wf)
        return templates

    def extract_Kesseli_2017_template_info(self, file_name):
        pattern = re.compile(r".+?([A-Z]\d+)_([-+]?\d+\.\d+)?_?(\w+)?\.fits")
        match = pattern.match(file_name)
        if match:
            spectral_type = match.group(1)
            metallicity = match.group(2)
            class_info = match.group(3)
            if metallicity:
                return f"{spectral_type}, [Fe/H]={metallicity}, {class_info}"
            else:
                return f"{spectral_type}, {class_info}"
        else:
            pattern_short = re.compile(r".+?([A-Z]\d+)\.fits")
            match = pattern_short.match(file_name)
            if match:
                spectral_type = match.group(1)
                return spectral_type
            else:
                return ""

    def get_Theissen_2022_templates(self, wave_range=None, spt=None, smooth_window=None):
        print("Loading Theissen+2022 templates ...")
        template_dir = join(self.module_path, "templates/Theissen+2022/")
        template_paths = [
            template_dir + f for f in os.listdir(template_dir) if f.endswith(".fits") and isfile(join(template_dir, f))
        ]
        templates = []
        for template_path in template_paths:
            template_info = self.extract_Theissen_2022_template_info(template_path)
            if spt and not template_info.startswith(spt):
                continue
            print(template_info)
            template = fits.open(template_path)
            data = template[0].data
            wave0, flux0, unc0 = data[3][0], data[3][1], data[3][2]
            wave1, flux1, unc1 = data[2][0], data[2][1], data[2][2]
            wave2, flux2, unc2 = data[1][0], data[1][1], data[1][2]
            wave3, flux3, unc3 = data[0][0], data[0][1], data[0][2]
            wave0, flux0 = trim(wave0, flux0, 1.00, 1.15)
            wave1, flux1 = trim(wave1, flux1, 1.15, 1.33)
            wave2, flux2 = trim(wave2, flux2, 1.43, 1.80)
            wave3, flux3 = trim(wave3, flux3, 1.93, 2.40)
            _, unc0 = trim(wave0, unc0, 1.00, 1.15)
            _, unc1 = trim(wave1, unc1, 1.15, 1.33)
            _, unc2 = trim(wave2, unc2, 1.43, 1.80)
            _, unc3 = trim(wave3, unc3, 1.93, 2.40)
            wavelength = np.concatenate([wave0, wave1, wave2, wave3]) * u.um
            flux = np.concatenate([flux0, flux1, flux2, flux3]) * u.Jy
            uncertainty = np.concatenate([unc0, unc1, unc2, unc3]) * u.Jy
            wf = WaveFlux(template_info, wavelength, flux, uncertainty, model_params={"spt": template_info})
            if wave_range:
                wf.trim(wave_range[0], wave_range[1])
            if smooth_window:
                wf.smooth(smooth_window)
            templates.append(wf)
        return templates

    def extract_Theissen_2022_template_info(self, file_name):
        pattern = re.compile(r".+?([A-Z]\d+).+?\.fits")
        match = pattern.match(file_name)
        if match:
            spectral_type = match.group(1)
            return f"{spectral_type}"
        else:
            return ""

    def get_Burgasser_2017_templates(self, wave_range=None, spt=None, smooth_window=None):
        print("Loading Burgasser+2017 templates ...")
        template_dir = join(self.module_path, "templates/Burgasser+2017/")
        template_paths = [
            template_dir + f for f in os.listdir(template_dir) if f.endswith(".fits") and isfile(join(template_dir, f))
        ]
        templates = []
        for template_path in template_paths:
            template_info = self.extract_Burgasser_2017_template_info(template_path)
            if spt and not template_info.startswith(spt):
                continue
            print(template_info)
            template = fits.open(template_path)
            data = template[0].data
            wavelength = data[0] * u.um
            flux = data[1] * u.Jy
            uncertainty = data[2] * u.Jy
            wf = WaveFlux(template_info, wavelength, flux, uncertainty, model_params={"spt": template_info})
            if wave_range:
                wf.trim(wave_range[0], wave_range[1])
            if smooth_window:
                wf.smooth(smooth_window)
            templates.append(wf)
        return templates

    def extract_Burgasser_2017_template_info(self, file_name):
        pattern = re.compile(r".+?([A-Z]\d+).+?\.fits")
        match = pattern.match(file_name)
        if match:
            spectral_type = match.group(1)
            return f"{spectral_type}"
        else:
            return ""


class Features:

    def __init__(self):
        self.features = [
            {"id": "h2o", "label": "H$_2$O", "type": "band", "wavelengths": [0.925, 0.95]},
            {"id": "h2o", "label": "H$_2$O", "type": "band", "wavelengths": [1.08, 1.2]},
            {"id": "h2o", "label": "H$_2$O", "type": "band", "wavelengths": [1.325, 1.55]},
            {"id": "h2o", "label": "H$_2$O", "type": "band", "wavelengths": [1.72, 2.14]},
            {"id": "ch4", "label": "CH$_4$", "type": "band", "wavelengths": [1.1, 1.24]},
            {"id": "ch4", "label": "CH$_4$", "type": "band", "wavelengths": [1.28, 1.44]},
            {"id": "ch4", "label": "CH$_4$", "type": "band", "wavelengths": [1.6, 1.76]},
            {"id": "ch4", "label": "CH$_4$", "type": "band", "wavelengths": [2.2, 2.35]},
            {"id": "co", "label": "CO", "type": "band", "wavelengths": [2.29, 2.39]},
            {"id": "tio", "label": "TiO", "type": "band", "wavelengths": [0.6569, 0.6852]},
            {"id": "tio", "label": "TiO", "type": "band", "wavelengths": [0.705, 0.727]},
            {"id": "tio", "label": "TiO", "type": "band", "wavelengths": [0.76, 0.8]},
            {"id": "tio", "label": "TiO", "type": "band", "wavelengths": [0.825, 0.831]},
            {"id": "tio", "label": "TiO", "type": "band", "wavelengths": [0.845, 0.86]},
            {"id": "vo", "label": "VO", "type": "band", "wavelengths": [1.04, 1.08]},
            {"id": "young vo", "label": "VO", "type": "band", "wavelengths": [1.17, 1.2]},
            {"id": "cah", "label": "CaH", "type": "band", "wavelengths": [0.6346, 0.639]},
            {"id": "cah", "label": "CaH", "type": "band", "wavelengths": [0.675, 0.705]},
            {"id": "crh", "label": "CrH", "type": "band", "wavelengths": [0.8611, 0.8681]},
            {"id": "feh", "label": "FeH", "type": "band", "wavelengths": [0.8692, 0.875]},
            {"id": "feh", "label": "FeH", "type": "band", "wavelengths": [0.98, 1.03]},
            {"id": "feh", "label": "FeH", "type": "band", "wavelengths": [1.19, 1.25]},
            {"id": "feh", "label": "FeH", "type": "band", "wavelengths": [1.57, 1.64]},
            {"id": "h2", "label": "H$_2$", "type": "band", "wavelengths": [1.5, 2.4]},
            {"id": "sb", "label": "*", "type": "band", "wavelengths": [1.6, 1.64]},
            {"id": "h", "label": "H I", "type": "line", "wavelengths": [1.004, 1.005]},
            {"id": "h", "label": "H I", "type": "line", "wavelengths": [1.093, 1.094]},
            {"id": "h", "label": "H I", "type": "line", "wavelengths": [1.281, 1.282]},
            {"id": "h", "label": "H I", "type": "line", "wavelengths": [1.944, 1.945]},
            {"id": "h", "label": "H I", "type": "line", "wavelengths": [2.166, 2.166]},
            {"id": "na", "label": "Na I", "type": "line", "wavelengths": [0.8186, 0.8195]},
            {"id": "na", "label": "Na I", "type": "line", "wavelengths": [1.136, 1.137]},
            {"id": "na", "label": "Na I", "type": "line", "wavelengths": [2.206, 2.209]},
            {"id": "cs", "label": "Cs I", "type": "line", "wavelengths": [0.8521, 0.8521]},
            {"id": "cs", "label": "Cs I", "type": "line", "wavelengths": [0.8943, 0.8943]},
            {"id": "rb", "label": "Rb I", "type": "line", "wavelengths": [0.78, 0.78]},
            {"id": "rb", "label": "Rb I", "type": "line", "wavelengths": [0.7948, 0.7948]},
            {"id": "mg", "label": "Mg I", "type": "line", "wavelengths": [1.7113336, 1.7113336]},
            {"id": "mg", "label": "Mg I", "type": "line", "wavelengths": [1.5745017, 1.577015]},
            {"id": "mg", "label": "Mg I", "type": "line", "wavelengths": [1.4881595, 1.4881847, 1.5029098, 1.5044356]},
            {"id": "mg", "label": "Mg I", "type": "line", "wavelengths": [1.1831422, 1.2086969]},
            {"id": "ca", "label": "Ca I", "type": "line", "wavelengths": [0.6573, 0.6573]},
            {"id": "ca", "label": "Ca I", "type": "line", "wavelengths": [2.26311, 2.265741]},
            {"id": "ca", "label": "Ca I", "type": "line", "wavelengths": [1.978219, 1.985852, 1.986764]},
            {"id": "ca", "label": "Ca I", "type": "line", "wavelengths": [1.931447, 1.94583, 1.951105]},
            {"id": "caii", "label": "Ca II", "type": "line", "wavelengths": [1.184224, 1.195301]},
            {"id": "caii", "label": "Ca II", "type": "line", "wavelengths": [0.985746, 0.993409]},
            {"id": "al", "label": "Al I", "type": "line", "wavelengths": [1.672351, 1.675511]},
            {"id": "al", "label": "Al I", "type": "line", "wavelengths": [1.3127006, 1.3154345]},
            {"id": "fe", "label": "Fe I", "type": "line", "wavelengths": [1.5081407, 1.549457]},
            {"id": "fe", "label": "Fe I", "type": "line", "wavelengths": [1.25604314, 1.28832892]},
            {
                "id": "fe",
                "label": "Fe I",
                "type": "line",
                "wavelengths": [
                    1.14254467,
                    1.15967616,
                    1.16107501,
                    1.16414462,
                    1.16931726,
                    1.18860965,
                    1.18873357,
                    1.19763233,
                ],
            },
            {"id": "k", "label": "K I", "type": "line", "wavelengths": [0.7699, 0.7665]},
            {"id": "k", "label": "K I", "type": "line", "wavelengths": [1.169, 1.177]},
            {"id": "k", "label": "K I", "type": "line", "wavelengths": [1.244, 1.252]},
        ]

    def get_all_features(self):
        return self.features

    def get_m_features(self):
        return self.select_features(["k", "na", "feh", "tio", "co", "h2o", "h2"])

    def get_l_features(self):
        return self.select_features(["k", "na", "feh", "tio", "co", "h2o", "h2"])

    def get_t_features(self):
        return self.select_features(["k", "ch4", "h2o", "h2"])

    def get_young_features(self):
        return self.select_features(["vo", "young vo"])

    def get_binary_features(self):
        return self.select_features(["sb"])

    def select_features(self, to_select):
        features = []
        for feature in self.features:
            if feature["id"] in to_select:
                features.append(feature)
        return features
