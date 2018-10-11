import os
import sys
import argparse
import itertools

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize
from astropy.table import Table, vstack
from astropy.io import ascii
from astropy.stats import median_absolute_deviation, sigma_clip
from astropy.modeling import models, fitting
from astropy.io import ascii
import statop as stat
import functools
import scipy

sys.path.append(os.path.join(os.environ["THESIS"], "scripts"))
sys.path.append(os.path.join(os.environ["HOME"], "papers", "rotletter18"))
import path_config as paths
import read_catalog as catin
import hrplots as hr
import astropy_util as au
import catalog
import sed
import data_splitting as split
import biovis_colors as bc
import aspcap_corrections as aspcor
import rotation_consistency as rot
import sample_characterization as samp
import mist
import dsep
import data_cache as cache
import eclipsing_binaries as ebs
import paperexport

plt.style.use("presentation")

DOCUMENT_PATH = paths.Path.cwd()
FIGURE_PATH = DOCUMENT_PATH / "img"
PLOT_SUFFIX = "png"

Protstr = r"$P_{\mathrm{rot}}$"
Teffstr = r"$T_{\mathrm{eff}}$"
MKstr = r"$M_{Ks}$"

dpi=100
figsize=(12, 12)

def write_plot(filename):
    return paperexport.write_plot(filename, PLOT_SUFFIX, FIGURE_PATH)

@write_plot("gyrochronology")
def gyro_figure():
    '''Make a figure illustrating gyrochronology.

    This figure should have the Pleiades, M34, and the Hyades for
    illustration.'''
    pleiades = catin.read_Rebull_Pleiades_Periods()
    stauffer_photometry = catin.read_Stauffer_Pleiades_photometry()
    hyades = catin.read_Radick_87_Periods()
    m34 = catin.read_Meibom_M34_periods()

    def meibom_gyro(t, BV):
        '''Return the Meibom gyrochronology relation.'''
        n = 0.52
        a = 0.7
        b = 0.472
        c = 0.553
        return t**n * (a * (BV - b)**c)

    pleiades_with_phot = au.join_by_ra_dec(
        pleiades, stauffer_photometry, "RAdeg", "DEdeg", "ra", "dec")

    bvvals = np.linspace(0.3, 1.5, 100)

    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(
        hyades["(B-V)"], hyades["Period"], color=bc.green, marker=".", ls="", 
        label="Hyades")
    ax.plot(
        m34["(B-V)0"], m34["Prot"], color=bc.blue, marker=".", ls="",
        label="M34")
    ax.plot(
        pleiades_with_phot["B-V"], pleiades_with_phot["Per1"], color=bc.red, 
        marker=".", ls="", label="Pleiades")

    ax.plot(
        bvvals, meibom_gyro(600, bvvals), color="g", marker="", ls="-",
        label="T=600 Myr")
    ax.plot(
        bvvals, meibom_gyro(220, bvvals), color="g", marker="", ls="-",
        label="T=220 Myr")
    ax.plot(
        bvvals, meibom_gyro(125, bvvals), color="r", marker="", ls="-",
        label="T=125 Myr")
    ax.set_xlabel("B-V")
    ax.set_ylabel("Period (day)")
    ax.set_xlim(0.3, 1.7)
    ax.legend(loc="upper left")

@write_plot("lurie_synch")
def Lurie_synch():
    '''Demonstrate tidal synchronization in Kepler.'''
    lurie = catin.read_Lurie_periods()
    spotted = lurie[lurie["Class"] == "sp"]
    spotted = spotted[spotted["Note"] != "b"]

    f, ax = plt.subplots(1, 1, figsize=(24, 12))
    ax.plot(spotted["Porb"], spotted["PACF"], 'ko')
    ax.plot([0, 20], [0, 20], 'k-')
    ax.set_xlabel("Orbital period (day)")
    ax.set_ylabel("Rotation period (day)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_title("Kepler Eclipsing Binaries")

@write_plot("met_trend")
def dwarf_metallicity():
    '''Show the metallicity distribution of the cool dwarfs.'''
    full = cache.apogee_splitter_with_DSEP()
    full_data = full.subsample(["Dwarfs", "APOGEE Statistics Teff"])

    f, ax2 = plt.subplots(1,1, figsize=(12,12), sharex=True)
    minorLocator = AutoMinorLocator()
    xminorLocator = AutoMinorLocator()
    med_loc = 50
    bottom_1sig = 50-67/2
    top_1sig = 50+67/2
    median = np.percentile(full_data["FE_H"], med_loc)
    bottom_percent = np.percentile(full_data["FE_H"], bottom_1sig)
    top_percent = np.percentile(full_data["FE_H"], top_1sig)

    # Make the empirical metallicity correction.
    apo_dwarfs = full.subsample(["Dwarfs", "APOGEE MetCor Teff"])
    metcoeff = samp.flatten_MS_metallicity(
        apo_dwarfs["K Excess"], apo_dwarfs["FE_H"], deg=2)
    metcorrect = np.poly1d(metcoeff)

    metspace = np.linspace(-1.25, 0.46, 20)
    k_mets = samp.calc_model_mag_fixed_age_alpha(
        5000, metspace, "Ks", age=1e9)
    corrected_k_mets = k_mets + metcorrect(metspace)
    ref_k = samp.calc_model_mag_fixed_age_alpha(
        5000, median, "Ks", age=1e9)
#   V_mets = samp.calc_model_mag_fixed_age_alpha(
#       5000, metspace, "V", age=1e9)
#   ref_V = samp.calc_model_mag_fixed_age_alpha(
#       5000, median, "V", age=1e9)
    ax2.plot(metspace, k_mets - ref_k, color=bc.blue, ls="-", marker="",
             label=r"MIST $\mathit{Ks}$", lw=5)
    ax2.plot(metspace, corrected_k_mets - (ref_k + metcorrect(median)), 
             color=bc.orange, ls="-", marker="", 
             label=r"Empirical $\mathit{Ks}$", lw=5)
#   ax2.plot(metspace, V_mets - ref_V, color=bc.blue, ls="-", marker="",
#            label="V")
    ax2.plot(
        [median, median], [0.9, -0.3], color=bc.black, lw=3, marker="", ls="-")
    ax2.plot(
        [bottom_percent, bottom_percent], [0.9, -0.3], color=bc.black, lw=1, 
        marker="", ls="-")
    ax2.plot(
        [top_percent, top_percent], [0.9, -0.3], color=bc.black, lw=1, 
        marker="", ls="-")
    ax2.fill_between(
        [-1.25, -0.5], [0.9, 0.9], [-0.3, -0.3], edgecolor=bc.black, 
        facecolor="white", hatch="/")
    ax2.plot([-1.25, 0.5], [0.0, 0.0], 'k--')
    ax2.yaxis.set_minor_locator(minorLocator)
    hr.invert_y_axis(ax2)
    ax2.set_xlabel("[Fe/H]")
    ax2.set_ylabel("Vertical displacement")
    ax2.set_ylim(0.9, -0.3)
    ax2.set_xlim(-1.25, 0.5)
    ax2.legend(loc="center left")

@write_plot("bad_gyro")
def bad_gyro():
    '''Show where in gyrochronology the rapid rotators cause problems.'''
    mcq = cache.mcquillan_corrected_splitter()
    dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    rapid_indices = dwarfs["Prot"] < 7

    bmags = samp.calc_model_mag_fixed_age_feh_alpha(
        dwarfs["teff"], 0.0, "B", age=1e9, model="MIST v1.2")
    vmags = samp.calc_model_mag_fixed_age_feh_alpha(
        dwarfs["teff"], 0.0, "V", age=1e9, model="MIST v1.2")
    bvcolor = bmags - vmags

    def meibom_age(P, BV):
        '''Return the Meibom gyrochronology relation.'''
        n = 0.52
        a = 0.7
        b = 0.472
        c = 0.553
        return (np.log10(P) - np.log10(a) - b*np.log10(BV))/n

    ok_logages = meibom_age(
        dwarfs["Prot"][~rapid_indices], bvcolor[~rapid_indices])
    bad_logages = meibom_age(
        dwarfs["Prot"][rapid_indices], bvcolor[rapid_indices])

    # Recreating the diagram in Reinhold and Gizon
    logage_bins = np.linspace(0.5, 3.8, 34, endpoint=True)
    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    n, bins, patches = ax.hist(
        [ok_logages, bad_logages], bins=logage_bins, normed=False, 
        histtype="step", stacked=True, color=["black", "red"], lw=2)

    ax.set_xlabel("Log age (Myr)")
    ax.set_ylabel("N")
    ax.set_xlim(0.5, 3.8)


@write_plot("binary_limit")
def photometric_binary_limit():
    '''Plot the fraction of mass ratios detected as photometric binaries.'''
    refmass = 0.725
    refmet = 0.00
    minmass = 0.1
    solmet = mist.MISTIsochrone.isochrone_from_file(refmet)
    tab = solmet.iso_table(1e9)
    lowmass = tab[tab[solmet.mass_col] <= refmass]
    qvals = lowmass[solmet.mass_col] / max(lowmass[solmet.mass_col])
    refk = solmet.interpolate_isochrone_cols(
        1e9, [refmass], solmet.mass_col, mist.band_translation["Ks"])
    refteff = 10**solmet.interpolate_isochrone_cols(
        1e9, [refmass], solmet.mass_col, solmet.logteff_col)

    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    incl_limit = -0.2
    cons_limit = -0.3

    combined_k = sed.sum_binary_mag(refk, lowmass[mist.band_translation["Ks"]])
    kdiff = combined_k - refk
    ax.plot(qvals, kdiff, marker="", ls="-", color=bc.black)
    ax.plot([0, 0.702, 0.702], [cons_limit, cons_limit, 0], color=bc.violet, ls="--",
            marker="")
    ax.set_xlabel("Mass ratio (q)")
    ax.set_ylabel(r"{0}(MIST; $M_1+M_2$) - {0}(MIST; $M_1$)".format(MKstr))
    hr.invert_y_axis()

@write_plot("sed_fit")
def binary_characterization():
    '''Plot the SED difference between components.'''
    primary_teff = 6000
    iso = mist.MISTIsochrone.isochrone_from_file(0.0)
    primary_mass = iso.interpolate_isochrone_cols(
        1e9, np.log10([primary_teff]), iso.logteff_col, iso.mass_col)

    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    bands = ["V", "R", "J", "H", "Ks"]
    band_wv = np.array([0.545, 0.641, 1.22, 1.63, 2.19]) # Microns
    band_energies = 6.626e-27 * 3e10 / (band_wv / 1e4)
    primary_mags = np.zeros(len(bands))
    for i, band in enumerate(bands):
        primary_mags[i] = iso.interpolate_isochrone_cols(
            1e9, np.log10([primary_teff]), iso.logteff_col,
            mist.band_translation[band])
    ax.plot(band_wv, primary_mags, color="k", marker="d", ls="", ms=5,
            label="Primary")

    massratios = [0.9, 0.7, 0.3]
    colors = [bc.red, bc.green, bc.violet]
    for col, q in zip(colors, massratios):
        secondary_mass = q * primary_mass
        secondary_mags = np.zeros(len(bands))
        for i, band in enumerate(bands):
            secondary_mags[i] = iso.interpolate_isochrone_cols(
                1e9, [secondary_mass], iso.mass_col,
                mist.band_translation[band])
        combined_mag = (primary_mags - 2.5 * np.log10(
            1 + 10**(-0.4*(secondary_mags - primary_mags))))
        ax.plot(band_wv, combined_mag, color=col, marker="o", ls="", ms=3,
                label="q={0:.1f}".format(q))


    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Absolute Mag")
    ax.set_xscale("log")
    hr.invert_y_axis(ax)
    ax.legend(loc="lower right")


#############
# OLD STUFF #
#############

@write_plot("mcquillan_selection")
def mcquillan_selection_coordinates():
    mcq = catin.mcquillan_with_stelparms()
    nomcq = catin.mcquillan_nondetections_with_stelparms()

    f, ax2 = plt.subplots(
        1,1, dpi=dpi, figsize=figsize)
    teff_bin_edges = np.arange(4000, 7000, 50)
    mk_bin_edges = np.arange(-3, 8, 0.02)

    count_cmap = plt.get_cmap("viridis")
    count_cmap.set_under("white")
    mcq_gaia_hist, xedges, yedges = np.histogram2d(
        mcq["SDSS-Teff"], mcq["M_K"], bins=(teff_bin_edges, mk_bin_edges))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax2.imshow(mcq_gaia_hist.T, origin="lower", extent=extent,
               aspect=(extent[1]-extent[0])/(extent[3]-extent[2]),
               cmap=count_cmap, norm=Normalize(vmin=1))
    f.colorbar(im, ax=ax2)

    # Add a representative error bar.
    stacked_columns = ["SDSS-Teff", "M_K", "M_K_err1", "M_K_err2"]
    stacked_mcq = vstack([mcq[stacked_columns], nomcq[stacked_columns]])
    dwarfs = np.logical_and(
        stacked_mcq["SDSS-Teff"] < 5500, stacked_mcq["M_K"] > 2.95)
    teff_error=100
    median_k_errup = np.median(stacked_mcq[dwarfs]["M_K_err1"]) 
    median_k_errdown = np.median(stacked_mcq[dwarfs]["M_K_err2"])
    ax2.errorbar(
        [6500], [7.0], yerr=[[median_k_errdown], [median_k_errup]], 
        xerr=teff_error, elinewidth=3)

    ax2.set_xlim(7000, 4000)
    ax2.set_ylim(8.2, -3)
    ax2.set_ylabel(MKstr)
    ax2.set_xlabel("{0} (K)".format(Teffstr))
    ax2.set_title("Period detection density")

@write_plot("ages")
def age_isochrones():
    '''Plot age isochrones on the APOGEE sample.'''
    targs = cache.apogee_splitter_with_DSEP()
    dwarfs = targs.subsample(["Dwarfs"])

    f, ax2 = plt.subplots(1, 1, dpi=dpi, figsize=figsize)
    hr.absmag_teff_plot(
        dwarfs["TEFF"], dwarfs["K Excess"], marker=".", color=bc.black, ls="", 
        label="APOGEE Dwarfs", axis=ax2, alpha=0.2)
    hr.absmag_teff_plot(
        [3700], [0.3], yerr=[
            [np.median(dwarfs["K Excess Error Down"])], 
            [np.median(dwarfs["K Excess Error Up"])]],
        xerr=[np.median(dwarfs["TEFF_ERR"])], marker="", color=bc.black, ls="",
        label="", axis=ax2, alpha=1.0)
    # Plot the bins.
    teff_bin_edges = np.linspace(6000, 4000, 15+1)
    teff_bin_indices = np.digitize(dwarfs["TEFF"], teff_bin_edges)
    percentiles = np.zeros(len(teff_bin_edges)-1)
    med_teff = np.zeros(len(teff_bin_edges)-1)
    mads = np.zeros(len(teff_bin_edges)-1)
    for ind in range(1, len(teff_bin_edges)):
        tablebin = dwarfs[teff_bin_indices == ind]
        percentiles[ind-1] = np.percentile(
            tablebin["K Excess"], 100-25)
        med_teff[ind-1] = np.mean(tablebin["TEFF"])
        median = np.percentile(tablebin["K Excess"], 50)
        singles = tablebin[tablebin["K Excess"] > median]
        mads[ind-1] = np.median(
            np.abs(singles["K Excess"] - percentiles[ind-1]))
    hr.absmag_teff_plot(
        med_teff, percentiles, marker="o", color=bc.sky_blue, ls="-", 
        label="25th percentile", lw=2, yerr=mads)
    ax2.plot([7000, 3000], [0, 0], 'k-')
    ax2.plot([7000, 3000], [-0.75, -0.75], 'k--')
    ax2.plot([5250, 5250], [-1.5, 0.5], 'k:')
    ax2.set_xlim([6500, 3500])
    ax2.set_ylim(0.5, -1.3)
    ax2.set_xlabel("$T_{\mathrm{eff}}$ (K)")
    ax2.set_ylabel("$M_{Ks}$ - $M_{Ks}$ (MIST; 1 Gyr)")

@write_plot("met")
def collapsed_met_histogram():
    '''Plot the distribution of K Excesses in the cool, unevolved sample.'''
    targs = cache.apogee_splitter_with_DSEP()
    cooldwarfs = targs.subsample(["Dwarfs", "Cool Noev"])

    f, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(24, 12), sharex=True, sharey=True)
    cons_limit = -0.3
    arr1, bins, patches = ax1.hist(
        cooldwarfs["Corrected K Excess"], bins=60, color=bc.blue, alpha=0.5,
        range=(-1.6, 1.1), histtype="bar", label="")
    arr2, bins, patches = ax2.hist(
        cooldwarfs["Corrected K Solar"], bins=60, color=bc.red, alpha=0.5,
        range=(-1.6, 1.1), histtype="bar", label="")
    metarray = arr1
    nometarray = arr2
    singlemodel = models.Gaussian1D(100, 0, 0.1, bounds={
        "mean": (-0.5, 0.5), "stddev": (0.01, 0.5)})
    binarymodel = models.Gaussian1D(20, -0.75, 0.1, bounds={
        "mean": (-1.5, 0.0), "stddev":(0.01, 0.5)})
    dualmodel = singlemodel+binarymodel
    fitter = fitting.SLSQPLSQFitter()
    fittedmet = fitter(dualmodel, (bins[1:]+bins[:-1])/2, metarray)
    inputexcesses = np.linspace(-1.6, 1.1, 200)
    metmodel = fittedmet(inputexcesses)
    fittednomet = fitter(
        dualmodel, (bins[1:]+bins[:-1])/2, nometarray)
    nometmodel = fittednomet(inputexcesses)
    ax1.plot(inputexcesses, metmodel, color=bc.blue, ls="-", lw=3, marker="",
            label="[Fe/H] Corrected")
    ax1.plot(inputexcesses, nometmodel, color=bc.red, ls="-", lw=3, marker="",
            label="[Fe/H] = 0.08")
    ax2.plot(inputexcesses, metmodel, color=bc.blue, ls="-", lw=3, marker="")
    ax2.plot(inputexcesses, nometmodel, color=bc.red, ls="-", lw=3, marker="")
    ax1.plot(
        [cons_limit, cons_limit], [0, 100], marker="", ls="--", color=bc.violet, 
        lw=4, zorder=3)
    ax2.plot(
        [cons_limit, cons_limit], [0, 100], marker="", ls="--", color=bc.violet, 
        lw=4, zorder=3)
    ax1.set_xlabel(r"Corrected {0} Excess".format(MKstr))
    ax1.set_ylabel("N")
    ax2.set_xlabel(r"Corrected {0} Excess".format(MKstr))
    ax1.set_ylabel("")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left")

@write_plot("Pbins")
def mcquillan_rapid_rotator_bins():
    '''Plot the rapid rotator bins in the full McQuillan sample.'''
    mcq = cache.mcquillan_corrected_splitter()
    ebs = cache.eb_splitter_with_DSEP()
    dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    eb_dwarfs = ebs.subsample(["Dwarfs", "Right Statistics Teff"])

    periodbins = np.flipud(np.array([1.5, 7, 10]))
    f, axes = plt.subplots(
        1, 2, figsize=(24, 12), sharex=True, sharey=True)
    cons_limit = -0.3
    mcq_period_indices = np.digitize(dwarfs["Prot"], periodbins)
    eb_period_indices = np.digitize(eb_dwarfs["period"], periodbins)
    Protstr = r"$P_{\mathrm{rot}}$"
    titles = ["{1} > {0:g} day".format(periodbins[0], Protstr), 
              "{0:g} day < {2} <= {1:g} day".format(
                  periodbins[2], periodbins[1], Protstr)]
    eblabel="EB"
    rotlabel="McQuillan"
    for i, title, ax in zip(range(0, 4, 2), titles, np.ravel(axes)):
        mcq_periodbin = dwarfs[mcq_period_indices == i]
        eb_periodbin = eb_dwarfs[eb_period_indices == i]
        hr.absmag_teff_plot(
            mcq_periodbin["SDSS-Teff"], mcq_periodbin["Corrected K Excess"], 
            marker=".", color=bc.black, ls="", axis=ax, zorder=1, label=rotlabel)
        hr.absmag_teff_plot(
            eb_periodbin["SDSS-Teff"], eb_periodbin["Corrected K Excess"], 
            marker="*", color=bc.pink, ls="", ms=24, axis=ax, zorder=2,
            label=eblabel)
        eblabel = rotlabel = ""
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(title)
        ax.plot(
            [4000, 5250], [cons_limit, cons_limit], marker="", ls="--", color=bc.violet, 
            lw=6, zorder=3)
        ax.plot([3500, 6500], [-0.0, -0.0], 'k-', lw=2, zorder=4)
    axes[0].set_ylabel("Corrected $M_{Ks}$ Excess")
    axes[0].set_xlabel("$T_{\mathrm{eff}}$ (K)")
    axes[1].set_xlabel("$T_{\mathrm{eff}}$ (K)")
    axes[0].set_xlim(5250, 4000)
    axes[0].set_ylim(0.3, -1.25)
    axes[0].legend(loc="upper right")

@write_plot("dist")
def distribution_singles_binaries():
    '''Plot the distribution of singles and binaries.'''
    mcq = cache.mcquillan_corrected_splitter()
    nomcq = cache.mcquillan_nondetections_corrected_splitter()
    ebs = cache.eb_splitter_with_DSEP()
    mcq_dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    nomcq_dwarfs = nomcq.subsample(["Dwarfs", "Right Statistics Teff"])
    generic_columns = ["Corrected K Excess"]
    dwarfs = vstack([
        mcq_dwarfs[generic_columns], nomcq_dwarfs[generic_columns]])
    dwarf_ebs = ebs.subsample(["Dwarfs", "Right Statistics Teff"])

    f, ax = plt.subplots(
        1, 1, dpi=dpi, figsize=figsize, sharex=True)
    cons_limit = -0.3

    # Create the histograms
    period_bins, dp = np.linspace(1.5, 21.5, 20+1, retstep=True, endpoint=True)
    # These are for the conservative sample.
    singles03 =  mcq_dwarfs["Prot"][mcq_dwarfs["Corrected K Excess"] >= cons_limit]
    binaries03 =  mcq_dwarfs["Prot"][mcq_dwarfs["Corrected K Excess"] < cons_limit]
    single_ebs03 = dwarf_ebs["period"][dwarf_ebs["Corrected K Excess"] >= cons_limit]
    binary_ebs03 = dwarf_ebs["period"][dwarf_ebs["Corrected K Excess"] < cons_limit]
    binary_rot_hist03, _ = np.histogram(binaries03, bins=period_bins)
    binary_eb_hist03, _ = np.histogram(binary_ebs03, bins=period_bins)
    single_rot_hist03, _ = np.histogram(singles03, bins=period_bins)
    single_eb_hist03, _ = np.histogram(single_ebs03, bins=period_bins)
    full_rot_hist, _ = np.histogram(mcq_dwarfs["Prot"], bins=period_bins)
    full_eb_hist, _ = np.histogram(dwarf_ebs["period"], bins=period_bins)

    # The total number of binaries and singles in the full sample.
    summed_binaries03 = (
        np.count_nonzero(mcq_dwarfs["Corrected K Excess"] < cons_limit) + 
        np.count_nonzero(dwarf_ebs["Corrected K Excess"] < cons_limit) + 
        np.count_nonzero(nomcq_dwarfs["Corrected K Excess"] < cons_limit))
    summed_singles03 = (
        np.count_nonzero(mcq_dwarfs["Corrected K Excess"] >= cons_limit) + 
        np.count_nonzero(dwarf_ebs["Corrected K Excess"] >= cons_limit) + 
        np.count_nonzero(nomcq_dwarfs["Corrected K Excess"] >= cons_limit))

    # The binary fraction for the full sample
    fullsamp_frac03 = (summed_binaries03 /
        (len(mcq_dwarfs) + len(dwarf_ebs) + len(nomcq_dwarfs)))

    # Sum up the binaries, single, and total histograms
    total_binaries03 = binary_rot_hist03 + binary_eb_hist03
    total_singles03 = single_rot_hist03 + single_eb_hist03
    total = full_rot_hist + full_eb_hist

    # Measure the binary fraction
    frac03 = total_binaries03 / total
    frac_uppers03 = au.binomial_upper(total_binaries03, total) - frac03
    frac_lowers03 = frac03 - au.binomial_lower(total_binaries03, total)

    normalized_binaries03 = total_binaries03 / summed_binaries03
    normalized_binaries_upper03 = (
        (au.poisson_upper(total_binaries03, 1) - total_binaries03) / 
        summed_binaries03)
    normalized_binaries_lower03= (
        (total_binaries03 - au.poisson_lower(total_binaries03, 1)) / 
        summed_binaries03)
    normalized_singles03 = total_singles03 / summed_singles03
    normalized_singles_upper03 = (
        (au.poisson_upper(total_singles03, 1) - total_singles03) / 
        summed_singles03)
    normalized_singles_lower03= (
        (total_singles03 - au.poisson_lower(total_singles03, 1)) / 
        summed_singles03)

    ax.step(period_bins, np.append(normalized_binaries03, [0]), where="post", 
            color=bc.algae, ls="-", label="Photometric Binaries")
    ax.step(period_bins, np.append(normalized_singles03, [0]), where="post", 
            color=bc.purple, linestyle="--", label="Photometric Singles")
    ax.errorbar(period_bins[:-1]+dp/2-dp/10, normalized_binaries03,
                yerr=[normalized_binaries_lower03, normalized_binaries_upper03],
                color=bc.algae, ls="", marker="")
    ax.errorbar(period_bins[:-1]+dp/2+dp/10, normalized_singles03,
                yerr=[normalized_singles_lower03, normalized_singles_upper03],
                color=bc.purple, ls="", marker="")
    ax.set_xlabel("Rotation Period (day)")
    ax.set_ylabel("Normalized Period Distribution")
    ax.set_ylim(0, 0.035)
    ax.legend(loc="lower right")

@write_plot("phot_frac")
def binary_fractions_with_period():
    '''Measure the binary fraction with period.'''
    mcq = cache.mcquillan_corrected_splitter()
    nomcq = cache.mcquillan_nondetections_corrected_splitter()
    ebs = cache.eb_splitter_with_DSEP()
    mcq_dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    nomcq_dwarfs = nomcq.subsample(["Dwarfs", "Right Statistics Teff"])
    generic_columns = ["Corrected K Excess"]
    dwarfs = vstack([
        mcq_dwarfs[generic_columns], nomcq_dwarfs[generic_columns]])
    dwarf_ebs = ebs.subsample(["Dwarfs", "Right Statistics Teff"])

    f, ax2 = plt.subplots(
        1, 1, dpi=dpi, figsize=figsize)    
    cons_limit = -0.3

    # Create the histograms
    period_bins, dp = np.linspace(1.5, 21.5, 20+1, retstep=True, endpoint=True)
    # These are for the conservative sample.
    singles03 =  mcq_dwarfs["Prot"][mcq_dwarfs["Corrected K Excess"] >= cons_limit]
    binaries03 =  mcq_dwarfs["Prot"][mcq_dwarfs["Corrected K Excess"] < cons_limit]
    single_ebs03 = dwarf_ebs["period"][dwarf_ebs["Corrected K Excess"] >= cons_limit]
    binary_ebs03 = dwarf_ebs["period"][dwarf_ebs["Corrected K Excess"] < cons_limit]
    binary_rot_hist03, _ = np.histogram(binaries03, bins=period_bins)
    binary_eb_hist03, _ = np.histogram(binary_ebs03, bins=period_bins)
    single_rot_hist03, _ = np.histogram(singles03, bins=period_bins)
    single_eb_hist03, _ = np.histogram(single_ebs03, bins=period_bins)
    full_rot_hist, _ = np.histogram(mcq_dwarfs["Prot"], bins=period_bins)
    full_eb_hist, _ = np.histogram(dwarf_ebs["period"], bins=period_bins)

    # The total number of binaries and singles in the full sample.
    summed_binaries03 = (
        np.count_nonzero(mcq_dwarfs["Corrected K Excess"] < cons_limit) + 
        np.count_nonzero(dwarf_ebs["Corrected K Excess"] < cons_limit) + 
        np.count_nonzero(nomcq_dwarfs["Corrected K Excess"] < cons_limit))
    summed_singles03 = (
        np.count_nonzero(mcq_dwarfs["Corrected K Excess"] >= cons_limit) + 
        np.count_nonzero(dwarf_ebs["Corrected K Excess"] >= cons_limit) + 
        np.count_nonzero(nomcq_dwarfs["Corrected K Excess"] >= cons_limit))

    # The binary fraction for the full sample
    fullsamp_frac03 = (summed_binaries03 /
        (len(mcq_dwarfs) + len(dwarf_ebs) + len(nomcq_dwarfs)))

    # Sum up the binaries, single, and total histograms
    total_binaries03 = binary_rot_hist03 + binary_eb_hist03
    total_singles03 = single_rot_hist03 + single_eb_hist03
    total = full_rot_hist + full_eb_hist


    # Measure the binary fraction
    frac03 = total_binaries03 / total
    frac_uppers03 = au.binomial_upper(total_binaries03, total) - frac03
    frac_lowers03 = frac03 - au.binomial_lower(total_binaries03, total)

    period_mids = (period_bins[1:] + period_bins[:-1])/2
    ax2.step(period_bins, np.append(frac03, [0]), where="post", 
            color=bc.black, ls="-", label="", lw=3)
    ax2.errorbar(period_bins[:-1]+dp/2, frac03,
                yerr=[frac_lowers03, frac_uppers03],
                color=bc.black, ls="", marker="")
    ax2.plot([1, 50], [fullsamp_frac03, fullsamp_frac03], ls=":", marker="",
            color=bc.black, lw=3)
    ax2.set_xlabel("Rotation Period (day)")
    ax2.set_ylabel("Photometric Binary Function")
    ax2.set_xlim(1.5, 21.5)
    ax2.set_ylim(0, 1)
    ax2.set_title(r"$\Delta {0} < {1:g}$ mag".format(MKstr.strip("$"), cons_limit))

@write_plot("ebdist")
def verify_eb_rapid_rotator_rate():
    '''Compare the rate of EBs to the rate of rapid rotators.'''
    mcq = cache.mcquillan_corrected_splitter()
    nomcq = cache.mcquillan_nondetections_corrected_splitter()
    eb_split = cache.eb_splitter_with_DSEP()
    mcq_dwarfs = mcq.subsample(["Dwarfs", "Right Statistics Teff"])
    nomcq_dwarfs = nomcq.subsample(["Dwarfs", "Right Statistics Teff"])
    generic_columns = ["kepid", "Corrected K Excess"]
    dwarfs = vstack([
        mcq_dwarfs[generic_columns], nomcq_dwarfs[generic_columns]])
    dwarf_ebs = eb_split.subsample(["Dwarfs", "Right Statistics Teff"])
    # We only want detached systems
    dwarf_ebs = dwarf_ebs[dwarf_ebs["period"] > 1]

    # Check the intersection between the two samples.
    dwarfs = au.filter_column_from_subtable(
        dwarfs, "kepid", dwarf_ebs["kepid"])

    f, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize)
    # Now bin the EBs
    period_bins, dp = np.linspace(1.5, 12.5, 11+1, retstep=True)
    period_bins = period_bins 
    period_bin_centers = np.sqrt(period_bins[1:] * period_bins[:-1])
    eb_hist, _ = np.histogram(dwarf_ebs["period"], bins=period_bins)
    totalobjs = len(dwarfs) + len(dwarf_ebs)
    normalized_ebs = eb_hist / totalobjs
    eb_upperlim = (au.poisson_upper(eb_hist, 1) - eb_hist) / totalobjs
    eb_lowerlim = (eb_hist - au.poisson_lower(eb_hist, 1)) / totalobjs
    ax.step(period_bins, np.append(normalized_ebs, [0]), where="post", 
            color=bc.red, ls="-", label="EBs", alpha=0.5)
    ax.set_xscale("linear")

    # Bin the rapid rotators
    rapid_hist, _ = np.histogram(mcq_dwarfs["Prot"], bins=period_bins)
    normalized_rapid = rapid_hist / totalobjs 
    rapid_upperlim = (au.poisson_upper(rapid_hist, 1) - rapid_hist) / totalobjs
    rapid_lowerlim = (rapid_hist - au.poisson_lower(rapid_hist, 1)) / totalobjs
    ax.step(period_bins, np.append(normalized_rapid, [0]), where="post", color=bc.blue,
            ls="-", label="Rapid rotators")
    ax.errorbar(period_bin_centers, normalized_rapid, 
            yerr=[rapid_lowerlim, rapid_upperlim], marker="", ls="", 
            color=bc.blue, capsize=5)

    # To calculate the eclipse probability, I need masses and radii.
    masses = samp.calc_model_over_feh_fixed_age_alpha(
        np.log10(dwarf_ebs["SDSS-Teff"]), mist.MISTIsochrone.logteff_col,
        mist.MISTIsochrone.mass_col, 0.08, 1e9)
    radii = masses
    eclipse_prob = ebs.eclipse_probability(
        dwarf_ebs["period"], radii*1.5, masses*1.5)
    # For empty bins, this is the default eclipse probability.
    default_probs = ebs.eclipse_probability(period_bin_centers, 0.7, 0.7)
    # To translate from eb fraction to rapid fraction.
    correction_factor = (np.maximum(0, 0.92 - eclipse_prob) / 
                         eclipse_prob)
    default_correction = (np.maximum(0, 0.92 - default_probs) /
                          default_probs)
    pred_hist, _ = np.histogram(
        dwarf_ebs["period"], bins=period_bins, weights=correction_factor)
    normalized_pred = pred_hist / totalobjs
    scale_factor = np.where(
        normalized_ebs, normalized_pred / normalized_ebs, default_correction)
    pred_upperlim =  eb_upperlim * scale_factor
    pred_lowerlim = eb_lowerlim * scale_factor
    ax.step(period_bins, np.append(normalized_pred, [0]), where="post", 
            color=bc.red, linestyle=":", 
            label="Predicted Rapid Rotators from EBs")
    ax.errorbar(period_bin_centers, normalized_pred, 
            yerr=[pred_lowerlim, pred_upperlim], marker="", ls="", 
            color=bc.red, capsize=5)
    ax.set_xlabel("Period (day)")
    ax.set_ylabel("Normalized Period Distribution")
    ax.legend(loc="upper left")
    ax.set_xlim(1.5, 12.5)

    # Calculate the total rate of synchronized binaries.
    num_rapid = np.count_nonzero(np.logical_and(
        mcq_dwarfs["Prot"] > 1.5, mcq_dwarfs["Prot"] < 7))
    total_rapid = num_rapid 
    total_rapid_upper = au.poisson_upper_exact(total_rapid, 1) - total_rapid
    total_rapid_lower = total_rapid - au.poisson_lower_exact(total_rapid, 1)
    print("Rapid rate is {0:.5f} + {1:.6f} - {2:.6f}".format(
        total_rapid / totalobjs, total_rapid_upper / totalobjs, 
        total_rapid_lower / totalobjs))

    # Print the predicted number of rapid rotators from the eclipsing binaries.
    rapid = np.logical_and(dwarf_ebs["period"] > 1.5, dwarf_ebs["period"] < 7)
    pred_rapid = np.sum(correction_factor[rapid])
    pred_num = np.count_nonzero(rapid)
    scale = pred_rapid / pred_num
    pred_rate = pred_rapid / totalobjs
    raw_upper = au.poisson_upper(pred_num, 1) - pred_num
    raw_lower = au.poisson_lower(pred_num, 1) - pred_num
    upper_pred = raw_upper * scale / totalobjs
    lower_pred = raw_lower * scale / totalobjs
    print("Predicted Rate is {0:.5f} + {1:.6f} - {2:.6f}".format(
        pred_rate, upper_pred, lower_pred))

    # Calculate the total rate of synchronized binaries.
    num_rapid = np.count_nonzero(np.logical_and(
        mcq_dwarfs["Prot"] > 1.5, mcq_dwarfs["Prot"] < 7))
    num_ebs = np.count_nonzero(np.logical_and(
        dwarf_ebs["period"] > 1.5, dwarf_ebs["period"] < 7))
    total_rapid = num_rapid + num_ebs
    total_rapid_upper = au.poisson_upper_exact(total_rapid, 1) - total_rapid
    total_rapid_lower = total_rapid - au.poisson_lower_exact(total_rapid, 1)
    total_rate = total_rapid / 0.92 / totalobjs
    total_rate_upper = total_rapid_upper / 0.92 / totalobjs
    total_rate_lower = total_rapid_lower / 0.92 / totalobjs
    print("Total rate is {0:.5f} + {1:.6f} - {2:.6f}".format(
        total_rate, total_rate_upper, total_rate_lower))
