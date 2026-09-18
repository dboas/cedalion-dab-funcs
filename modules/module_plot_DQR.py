
import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.sigproc.motion as motion_correct
import cedalion.xrutils as xrutils
import xarray as xr
import matplotlib.pyplot as p
import matplotlib.colors as clrs
import cedalion.vis as plots
from cedalion import units
import numpy as np

from scipy.signal import filtfilt
from scipy.signal.windows import gaussian

import pdb


def plotDQR( rec = None, chs_pruned = None, cfg_preprocess = None, filenm = None, cfg_dataset = None):

    stim_lst_str = cfg_dataset['cfg_hrf']['stim_lst']
    
    f, ax = p.subplots(3, 2, figsize=(11, 14))
    
    #
    # Plot GVTD
    #
    ax[0][0].plot( rec.aux_ts["gvtd"].time, rec.aux_ts["gvtd"], color='b', label="GVTD")
    if 'gvtd_corrected' in rec.aux_ts.keys():
        ax[0][0].plot( rec.aux_ts["gvtd"].time, rec.aux_ts["gvtd_corrected"], color='#ff4500', label="GVTD corrected")
    ax[0][0].set_xlabel("time / s")
    ax[0][0].set_title(f"{filenm}")
    thresh = quality._get_gvtd_threshold(rec.aux_ts['gvtd'], 'histogram_mode', n_std = 10)
    ax[0][0].axhline(thresh.values, color='b', linestyle='--', label=f'Thresh {thresh:.1e}')
    if 'gvtd_corrected' in rec.aux_ts.keys():
        thresh_corrected = quality._get_gvtd_threshold(rec.aux_ts['gvtd_corrected'], 'histogram_mode', n_std = 10)
        ax[0][0].axhline(thresh_corrected.values, color='#ff4500', linestyle='--', label=f'Thresh {thresh_corrected:.1e}')
    ax[0][0].legend()
    ax[0][0].set_ylim(0, 3*thresh)

    stim = rec.stim.copy()
    if stim_lst_str is not None:
        plots.plot_stim_markers(ax[0][0], stim[stim.trial_type.isin(stim_lst_str)], y=1)
    # add stim_lst_str to the legend
    handles, labels = ax[0][0].get_legend_handles_labels()
    labels.append(stim_lst_str)
    ax[0][0].legend(handles, labels)


    #
    # Plot the pruned channels
    #
    # create cmap
    
    colors = ['cyan', 'blue', (1,0.9,0.4), (0.3, 1, 0.3), 'magenta', 'red']  # Change these colors if needed
    bounds = [0, 0.16, 0.32, 0.48, 0.68, 0.84, 1]
    
    cmap = clrs.ListedColormap(colors)
    norm = clrs.BoundaryNorm(bounds, cmap.N)
        
    cb_ticks_labels = [(0.08,'SDS'), (0.24,'Low Signal'), (0.4,'Poor SNR'), (0.58,'Good SNR'), (0.76,'SCI/PSP'), (0.92,'Saturated')]

    idx_good = np.where(chs_pruned.values == 0.58)[0]
    plots.scalp_plot( 
            rec["amp"],
            rec.geo3d,
            chs_pruned,
            ax[0][1],
            min_dist = cfg_preprocess['cfg_prune']['sd_threshs'][0],
            #max_dist = cfg_preprocess['cfg_prune']['sd_threshs'][1], 
            cmap=cmap, #'gist_rainbow',
            norm=norm,
            vmin=0,
            vmax=1,
            optode_labels=False,
            title=f"Pruned Channels {(len(chs_pruned)-len(idx_good))/len(chs_pruned)*100:.1f}%",
            optode_size=6,
            cb_ticks_labels = cb_ticks_labels
        )
    
    
    #
    # Plot variance of OD along time axis for wavelength 1 (post corrected) 
    #
    ax1 = ax[1][0]
    variance_vals = np.log10( rec['od_corrected'].values.var(axis=2))
    variance_vals_da = xr.DataArray(variance_vals, dims=["channel", "wavelength"], coords={"channel": rec["od"].channel, "wavelength": rec["od"].wavelength})
    max_variance = np.nanmax(variance_vals)
    min_variance = np.nanmin(variance_vals)
    wav = rec['amp'].wavelength.values[0]  # first wav
    plots.scalp_plot(
            rec["od"],
            rec.geo3d,
            variance_vals_da.isel(wavelength=0),  # first wav
            ax1,
            min_dist = cfg_preprocess['cfg_prune']['sd_threshs'][0],
            #max_dist = cfg_preprocess['cfg_prune']['sd_threshs'][1], 
            cmap='jet',
            vmin=min_variance,
            vmax=max_variance,
            optode_labels=False,
            title=f"OD Variance - {wav} nm",
            optode_size=6
        )
    
    #
    # Plot variance of OD along time axis for wavelength 2 (post correction) 
    #
    ax1 = ax[1][1]
    variance_vals = np.log10( rec['od_corrected'].values.var(axis=2))
    variance_vals_da = xr.DataArray(variance_vals, dims=["channel", "wavelength"], coords={"channel": rec["od"].channel, "wavelength": rec["od"].wavelength})
    max_variance = np.nanmax(variance_vals)
    min_variance = np.nanmin(variance_vals)
    wav = rec['amp'].wavelength.values[1]  # 2nd wav
    plots.scalp_plot(
            rec["od"],
            rec.geo3d,
            variance_vals_da.isel(wavelength=1),  # 2nd wav
            ax1,
            min_dist = cfg_preprocess['cfg_prune']['sd_threshs'][0],
            #max_dist = cfg_preprocess['cfg_prune']['sd_threshs'][1], 
            cmap='jet',
            vmin=min_variance,
            vmax=max_variance,
            optode_labels=False,
            title=f"OD  Variance - {wav} nm",
            optode_size=6
        )

    #
    # Plot SNR (for wav 1)
    #
    ax1 = ax[2][0]
    snr_thresh = cfg_preprocess['cfg_prune']['snr_thresh']
    snr, snr_mask = quality.snr(rec['amp'], snr_thresh)
    
    snr_mask_wav = snr_mask.isel(wavelength=0)
    num_above_thresh = snr_mask_wav.sum().item() # count 'True' (num chans where SNR > thresh)
    
    wav = rec['amp'].wavelength.values[0]
    plots.scalp_plot(
            rec["amp"],
            rec.geo3d,
            snr.isel(wavelength=0),
            ax1,
            min_dist = cfg_preprocess['cfg_prune']['sd_threshs'][0],
            #max_dist = cfg_preprocess['cfg_prune']['sd_threshs'][1], 
            cmap='jet',
            vmin = 0,  #np.min(snr.isel(wavelength=0)),
            vmax = 25,  #np.max(snr.isel(wavelength=0)),
            optode_labels=False,
            title=f"SNR - {wav} nm ({num_above_thresh/len(chs_pruned)*100:.1f}% chans > threshold = {snr_thresh})",
            optode_size=6
        )
    
    #
    # Plot SNR (for wav 2)
    #
    ax1 = ax[2][1]
    snr_thresh = cfg_preprocess['cfg_prune']['snr_thresh']
    snr, snr_mask = quality.snr(rec['amp'], snr_thresh)
    
    snr_mask_wav = snr_mask.isel(wavelength=1)
    num_above_thresh = snr_mask_wav.sum().item() # count 'True' (num chans where SNR > thresh)
    
    wav = rec['amp'].wavelength.values[1]
    plots.scalp_plot(
            rec["amp"],
            rec.geo3d,
            snr.isel(wavelength=1),
            ax1,
            min_dist = cfg_preprocess['cfg_prune']['sd_threshs'][0],
            #max_dist = cfg_preprocess['cfg_prune']['sd_threshs'][1], 
            cmap='jet',
            vmin = 0,  #np.min(snr.isel(wavelength=1)),
            vmax = 25,  #np.max(snr.isel(wavelength=1)),
            optode_labels=False,
            title=f"SNR - {wav} nm ({num_above_thresh/len(chs_pruned)*100:.1f}% chans > threshold = {snr_thresh})",
            optode_size=6
        )

        
    # give a title to the figure
    if cfg_preprocess['flag_prune_channels']:  # !!! add if puned or unpruned to title and file name? - matters for gvtd and variance
        flag_prune = '_pruned'
    else:
        flag_prune = '_unpruned'
        
    fig_title = filenm + flag_prune
    
    p.suptitle(fig_title)

    p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'],'plots', 'DQR', fig_title + "_DQR.png") )
    p.close()

    
    # Plot the GVTD Histograms
    # thresh = make_gvtd_hist(rec.aux_ts['gvtd'], plot_thresh=True, stat_type='histogram_mode', n_std=10) # !!! why recalc thresh? do we need to?
    # p.suptitle(fig_title)
    # p.savefig( os.path.join(filepath, 'derivatives', 'plots', 'DQR','gvtd', fig_title + "_DQR_gvtd_hist.png") )
    # p.close()
    
    # if 'gvtd_corrected' in rec.aux_ts.keys():
        # thresh_corrected = make_gvtd_hist(rec.aux_ts['gvtd_corrected'], plot_thresh=True, stat_type='histogram_mode', n_std=10)
        # p.suptitle(fig_title)
        # p.savefig( os.path.join(filepath, 'derivatives', 'plots', 'DQR','gvtd', fig_title + "_DQR_gvtd_hist_corrected.png") )
        # p.close()
    
    
    # GVTD plots
    if 'gvtd_corrected' in rec.aux_ts.keys():
        der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', 'gvtd')
        if not os.path.exists(der_dir):
            os.makedirs(der_dir)
        
        thresh_b4, thresh_corrected = make_gvtd_hist_compare_corrected(rec.aux_ts['gvtd'], rec.aux_ts['gvtd_corrected'], plot_thresh=True, stat_type='histogram_mode', n_std=10)
        p.suptitle(filenm)
        p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR','gvtd', fig_title + "_DQR_gvtd_hist_compare.png") )
        p.close()
    

    return


def make_gvtd_hist_compare_corrected(gvtd_time_trace_1, gvtd_time_trace_2, plot_thresh=True, stat_type=None, n_std=None):
    """Generate a histogram of GVTD values and optionally overlay a threshold line.

    Code from:
    https://github.com/sherafatia/GVTD
    Sherafati, A., Snyder, A. Z., Eggebrecht, A. T., Bergonzi, K. M., Burns-Yocum,
    T. M., Lugar, H. M., Ferradal, S. L., Robichaux-Viehoever, A., Smyser, C. D.,
    Palanca, B. J., Hershey, T. & Culver, J. P. Global motion detection and censoring
    in high-density diffuse optical tomography. Hum. Brain Mapp. 41, 4093?4112 (2020).
    converted from matlab by chatGPT

    Args:
        gvtd_time_trace (array-like): GVTD time trace (1D array).
        plot_thresh (bool): Whether to plot the threshold on the histogram. Default is
            True.
        stat_type (str): Statistic type for threshold calculation (e.g., from StatType).
            Default is Histogram_Mode.
        n_std (int or float): Number of standard deviations for threshold calculation.
            Default is 4.
        bin_size (float): Size of histogram bins. If None, it's calculated based on
            data.

    Returns:
        float: Calculated threshold value if `plot_thresh` is True; otherwise, None.
    """
    # Set default values
    if stat_type is None:
        stat_type = 'histogram_mode'
    if plot_thresh is None:
        plot_thresh = True
    if n_std is None and plot_thresh:
        n_std = 4

    # Calculate bin size if not provided
    min_counts_each_bin = 5
    n_bins_1 = round(len(gvtd_time_trace_1) / min_counts_each_bin)
    bin_size_1 = np.max(gvtd_time_trace_1) / n_bins_1
    
    n_bins_2 = round(len(gvtd_time_trace_2) / min_counts_each_bin)
    bin_size_2 = np.max(gvtd_time_trace_2) / n_bins_2

    f, ax = p.subplots(1, 2, figsize=(11, 5))

    # B4 correction
    # Create the histogram
    bins_1 = np.arange(0, np.max(gvtd_time_trace_1) + bin_size_1, bin_size_1)
    ax[0].hist(gvtd_time_trace_1, bins=bins_1, edgecolor='black', alpha=0.75)
    ax[0].set_title('GVTD Histogram')
    ax[0].set_xlabel('GVTD')
    ax[0].set_ylabel('Counts')

    threshold_1 = None
    if plot_thresh:
        # Calculate the threshold
        threshold_1 = quality._get_gvtd_threshold(gvtd_time_trace_1, stat_type, n_std)

        # Plot the threshold line
        ax[0].axvline(threshold_1.values, color='red', linestyle='--', label=f'Threshold: {threshold_1:.4f}')
        ax[0].legend()
        
        
    # After correction
    # Create the histogram
    bins_2 = np.arange(0, np.max(gvtd_time_trace_2) + bin_size_2, bin_size_2)
    ax[1].hist(gvtd_time_trace_2, bins=bins_2, edgecolor='black', alpha=0.75)
    ax[1].set_title('GVTD Histogram - corrected')
    ax[1].set_xlabel('GVTD')
    ax[1].set_ylabel('Counts')

    threshold_2 = None
    if plot_thresh:
        # Calculate the threshold
        threshold_2 = quality._get_gvtd_threshold(gvtd_time_trace_2, stat_type, n_std)

        # Plot the threshold line
        ax[1].axvline(threshold_2.values, color='red', linestyle='--', label=f'Threshold: {threshold_2:.4f}')
        ax[1].legend()

    return threshold_1, threshold_2
    



def make_gvtd_hist(gvtd_time_trace, plot_thresh=True, stat_type=None, n_std=None, 
                   bin_size=None):
    """Generate a histogram of GVTD values and optionally overlay a threshold line.

    Code from:
    https://github.com/sherafatia/GVTD
    Sherafati, A., Snyder, A. Z., Eggebrecht, A. T., Bergonzi, K. M., Burns-Yocum,
    T. M., Lugar, H. M., Ferradal, S. L., Robichaux-Viehoever, A., Smyser, C. D.,
    Palanca, B. J., Hershey, T. & Culver, J. P. Global motion detection and censoring
    in high-density diffuse optical tomography. Hum. Brain Mapp. 41, 4093?4112 (2020).
    converted from matlab by chatGPT

    Args:
        gvtd_time_trace (array-like): GVTD time trace (1D array).
        plot_thresh (bool): Whether to plot the threshold on the histogram. Default is
            True.
        stat_type (str): Statistic type for threshold calculation (e.g., from StatType).
            Default is Histogram_Mode.
        n_std (int or float): Number of standard deviations for threshold calculation.
            Default is 4.
        bin_size (float): Size of histogram bins. If None, it's calculated based on
            data.

    Returns:
        float: Calculated threshold value if `plot_thresh` is True; otherwise, None.
    """
    # Set default values
    if stat_type is None:
        stat_type = 'histogram_mode'
    if plot_thresh is None:
        plot_thresh = True
    if n_std is None and plot_thresh:
        n_std = 4

    # Calculate bin size if not provided
    if bin_size is None:
        min_counts_each_bin = 5
        n_bins = round(len(gvtd_time_trace) / min_counts_each_bin)
        bin_size = np.max(gvtd_time_trace) / n_bins

    f, ax = p.subplots(1, 1, figsize=(7, 5))

    # Create the histogram
    bins = np.arange(0, np.max(gvtd_time_trace) + bin_size, bin_size)
    ax.hist(gvtd_time_trace, bins=bins, edgecolor='black', alpha=0.75)
    ax.set_title('GVTD Histogram')
    ax.set_xlabel('GVTD')
    ax.set_ylabel('Counts')

    threshold = None
    if plot_thresh:
        # Calculate the threshold
        threshold = quality._get_gvtd_threshold(gvtd_time_trace, stat_type, n_std)

        # Plot the threshold line
        ax.axvline(threshold.values, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
        ax.legend()

    return threshold




def plot_slope(rec = None, slope = None, cfg_preprocess=None, filenm = None, cfg_dataset = None):
    '''
    Plot slope before and after correction on a scalp plot.
    '''
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', 'slope')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    
    f, ax = p.subplots(2, 1, figsize=(10, 10))
    # plot the base slope as a scalp plot
    # get the slope values for each channel and change units to per 10min rather than per second
    if slope[0] is not None:
        ax1 = ax[0]
        slope_vals = slope[0].slope.values * 60 * 10
        # create a data array of the slope values
        slope_vals_da = xr.DataArray(slope_vals, dims=["channel", "wavelength"], coords={"channel": rec["od"].channel, "wavelength": rec["od"].wavelength})
        # get max of the absolute value of the slope values
        max_slope = np.nanmax(np.abs(slope_vals))
        plots.scalp_plot(
                rec["od"],
                rec.geo3d,
                slope_vals_da.isel(wavelength=0),
                ax1,
                cmap='jet',
                vmin=-max_slope,
                vmax=max_slope,
                optode_labels=False,
                title="Baseline Slope",
                optode_size=6
            )

    # plot the tddr slope as a scalp plot
    ax1 = ax[1]
    # get the slope values for each channel and change units to per 10min rather than per second
    slope_vals = slope[1].slope.values * 60 * 10
    # create a data array of the slope values
    slope_vals_da = xr.DataArray(slope_vals, dims=["channel", "wavelength"], coords={"channel": rec["od_corrected"].channel, "wavelength": rec["od_corrected"].wavelength})
    # get max of the absolute value of the slope values
    max_slope = np.nanmax(np.abs(slope_vals))
    plots.scalp_plot(
            rec["od_corrected"],
            rec.geo3d,
            slope_vals_da.isel(wavelength=0),
            ax1,
            cmap='jet',
            vmin=-max_slope,
            vmax=max_slope,
            optode_labels=False,
            title="OD Corrected Slope",
            optode_size=6
        )
    
    # give a title to the figure
    if cfg_preprocess['flag_prune_channels']:  # !!! add if puned or unpruned to title and file name? - matters for gvtd and variance
        flag_prune = '_pruned'
    else:
        flag_prune = '_unpruned'
        
    fig_title = filenm + flag_prune
    
    p.suptitle(fig_title)

    p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', 'slope', fig_title + "_slope.png") )
    p.close()
    
    return
    
    
    

def plotDQR_sidecar(file_json, rec, cfg_dataset, filenm):

    # get the variables from the json file
    dataSDWP_LowHigh = file_json['dataSDWP_LowHigh']
    powerLevelSetting = file_json['powerLevelSetting']
    powerLevelSetLowHigh = file_json['powerLevelSetLowHigh']
    srcModuleGroups = file_json['srcModuleGroups']
    dataSDWP_LowHigh = file_json['dataSDWP_LowHigh']

    SDj = file_json['SD']
    SD = SDj.copy()
    SD['DetPos2D'] = np.array(SDj['DetPos2D'])
    SD['SrcPos2D'] = np.array(SDj['SrcPos2D'])
    SD['SrcPos3D'] = np.array(SDj['SrcPos3D'])
    SD['DetPos3D'] = np.array(SDj['DetPos3D'])
    SD['Lambda'] = np.array(SDj['Lambda'])
    SD['MeasList'] = np.array(SDj['MeasList'])

    #srcModuleGroups[2]
    # get dimensions of dataSDWP_LowHigh
    nSrc = len(dataSDWP_LowHigh)
    nDet = len(dataSDWP_LowHigh[0])
    nWav = len(dataSDWP_LowHigh[0][0])
    nPower = len(dataSDWP_LowHigh[0][0][0])

    # convert the dataSDWP_LowHigh to a numpy array
    dataSDWP_LowHigh_np = np.zeros((nSrc, nDet, nWav, nPower))
    for iSrc in range(nSrc):
        for iDet in range(nDet):
            for iWav in range(nWav):
                for iPower in range(nPower):
                    dataSDWP_LowHigh_np[iSrc, iDet, iWav, iPower] = dataSDWP_LowHigh[iSrc][iDet][iWav][iPower]

    # get rho_sds
    nS = SD['SrcPos3D'].shape[0]
    nD = SD['DetPos3D'].shape[0]
    rho_sds = np.zeros((nS, nD))

    for iS in range(nS):
        posS = np.ones((nD, 1)) * SD['SrcPos3D'][iS, :]
        rho_sds[iS, :] = np.sqrt(np.sum((posS - SD['DetPos3D'])**2, axis=1))

    # Identify the first short separation detector
    lstSSr, lstSSc = np.where(rho_sds < 12)
    SSd1 = nD
    if lstSSc.size > 0:
        SSd1 = np.min(lstSSc)  # Assume 1 SS bundle for now
        for ii in range(len(lstSSr)):
            rho_sds[lstSSr[ii], SSd1] = rho_sds[lstSSr[ii], lstSSc[ii]]
        nD = SSd1 + 1
        rho_sds = rho_sds[:, :SSd1+1]


    #
    # Plot Signal vs Distance and LED Power Levels
    #
    alpha = 0.4

    fig, ax = p.subplots(2, 2, figsize=(11, 11))

    # Low Power
    ax1 = ax[0, 0]
    foo = dataSDWP_LowHigh_np[:nS, :nD, 0, 0]
    boo = rho_sds
    scatter1 = ax1.scatter(boo.flatten(), np.log10(np.maximum(foo.flatten(), 1e-8)), 
                        facecolor='b', edgecolor='none', alpha=alpha)
    foo = dataSDWP_LowHigh_np[:nS, :nD, 1, 0]
    scatter2 = ax1.scatter(boo.flatten(), np.log10(np.maximum(foo.flatten(), 1e-8)), 
                        facecolor='r', edgecolor='none', alpha=alpha)

    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_xlabel('Distance (mm)')
    ax1.set_ylabel('log$_{10}$( Signal )')
    ax1.set_title('Low Power')
    ax1.legend([f'{SD["Lambda"][0]} nm', f'{SD["Lambda"][1]} nm'])
    ax1.set_xlim([0, 100])
    ax1.set_ylim([-6, 0])
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    # move ax1 up a bit
    pos1 = ax1.get_position()
    pos2 = [pos1.x0, pos1.y0 + 0.03, pos1.width, pos1.height]
    ax1.set_position(pos2)

    # High Power
    ax1 = ax[0, 1]
    foo = dataSDWP_LowHigh_np[:nS, :nD, 0, 1]
    boo = rho_sds
    scatter1 = ax1.scatter(boo.flatten(), np.log10(np.maximum(foo.flatten(), 1e-8)), 
                        facecolor='b', edgecolor='none', alpha=alpha)
    foo = dataSDWP_LowHigh_np[:nS, :nD, 1, 1]
    scatter2 = ax1.scatter(boo.flatten(), np.log10(np.maximum(foo.flatten(), 1e-8)), 
                        facecolor='r', edgecolor='none', alpha=alpha)

    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_xlabel('Distance (mm)')
    ax1.set_title('High Power')
    ax1.legend([f'{SD["Lambda"][0]} nm', f'{SD["Lambda"][1]} nm'])
    ax1.set_xlim([0, 100])
    ax1.set_ylim([-6, 0])
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    # move ax1 up a bit
    pos1 = ax1.get_position()
    pos2 = [pos1.x0, pos1.y0 + 0.03, pos1.width, pos1.height]
    ax1.set_position(pos2)

    # power level setting lambda0
    lst0 = np.where(SD['MeasList'][:, 3]==1)[0]
    power_level = []
    for i in lst0:
        power_level.append(powerLevelSetting[i])
    power_level = xr.DataArray(
        power_level,
        dims="channel",
        coords={"channel": rec["amp"].channel},
    )
    plots.scalp_plot(
            rec["conc"],
            rec.geo3d,
            power_level,
            ax[1,0],
            cmap='jet',
            optode_labels=False,
            optode_size=5,
            vmin=0,
            vmax=7,
            title=f"Power Level {SD['Lambda'][0]} nm",
        )

    # power level setting lambda1
    lst0 = np.where(SD['MeasList'][:, 3]==2)[0]
    power_level = []
    for i in lst0:
        power_level.append(powerLevelSetting[i])
    power_level = xr.DataArray(
        power_level,
        dims="channel",
        coords={"channel": rec["amp"].channel},
    )
    plots.scalp_plot(
            rec["conc"],
            rec.geo3d,
            power_level,
            ax[1,1],
            cmap='jet',
            optode_labels=False,
            optode_size=5,
            vmin=0,
            vmax=7,
            title=f"Power Level {SD['Lambda'][1]} nm",
        )

    # give a title to the figure
    p.suptitle(filenm)

    p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', filenm + "_DQR_sigVdis.png") )
    p.close()


    #
    # Plot calibration data
    #
    ml = SD["MeasList"]
    dataCrosstalk = np.zeros((len(ml), 1))
    dataCrosstalkLow = np.zeros((len(ml), 1))

    # Convert dataSDWP_LowHigh to numpy array
    dataSDWP_LowHigh_np = np.array(dataSDWP_LowHigh)

    for iML in range(len(ml)):
        iS = ml[iML, 0]
        iD = ml[iML, 1]
        iW = ml[iML, 3]

        # Check if short separation detector
        if iD > SSd1:
            iD = SSd1

        # Determine source group for the given iS
        iSrcModule = (np.ceil(iS / 8).astype(int))
        iSrc = (iS - (iSrcModule - 1) * 8)
        iSg = 0
        for ii in range(len(srcModuleGroups)):
            if np.sum(np.isin(srcModuleGroups[ii], iSrcModule)) > 0:
                iSg = ii
                break

        # High power
        data = dataSDWP_LowHigh_np[int(iSrc)-1::8, int(iD-1), int(iW-1), 1]  

        if data[iSrcModule-1] > 1e-2:
            data = data / data[iSrcModule-1]  # Normalize to get cross-talk from other source modules
            data[iSrcModule-1] = 0  # No cross talk from itself
            data = data[np.array(srcModuleGroups[iSg]) - 1]  # Only consider modules within the group for high power
            dataCrosstalk[iML] = np.sum(np.abs(data))  # Sum up cross talk from other modules

        # Low power
        data = dataSDWP_LowHigh_np[int(iSrc)-1::8, int(iD-1), int(iW-1), 0]  # Adjusted for 0-based indexing in Python
        if data[iSrcModule-1] > 1e-2:
            data = data / data[iSrcModule-1]  # Normalize to get cross-talk from other source modules
            data[iSrcModule-1] = 0  # No cross talk from itself
            dataCrosstalkLow[iML] = np.sum(np.abs(data))  # Sum up cross talk from other modules

    # Plot the calibration data
    f, ax = p.subplots(2, 2, figsize=(11,11))

    ax1 = ax[0,0]
    ax1.imshow(np.log10(np.abs(dataSDWP_LowHigh_np[:, :, 0, 0])), vmin=-6, vmax=0, aspect='auto')
    ax1.set_ylabel("Source")
    ax1.set_title(f'LOW Power, {SD["Lambda"][0]} nm')

    ax1 = ax[0,1]
    ax1.imshow(np.log10(np.abs(dataSDWP_LowHigh_np[:, :, 0, 1])), vmin=-6, vmax=0, aspect='auto')
    ax1.set_title(f'HIGH Power, {SD["Lambda"][0]} nm')

    ax1 = ax[1,0]
    ax1.imshow(np.log10(np.abs(dataSDWP_LowHigh_np[:, :, 1, 0])), vmin=-6, vmax=0, aspect='auto')
    ax1.set_ylabel("Source")
    ax1.set_title(f'LOW Power, {SD["Lambda"][1]} nm')
    ax1.set_xlabel('Detector')

    ax1 = ax[1,1]
    ax1.imshow(np.log10(np.abs(dataSDWP_LowHigh_np[:, :, 1, 1])), vmin=-6, vmax=0, aspect='auto')
    ax1.set_title(f'HIGH Power, {SD["Lambda"][1]} nm')
    ax1.set_xlabel('Detector')

    # give a title to the figure
    p.suptitle(filenm)

    p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', filenm + "_DQR_calib.png") )
    p.close()


    #
    # Plot the cross-talk data
    #
    f, ax = p.subplots(2, 2, figsize=(11,11))

    ax1 = ax[0,0]
    lst1 = np.where(SD['MeasList'][:, 3]==1)[0]
    strTitle = f'Low Power {SD["Lambda"][0]} nm'
    plot_crosstalk(SD, dataCrosstalkLow, ax1, lst1, strTitle )

    ax1 = ax[0,1]
    strTitle = f'High Power {SD["Lambda"][0]} nm'
    plot_crosstalk(SD, dataCrosstalk, ax1, lst1, strTitle )

    ax1 = ax[1,0]
    lst1 = np.where(SD['MeasList'][:, 3]==2)[0]
    strTitle = f'Low Power {SD["Lambda"][1]} nm'
    plot_crosstalk(SD, dataCrosstalkLow, ax1, lst1, strTitle )

    ax1 = ax[1,1]
    strTitle = f'High Power {SD["Lambda"][1]} nm'
    plot_crosstalk(SD, dataCrosstalk, ax1, lst1, strTitle )

    # give a title to the figure
    p.suptitle(filenm)

    p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', filenm + "_DQR_crosstalk.png") )
    p.close()



def plot_crosstalk(SD, dataCrosstalk, ax1, lst1, strTitle ):
    nS = SD['SrcPos3D'].shape[0]
    nD = SD['DetPos3D'].shape[0]
    ml = SD['MeasList']

    for iS in range(nS):
        ax1.plot(SD['SrcPos2D'][iS-1, 0], SD['SrcPos2D'][iS-1, 1], 'r.', markersize=5)
    #        plt.hold(True)
    for iD in range(nD):
        ax1.plot(SD['DetPos2D'][iD-1, 0], SD['DetPos2D'][iD-1, 1], 'b.', markersize=5)
    for iML in lst1:
        iS = ml[iML, 0]
        iD = ml[iML, 1]
        iW = ml[iML, 3]
        ps = SD['SrcPos2D'][int(iS-1), :]
        pd = SD['DetPos2D'][int(iD-1), :]
        hl, = ax1.plot([ps[0], pd[0]], [ps[1], pd[1]], '-')
        hl.set_linewidth(2)
        cmIdx = int(np.ceil((max(min(np.log10(dataCrosstalk[iML][0]), 0), -3) + 3) / 0.1 + np.finfo(float).eps))
        if cmIdx < 2:
            hl.set_linewidth(0.25)
            hl.set_color([0,1,0])
        elif cmIdx < 11:
            hl.set_linewidth(2)
            hl.set_color([0,1,0])
        elif cmIdx < 21:
            hl.set_linewidth(2)
            hl.set_color([1, 0.7, 0])
        else:
            hl.set_linewidth(2)
            hl.set_color([1,0,0])
    #    plt.hold(False)
    ax1.axis('image')
    ax1.axis('off')
    ax1.set_title(strTitle)

    cmap = p.cm.colors.ListedColormap([[0, 1, 0], [1, 0.7, 0], [1, 0, 0]])
    norm = p.cm.colors.Normalize(vmin=-3, vmax=0)
    hc = p.colorbar(p.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
    hc.set_ticks([-3, -2, -1, 0])
    hc.set_ticklabels(['-3', '-2', '-1', '0'])
    ax1.tick_params(axis='both', which='major', labelsize=16)



def plot_tIncCh_dqr( rec, cfg_dataset, filenm_lst, iqr_threshold_std=2, iqr_threshold_grad=1.5, flag_plot=False ):

    n_subjects = len(rec)
    n_files_per_subject = len(rec[0])

    # loop over the subjects
    for subj_idx in range( n_subjects ):
        for file_idx in range( n_files_per_subject ):

            nan_chs = rec[subj_idx][file_idx]['od'].values
            nan_chs = np.mean(nan_chs, axis=2) # mean over time
            nan_chs = np.mean(nan_chs, axis=1) # mean over wavelengths
            nan_chs = np.isnan(nan_chs) # identify channels that are NaN
            idx_good = np.where( nan_chs == False )[0]
            idx_pruned = np.where( nan_chs == True )[0]

            #
            # Analyze the 'od' data for motion artifacts
            #
            M = quality.detect_outliers(rec[subj_idx][file_idx]["od"], 1 * units.s, iqr_threshold_std, iqr_threshold_grad)
            # get percent of unpruned channels that have no motion at each time point
            tInc_all = M.sum( axis=1 ).sum( axis=0 ) # sum over wavelengths
            #tInc_all = (tInc_all//2 - (len(nan_chs) - len(idx_good))) / len(nan_chs)
            tInc_all = (1-(tInc_all//2) / len(nan_chs)) * 100

            # get number of motion events per channel
            foo = M.values
            foo = foo.astype(int) # False=0, True=1
            foo = np.diff(foo, axis=2) # diff along time to identify start of motion epochs
            foo = np.where(foo==-1, 1, 0) # 1 where motion starts, 0 otherwise
            foo = np.sum(foo, axis=2) # sum motion starts along time
            foo = np.mean(foo, axis=1) # mean across wavelengths

            tIncCh_n_per_ch = xr.DataArray(foo, dims=['channel'], coords={'channel':M.channel})
            tIncCh_n_per_ch['source'] = M.source # add source coordinate from M
            tIncCh_n_per_ch[idx_pruned] = np.nan # set pruned channels to nan

            #
            # analyze the 'od_tddr' data for motion artifacts
            #
            foo = rec[subj_idx][file_idx]['od_tddr'].copy()
            # foo = cedalion.sigproc.frequency.freq_filter(foo, 0 * units.Hz, 1 * units.Hz)
            # foo = foo[:,:,::3]
            # foo = foo.interp(time=rec[subj_idx][file_idx]['od_tddr'].time) # this is done to handle when we downsample before ICA
            M = quality.detect_outliers(foo, 1 * units.s, iqr_threshold_std, iqr_threshold_grad)
            # get percent of unpruned channels that have no motion at each time point
            tInc_all_tddr = M.sum( axis=1 ) # sum over wavelengths
            tInc_all_tddr = tInc_all_tddr.sum( axis=0 ) # sum over channels
            #tInc_tddr_all = (tIncCh_tddr_all//2 - (len(nan_chs) - len(idx_good))) / len(nan_chs)
            tInc_all_tddr = (1-(tInc_all_tddr//2) / len(nan_chs))*100
            rec[subj_idx][file_idx].aux_ts['tInc_all_tddr'] = tInc_all_tddr

            # get number of motion events per channel
            foo = M.values
            foo = foo.astype(int) # False=0, True=1
            foo = np.diff(foo, axis=2) # diff along time to identify start of motion epochs
            foo = np.where(foo==-1, 1, 0) # 1 where motion starts, 0 otherwise
            foo = np.sum(foo, axis=2) # sum motion starts along time
            foo = np.mean(foo, axis=1) # mean across wavelengths

            tIncCh_n_per_ch_tddr = xr.DataArray(foo, dims=['channel'], coords={'channel':M.channel})
            tIncCh_n_per_ch_tddr['source'] = M.source # add source coordinate from M
            tIncCh_n_per_ch_tddr[idx_pruned] = np.nan # set pruned channels to nan

            #
            # analyze the 'od_tddr_ica' data for motion artifacts
            #
            if 'od_tddr_ica' in rec[subj_idx][file_idx].timeseries.keys():
                foo = rec[subj_idx][file_idx]['od_tddr_ica'].copy()
                foo = foo.interp(time=rec[subj_idx][file_idx]['od_tddr'].time) # this is done to handle when we downsample before ICA
                M = quality.detect_outliers(foo, 1 * units.s, iqr_threshold_std, iqr_threshold_grad)
                # get percent of unpruned channels that have no motion at each time point
                tInc_all_tddr_ica = M.sum( axis=1 ) # sum over wavelengths
                tInc_all_tddr_ica = tInc_all_tddr_ica.sum( axis=0 ) # sum over channels
                #tInc_tddr_all = (tIncCh_tddr_all//2 - (len(nan_chs) - len(idx_good))) / len(nan_chs)
                tInc_all_tddr_ica = (1-(tInc_all_tddr_ica//2) / len(nan_chs))*100
                rec[subj_idx][file_idx].aux_ts['tInc_all_tddr_ica'] = tInc_all_tddr_ica

                # get number of motion events per channel
                foo = M.values
                foo = foo.astype(int) # False=0, True=1
                foo = np.diff(foo, axis=2) # diff along time to identify start of motion epochs
                foo = np.where(foo==-1, 1, 0) # 1 where motion starts, 0 otherwise
                foo = np.sum(foo, axis=2) # sum motion starts along time
                foo = np.mean(foo, axis=1) # mean across wavelengths

                tIncCh_n_per_ch_tddr_ica = xr.DataArray(foo, dims=['channel'], coords={'channel':M.channel})
                tIncCh_n_per_ch_tddr_ica['source'] = M.source # add source coordinate from M
                tIncCh_n_per_ch_tddr_ica[idx_pruned] = np.nan # set pruned channels to nan


            if 'od_tddr_ica' in rec[subj_idx][file_idx].timeseries.keys():
                f, ax = p.subplots(3, 2, figsize=(9, 10))
            else:
                f, ax = p.subplots(2, 2, figsize=(9, 10))

            plots.scalp_plot(
                    rec[0][0]['od'],
                    rec[0][0].geo3d,
                    tIncCh_n_per_ch_tddr,
                    ax[0][0],
                    cmap='jet',
                    optode_labels=False,
                    optode_size=5,
                    vmin = 0,
                    vmax = np.min((np.nanmax(tIncCh_n_per_ch_tddr), 100)),
                    title='# Motion Artifacts after TDDR'
                )

            plots.scalp_plot(
                    rec[0][0]['od'],
                    rec[0][0].geo3d,
                    100*(1 - tIncCh_n_per_ch_tddr / tIncCh_n_per_ch),
                    ax[0][1],
                    cmap='jet',
                    optode_labels=False,
                    optode_size=5,
                    vmin = 0,
                    vmax = 100,
                    title='Percent reduction '
                )

            if 'od_tddr_ica' in rec[subj_idx][file_idx].timeseries.keys():
                plots.scalp_plot(
                        rec[0][0]['od'],
                        rec[0][0].geo3d,
                        tIncCh_n_per_ch_tddr_ica,
                        ax[1][0],
                        cmap='jet',
                        optode_labels=False,
                        optode_size=5,
                        vmin = 0,
                        vmax = np.min((np.nanmax(tIncCh_n_per_ch_tddr_ica), 100)),
                        title='# after TDDR ICA'
                    )

                plots.scalp_plot(
                        rec[0][0]['od'],
                        rec[0][0].geo3d,
                        100*(1 - tIncCh_n_per_ch_tddr_ica / tIncCh_n_per_ch_tddr),
                        ax[1][1],
                        cmap='jet',
                        optode_labels=False,
                        optode_size=5,
                        vmin = 0,
                        vmax = 100,
                        title='Percent reduction '
                    )
                ax1 = ax[2][0]
                ax2 = ax[2][1]
            else:
                ax1 = ax[1][0]
                ax2 = ax[1][1]

            # plot the tInc_all with time and stim markers
            ax1.plot( rec[subj_idx][file_idx]['od'].time, tInc_all_tddr, label='tInc_tddr', color='b' )
            if 'od_tddr_ica' in rec[subj_idx][file_idx].timeseries.keys():
                ax1.plot( rec[subj_idx][file_idx]['od'].time, tInc_all_tddr_ica, label='tInc_tddr_ica', color='m' )
            plots.plot_stim_markers(ax1, rec[subj_idx][file_idx].stim, y=1)
            ax1.set_title( f"Subject:{subj_idx+1}, Pruned: {(len(nan_chs)-len(idx_good))*100/len(nan_chs):.1f}%" )
            #    p.xlabel( 'Time' )
            #    p.ylabel( 'tInc_all' )
            ax1.set_xlabel("time (s)")
            ax1.grid()
            ax1.legend()

            # Plot GVTD
            ax2.plot( rec[subj_idx][file_idx].aux_ts["gvtd"].time, rec[subj_idx][file_idx].aux_ts["gvtd_tddr"], label="GVTD TDDR", color='b') # color='#ff4500', 
            thresh_tddr = quality.find_gvtd_thresh(rec[subj_idx][file_idx].aux_ts['gvtd_tddr'].values, quality.gvtd_stat_type.Histogram_Mode, n_std = 10)
            ax2.axhline(thresh_tddr, color='b', linestyle='--', label=f'Thresh {thresh_tddr:.1e}')
            if 'od_tddr_ica' in rec[subj_idx][file_idx].timeseries.keys():
                amp_tddr = rec[subj_idx][file_idx]['od_tddr_ica'].copy()
                amp_tddr.values = np.exp(-amp_tddr.values)
                gvtd_tddr_ica = quality.gvtd(amp_tddr)
                ax2.plot( rec[subj_idx][file_idx]['od_tddr_ica'].time, gvtd_tddr_ica, label="GVTD TDDR ICA", color='m' )
                thresh_tddr = quality.find_gvtd_thresh(gvtd_tddr_ica.values, quality.gvtd_stat_type.Histogram_Mode, n_std = 10)
                ax2.axhline(thresh_tddr, color='m', linestyle='--', label=f'Thresh {thresh_tddr:.1e}')
            ax2.set_xlabel("time (s)")
            plots.plot_stim_markers(ax2, rec[subj_idx][file_idx].stim, y=1)
            ax2.legend()

            # give a title to the figure and save it
            filenm = filenm_lst[subj_idx][file_idx]
            p.suptitle(filenm)
            p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', filenm + '_DQR_tIncCh.png') )
#            p.close()

            if flag_plot:
                p.show()
            else:
                p.close()

    return rec

# !!! plot_group_dqr will fail if no tddr ?
def plot_group_dqr( n_subjects, n_files_per_subject, chs_pruned_subjs, slope_base_subjs, slope_corrected_subjs, gvtd_corrected_subjs, snr0_subjs, snr1_subjs, rec, cfg_dataset, flag_plot = True):   
    
    subj_ids = cfg_dataset['subj_ids']
    subj_id_exclude = cfg_dataset['subj_id_exclude']
    
    n_subjects = n_subjects = len(subj_ids) - len(subj_id_exclude)
    subj_ids_new = [s for s in subj_ids if s not in subj_id_exclude]
    
    chs_pruned_percent = np.zeros( (n_subjects, n_files_per_subject) )
    for subj_idx in range(n_subjects):
        
        for file_idx in range(n_files_per_subject):
            n_chs = len(chs_pruned_subjs[subj_idx][file_idx])
            n_chs_pruned = len( np.where( chs_pruned_subjs[subj_idx][file_idx] != 0.4 )[0] )
            chs_pruned_percent[subj_idx, file_idx] = 100 * n_chs_pruned / n_chs

    slope_base_abs = np.zeros( (n_subjects, n_files_per_subject) )
    slope_corrected_abs = np.zeros( (n_subjects, n_files_per_subject) )
    for subj_idx in range(n_subjects):
        for file_idx in range(n_files_per_subject):
            slope_base_abs[subj_idx, file_idx] = np.nanmax(np.abs(slope_base_subjs[subj_idx][file_idx].slope.values)) * 60 * 10
            slope_corrected_abs[subj_idx, file_idx] = np.nanmax(np.abs(slope_corrected_subjs[subj_idx][file_idx].slope.values)) * 60 * 10

    gvtd_corrected_mean = np.zeros( (n_subjects, n_files_per_subject) )
    for subj_idx in range(n_subjects):
        for file_idx in range(n_files_per_subject):
            gvtd_corrected_mean[subj_idx, file_idx] = gvtd_corrected_subjs[subj_idx][file_idx]

    snr0_mean = np.zeros( (n_subjects, n_files_per_subject) )
    snr1_mean = np.zeros( (n_subjects, n_files_per_subject) )
    for subj_idx in range(n_subjects):
        for file_idx in range(n_files_per_subject):
            snr0_mean[subj_idx, file_idx] = snr0_subjs[subj_idx][file_idx]
            snr1_mean[subj_idx, file_idx] = snr1_subjs[subj_idx][file_idx]
    lambda0 = rec[0][0]["amp"].wavelength[0].wavelength.values
    lambda1 = rec[0][0]["amp"].wavelength[1].wavelength.values

    # bar graph of the percentage of channels pruned
    f, ax = p.subplots(2,2, figsize=(9,10))

    # channels pruned
    axtmp = ax[0][0]
    axtmp.bar( np.arange(n_subjects), np.mean(chs_pruned_percent,axis=1), color='k' )
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject), chs_pruned_percent[subj_idx,:], color='gray', marker='x' )
    axtmp.set_xticks( np.arange(n_subjects), subj_ids_new )
    axtmp.set_xlabel('Subject')
    axtmp.set_ylabel('Percentage')
    axtmp.set_title('Channels Pruned')

    # gvtd mean
    axtmp = ax[0][1]
    axtmp.bar( np.arange(n_subjects), np.mean(gvtd_corrected_mean,axis=1), color='k' )
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject), gvtd_corrected_mean[subj_idx,:], color='gray', marker='x' )
    axtmp.set_xticks( np.arange(n_subjects), subj_ids_new )
    axtmp.set_xlabel('Subject')
    foo_exp = np.max(np.round(np.log10(axtmp.get_yticks())))
    foo = axtmp.get_yticklabels()
    foo =  ( axtmp.get_yticks() ) / 10**foo_exp
    foo = np.round(foo, decimals=1)
    foo = [f"{x:.1f}e{foo_exp:.0f}" for x in foo]
    axtmp.set_yticklabels(foo, rotation=60 )
    axtmp.set_title('GVTD Mean')

    # snr
    axtmp = ax[1][0]
    h = 0.2
    axtmp.bar( np.arange(n_subjects)-h, np.mean(snr0_mean,axis=1), color='b', label=f'lambda={lambda0}', width=0.4)
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject)-h, snr0_mean[subj_idx,:], color='c', marker='x' )
    axtmp.bar( np.arange(n_subjects)+h, np.mean(snr1_mean,axis=1), color='r', label=f'lambda={lambda1}', width=0.4)
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject)+h, snr1_mean[subj_idx,:], color='m', marker='x' )
    axtmp.set_xticks( np.arange(n_subjects), subj_ids_new )
    axtmp.set_xlabel('Subject')
    axtmp.legend()
    axtmp.set_title('SNR')

    # slopes
    axtmp = ax[1][1]
    axtmp.bar( np.arange(n_subjects)-h, np.mean(slope_base_abs,axis=1), color='b', label='Base', width=0.4)
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject)-h, slope_base_abs[subj_idx,:], color='c', marker='x' )
    axtmp.bar( np.arange(n_subjects)+h, np.mean(slope_corrected_abs,axis=1), color='r', label='Corrected', width=0.4)
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject)+h, slope_corrected_abs[subj_idx,:], color='m', marker='x' )
    axtmp.set_xticks( np.arange(n_subjects), subj_ids_new )
    axtmp.set_xlabel('Subject')
    foo_exp = np.max(np.round(np.log10(axtmp.get_yticks())))
    foo =  ( axtmp.get_yticks() ) / 10**foo_exp
    foo = np.round(foo, decimals=1)
    if round(foo_exp) == 0:
        foo = [f"{x:.1f}" for x in foo]
    else:
        foo = [f"{x:.1f}e{foo_exp:.0f}" for x in foo]
    axtmp.set_yticklabels(foo, rotation=60 )
    axtmp.legend()
    axtmp.set_title('Max Slope Corrected')

    # give a title to the figure
    dirnm = os.path.basename(os.path.normpath(cfg_dataset['root_dir']))
    p.suptitle(f'Data set - {dirnm}')

    p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', "DQR_group.png") )
    
    if flag_plot:
        p.show()



def plot_gradCPT_VTC( stim, cfg_dataset, filenm ):

    RT = np.zeros((stim.shape[0], 4))
    RT[:,0] = np.array([stim.reaction_time.values])
    t = stim.onset.values

    lst_commision_error = np.where( stim.response_code == -2 )[0]
    lst_omision_error = np.where( stim.response_code == -1 )[0]

    meanRT = np.nanmean(RT[:, 0])
    stdRT = np.nanstd(RT[:, 0], ddof=1)

    # Interpolate to fill NaNs (or replace with the mean RT)
    RT[:, 1] = np.where(np.isnan(RT[:, 0]), meanRT, RT[:, 0])

    RT[:, 2] = (RT[:, 1] - meanRT) / stdRT
    RT[:, 3] = np.abs(RT[:, 2])

    # Smooth the VTC and compute the median
    L = 20
    W = gaussian(L, std=L/2) / 2  # Creates a Gaussian window with width L
    VTC_smoothed = filtfilt(W, np.sum(W), RT[:, 3])
    median_VTC = np.median(VTC_smoothed)


    # Plot the RT deviance z-score

    f, ax1 = p.subplots(1, 1, figsize=(12, 7))

    # Plot RT deviance z-score
    ax1.plot(t, RT[:, 3], color='m', linewidth=0.5)
    ax1.set_ylabel('RT deviance z-score', fontsize=16, color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    # # Smooth pupil diameter
    # pd_smoothed = filtfilt(np.ones(100), 100, np.mean(pd, axis=1))

    # # Create a second y-axis for pupil diameter
    # ax2 = ax1.twinx()
    # ax2.plot(t_pd, pd_smoothed, 'c', linewidth=1)
    # ax2.set_ylabel('Pupil Diameter (mm)', fontsize=16, color='c')
    # ax2.tick_params(axis='y', labelcolor='c')

    # Highlight VTC smoothed values
    VTC_smoothed_in = np.full_like(VTC_smoothed, np.nan)
    VTC_smoothed_out = np.full_like(VTC_smoothed, np.nan)
    VTC_smoothed_in[VTC_smoothed < median_VTC] = VTC_smoothed[VTC_smoothed < median_VTC]
    VTC_smoothed_out[VTC_smoothed >= median_VTC] = VTC_smoothed[VTC_smoothed >= median_VTC]

    ax1.plot(t, VTC_smoothed_in, 'r', linewidth=2)
    ax1.plot(t, VTC_smoothed_out, 'b', linewidth=2)

    # Plot omission and commission errors
    ax1.plot(t[lst_omision_error], 2 * np.ones(len(lst_omision_error)), 'ko', markersize=10, markeredgecolor='k', markerfacecolor='k')
    ax1.plot(t[lst_commision_error], 2 * np.ones(len(lst_commision_error)), 'kd', markersize=10, markeredgecolor='k', markerfacecolor=[0.6, 0.6, 0.6])

    # Plot event markers
    lst_mnt = np.where( stim.trial_type == 'mnt' )[0]
    for i_stim in lst_mnt:
        ax1.axvline(x=t[i_stim], color='k')

    ax1.set_xlim([0, t[-1]])
    # ax2.set_xlim([0, t[-1]])

    # Set legend
    legend_labels = ['VTC', 'VTC in', 'VTC out','Omission', 'Commission']
    ax1.legend(legend_labels, loc='upper right')
    
    ax1.set_title( f'mean RT = {1e3*meanRT:.0f}ms,  commision {len(lst_commision_error)} / {len(lst_mnt)}' )

    # f.tight_layout()
    p.xlabel('Time (s)', fontsize=16)
    # p.show()
    p.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # give a title to the figure
    p.suptitle(filenm)

    p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', filenm + "_DQR_gradCPT_VTC.png") )
    p.close()


#%%


# !!! keeping old plotting func to go back and use slope stuff for diff fig
def plotDQR_old( rec = None, chs_pruned = None, slope = None, filenm = None, filepath = None, stim_lst_str = None ):

    f, ax = p.subplots(3, 2, figsize=(11, 14))

    # Plot GVTD
    ax[0][0].plot( rec.aux_ts["gvtd"].time, rec.aux_ts["gvtd"], color='b', label="GVTD")
    ax[0][0].plot( rec.aux_ts["gvtd"].time, rec.aux_ts["gvtd_corrected"], color='#ff4500', label="GVTD Corrected")
    ax[0][0].set_xlabel("time / s")
    ax[0][0].set_title(f"{filenm}")
    thresh = quality._get_gvtd_threshold(rec.aux_ts['gvtd'], 'histogram_mode', n_std = 10)
    ax[0][0].axhline(thresh.values, color='b', linestyle='--', label=f'Thresh {thresh:.1e}')
    thresh_corrected = quality._get_gvtd_threshold(rec.aux_ts['gvtd_corrected'], 'histogram_mode', n_std = 10)

    ax[0][0].axhline(thresh_corrected.values, color='#ff4500', linestyle='--', label=f'Thresh {thresh_corrected:.1e}')
    ax[0][0].legend()

    stim = rec.stim.copy()
    if stim_lst_str is not None:
        plots.plot_stim_markers(ax[0][0], stim[stim.trial_type.isin(stim_lst_str)], y=1)
    # add stim_lst_str to the legend
    handles, labels = ax[0][0].get_legend_handles_labels()
    labels.append(stim_lst_str)
    ax[0][0].legend(handles, labels)

    # plot the pruned channels
    idx_good = np.where(chs_pruned.values == 0.4)[0]
    plots.scalp_plot( 
            rec["amp"],
            rec.geo3d,
            chs_pruned,
            ax[0][1],
            cmap='gist_rainbow',
            vmin=0,
            vmax=1,
            optode_labels=False,
            title=f"Pruned Channels {(len(chs_pruned)-len(idx_good))/len(chs_pruned)*100:.1f}%",
            optode_size=6
        )

    ax1 = ax[1][0]
    variance_vals = np.log10( rec['od_o_tddr'].values.var(axis=2))
    variance_vals_da = xr.DataArray(variance_vals, dims=["channel", "wavelength"], coords={"channel": rec["od"].channel, "wavelength": rec["od"].wavelength})
    max_variance = np.nanmax(variance_vals)
    min_variance = np.nanmin(variance_vals)
    wav =rec['amp'].wavelength.values[0]
    plots.scalp_plot(
            rec["od"],
            rec.geo3d,
            variance_vals_da.isel(wavelength=0),
            ax1,
            cmap='jet',
            vmin=min_variance,
            vmax=max_variance,
            optode_labels=False,
            title=f"OD_o_tddr Variance - {wav} nm",
            optode_size=6
        )

    ax1 = ax[1][1]
    variance_vals = np.log10( rec['od_tddr'].values.var(axis=2))
    variance_vals_da = xr.DataArray(variance_vals, dims=["channel", "wavelength"], coords={"channel": rec["od"].channel, "wavelength": rec["od"].wavelength})
    max_variance = np.nanmax(variance_vals)
    min_variance = np.nanmin(variance_vals)
    wav = rec['amp'].wavelength.values[0]
    plots.scalp_plot(
            rec["od"],
            rec.geo3d,
            variance_vals_da.isel(wavelength=1),
            ax1,
            cmap='jet',
            vmin=min_variance,
            vmax=max_variance,
            optode_labels=False,
            title=f"OD_tddr Variance = {wav} nm",
            optode_size=6
        )

    # plot the base slope as a scalp plot
    # get the slope values for each channel and change units to per 10min rather than per second
    if slope[0] is not None:
        ax1 = ax[2][0]
        slope_vals = slope[0].slope.values * 60 * 10
        # create a data array of the slope values
        slope_vals_da = xr.DataArray(slope_vals, dims=["channel", "wavelength"], coords={"channel": rec["od"].channel, "wavelength": rec["od"].wavelength})
        # get max of the absolute value of the slope values
        max_slope = np.nanmax(np.abs(slope_vals))
        plots.scalp_plot(
                rec["od"],
                rec.geo3d,
                slope_vals_da.isel(wavelength=0),
                ax1,
                cmap='jet',
                vmin=-max_slope,
                vmax=max_slope,
                optode_labels=False,
                title="Baseline Slope",
                optode_size=6
            )

    # plot the tddr slope as a scalp plot
    ax1 = ax[2][1]
    # get the slope values for each channel and change units to per 10min rather than per second
    slope_vals = slope[1].slope.values * 60 * 10
    # create a data array of the slope values
    slope_vals_da = xr.DataArray(slope_vals, dims=["channel", "wavelength"], coords={"channel": rec["od_tddr"].channel, "wavelength": rec["od_tddr"].wavelength})
    # get max of the absolute value of the slope values
    max_slope = np.nanmax(np.abs(slope_vals))
    plots.scalp_plot(
            rec["od_tddr"],
            rec.geo3d,
            slope_vals_da.isel(wavelength=0),
            ax1,
            cmap='jet',
            vmin=-max_slope,
            vmax=max_slope,
            optode_labels=False,
            title="TDDR Slope",
            optode_size=6
        )
        
    # give a title to the figure
    p.suptitle(filenm)

    p.savefig( os.path.join(filepath, 'derivatives', 'plots', 'DQR', filenm + "_DQR.png") )
    p.close()

    # Plot the GVTD Histograms
    # thresh = make_gvtd_hist(rec.aux_ts['gvtd'], plot_thresh=True, stat_type='histogram_mode', n_std=10) # !!! why recalc thresh? do we need to?
    # p.suptitle(filenm)
    # p.savefig( os.path.join(filepath, 'derivatives', 'plots', 'DQR', filenm + "_DQR_gvtd_hist.png") )
    # p.close()

    # thresh_corrected = make_gvtd_hist(rec.aux_ts['gvtd_tddr'], plot_thresh=True, stat_type='histogram_mode', n_std=10)
    # p.suptitle(filenm)
    # p.savefig( os.path.join(filepath, 'derivatives', 'plots', 'DQR', filenm + "_DQR_gvtd_hist_corrected.png") )
    # p.close()
    
    der_dir = os.path.join(filepath, 'derivatives', 'plots', 'DQR', 'gvtd')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    
    thresh_b4, thresh_corrected = make_gvtd_hist(rec.aux_ts['gvtd_tddr'], plot_thresh=True, stat_type='histogram_mode', n_std=10)
    p.suptitle(filenm)
    p.savefig( os.path.join(filepath, 'derivatives', 'plots', 'DQR','gvtd', filenm + "_DQR_gvtd_hist.png") )
    p.close()

    return