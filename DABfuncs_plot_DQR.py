
import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.sigproc.motion_correct as motion_correct
import cedalion.xrutils as xrutils
import cedalion.datasets as datasets
import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
#import cedalion.vis.time_series as vTimeSeries
from cedalion import units
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from sklearn.decomposition import PCA
from cedalion.sigdecomp.ERBM import ERBM
from scipy import stats



def plotDQR( rec = None, chs_pruned = None, slope = None, filenm = None, flagSave = False, filepath = None, stim_lst_str = None ):

    f, ax = p.subplots(2, 2, figsize=(11, 11))

    # Plot GVTD
    ax[0][0].plot( rec.aux_ts["gvtd"].time, rec.aux_ts["gvtd"], color='b', label="GVTD")
    ax[0][0].plot( rec.aux_ts["gvtd"].time, rec.aux_ts["gvtd_tddr"], color='#ff4500', label="GVTD TDDR")
    ax[0][0].set_xlabel("time / s")
    ax[0][0].set_title(f"{filenm}")
    thresh = quality.find_gvtd_thresh(rec.aux_ts['gvtd'].values, quality.gvtd_stat_type.Histogram_Mode, n_std = 10)
    ax[0][0].axhline(thresh, color='b', linestyle='--', label=f'Thresh {thresh:.1e}')
    thresh_tddr = quality.find_gvtd_thresh(rec.aux_ts['gvtd_tddr'].values, quality.gvtd_stat_type.Histogram_Mode, n_std = 10)
    ax[0][0].axhline(thresh_tddr, color='#ff4500', linestyle='--', label=f'Thresh {thresh_tddr:.1e}')
    ax[0][0].legend()

    stim = rec.stim.copy()
    if stim_lst_str is not None:
        plots.plot_stim_markers(ax[0][0], stim[stim.trial_type.isin(stim_lst_str)], y=1)
    # add stim_lst_str to the legend
    handles, labels = ax[0][0].get_legend_handles_labels()
    labels.append(stim_lst_str)
    ax[0][0].legend(handles, labels)

    # # Plot the GVTD Histograms
    # #thresh = pfDAB.find_gvtd_thresh(rec[idx_file].aux_ts['gvtd'].values, statType, nStd)
    # thresh = quality.make_gvtd_hist(recTmp.aux_ts['gvtd'].values, plot_thresh=True, stat_type=stat_type, n_std=n_std)

    # # #thresh = pfDAB.find_gvtd_thresh(rec[idx_file].aux_ts['gvtd_tddr'].values, statType, nStd)
    # thresh_tddr = quality.make_gvtd_hist(recTmp.aux_ts['gvtd_tddr'].values, plot_thresh=True, stat_type=stat_type, n_std=n_std)




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

    # plot the base slope as a scalp plot
    # get the slope values for each channel and change units to per 10min rather than per second
    if slope[0] is not None:
        slope_vals = slope[0].slope.values * 60 * 10
        # create a data array of the slope values
        slope_vals_da = xr.DataArray(slope_vals, dims=["channel", "wavelength"], coords={"channel": rec["od"].channel, "wavelength": rec["od"].wavelength})
        # get max of the absolute value of the slope values
        max_slope = np.nanmax(np.abs(slope_vals))
        plots.scalp_plot(
                rec["od"],
                rec.geo3d,
                slope_vals_da.isel(wavelength=0),
                ax[1][0],
                cmap='jet',
                vmin=-max_slope,
                vmax=max_slope,
                optode_labels=False,
                title="Baseline Slope",
                optode_size=6
            )

    # plot the tddr slope as a scalp plot
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
            ax[1][1],
            cmap='jet',
            vmin=-max_slope,
            vmax=max_slope,
            optode_labels=False,
            title="TDDR Slope",
            optode_size=6
        )
        
    # give a title to the figure
    p.suptitle(filenm)

    if flagSave:
        p.savefig( os.path.join(filepath, 'derivatives', 'plots', filenm + "_DQR.png") )
        p.close()

        # Plot the GVTD Histograms
        #thresh = pfDAB.find_gvtd_thresh(rec[idx_file].aux_ts['gvtd'].values, statType, nStd)
        thresh = quality.make_gvtd_hist(rec.aux_ts['gvtd'].values, plot_thresh=True, stat_type=quality.gvtd_stat_type.Histogram_Mode, n_std=10)
        p.suptitle(filenm)
        p.savefig( os.path.join(filepath, 'derivatives', 'plots', filenm + "_DQR_gvtd_hist.png") )
        p.close()

        # #thresh = pfDAB.find_gvtd_thresh(rec[idx_file].aux_ts['gvtd_tddr'].values, statType, nStd)
        thresh_tddr = quality.make_gvtd_hist(rec.aux_ts['gvtd_tddr'].values, plot_thresh=True, stat_type=quality.gvtd_stat_type.Histogram_Mode, n_std=10)
        p.suptitle(filenm)
        p.savefig( os.path.join(filepath, 'derivatives', 'plots', filenm + "_DQR_gvtd_hist_tddr.png") )
        p.close()


    return


def plotDQR_sidecar(file_json, rec, filepath, filenm):

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
            rec["conc_tddr"],
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
            rec["conc_tddr"],
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

    p.savefig( os.path.join(filepath, 'derivatives', 'plots', filenm + "_DQR_sigVdis.png") )
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

    p.savefig( os.path.join(filepath, 'derivatives', 'plots', filenm + "_DQR_calib.png") )
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

    p.savefig( os.path.join(filepath, 'derivatives', 'plots', filenm + "_DQR_crosstalk.png") )
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


def plot_group_dqr( n_subjects, n_files_per_subject, chs_pruned_subjs, slope_base_subjs, slope_tddr_subjs, gvtd_tddr_subjs, snr0_subjs, snr1_subjs, subj_ids, rec, filepath, flag_plot = True):   

    chs_pruned_percent = np.zeros( (n_subjects, n_files_per_subject) )
    for subj_idx in range(n_subjects):
        for file_idx in range(n_files_per_subject):
            n_chs = len(chs_pruned_subjs[subj_idx][file_idx])
            n_chs_pruned = len( np.where( chs_pruned_subjs[subj_idx][file_idx] != 0.4 )[0] )
            chs_pruned_percent[subj_idx, file_idx] = 100 * n_chs_pruned / n_chs

    slope_base_abs = np.zeros( (n_subjects, n_files_per_subject) )
    slope_tddr_abs = np.zeros( (n_subjects, n_files_per_subject) )
    for subj_idx in range(n_subjects):
        for file_idx in range(n_files_per_subject):
            slope_base_abs[subj_idx, file_idx] = np.nanmax(np.abs(slope_base_subjs[subj_idx][file_idx].slope.values)) * 60 * 10
            slope_tddr_abs[subj_idx, file_idx] = np.nanmax(np.abs(slope_tddr_subjs[subj_idx][file_idx].slope.values)) * 60 * 10

    gvtd_tddr_mean = np.zeros( (n_subjects, n_files_per_subject) )
    for subj_idx in range(n_subjects):
        for file_idx in range(n_files_per_subject):
            gvtd_tddr_mean[subj_idx, file_idx] = gvtd_tddr_subjs[subj_idx][file_idx]

    snr0_mean = np.zeros( (n_subjects, n_files_per_subject) )
    snr1_mean = np.zeros( (n_subjects, n_files_per_subject) )
    for subj_idx in range(n_subjects):
        for file_idx in range(n_files_per_subject):
            snr0_mean[subj_idx, file_idx] = snr0_subjs[subj_idx][file_idx]
            snr1_mean[subj_idx, file_idx] = snr1_subjs[subj_idx][file_idx]
    lambda0 = rec[0][0]["amp_pruned"].wavelength[0].wavelength.values
    lambda1 = rec[0][0]["amp_pruned"].wavelength[1].wavelength.values

    # bar graph of the percentage of channels pruned
    f, ax = p.subplots(2,2, figsize=(9,10))

    # channels pruned
    axtmp = ax[0][0]
    axtmp.bar( np.arange(n_subjects), np.mean(chs_pruned_percent,axis=1), color='k' )
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject), chs_pruned_percent[subj_idx,:], color='gray', marker='x' )
    axtmp.set_xticks( np.arange(n_subjects), subj_ids )
    axtmp.set_xlabel('Subject')
    axtmp.set_ylabel('Percentage')
    axtmp.set_title('Channels Pruned')

    # gvtd mean
    axtmp = ax[0][1]
    axtmp.bar( np.arange(n_subjects), np.mean(gvtd_tddr_mean,axis=1), color='k' )
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject), gvtd_tddr_mean[subj_idx,:], color='gray', marker='x' )
    axtmp.set_xticks( np.arange(n_subjects), subj_ids )
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
    axtmp.set_xticks( np.arange(n_subjects), subj_ids )
    axtmp.set_xlabel('Subject')
    axtmp.legend()
    axtmp.set_title('SNR')

    # slopes
    axtmp = ax[1][1]
    axtmp.bar( np.arange(n_subjects)-h, np.mean(slope_base_abs,axis=1), color='b', label='Base', width=0.4)
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject)-h, slope_base_abs[subj_idx,:], color='c', marker='x' )
    axtmp.bar( np.arange(n_subjects)+h, np.mean(slope_tddr_abs,axis=1), color='r', label='TDDR', width=0.4)
    for subj_idx in range(n_subjects):
        axtmp.scatter( subj_idx*np.ones(n_files_per_subject)+h, slope_tddr_abs[subj_idx,:], color='m', marker='x' )
    axtmp.set_xticks( np.arange(n_subjects), subj_ids )
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
    axtmp.set_title('Max Slope TDDR')

    # give a title to the figure
    dirnm = os.path.basename(os.path.normpath(filepath))
    p.suptitle(f'Data set - {dirnm}')

    p.savefig( os.path.join(filepath, 'derivatives', 'plots', "DQR_group.png") )
    
    if flag_plot:
        p.show()
