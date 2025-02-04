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
from cedalion import units
import numpy as np
import pandas as pd

import json



# import my own functions from a different directory
import sys
import DABfuncs_plot_DQR as pfDAB_dqr





def load_and_preprocess( cfg_dataset, cfg_preprocess, cfg_dqr ):

    # make sure derivatives folders exist
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'ica')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)



    n_subjects = len(cfg_dataset['subj_ids'])
    n_files_per_subject = len(cfg_dataset['file_ids'])

    # loop over subjects and files
    for subj_idx in range(n_subjects):
        for file_idx in range(n_files_per_subject):

            filenm = cfg_dataset['filenm_lst'][subj_idx][file_idx]

            print( f"Loading {subj_idx+1} of {n_subjects} subjects, {file_idx+1} of {n_files_per_subject} files : {filenm}" )

            subStr = filenm.split('_')[0]
            subDir = os.path.join(cfg_dataset['root_dir'], subStr, 'nirs')

            file_path = os.path.join(subDir, filenm )
            records = cedalion.io.read_snirf( file_path ) 

            recTmp = records[0]

            foo = file_path[:-5] + '_events.tsv'
            # check if the events.tsv file exists
            if not os.path.exists( foo ):
                print( f"Error: File {foo} does not exist" )
            else:
                stim_df = pd.read_csv( file_path[:-5] + '_events.tsv', sep='\t' )
                recTmp.stim = stim_df

            

            recTmp = preprocess( recTmp, cfg_preprocess['median_filt'] )
            recTmp, chs_pruned, sci, psp = pruneChannels( recTmp, cfg_preprocess['cfg_prune'] )

            # calculate optical density
            recTmp["od"] = cedalion.nirs.int2od(recTmp['amp_pruned'])
            recTmp["od_o"] = cedalion.nirs.int2od(recTmp['amp'])
            

            # Calculate gvtd
            recTmp.aux_ts["gvtd"], _ = quality.gvtd(recTmp['amp_pruned'])

        
            # Get the slope of 'od' before motion correction and any bandpass filtering
            foo = recTmp['od'].copy()
            foo = foo.pint.dequantify()
            slope_base = foo.polyfit(dim='time', deg=1).sel(degree=1)
            slope_base = slope_base.rename({"polyfit_coefficients": "slope"})
            slope_base = slope_base.assign_coords(channel = recTmp['od'].channel)
            slope_base = slope_base.assign_coords(wavelength = recTmp['od'].wavelength)


            # FIXME: could have dictionary slope['base'], slope['tddr'], slope['splineSG'] etc


            # Spline SG
            if cfg_preprocess['cfg_motion_correct']['flag_do_splineSG']:
                recTmp, slope = motionCorrect_SplineSG( recTmp, cfg_preprocess['cfg_bandpass'] )
            else:
                slope = None

            # TDDR
            recTmp['od_tddr'] = motion_correct.tddr( recTmp['od'] )
            recTmp['od_o_tddr'] = motion_correct.tddr( recTmp['od_o'] )


            # Get slopes after TDDR before bandpass filtering
            slope_tddr = recTmp['od_tddr'].polyfit(dim='time', deg=1).sel(degree=1)
            slope_tddr = slope_tddr.rename({"polyfit_coefficients": "slope"})
            slope_tddr = slope_tddr.assign_coords(channel = recTmp['od_tddr'].channel)
            slope_tddr = slope_tddr.assign_coords(wavelength = recTmp['od_tddr'].wavelength)


            # GVTD for TDDR before bandpass filtering
            amp_tddr = recTmp['od_tddr'].copy()
            amp_tddr.values = np.exp(-amp_tddr.values)
            recTmp.aux_ts['gvtd_tddr'], _ = quality.gvtd(amp_tddr)
            
            

            # bandpass filter od_tddr
            fmin = cfg_preprocess['cfg_bandpass']['fmin']
            fmax = cfg_preprocess['cfg_bandpass']['fmax']
            recTmp['od_tddr'] = cedalion.sigproc.frequency.freq_filter(recTmp['od_tddr'], fmin, fmax)
            recTmp['od_o_tddr'] = cedalion.sigproc.frequency.freq_filter(recTmp['od_o_tddr'], fmin, fmax)


            # SplineSG Conc
            dpf = xr.DataArray(
                [1, 1],
                dims="wavelength",
                coords={"wavelength": recTmp['amp'].wavelength},
            )
            if cfg_preprocess['cfg_motion_correct']['flag_do_splineSG']:
                recTmp['conc_splineSG'] = cedalion.nirs.od2conc(recTmp['od_splineSG'], recTmp.geo3d, dpf, spectrum="prahl")

            # TDDR Conc
            recTmp['conc_tddr'] = cedalion.nirs.od2conc(recTmp['od_tddr'], recTmp.geo3d, dpf, spectrum="prahl")
            recTmp['conc_o_tddr'] = cedalion.nirs.od2conc(recTmp['od_o_tddr'], recTmp.geo3d, dpf, spectrum="prahl")

            #
            # Plot DQRs
            #
            lambda0 = recTmp['amp_pruned'].wavelength[0].wavelength.values
            lambda1 = recTmp['amp_pruned'].wavelength[1].wavelength.values
            snr0, _ = quality.snr(recTmp['amp_pruned'].sel(wavelength=lambda0), cfg_preprocess['cfg_prune']['snr_thresh'])
            snr1, _ = quality.snr(recTmp['amp_pruned'].sel(wavelength=lambda1), cfg_preprocess['cfg_prune']['snr_thresh'])


            pfDAB_dqr.plotDQR( recTmp, chs_pruned, [slope_base, slope_tddr], filenm, cfg_dataset['root_dir'], cfg_dqr['stim_lst_dqr'] )

            # load the sidecar json file 
            if os.path.exists(file_path + '.json'):
                with open(file_path + '.json') as json_file:
                    file_json = json.load(json_file)
                if 'dataSDWP_LowHigh' in file_json:
                    pfDAB_dqr.plotDQR_sidecar(file_json, recTmp, cfg_dataset['root_dir'], filenm )

            snr0 = np.nanmedian(snr0.values)
            snr1 = np.nanmedian(snr1.values)

            #
            # Organize the processed data
            #
            if subj_idx == 0 and file_idx == 0:
                rec = []
                chs_pruned_subjs = []
                slope_base_subjs = []
                slope_tddr_subjs = []
                gvtd_tddr_subjs = []
                snr0_subjs = []
                snr1_subjs = []

                rec.append( [recTmp] )
                chs_pruned_subjs.append( [chs_pruned] )
                slope_base_subjs.append( [slope_base] )
                slope_tddr_subjs.append( [slope_tddr] )
                gvtd_tddr_subjs.append( [np.nanmean(recTmp.aux_ts['gvtd_tddr'].values)] )
                snr0_subjs.append( [snr0] )
                snr1_subjs.append( [snr1] )
            elif file_idx == 0:
                rec.append( [recTmp] )
                chs_pruned_subjs.append( [chs_pruned] )
                slope_base_subjs.append( [slope_base] )
                slope_tddr_subjs.append( [slope_tddr] )
                gvtd_tddr_subjs.append( [np.nanmean(recTmp.aux_ts['gvtd_tddr'].values)] )
                snr0_subjs.append( [snr0] )
                snr1_subjs.append( [snr1] )
            else:
                rec[subj_idx].append( recTmp )
                chs_pruned_subjs[subj_idx].append( chs_pruned )
                slope_base_subjs[subj_idx].append( slope_base )
                slope_tddr_subjs[subj_idx].append( slope_tddr )
                gvtd_tddr_subjs[subj_idx].append( np.nanmean(recTmp.aux_ts['gvtd_tddr'].values) )
                snr0_subjs[subj_idx].append( snr0 )
                snr1_subjs[subj_idx].append( snr1 )

        # End of file loop
    # End of subject loop

    # plot the group DQR
    pfDAB_dqr.plot_group_dqr( n_subjects, n_files_per_subject, chs_pruned_subjs, slope_base_subjs, slope_tddr_subjs, gvtd_tddr_subjs, snr0_subjs, snr1_subjs, cfg_dataset['subj_ids'], rec, cfg_dataset['root_dir'], flag_plot=False )

    return rec, chs_pruned_subjs


def preprocess(rec, median_filt ):

    # replace negative values and NaNs with a small positive value
    rec['amp'] = rec['amp'].where( rec['amp']>0, 1e-18 ) 
    rec['amp'] = rec['amp'].where( ~rec['amp'].isnull(), 1e-18 ) 

    # if first value is 1e-18 then replace with second value
    indices = np.where(rec['amp'][:,0,0] == 1e-18)
    rec['amp'][indices[0],0,0] = rec['amp'][indices[0],0,1]
    indices = np.where(rec['amp'][:,1,0] == 1e-18)
    rec['amp'][indices[0],1,0] = rec['amp'][indices[0],1,1]

    # apply a median filter to rec['amp'] along the time dimension
    # FIXME: this is to handle spikes that arise from the 1e-18 values inserted above or from other causes, 
    #        but this is an effective LPF. TDDR may handle this
    # Pad the data before applying the median filter
    pad_width = 1  # Adjust based on the kernel size
    padded_amp = rec['amp'].pad(time=(pad_width, pad_width), mode='edge')
    # Apply the median filter to the padded data
    filtered_padded_amp = padded_amp.rolling(time=median_filt, center=True).reduce(np.median)
    # Trim the padding after applying the filter
    rec['amp'] = filtered_padded_amp.isel(time=slice(pad_width, -pad_width))

    return rec


def pruneChannels( rec, cfg_prune ):

    amp_threshs = cfg_prune['amp_threshs']
    snr_thresh = cfg_prune['snr_thresh']
    sd_threshs = cfg_prune['sd_threshs']

    amp_threshs_sat = [0., amp_threshs[1]]
    amp_threshs_low = [amp_threshs[0], 1]

    # then we calculate the masks for each metric: SNR, SD distance and mean amplitude
    snr, snr_mask = quality.snr(rec['amp'], snr_thresh)
    _, sd_mask = quality.sd_dist(rec['amp'], rec.geo3d, sd_threshs)
    _, amp_mask = quality.mean_amp(rec['amp'], amp_threshs)
    _, amp_mask_sat = quality.mean_amp(rec['amp'], amp_threshs_sat)
    _, amp_mask_low = quality.mean_amp(rec['amp'], amp_threshs_low)

    # create an xarray of channel labels with values indicated why pruned
    chs_pruned = xr.DataArray(np.zeros(rec['amp'].shape[0]), dims=["channel"], coords={"channel": rec['amp'].channel})

    #i initialize chs_pruned to 0.4
    chs_pruned[:] = 0.4

    # get indices for where snr_mask = false
    snr_mask_false = np.where(snr_mask == False)[0]
    chs_pruned[snr_mask_false] = 0.19 # poor snrf channels

    # get indices for where amp_mask_sat = false
    amp_mask_false = np.where(amp_mask_sat == False)[0]
    chs_pruned[amp_mask_false] = 0.0 # saturated channels

    # get indices for where amp_mask_low = false
    amp_mask_false = np.where(amp_mask_low == False)[0]
    chs_pruned[amp_mask_false] = 0.8 # low signal channels

    # get indices for where sd_mask = false
    sd_mask_false = np.where(sd_mask == False)[0]
    chs_pruned[sd_mask_false] = 0.65 # SDS channels


    # put all masks in a list
    masks = [snr_mask, sd_mask, amp_mask]

    # prune channels using the masks and the operator "all", which will keep only channels that pass all three metrics
    amp_pruned, drop_list = quality.prune_ch(rec['amp'], masks, "all", flag_drop=False)

    # record the pruned array in the record
    rec['amp_pruned'] = amp_pruned



    perc_time_clean_thresh = cfg_prune['perc_time_clean_thresh']
    sci_threshold = cfg_prune['sci_threshold']
    psp_threshold = cfg_prune['psp_threshold']
    window_length = cfg_prune['window_length']

    # Here we can assess the scalp coupling index (SCI) of the channels
    sci, sci_mask = quality.sci(rec['amp_pruned'], window_length, sci_threshold)

    # We can also look at the peak spectral power which takes the peak power of the cross-correlation signal between the cardiac band of the two wavelengths
    psp, psp_mask = quality.psp(rec['amp_pruned'], window_length, psp_threshold)

    # create a mask based on SCI or PSP or BOTH
    if cfg_prune['flag_use_sci'] and cfg_prune['flag_use_psp']:
        sci_x_psp_mask = sci_mask & psp_mask
    elif cfg_prune['flag_use_sci']:
        sci_x_psp_mask = sci_mask
    elif cfg_prune['flag_use_psp']:
        sci_x_psp_mask = psp_mask
    else:
        return rec, chs_pruned, sci, psp

    perc_time_clean = sci_x_psp_mask.sum(dim="time") / len(sci.time)
    perc_time_mask = xrutils.mask(perc_time_clean, True)
    perc_time_mask = perc_time_mask.where(perc_time_clean > perc_time_clean_thresh, False)

    # add the lambda dimension to the perc_time_mask with two entries, 760 and 850, and duplicate the existing column of data to it
    perc_time_mask = xr.concat([perc_time_mask, perc_time_mask], dim="lambda")

    # prune channels using the masks and the operator "all", which will keep only channels that pass all three metrics
    perc_time_pruned, drop_list = quality.prune_ch(rec['amp_pruned'], perc_time_mask, "all", flag_drop=False)

    # record the pruned array in the record
    rec['amp_pruned'] = perc_time_pruned

    # modify xarray of channel labels with value of 0.95 for channels that are pruned by SCI and/or PSP
    chs_pruned.loc[drop_list] = 0.95

    return rec, chs_pruned, sci, psp






def motionCorrect_SplineSG( rec, cfg_bandpass ):
    
    # FIXME: need to pass cfg_motion_correct and consider spline only and any other spline or splineSG params

    fmin = cfg_bandpass['fmin']
    fmax = cfg_bandpass['fmax']

    if 0: # do just the Spline correct
        M = quality.detect_outliers(rec['od'], 1 * units.s)

        tIncCh = quality.detect_baselineshift(rec['od'], M)

        fNIRSdata = rec['od'].pint.dequantify()
        fs = quality.sampling_rate(fNIRSdata)
        fNIRSdata_lpf2 = fNIRSdata.cd.freq_filter(0, 2, butter_order=4)
            
        PADDING_TIME = 12 * units.s # FIXME configurable?
        extend = int(np.round(PADDING_TIME  * fs))  # extension for padding

        # pad fNIRSdata and tIncCh for motion correction
        fNIRSdata_lpf2_pad = fNIRSdata_lpf2.pad(time=extend, mode="edge")

        tIncCh_pad = tIncCh.pad(time=extend, mode="constant", constant_values=True)

        rec['od_splineSG'] = motion_correct.motion_correct_spline(fNIRSdata_lpf2_pad, tIncCh_pad, 0.99)

    else: # Do SplineSG
        frame_size = 10 * units.s
        rec['od_splineSG'] = motion_correct.motion_correct_splineSG(rec['od'], p=0.99, frame_size=frame_size)

    # fit a line to the time course for each channel in od_splineSG
    slope = rec['od_splineSG'].polyfit(dim='time', deg=1).sel(degree=1)
    slope = slope.rename({"polyfit_coefficients": "slope"})
    slope = slope.assign_coords(channel=rec['od_splineSG'].channel)
    slope = slope.assign_coords(wavelength=rec['od_splineSG'].wavelength)

    rec['od_splineSG'] = cedalion.sigproc.frequency.freq_filter(rec['od_splineSG'], fmin, fmax)


    return rec, slope



def Conc( rec = None ):

    dpf = xr.DataArray(
        [1, 1],
        dims="wavelength",
        coords={"wavelength": rec['amp'].wavelength},
    )

    # calculate concentrations
    rec['conc_splineSG'] = cedalion.nirs.od2conc(rec['od_splineSG'], rec.geo3d, dpf, spectrum="prahl")

    return rec






