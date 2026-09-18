import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.sigproc.motion as motion_correct
import cedalion.xrutils as xrutils
import cedalion.models.glm as glm
import xarray as xr
import matplotlib.pyplot as p
import cedalion.vis as plots
from cedalion import units
import numpy as np
import pandas as pd

import json


# import my own functions from a different directory
import sys
import module_plot_DQR as pfDAB_dqr
# import module_imu_glm_filter as pfDAB_imu

import pdb


def load_and_preprocess( cfg_dataset, cfg_preprocess ):
    '''
    This function will load all the data for the specified subject and file IDs, and preprocess the data.
    This function will also create several data quality report (DQR) figures that are saved in /derivatives/plots.
    The function will return the preprocessed data and a list of the filenames that were loaded, both as 
    two dimensional lists [subj_idx][file_idx].
    The data is returned as a recording container with the following fields:
      timeseries - the data matrices with dimensions of ('channel', 'wavelength', 'time') 
         or ('channel', 'HbO/HbR', 'time') depending on the data type. 
         The following sub-fields are included:
            'amp' - the original amplitude data slightly processed to remove negative and NaN values and to 
               apply a 3 point median filter to remove outliers.
            'amp_pruned' - the 'amp' data pruned according to the SNR, SD, and amplitude thresholds.
            'od' - the optical density data
            'od_tddr' - the optical density data after TDDR motion correction is applied
            'conc_tddr' - the concentration data obtained from 'od_tddr'
            'od_splineSG' and 'conc_splineSG' - returned if splineSG motion correction is applied (i.e. flag_do_splineSG=True)
      stim - the stimulus data with 'onset', 'duration', and 'trial_type' fields and more from the events.tsv files.
      aux_ts - the auxiliary time series data from the SNIRF files.
         In addition, the following aux sub-fields are added during pre-processing:
            'gvtd' - the global variance of the time derivative of the 'od' data.
            'gvtd_tddr' - the global variance of the time derivative of the 'od_tddr' data.
    '''

    if not cfg_dataset['derivatives_subfolder']:
        cfg_dataset['derivatives_subfolder'] = ''

    # make sure derivatives folders exist
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives',  cfg_dataset['derivatives_subfolder'])
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    # der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'ica')  # !!! not currently using
    # if not os.path.exists(der_dir):
    #     os.makedirs(der_dir)
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'processed_data')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)

    subj_ids = cfg_dataset['subj_ids']
    n_subjects = len(cfg_dataset['subj_ids'])
    n_files_per_subject = len(cfg_dataset['file_ids'])
    
    n_subjects_corrected =  len(cfg_dataset['subj_ids']) - len(cfg_dataset['subj_id_exclude'])
    
    # init variables outside loop
    rec = []
    chs_pruned_subjs = []
    slope_base_subjs = []
    slope_corrected_subjs = []
    gvtd_corrected_subjs = []
    snr0_subjs = []
    snr1_subjs = []
    subj_idx_included = 0  # NEW: index for included subjects only

    # loop over subjects and files
    for subj_idx in range(n_subjects):
        
        # if subject id is in excluded list, skip processing
        if subj_ids[subj_idx] in cfg_dataset['subj_id_exclude']:
            print(f"Subject {subj_ids[subj_idx]} listed in subj_id_exclude. Skipping processing for this subject.")
            continue
        
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
            if not os.path.exists( foo ):  # !!! assert?
                print( f"Error: File {foo} does not exist" )
            else:
                stim_df = pd.read_csv( file_path[:-5] + '_events.tsv', sep='\t' )
                recTmp.stim = stim_df
                
            # Walking filter checks:
            if cfg_preprocess['cfg_motion_correct']['flag_do_imu_glm']:
                
                # Check if walking condition exists in rec.stim, if no then sets flag_do_imu_glm to false
                if not recTmp.stim.isin(["start_walk", "end_walk"]).any().any():
                    cfg_preprocess['cfg_motion_correct']['flag_do_imu_glm'] = False
                    print("No walking condition found in events.tsv. Skipping imu glm filtering step.")
                    
                 # Check if at least 1 imu value (using ACCEL_X) is non-zero (making sure there is imu data in snirf)
                if not np.any(recTmp.aux_ts['ACCEL_X'] != 0):   # !!! this might not always work and I'm only checking aux_x
                    cfg_preprocess['cfg_motion_correct']['flag_do_imu_glm'] = False
                    print("There is no valid imu data in aux, skipping walking filter")

            recTmp = preprocess( recTmp, cfg_preprocess['median_filt'] )
            recTmp, chs_pruned, sci, psp = pruneChannels( recTmp, cfg_preprocess['cfg_prune'] )
            
            pruned_chans = chs_pruned.where(chs_pruned != 0.58, drop=True).channel.values # get array of channels that were pruned

            # Calculate OD 
            # if flag pruned channels is True, then do rest of preprocessing on pruned amp, if not then do preprocessing on unpruned data
            if cfg_preprocess['flag_prune_channels']:
                recTmp["od"] = cedalion.nirs.int2od(recTmp['amp_pruned'])                
            else:
                recTmp["od"] = cedalion.nirs.int2od(recTmp['amp'])
                del recTmp.timeseries['amp_pruned']   # delete pruned amp from time series
            
            # Calculate GVTD on pruned data
            amp_masked = prune_mask_ts(recTmp['amp'], pruned_chans)  # use chs_pruned to get gvtd w/out pruned data (could also zscore in gvtd func)
            recTmp.aux_ts["gvtd"], _ = quality.gvtd(amp_masked) 
            
            # Walking filter
            if cfg_preprocess['cfg_motion_correct']['flag_do_imu_glm']: 
                print('Starting imu glm filtering step on walking portion of data.')
                recTmp["od_corrected"] = pfDAB_imu.filterWalking(recTmp, "od", cfg_preprocess['cfg_motion_correct']['cfg_imu_glm'], filenm, cfg_dataset)
                
            # Get the slope of 'od' before motion correction and any bandpass filtering
            slope_base = quant_slope(recTmp, "od")

            # Spline SG # !!! fix me in future
            # if cfg_preprocess['cfg_motion_correct']['flag_do_splineSG']:
            #     recTmp, slope = motionCorrect_SplineSG( recTmp, cfg_preprocess['cfg_bandpass'] ) 
            # else:
            #     slope = None
            
            # TDDR
            if cfg_preprocess['cfg_motion_correct']['flag_do_tddr']:
                if 'od_corrected' in recTmp.timeseries.keys():
                    recTmp['od_corrected'] = motion_correct.tddr( recTmp['od_corrected'] )  
                else:   # do tddr on uncorrected od
                    recTmp['od_corrected'] = motion_correct.tddr( recTmp['od'] )  
                slope_corrected = quant_slope(recTmp, "od_corrected")  # Get slopes after correction before bandpass filtering
            else:
                if 'od_corrected' not in recTmp.timeseries.keys():
                    recTmp['od_corrected'] = recTmp['od']
                    slope_corrected = quant_slope(recTmp, "od_corrected")  # Get slopes after correction before bandpass filtering
            
            
            recTmp['od_corrected'] = recTmp['od_corrected'].where( ~recTmp['od_corrected'].isnull(), 0)  #1e-18 )  # replace any NaNs after TDDR
            
            # GVTD for Corrected od before bandpass filtering
            amp_corrected = recTmp['od_corrected'].copy()  
            amp_corrected.values = np.exp(-amp_corrected.values)
            amp_corrected_masked = prune_mask_ts(amp_corrected, pruned_chans)  # get "pruned" amp data post tddr
            recTmp.aux_ts['gvtd_corrected'], _ = quality.gvtd(amp_corrected_masked)    
            
            
            # Bandpass filter od_tddr
            if cfg_preprocess['cfg_bandpass']['flag_bandpass_filter']:
                recTmp['od_corrected'] = cedalion.sigproc.frequency.freq_filter(recTmp['od_corrected'], 
                                                                                cfg_preprocess['cfg_bandpass']['fmin'], 
                                                                                cfg_preprocess['cfg_bandpass']['fmax'])  
                
            # Convert OD to Conc
            dpf = xr.DataArray(
                [1, 1],
                dims="wavelength",
                coords={"wavelength": recTmp['amp'].wavelength},
            )
            
           
            # Conc
            recTmp['conc'] = cedalion.nirs.od2conc(recTmp['od_corrected'], recTmp.geo3d, dpf, spectrum="prahl")
            
            # GLM filtering step
            if cfg_preprocess['flag_do_GLM_filter']:
                
                recTmp = GLM(recTmp, 'conc', cfg_preprocess['cfg_GLM'], pruned_chans) # passing in pruned channels
                
                recTmp['od_corrected'] = cedalion.nirs.conc2od(recTmp['conc'], recTmp.geo3d, dpf)  # Convert GLM filtered data back to OD
                recTmp['od_corrected'] = recTmp['od_corrected'].transpose('channel', 'wavelength', 'time') # need to transpose to match recTmp['od'] bc conc2od switches the axes
            
            #
            # Plot DQRs
            #
           
            lambda0 = amp_masked.wavelength[0].wavelength.values
            lambda1 = amp_masked.wavelength[1].wavelength.values
            snr0, _ = quality.snr(amp_masked.sel(wavelength=lambda0), cfg_preprocess['cfg_prune']['snr_thresh'])
            snr1, _ = quality.snr(amp_masked.sel(wavelength=lambda1), cfg_preprocess['cfg_prune']['snr_thresh'])

            pfDAB_dqr.plotDQR( recTmp, chs_pruned, cfg_preprocess, filenm, cfg_dataset )
            
            # Plot slope before and after MA
            if cfg_preprocess['cfg_motion_correct']['flag_do_tddr']:
                pfDAB_dqr.plot_slope(recTmp, [slope_base, slope_corrected], cfg_preprocess, filenm, cfg_dataset)

            # !!! update this code adn the func in DQR
            # load the sidecar json file 
            file_path_json = os.path.join(cfg_dataset['root_dir'], 'sourcedata', 'raw', subStr, filenm + '.json')
            if os.path.exists(file_path_json):
                with open(file_path_json) as json_file:
                    file_json = json.load(json_file)
                if 'dataSDWP_LowHigh' in file_json:
                    pfDAB_dqr.plotDQR_sidecar(file_json, recTmp, cfg_dataset, filenm )

            snr0 = np.nanmedian(snr0.values)
            snr1 = np.nanmedian(snr1.values)

            #
            # Organize the processed data
            #
                
            if file_idx == 0:
                rec.append([recTmp])
                chs_pruned_subjs.append([chs_pruned])
                slope_base_subjs.append([slope_base])
                slope_corrected_subjs.append([slope_corrected])
                gvtd_corrected_subjs.append([np.nanmean(recTmp.aux_ts['gvtd_corrected'].values)])
                snr0_subjs.append([snr0])
                snr1_subjs.append([snr1])
            else:
                rec[subj_idx_included].append(recTmp)
                chs_pruned_subjs[subj_idx_included].append(chs_pruned)
                slope_base_subjs[subj_idx_included].append(slope_base)
                slope_corrected_subjs[subj_idx_included].append(slope_corrected)
                gvtd_corrected_subjs[subj_idx_included].append(np.nanmean(recTmp.aux_ts['gvtd_corrected'].values))
                snr0_subjs[subj_idx_included].append(snr0)
                snr1_subjs[subj_idx_included].append(snr1)
            
        subj_idx_included += 1  # Only incremented if subject is included
        
        # End of file loop
    # End of subject loop
    # plot the group DQR
    pfDAB_dqr.plot_group_dqr( n_subjects_corrected, n_files_per_subject, chs_pruned_subjs, slope_base_subjs, slope_corrected_subjs, gvtd_corrected_subjs, snr0_subjs, snr1_subjs, rec, cfg_dataset, flag_plot=False )
    # !!! plot_group_dqr will fail if no tddr ?

    
    return rec, chs_pruned_subjs


#%%

def prune_mask_ts(ts, pruned_chans):
    '''
    Function to mask pruned channels with NaN .. essentially repruning channels
    Parameters
    ----------
    ts : data array
        time series from rec[rec_str].
    pruned_chans : list or array
        list or array of channels that were pruned prior.

    Returns
    -------
    ts_masked : data array
        time series that has been "repruned" or masked with data for the pruned channels as NaN.

    '''
    mask = np.isin(ts.channel.values, pruned_chans)
    
    if ts.ndim == 3 and ts.shape[0] == len(ts.channel):
        mask_expanded = mask[:, None, None]  # (chan, wav, time)
    elif ts.ndim == 3 and ts.shape[1] == len(ts.channel):
        mask_expanded = mask[None, :, None]  # (chrom, chan, time)
    else:
        raise ValueError("Expected input shape to be either (chan, dim, time) or (dim, chan, time)")

    ts_masked = ts.where(~mask_expanded, np.nan)
    return ts_masked

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
    ''' Function that prunes channels based on cfg params.
        *Pruned channels are not dropped, instead they are set to NaN 
        '''

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
    chs_pruned[:] = 0.58      # good snr   # was 0.4

    # get indices for where snr_mask = false
    snr_mask_false = np.where(snr_mask == False)[0]
    chs_pruned[snr_mask_false] = 0.4 # poor snr channels      # was 0.19

    # get indices for where amp_mask_sat = false
    amp_mask_false = np.where(amp_mask_sat == False)[0]
    chs_pruned[amp_mask_false] = 0.92 # saturated channels  # was 0.0

    # get indices for where amp_mask_low = false
    amp_mask_false = np.where(amp_mask_low == False)[0]
    chs_pruned[amp_mask_false] = 0.24  # low signal channels    # was 0.8

    # get indices for where sd_mask = false
    sd_mask_false = np.where(sd_mask == False)[0]
    chs_pruned[sd_mask_false] = 0.08 # SDS channels    # was 0.65


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
    chs_pruned.loc[drop_list] = 0.76     # was 0.95

    return rec, chs_pruned, sci, psp


def GLM(rec, rec_str, cfg_GLM, pruned_chans):
    
    # get pruned data for SSR
    rec_pruned = prune_mask_ts(rec[rec_str], pruned_chans) # !!! how is this affected when using pruned data
    
    rec_pruned = rec_pruned.where( ~rec_pruned.isnull(), 0)  #1e-18 )   # set nan to 0

    #### build design matrix
    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        rec[rec_str], rec.geo3d, distance_threshold= cfg_GLM['distance_threshold']  # !!! change to rec_pruned once NaN prob fixed
    )
    
    # build regressors
    # dm, channel_wise_regressors = glm.make_design_matrix(
    #     rec[rec_str],
    #     ts_short,
    #     rec.stim,
    #     rec.geo3d,
    #     basis_function = glm.GaussianKernels(cfg_GLM['cfg_hrf']['t_pre'], cfg_GLM['cfg_hrf']['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std']),
    #     drift_order = cfg_GLM['drift_order'],
    #     short_channel_method = cfg_GLM['short_channel_method']
    # )
    
    
    dm = (
    glm.design_matrix.hrf_regressors(
        rec[rec_str], rec.stim, glm.GaussianKernels(cfg_GLM['cfg_hrf']['t_pre'], cfg_GLM['cfg_hrf']['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
    )
    & glm.design_matrix.drift_regressors(rec[rec_str], drift_order = cfg_GLM['drift_order'])
    & glm.design_matrix.closest_short_channel_regressor(rec[rec_str], ts_short, rec.geo3d)
)

    #### fit the model 
    #pdb.set_trace()
    #betas = glm.fit(rec[rec_str], dm, channel_wise_regressors, noise_model=cfg_GLM['noise_model'])
    results = glm.fit(rec[rec_str], dm, noise_model= cfg_GLM['noise_model'])  #, max_jobs=1)

    betas = results.sm.params

    pred_all = glm.predict(rec[rec_str], betas, dm)  #, channel_wise_regressors)
    pred_all = pred_all.pint.quantify('micromolar')
    
    residual = rec[rec_str] - pred_all
    
    # prediction of all HRF regressors, i.e. all regressors that start with 'HRF '
    pred_hrf = glm.predict(
                            rec[rec_str],
                            betas.sel(regressor=betas.regressor.str.startswith("HRF ")),
                            dm ) #,
                            #channel_wise_regressors)
    
    pred_hrf = pred_hrf.pint.quantify('micromolar')
    
    rec[rec_str] = pred_hrf + residual 
    
    #### get average HRF prediction 
    rec[rec_str] = rec[rec_str].transpose('chromo', 'channel', 'time')
    rec[rec_str] = rec[rec_str].assign_coords(samples=("time", np.arange(len(rec[rec_str].time))))
    rec[rec_str]['time'] = rec[rec_str].time.pint.quantify(units.s) 
             
    
    return rec



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


def quant_slope(rec, timeseries):
    foo = rec[timeseries].copy()
    foo = foo.pint.dequantify()
    slope = foo.polyfit(dim='time', deg=1).sel(degree=1)
    slope = slope.rename({"polyfit_coefficients": "slope"})
    slope = slope.assign_coords(channel = rec[timeseries].channel)
    slope = slope.assign_coords(wavelength = rec[timeseries].wavelength)

    return slope


