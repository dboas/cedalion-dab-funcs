# %% Imports
##############################################################################
#%matplotlib widget



import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.xrutils as xrutils
from cedalion.sigdecomp.ERBM import ERBM
import cedalion.models.glm as glm

import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
from cedalion import units
import numpy as np
import pandas as pd
from math import ceil

import gzip
import pickle
import json


# import my own functions from a different directory
import sys
sys.path.append('/Users/dboas/Documents/GitHub/cedalion-dab-funcs/modules')
#sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-dab-funcs2/modules')
import module_load_and_preprocess as pfDAB
import module_plot_DQR as pfDAB_dqr
import module_group_avg as pfDAB_grp_avg    
import module_ERBM_ICA as pfDAB_ERBM
import module_image_recon as pfDAB_img
import module_spatial_basis_funs_ced as sbf 




# %% 
##############################################################################
import importlib
importlib.reload(pfDAB)




# %% Initial root directory and analysis parameters
##############################################################################

cfg_hrf = {
    'stim_lst' : ['mnt'], 
    't_pre' : 5 *units.s, 
    't_post' : 33 *units.s
    #'t_post' : [ 33, 33 ] *units.s   # !!! GLM does not let you have different time ranges for diff stims right now
    }

cfg_dataset = {
    'root_dir' : '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/gradCPT_NN24/',
    'subj_ids' : ['629', '634', '636'],
    'file_ids' : ['gradCPT_run-01','gradCPT_run-02', 'gradCPT_run-03'],#'WM_run-01','WM_run-02','WM_run-03','WM_run-04'],
    # 'root_dir' : '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/BS_Laura_Miray_2025/',
    # 'subj_ids' : ['586','587','592','613','621'],#['547','568','577','580','581','583','586','587','588','592','613','618','619','621','633'], 
    # 'file_ids' : ['BS_run-01', 'BS_run-02', 'BS_run-03'],
    'subj_id_exclude' : [], #['05','07'] # if you want to exclude a subject from the group average
    'cfg_hrf' : cfg_hrf,
    'derivatives_subfolder' : ''    
}

# Add 'filenm_lst' separately after cfg_dataset is initialized
cfg_dataset['filenm_lst'] = [
    [f"sub-{subj_id}_task-{file_id}_nirs"] 
    for subj_id in cfg_dataset['subj_ids'] 
    for file_id in cfg_dataset['file_ids']
    ]



cfg_prune = {
    'snr_thresh' : 5, # the SNR (std/mean) of a channel. 
    'sd_threshs' : [1, 38]*units.mm, # defines the lower and upper bounds for the source-detector separation that we would like to keep
    'amp_threshs' : [1e-5, 0.84], # define whether a channel's amplitude is within a certain range
    'perc_time_clean_thresh' : 0.6,
    'sci_threshold' : 0.6,
    'psp_threshold' : 0.1,
    'window_length' : 5 * units.s,
    'flag_use_sci' : True,
    'flag_use_psp' : False
}

cfg_imu_glm = {'statesPerDataFrame' : 89,   # FOR WALKING DATA
		'hWin' : np.arange(-3,5,1), # window for impulse response function 
		'statesPerDataFrame' : 89,
		'n_components' : [3, 2],  # [gyro, accel]       # !!! note: changing this will change fig sizes - add that in?
        'butter_order' : 4,   # butterworth filter order
        'Fc' : 0.1,   # cutoff freq (Hz)
        'plot_flag_imu' : True  
}

cfg_motion_correct = {
    #'flag_do_splineSG' : False, # !!! This is not doing anything. left out for now. if True, will do splineSG motion correction
    #'splineSG_p' : 0.99, 
    #'splineSG_frame_size' : 10 * units.s,
    'flag_do_tddr' : True,  
    'flag_do_imu_glm' : False,
    'cfg_imu_glm' : cfg_imu_glm,
}

cfg_bandpass = { 
    'fmin' : 0.01 * units.Hz, #0.02 * units.Hz,
    'fmax' : 1 * units.Hz,  #3 * units.Hz
    'flag_bandpass_filter' : True
}


cfg_GLM = {
    'drift_order' : 1,
    'distance_threshold' : 20 *units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : "ols",    # !!! add choice of basis func 
    't_delta' : 1 *units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1 *units.s ,     #  the temporal spacing between consecutive gaussians
    'cfg_hrf' : cfg_hrf
    }           


cfg_preprocess = {
    'flag_prune_channels' : False,  # FALSE = does not prune chans and does weighted averaging, TRUE = prunes channels and no weighted averaging
    'median_filt' : 3, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass,
    'flag_do_GLM_filter' : False,
    'cfg_GLM' : cfg_GLM 
}


cfg_mse_conc = {                
    'mse_val_for_bad_data' : 1e7 * units.micromolar**2, 
    'mse_amp_thresh' : 1.1e-6,
    'mse_min_thresh' : 1e0 * units.micromolar**2, 
    'blockaverage_val' : 0 * units.micromolar
    }

# if block averaging on OD:
cfg_mse_od = {
    'mse_val_for_bad_data' : 1e1, 
    'mse_amp_thresh' : 1.1e-6,
    'mse_min_thresh' : 1e-6,  # LC using 1e-3 ?
    'blockaverage_val' : 0      # blockaverage val for bad data?
    }

cfg_blockavg = {
    'rec_str' : 'conc',   # what you want to block average (will be either 'od_corrected' or 'conc')
    'flag_prune_channels' : cfg_preprocess['flag_prune_channels'],
    'cfg_prune' : cfg_prune,
    'cfg_hrf' : cfg_hrf,
    'trange_hrf_stat' : [4, 7],  
    'flag_save_group_avg_hrf': True,
    'flag_save_each_subj' : False,  # if True, will save the block average data for each subject
    'cfg_mse_conc' : cfg_mse_conc,
    'cfg_mse_od' : cfg_mse_od
    }               



cfg_erbmICA = {}

save_path = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data')

flag_load_preprocessed_data = True  
flag_save_preprocessed_data = False   # SAVE or no save

flag_load_blockaveraged_data = False


# %% Load and preprocess the data
##############################################################################

# determine the number of subjects and files. Often used in loops.
n_subjects = len(cfg_dataset['subj_ids'])
n_files_per_subject = len(cfg_dataset['file_ids'])

# files to load
for subj_id in cfg_dataset['subj_ids']:
    subj_idx = cfg_dataset['subj_ids'].index(subj_id)
    for file_id in cfg_dataset['file_ids']:
        file_idx = cfg_dataset['file_ids'].index(file_id)
        filenm = f'sub-{subj_id}_task-{file_id}_nirs'
        if subj_idx == 0 and file_idx == 0:
            cfg_dataset['filenm_lst'] = []
            cfg_dataset['filenm_lst'].append( [filenm] )
        elif file_idx == 0:
            cfg_dataset['filenm_lst'].append( [filenm] )
        else:
            cfg_dataset['filenm_lst'][subj_idx].append( filenm )

import importlib
importlib.reload(pfDAB)


# File naming stuff
p_save_str = ''
if cfg_motion_correct['flag_do_imu_glm']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_imuGLM' 
else:
    p_save_str =  p_save_str
if cfg_motion_correct['flag_do_tddr']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_tddr' 
else:
    p_save_str =  p_save_str 
if cfg_preprocess['flag_do_GLM_filter']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_GLMfilt' 
else:
    p_save_str =  p_save_str   
if cfg_preprocess['flag_prune_channels']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_pruned' 
else:
    p_save_str =  p_save_str + '_unpruned' 
    
    
# RUN PREPROCESSING
if not flag_load_preprocessed_data:
    print("Running load and process function")
    
    # RUN preprocessing
    rec, chs_pruned_subjs = pfDAB.load_and_preprocess( cfg_dataset, cfg_preprocess ) 

    
    # SAVE preprocessed data 
    if flag_save_preprocessed_data:
        print(f"Saving preprocessed data for {cfg_dataset['file_ids']}")
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'wb') as f: # !!! FIX ME: naming convention assumes file_ids only includes ONE task
            pickle.dump(chs_pruned_subjs, f, protocol=pickle.HIGHEST_PROTOCOL )
            
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'cfg_dataset': cfg_dataset}, f, protocol=pickle.HIGHEST_PROTOCOL )
            
            
        # SAVE cfg params to json file
        # !!! ADD image recon cfg  ?? - or make it its own .json since i am planning to separate into 2 scripts
        dict_cfg_save = {"cfg_hrf": cfg_hrf, "cfg_dataset" : cfg_dataset, "cfg_preprocess" : cfg_preprocess, "cfg_GLM" : cfg_GLM, "cfg_blockavg" : cfg_blockavg}
        
        cfg_save_str = 'cfg_params_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.json'
            
        with open(os.path.join(save_path, cfg_save_str), "w", encoding="utf-8") as f:
            json.dump(dict_cfg_save, f, indent=4, default = str)  # Save as JSON with indentation
        print("Preprocessed data successfully saved.")
        
        
# LOAD IN SAVED DATA
else:
    print("Loading saved data")   # !!! update with new naming for pruned or unpruned above
    with gzip.open( os.path.join(save_path, 'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'rb') as f: # !!! FIX ME: this assumes file_ids only includes ONE task
         rec = pickle.load(f)
    with gzip.open( os.path.join(save_path, 'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'rb') as f:
         chs_pruned_subjs = pickle.load(f)
    print(f'Data loaded successfully for {cfg_dataset["file_ids"][0].split("_")[0]}')







# # %%
# # problem with onset time of gradCPT
##############################################################################

# for idx_subj in range(len(cfg_dataset['subj_ids'])):
#     for idx_file in range(len(cfg_dataset['file_ids'])):
#         print(f"{idx_subj},{idx_file}  {rec[idx_subj][idx_file]['amp'].time.values[-1]:.2f},{rec[idx_subj][idx_file].stim.onset.values[-1]}")

# # %%
# # plot rec[0][0].aux_ts['digital']

# idx_subj = 0
# idx_file = 0

# rec[idx_subj][idx_file].aux_ts['digital'].plot()

# idx = np.where(rec[idx_subj][idx_file].aux_ts['digital'].values > 1)[0][0] # find where the digital signal is 1
# print(rec[idx_subj][idx_file].aux_ts['digital'].time[idx].values)
# print(rec[idx_subj][idx_file].stim.onset.values[0]) # find the first stimulus onset time

# %%
# GLM
idx_subj = 0
idx_file = 0

stim = rec[idx_subj][idx_file].stim.copy()

#stim = stim.loc[stim['trial_type'] == 'mnt']

idx_subj = 0
idx_file = 0

# ts_long, ts_short = cedalion.nirs.split_long_short_channels(
#     rec[idx_subj][idx_file]["conc"], rec[idx_subj][idx_file].geo3d, distance_threshold=15 * units.mm
# )

dms = glm.design_matrix.hrf_regressors(
        rec[idx_subj][idx_file]['conc'], stim, glm.Gamma(
            tau={"HbO": 0 * units.s, "HbR": 1 * units.s}, sigma={"HbO": 3 * units.s, "HbR": 3 * units.s}, T=0 * units.s
            )
    )

# %%
# get the design matrix for VTC

#import numpy as np
from scipy.signal import filtfilt, windows

t_vtc = stim.onset.values
vtc = stim.VTC.values

# smooth vtc
# Create the Gaussian window
L = 20
W = windows.gaussian(M=L, std=4) / 2  # std=7 approximates MATLAB's gausswin(L)
W = W / np.sum(W)  # normalize so sum(W) = 1
# Apply zero-phase filtering
vtc = filtfilt(W, [1.0], vtc)

# interpolate the VTC to the time of the ts
vtc_interp = np.interp(rec[idx_subj][idx_file]['conc'].time.values, t_vtc, vtc)

vtc_interp_xr = xr.DataArray(
    vtc_interp,
    coords=[rec[idx_subj][idx_file]['conc'].time],
    dims=['time']
)

# expand dimensions to match the design matrix
vtc_interp_xr = vtc_interp_xr.expand_dims( regressor=['VTC'], chromo=['HbO', 'HbR'] )
vtc_interp_xr = vtc_interp_xr.transpose('time', 'regressor', 'chromo' )

vtc_xr = xr.DataArray(
    vtc,
    coords={
        'time': t_vtc },
    dims=['time']
)
vtc_xr = vtc_xr.expand_dims( regressor=['VTC'], chromo=['HbO', 'HbR'] )
vtc_xr = vtc_xr.transpose('time', 'regressor', 'chromo' )


from cedalion.models.glm.design_matrix import DesignMatrix

dms_vtc = DesignMatrix(
            common=vtc_interp_xr,
            channel_wise=[]
        )

# %%
# display design matrix
f,ax = p.subplots(1,1,figsize=(12,5))
foo = vtc_interp_xr.sel(chromo="HbO")#, time=slice(0, 170))
ax.plot(foo.time,foo.values)
# foo = vtc_xr.sel(chromo="HbO", time=slice(0, 170))
# ax.plot(foo.time, foo.values, marker='x')
p.show



# %%
# fit the model
results = glm.fit(rec[idx_subj][idx_file]['conc'], dms_vtc, noise_model="ols", max_jobs=1) # ols, ar_irls

display(results)

betas = results.sm.params
betas

betas.rename("betas").to_dataframe()

# %%

# check if any of the values are NaN along the time dimension of rec[idx_subj][idx_file]['conc']
foo1 = rec[idx_subj][idx_file]['conc'].copy()

foo = rec[idx_subj][idx_file]['conc'].mean('time').values
foo = np.isnan(foo)

# replace NaN values in foo1 with 0
# foo1.values[np.isnan(foo)] = 0
# foo1.values[np.isinf(foo)] = 0
# foo1.values[np.isneginf(foo)] = 0

foo1 = foo1.where( ~foo1.isnull(), 0)

results = glm.fit(foo1, dms, noise_model="ar_irls", max_jobs=1) # ols, ar_irls


# %%
# scalp plots
importlib.reload(plots)


f, ax = p.subplots(2, 2, figsize=(16, 16))
vlims = {"HbO" : [0.,0.03], "HbR" : [-0.1, 0.05]}
vminmax = 10
for i_chr, chromo in enumerate(betas.chromo.values):
    vmin, vmax = vlims[chromo]
    for i_reg, reg in enumerate(["VTC"]):
        # vminmax = np.max(np.abs(betas.sel(chromo=chromo, regressor=reg)))
        vmin = -vminmax
        vmax = vminmax
        cedalion.plots.scalp_plot(
            rec[idx_subj][idx_file]["amp"],
            rec[idx_subj][idx_file].geo3d,
            betas.sel(chromo=chromo, regressor=reg),
            ax[i_chr][i_reg],
            min_dist=1.5 * cedalion.units.cm,
            max_dist=3.5 * cedalion.units.cm,
            title=f"{chromo} {reg}",
            vmin=vmin,
            vmax=vmax,
            optode_labels=True,
            cmap="jet",
            cb_label=r"$\beta$"
        )
p.tight_layout()






# %% Block Average - unweighted and weighted
##############################################################################
cfg_blockavg = {
    'rec_str' : 'conc',   # what you want to block average (will be either 'od_corrected' or 'conc')
    'flag_prune_channels' : cfg_preprocess['flag_prune_channels'],
    'cfg_prune' : cfg_prune,
    'cfg_hrf' : cfg_hrf,
    'trange_hrf_stat' : [4, 7],  
    'flag_save_group_avg_hrf': True,
    'flag_save_each_subj' : False,  # if True, will save the block average data for each subject
    'cfg_mse_conc' : cfg_mse_conc,
    'cfg_mse_od' : cfg_mse_od
    }               

import importlib
importlib.reload(pfDAB_grp_avg)

#flag_load_blockaveraged_data = False


# for saving file name 
    
if 'conc' in cfg_blockavg['rec_str']:  
    save_str = p_save_str + '_CONC' 
else:
    save_str = p_save_str + '_OD' 
    
# FIXME: use cfg_GLM? Or is that for the GLM filter and not the HRF estimation? Can probably use it for both
_, blockaverage_mean, blockaverage_stderr, blockaverage_subj, blockaverage_mse_subj = pfDAB_grp_avg.run_group_glm( rec, cfg_blockavg['rec_str'], chs_pruned_subjs, cfg_dataset, cfg_blockavg )

#_, blockaverage_mean, blockaverage_stderr, blockaverage_subj, blockaverage_mse_subj = pfDAB_grp_avg.run_group_block_average( rec, cfg_blockavg['rec_str'], chs_pruned_subjs, cfg_dataset, cfg_blockavg )


# %%

# find channels that are NaN in blockaverage_mean
idxO = np.where(np.isnan(blockaverage_mean.sel(trial_type='right',chromo='HbO').values))[0]
idxR = np.where(np.isnan(blockaverage_mean.sel(trial_type='right',chromo='HbR').values))[0]








# %%

# Compute block average
if not flag_load_blockaveraged_data:  
    
    if cfg_preprocess['flag_prune_channels']:   # if using pruned data, don't save weighted
        blockaverage_mean, _, blockaverage_stderr, blockaverage_subj, blockaverage_mse_subj = pfDAB_grp_avg.run_group_block_average( rec, cfg_blockavg['rec_str'], chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    
    else:    # if not pruning, save weighted blockaverage data
        _, blockaverage_mean, blockaverage_stderr, blockaverage_subj, blockaverage_mse_subj = pfDAB_grp_avg.run_group_block_average( rec, cfg_blockavg['rec_str'], chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    
    groupavg_results = {'blockaverage': blockaverage_mean,
               'blockaverage_stderr': blockaverage_stderr,
               'blockaverage_subj': blockaverage_subj,
               'blockaverage_mse_subj': blockaverage_mse_subj,
               'geo2d' : rec[0][0].geo2d,
               'geo3d' : rec[0][0].geo3d
               }
    
    if cfg_blockavg['flag_save_group_avg_hrf']:
        file_path_pkl = os.path.join(save_path, 'blockaverage_' + cfg_dataset["file_ids"][0].split('_')[0] + '_' + save_str + '.pkl.gz')
        file = gzip.GzipFile(file_path_pkl, 'wb')
        file.write(pickle.dumps(groupavg_results))
        file.close()
        print('Saved group average HRF to ' + file_path_pkl)

else: # LOAD data
    filname =  'blockaverage_' + cfg_dataset["file_ids"][0].split('_')[0] + '_' + save_str + '.pkl.gz'
    filepath_bl = os.path.join(save_path , filname)
    
    if os.path.exists(filepath_bl):
        with gzip.open(filepath_bl, 'rb') as f:
            groupavg_results = pickle.load(f)
        blockaverage_mean = groupavg_results['blockaverage']
        blockaverage_stderr = groupavg_results['blockaverage_stderr']
        blockaverage_subj = groupavg_results['blockaverage_subj']
        blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
        geo2d = groupavg_results['geo2d']
        geo2d = groupavg_results['geo3d']
        print("Blockaverage file loaded successfully!")
    
    else:
        print(f"Error: File '{filepath_bl}' not found!")
        
blockaverage_all = blockaverage_mean.copy()



















# %% Load the Sensitivity Matrix and Head Model
##############################################################################


import importlib
importlib.reload(pfDAB_img)


cfg_sb = {
    'mask_threshold': -2,
    'threshold_brain': 5*units.mm,      # threshold_brain / threshold_scalp: Defines spatial limits for brain vs. scalp contributions.
    'threshold_scalp': 20*units.mm,
    'sigma_brain': 5*units.mm,      # sigma_brain / sigma_scalp: Controls smoothing or spatial regularization strength.
    'sigma_scalp': 20*units.mm,
    'lambda1': 0.01,        # regularization params
    'lambda2': 0.1
}


cfg_img_recon = {
    'probe_dir' : '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/BallSqueezing_WHHD/derivatives/fw',
    'head_model' : 'ICBM152',
    't_win' : (10, 20), 
    'flag_Cmeas' : True,   # if True make sure you are using the correct y_stderr_weighted below (or blockaverage_stderr now)-- covariance
    'BRAIN_ONLY' : False,
    'SB' : False,    # spatial basis
    'alpha_meas_list' : [1e0],  #[1e0]    measurement regularization (w/ Cmeas, 1 is good)  (w/out Cmeas do 1e-2?)
    'alpha_spatial_list' : [1e-3],    #[1e-2, 1e-4, 1e-5, 1e-3, 1e-1] #[1e-3]    spatial reg , small pushes deeper into the brain   -- # use smaller alpha spatial od 10^-2 or -3 w/out cmeas
    'spectrum' : 'prahl',
    'cfg_sb' : cfg_sb,
    'flag_save_img_results' : False
    }

wavelength = rec[0][0]['amp'].wavelength.values
#trial_type_img = 'ST_o_tddr'  # 'DT-o-imu' # 'DT', ST', 

#
# Load the Sensitivity Matrix and Head Model
#

# if Adot is not already loaded
if 'Adot' not in locals():
    Adot, head = pfDAB_img.load_Adot( cfg_img_recon['probe_dir'], cfg_img_recon['head_model'])

# !!! add flag for if doing image recon on group avg or direct or indirect
# !!! ADD flag for if doing image recon on ts or hrf mag


# %% Do the image reconstruction
##############################################################################

#
# Get the group average image
#
all_trial_X_grp = None

for idx, trial_type in enumerate(blockaverage_all.trial_type):  #enumerate([blockaverage_all.trial_type.values[2]]): 

    # !!! ADD if new_rec_str_use_weighted_lst is NONE then assume all are weighted and do img recon on ALL TRIAL TYPES
    if not new_rec_str_lst_use_weighted[idx]:  # !!! assumes user used rec_str_lst_use_weighted correctly 
        print(f'trial type = {trial_type.values} is assumed to be unweighted. Skipping image reconstruction \n')
        continue      # !!! might want to just run w/out Cmeas in future. skipping for now
    
    print(f'Getting images for trial type = {trial_type.values}')
    
    if 'chromo' in blockaverage_all.dims:
        # get the group average HRF over a time window
        hrf_conc_mag = blockaverage_all.sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0],cfg_img_recon['t_win'][1])).mean('reltime')
        hrf_conc_ts = blockaverage_all.sel(trial_type=trial_type)
        
        blockaverage_stderr_conc = blockaverage_stderr.sel(trial_type=trial_type) # need to convert blockaverage_stderr to od if its in conc
    
        # convert back to OD
        E = cedalion.nirs.get_extinction_coefficients(cfg_img_recon['spectrum'], wavelength)
        hrf_od_mag = xr.dot(E, hrf_conc_mag * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
        hrf_od_ts = xr.dot(E, hrf_conc_ts * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
        
        blockaverage_stderr = xr.dot(E, blockaverage_stderr_conc * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
            
    else:
        hrf_od_mag = blockaverage_all.sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
        hrf_od_ts = blockaverage_all.sel(trial_type=trial_type)
    
    if not cfg_img_recon['flag_Cmeas']:  
        cov_str = '' # for name
        X_grp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, None, wavelength, cfg_img_recon, trial_type, save_path)
    
    else:
        cov_str = 'cov'
       
        C_meas = blockaverage_stderr.sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime') 
        C_meas = C_meas.pint.dequantify()     # remove units
        C_meas = C_meas**2  # get variance
        C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')  
        X_grp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, cfg_img_recon, trial_type, save_path)
    
    print(f'Done with Image Reconstruction for trial type = {trial_type.values}')
    
    # Unweighted avg - can still get standard error (for Cmeas)
        # but don't wanna spend time coding it
    # !!! Therefore, Ditch image recon of pruned data ? !!!
        # if not weighted avg trial type -> make Cmeas = false
        
    X_grp = X_grp.assign_coords(trial_type = trial_type)
    
    #
    #  Calculate the image noise and image CNR
    #
    if cfg_img_recon['flag_Cmeas']:
        X_noise, X_tstat = pfDAB_img.img_noise_tstat(X_grp, W, C_meas)
        
        if cfg_img_recon['flag_save_img_results']:
            pfDAB_img.save_image_results(X_noise, 'X_noise', save_path, trial_type, cfg_img_recon)
            pfDAB_img.save_image_results(X_tstat, 'X_tstat', save_path, trial_type, cfg_img_recon)
        
        X_noise = X_noise.assign_coords(trial_type = trial_type)
        X_tstat = X_tstat.assign_coords(trial_type = trial_type)
        
        # save results for all trial types
        if all_trial_X_grp is None:
            all_trial_X_grp = X_grp
            all_trial_X_noise = X_noise  # comes from diag of covariance matrix
            all_trial_X_tstat = X_tstat 
        else:
            all_trial_X_grp = xr.concat([all_trial_X_grp, X_grp], dim='trial_type')
            all_trial_X_noise = xr.concat([all_trial_X_noise, X_noise], dim='trial_type')
            all_trial_X_tstat = xr.concat([all_trial_X_tstat, X_tstat], dim='trial_type')
            
        results_img_grp = {'X_grp_all_trial': all_trial_X_grp,
                   'X_noise_grp_all_trial': all_trial_X_noise,
                   'X_tstat_grp_all_trial': all_trial_X_tstat
                   }
    
    # if flag_Cmeas is false, can't calc tstat and noise
    else:
        if all_trial_X_grp is None:
            all_trial_X_grp = X_grp
        else:
            all_trial_X_grp = xr.concat([all_trial_X_grp, X_grp], dim='trial_type')
    
tasknm = cfg_dataset["file_ids"][0].split('_')[0] # get task name

filepath = os.path.join(cfg_dataset['root_dir'], f'X_{tasknm}_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
print(f'Saving to X_{tasknm}_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps(results_img_grp))
file.close()    


# %% 
#
# Get image for each subject and do weighted average
#
##############################################################################
import importlib
importlib.reload(pfDAB_img)


# add if chromo in blockaverage_subj.dims -> convert to OD --- 
    # !!! ^^ I think if in conc it will give error bc of blockaverage_subj_mse - check
# !!! ADD flag for if doing image recon on ts or hrf mag 

X_hrf_mag_subj = None
C = None # spatial regularization 
D = None

# !!! go thru each trial type (outside function)
    # ISSUE: some trial_types now have used weighted avg and some are pruned -> meaning can't use Cmeas for all ??? 
    # SKIPPING unweighted trial types

all_trial_X_hrf_mag = None

for idx_trial, trial_type in enumerate(blockaverage_subj.trial_type):
    
    if not new_rec_str_lst_use_weighted[idx_trial]:  # !!! assumes user used rec_str_lst_use_weighted correctly 
        print(f'trial type = {trial_type.values} is assumed to be unweighted. Skipping image reconstruction. \n')
        continue      # !!! might want to just run w/out Cmeas in future. skipping for now
    
    print(f'Getting images for trial type = {trial_type.values}')
    all_subj_X_hrf_mag = None
    
    for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):

        print(f'Starting image recon on subject {curr_subj}')
        
        # Check if rec_str exists for current subject
        if curr_subj in cfg_dataset['subj_id_exclude']:
            print(f'   Subject {cfg_dataset["subj_ids"][idx_subj]} excluded from group average')
            continue  # if subject is excluded, skip this loop
        
        if 'chromo' in blockaverage_subj.dims:
            # get the group average HRF over a time window
            hrf_conc_mag = blockaverage_subj.sel(subj= curr_subj).sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0],cfg_img_recon['t_win'][1])).mean('reltime')
            hrf_conc_ts = blockaverage_subj.sel(subj= curr_subj).sel(trial_type=trial_type)
            
            blockaverage_mse_subj_conc = blockaverage_mse_subj.sel(subj= curr_subj).sel(trial_type=trial_type)
            
            # convert back to OD
            E = cedalion.nirs.get_extinction_coefficients(cfg_img_recon['spectrum'], wavelength)
            hrf_od_mag = xr.dot(E, hrf_conc_mag * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # !!! assumes DPF = 1
            hrf_od_ts = xr.dot(E, hrf_conc_ts * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
                
            blockaverage_mse_subj= xr.dot(E, blockaverage_mse_subj_conc * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1

        else:
            hrf_od_mag = blockaverage_subj.sel(subj= curr_subj).sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
            hrf_od_ts = blockaverage_subj.sel(subj= curr_subj).sel(trial_type=trial_type)

        #
        #hrf_od_mag = blockaverage_subj.sel(subj=cfg_dataset['subj_ids'][idx_subj]).sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime') 
        # hrf_od_ts = blockaverage_all.sel(trial_type=trial_type)
    
        # get the image
        
        C_meas = blockaverage_mse_subj.sel(subj=cfg_dataset['subj_ids'][idx_subj]).sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime') 
    
        C_meas = C_meas.pint.dequantify()
        C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
        
        if cfg_img_recon['flag_Cmeas']:
            cov_str = 'cov'
            if C is None or D is None:
                #X_hrf_mag_tmp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type) 
                X_hrf_mag_tmp, W, C, D = pfDAB_img.do_image_recon( hrf_od = hrf_od_mag, head = head, Adot = Adot, C_meas = C_meas,
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = trial_type, save_path = save_path,
                                                                  W = None, C = None, D = None) 
        
            else:
                X_hrf_mag_tmp, W, _, _ = pfDAB_img.do_image_recon( hrf_od = hrf_od_mag, head = head, Adot = Adot, C_meas = C_meas, 
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = trial_type, save_path = save_path, 
                                                                  W = None, C = C, D = D)
        else:
            cov_str = ''
            if C is None or D is None:
                X_hrf_mag_tmp, W, C, D = pfDAB_img.do_image_recon( hrf_od = hrf_od_mag, head = head, Adot = Adot, C_meas = None,
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = trial_type, save_path = save_path,
                                                                  W = None, C = None, D = None) 
        
            else:
                X_hrf_mag_tmp, W, _, _ = pfDAB_img.do_image_recon( hrf_od = hrf_od_mag, head = head, Adot = Adot, C_meas = None, 
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = trial_type, save_path = save_path, 
                                                                  W = None, C = C, D = D)
        

        # get image noise
        cov_img_tmp = W * np.sqrt(C_meas.values) # get diag of image covariance
        cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
    
        nV = X_hrf_mag_tmp.vertex.size
        cov_img_diag = np.reshape( cov_img_diag, (2,nV) ).T
    
        X_mse = X_hrf_mag_tmp.copy() 
        X_mse.values = cov_img_diag # !!! SAVE nult trial types
        
        
        # weighted average -- same as chan space - but now is vertex space
        if all_subj_X_hrf_mag is None:
            all_subj_X_hrf_mag = X_hrf_mag_tmp
            all_subj_X_hrf_mag = all_subj_X_hrf_mag.expand_dims('subj')
            all_subj_X_hrf_mag = all_subj_X_hrf_mag.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])
    
            X_mse_subj = X_mse.copy()
            X_mse_subj = X_mse_subj.expand_dims('subj')
            X_mse_subj = X_mse_subj.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])
            
            X_hrf_mag_weighted = X_hrf_mag_tmp / X_mse
            X_mse_inv_weighted = 1 / X_mse
            X_mse_inv_weighted_max = 1 / X_mse
        elif cfg_dataset['subj_ids'][idx_subj] not in cfg_dataset['subj_id_exclude']:
            X_hrf_mag_subj_tmp = X_hrf_mag_tmp.expand_dims('subj') # !!! will need to expand dims to get back trial type -- can do in function 
            X_hrf_mag_subj_tmp = X_hrf_mag_subj_tmp.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])
    
            X_mse_subj_tmp = X_mse.copy().expand_dims('subj')
            X_mse_subj_tmp = X_mse_subj_tmp.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])
    
            all_subj_X_hrf_mag = xr.concat([all_subj_X_hrf_mag, X_hrf_mag_subj_tmp], dim='subj')
            X_mse_subj = xr.concat([X_mse_subj, X_mse_subj_tmp], dim='subj')
    
            X_hrf_mag_weighted = X_hrf_mag_weighted + X_hrf_mag_tmp / X_mse
            X_mse_inv_weighted = X_mse_inv_weighted + 1 / X_mse
            X_mse_inv_weighted_max = np.maximum(X_mse_inv_weighted_max, 1 / X_mse)
        else:
            print(f"   Subject {cfg_dataset['subj_ids'][idx_subj]} excluded from group average")
            
    
    # END OF SUBJECT LOOP
    
    # get the average
    X_hrf_mag_mean = all_subj_X_hrf_mag.mean('subj')
    X_hrf_mag_mean_weighted = X_hrf_mag_weighted / X_mse_inv_weighted
    
    X_mse_mean_within_subject = 1 / X_mse_inv_weighted
    
    X_mse_subj_tmp = X_mse_subj.copy()
    X_mse_subj_tmp = xr.where(X_mse_subj_tmp < 1e-6, 1e-6, X_mse_subj_tmp)
    X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_mag - X_hrf_mag_mean)**2 / X_mse_subj_tmp # X_mse_subj_tmp is weights for each sub
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp.mean('subj')
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects / (X_mse_subj**-1).mean('subj')
    
    X_stderr_weighted = np.sqrt( X_mse_mean_within_subject + X_mse_weighted_between_subjects )
    
    X_tstat = X_hrf_mag_mean_weighted / X_stderr_weighted
    
    X_weight_sum = X_mse_inv_weighted / X_mse_inv_weighted_max  # tstat = weighted group avg / noise # !!! not saving?
    
    # Assign trial type coord
    X_hrf_mag_mean = X_hrf_mag_mean.assign_coords(trial_type = trial_type)
    X_hrf_mag_mean_weighted = X_hrf_mag_mean_weighted.assign_coords(trial_type = trial_type)
    X_stderr_weighted = X_stderr_weighted.assign_coords(trial_type = trial_type)
    X_tstat = X_tstat.assign_coords(trial_type = trial_type)

    if all_trial_X_hrf_mag is None:
        
        all_trial_X_hrf_mag = X_hrf_mag_mean
        all_trial_X_hrf_mag_weighted = X_hrf_mag_mean_weighted
        all_trial_X_stderr = X_stderr_weighted # noise
        all_trial_X_tstat = X_tstat # tstat
    else:
    
        all_trial_X_hrf_mag = xr.concat([all_trial_X_hrf_mag, X_hrf_mag_mean], dim='trial_type')
        all_trial_X_hrf_mag_weighted = xr.concat([all_trial_X_hrf_mag_weighted, X_hrf_mag_mean_weighted], dim='trial_type')
        all_trial_X_stderr = xr.concat([all_trial_X_stderr, X_stderr_weighted], dim='trial_type')
        all_trial_X_tstat = xr.concat([all_trial_X_tstat, X_tstat], dim='trial_type')

# END OF TRIAL TYPE LOOP

# FIXME: I am trying to get something like number of subjects per vertex...
# maybe I need to change X_mse_inv_weighted_max to be some typical value 
# because when all subjects have a really low value, then it won't scale the way I want

results_img_s = {'X_hrf_mag_all_trial': all_trial_X_hrf_mag,
           'X_hrf_mag_weighted_all_trial': all_trial_X_hrf_mag_weighted,
           'X_std_err_all_trial': all_trial_X_stderr,  # noise
           'X_tstat_all_trial': all_trial_X_tstat
           }

tasknm = cfg_dataset["file_ids"][0].split('_')[0]

# !!! chang name when indirect is implemented
if not cfg_img_recon['SB']:
    filepath = os.path.join(save_path, f'Xs_{tasknm}_direct_alltrial_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
    print(f'   Saving to Xs_{tasknm}_direct_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
    file = gzip.GzipFile(filepath, 'wb')
    file.write(pickle.dumps(results_img_s))
    file.close()     
else:
    filepath = os.path.join(save_path, f'Xs_{tasknm}_direct_alltrial_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}_SB_sigma_brain_{cfg_img_recon["sigma_brain"]}_sigma_scalp_{cfg_img_recon["sigma_scalp"]}.pkl.gz')
    print(f'   Saving to Xs_{tasknm}_direct_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
    file = gzip.GzipFile(filepath, 'wb')
    file.write(pickle.dumps(results_img_s))
    file.close()     



# %% Plot the images
##############################################################################

# !!! CHANGE FILE NAME IF GROUP INSTEAD OF XS


if all_trial_X_hrf_mag.trial_type.values.ndim > 0:
    
    threshold = -2 # log10 absolute
    wl_idx = 1
    M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)
    SAVE = True
    flag_hbo = True
    flag_brain = True
    flag_img_list = ['mag','tstat', 'noise']    # ['mag', 'tstat', 'noise'] #, 'noise'
    #flag_condition_list =['ST_o_tddr', 'ST_o_imu_tddr', 'DT_o_tddr', 'DT_o_imu_tddr'] #
    flag_condition_list = all_trial_X_hrf_mag.trial_type.values
    
    
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'image_recon')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    
    direct_name = 'Direct'  # !!! Change when implementing indirect method
    
    for flag_condition in flag_condition_list:
        
        for flag_img in flag_img_list:
            
            if flag_hbo:
                title_str = flag_condition + ' HbO'
                hbx_brain_scalp = 'hbo'
            else:
                title_str = flag_condition + ' HbR'
                hbx_brain_scalp = 'hbr'
            
            if flag_brain:
                title_str = title_str + ' brain'
                hbx_brain_scalp = hbx_brain_scalp + '_brain'
            else:
                title_str = title_str + ' scalp'
                hbx_brain_scalp = hbx_brain_scalp + '_scalp'
            
            if flag_img == 'tstat':
                foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
                title_str = title_str + ' t-stat'
            elif flag_img == 'mag':
                foo_img = all_trial_X_hrf_mag_weighted.sel(trial_type=flag_condition).copy()  # plotting weighted
                title_str = title_str + ' magnitude'
            elif flag_img == 'noise':
                foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                title_str = title_str + ' noise'
    
            foo_img = foo_img.pint.dequantify()
            foo_img = foo_img.transpose('vertex', 'chromo') # why r we transposing these?
            foo_img[~M] = np.nan
            
            clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())
            # if flag_img == 'tstat':
            #     clim = [-5, 5]
            p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,1), clim, hbx_brain_scalp, 'scale_bar',
                                      None, title_str, off_screen=SAVE )
            p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,0), clim, hbx_brain_scalp, 'left', p0)
            p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,1), clim, hbx_brain_scalp, 'superior', p0)
            p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,2), clim, hbx_brain_scalp, 'right', p0)
            p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,0), clim, hbx_brain_scalp, 'anterior', p0)
            p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,2), clim, hbx_brain_scalp, 'posterior', p0)
            
            if SAVE:
                if not cfg_img_recon['SB']:
                    filname = f'IMG_{flag_condition}_{direct_name}_{cov_str}_{flag_img}_{hbx_brain_scalp}.png'
                else:
                    filname = f'IMG_{flag_condition}_{direct_name}_{cov_str}_{flag_img}_{hbx_brain_scalp}_SB.png'
                p0.screenshot( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'image_recon', filname) )
                p0.close()
            else:
                p0.show()
                



# %% Functional Connectivity
# Get the correlation matrices
##############################################################################
import importlib
importlib.reload(pfDAB_img)

flag_use_17_networks = False
flag_use_parcel_networks = False

# only calculate this once and then use for all other runs for all other subjects
if 1:
    C = None # spatially regularized (A Linv) (A Linv).T
    D = None # Linv**2 A.T

# Loop over subjects
for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):

    # if idx_subj > 0:
    #     break

    # Loop over runs
    for idx_file, curr_file in enumerate(cfg_dataset['file_ids']):

        # if idx_file >0:
        #     break

        print(f'\nStarting image recon on SUBJECT {curr_subj}, RUN {curr_file}')

        # get the OD time series
        od_ts = rec[idx_subj][idx_file]['od_o_tddr']


        # get the measurement variance and correct the 'bad' channels
        foo = od_ts.stack(measurement=('channel', 'wavelength')).sortby('wavelength')    
        C_meas = foo.var('time').values # get the variance along the time dimension

        # handle low amplitude data
        n_chs = len(od_ts.channel)
        amp = rec[idx_subj][idx_file]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
        idx_amp = np.where(amp < cfg_blockavg['cfg_mse_od']['mse_amp_thresh'])[0]
        C_meas[idx_amp] = cfg_blockavg['cfg_mse_od']['mse_val_for_bad_data']
        C_meas[idx_amp + n_chs] = cfg_blockavg['cfg_mse_od']['mse_val_for_bad_data']
        # Update bad data with predetermined value
        bad_vals = od_ts.isel(channel=idx_amp)
        od_ts.loc[dict(channel=bad_vals.channel.data)] = cfg_blockavg['cfg_mse_od']['blockaverage_val']

        # handle saturated channels
        idx_sat = np.where(chs_pruned_subjs[subj_idx][file_idx] == 0.0)[0] 
        C_meas[idx_sat] = cfg_blockavg['cfg_mse_od']['mse_val_for_bad_data']
        C_meas[idx_sat + n_chs] = cfg_blockavg['cfg_mse_od']['mse_val_for_bad_data']
        # Update bad data with predetermined value        
        bad_vals = od_ts.isel(channel=idx_sat)
        od_ts.loc[dict(channel=bad_vals.channel.data)] = cfg_blockavg['cfg_mse_od']['blockaverage_val']

        # handle rare instances where C_meas is 0
        idx_bad = np.where(C_meas == 0)[0]
        idx_bad1 = idx_bad[idx_bad<n_chs]
        idx_bad2 = idx_bad[idx_bad>=n_chs] - n_chs
        C_meas[idx_bad] = cfg_blockavg['cfg_mse_od']['mse_val_for_bad_data']
        # Update bad data with predetermined value        
        bad_vals = od_ts.isel(channel=idx_bad1)
        od_ts.loc[dict(channel=bad_vals.channel.data)] = cfg_blockavg['cfg_mse_od']['blockaverage_val']
        bad_vals = od_ts.isel(channel=idx_bad2)
        od_ts.loc[dict(channel=bad_vals.channel.data)] = cfg_blockavg['cfg_mse_od']['blockaverage_val']
        

        # Get the image time series for each run
        if cfg_img_recon['flag_Cmeas']:
            if C is None or D is None:
                #X_hrf_mag_tmp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type) 
                X_ts, W, C, D = pfDAB_img.do_image_recon( hrf_od = od_ts, head = head, Adot = Adot, C_meas = C_meas,
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = None, save_path = None,
                                                                  W = None, C = None, D = None) 
        
            else:
                X_ts, W, _, _ = pfDAB_img.do_image_recon( hrf_od = od_ts, head = head, Adot = Adot, C_meas = C_meas, 
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = None, save_path = None, 
                                                                  W = None, C = C, D = D)
        else:
            if C is None or D is None:
                X_ts, W, C, D = pfDAB_img.do_image_recon( hrf_od = od_ts, head = head, Adot = Adot, C_meas = None,
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = None, save_path = None,
                                                                  W = None, C = None, D = None) 
        
            else:
                X_ts, W, _, _ = pfDAB_img.do_image_recon( hrf_od = od_ts, head = head, Adot = Adot, C_meas = None, 
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = None, save_path = None, 
                                                                  W = None, C = C, D = D)

        # set time.attrs['units'] = 's' for the time dimension
        X_ts = X_ts.assign_coords(time = X_ts.time.values * units.s)
        X_ts.time.attrs['units'] = 's'

        # Get the parcel time series for each run
        X_parcel_ts = X_ts.groupby('parcel').mean('vertex')

        # remove the scalp parcel
        # what about first two parcels 'Background+Freesurfer...'
        X_parcel_ts = X_parcel_ts.sel(parcel=X_parcel_ts.parcel != 'scalp')
        X_parcel_ts = X_parcel_ts.sel(parcel=X_parcel_ts.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
        X_parcel_ts = X_parcel_ts.sel(parcel=X_parcel_ts.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')

        # bandpass filter the parcel time series
        fmin = 0.01 * units.Hz
        fmax = 0.2 * units.Hz
        X_parcel_ts = cedalion.sigproc.frequency.freq_filter(X_parcel_ts, fmin, fmax)

        # Global mean subtraction for each chromo
        gms = X_parcel_ts.mean('parcel')
        numerator = (X_parcel_ts * gms).sum(dim="time")
        denominator = (gms * gms).sum(dim="time")
        scl = numerator / denominator

        X_parcel_ts = X_parcel_ts - scl*gms
        
        if flag_use_17_networks:
            # reduce parcels to 17 network parcels plus 'Background+Freesurfer...'
            # get the unique 17 network parcels
            parcel_list = []
            for parcel in X_parcel_ts.parcel.values:
                parcel_list.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )
            unique_parcels = np.unique(parcel_list)

            parcel_list_lev2 = []
            for parcel in X_parcel_ts.parcel.values:
                if parcel.split('_')[1].isdigit():
                    parcel_list_lev2.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )
                else:
                    parcel_list_lev2.append( parcel.split('_')[0] + '_' + parcel.split('_')[1] + '_' + parcel.split('_')[-1] )
            unique_parcels_lev2 = np.unique(parcel_list_lev2)

            # get indices for each unique 17 network parcels and get mean of those parcels
            foo_17 = np.zeros( [36, len(X_parcel_ts.time), 2])
            jj = 0
            for parcel in unique_parcels:
                idx = [i for i, x in enumerate(parcel_list) if x == parcel]
                foo_17[jj,:,0] = X_parcel_ts.isel(parcel=idx).sel(chromo='HbO').mean('parcel').values
                foo_17[jj,:,1] = X_parcel_ts.isel(parcel=idx).sel(chromo='HbR').mean('parcel').values
                jj += 1
            X_parcel_ts = xr.DataArray(foo_17,
                                        dims=['parcel', 'time','chromo'],
                                        coords={'parcel': unique_parcels, 'time': X_parcel_ts.time.values, 'chromo': ['HbO', 'HbR']})

        # check if parcelClusterIdx exists
        if 'parcelClusterIdx' in locals() and flag_use_parcel_networks:
            foo_cluster = np.zeros( [len(np.unique(parcelClusterIdx)), len(X_parcel_ts.time), 2])
            jj = 0
            for parcel in np.unique(parcelClusterIdx):
                idx = [i for i, x in enumerate(parcelClusterIdx) if x == parcel]
                foo_cluster[jj,:,0] = X_parcel_ts.isel(parcel=idx).sel(chromo='HbO').mean('parcel').values
                foo_cluster[jj,:,1] = X_parcel_ts.isel(parcel=idx).sel(chromo='HbR').mean('parcel').values
                jj += 1
            X_parcel_ts = xr.DataArray(foo_cluster,
                                        dims=['parcel', 'time','chromo'],
                                        coords={'parcel': np.unique(parcelClusterIdx), 'time': X_parcel_ts.time.values, 'chromo': ['HbO', 'HbR']})


        # get xarray of data during each trial type
        idx_trial_type = 0
        unique_trial_types = np.unique(rec[0][0].stim.trial_type)
        X_parcel_ts_tmp = {}
        for trial_type in unique_trial_types:
            idx = np.where(rec[idx_subj][idx_file].stim.trial_type==trial_type)[0]
            t_indices_tmp = np.array([])
            dt = np.median(np.diff(od_ts.time)) # same time as X_parcel_ts
            for ii in idx:
                t_indices_tmp = np.concatenate( (t_indices_tmp, np.where( 
                                    (od_ts.time >  rec[idx_subj][idx_file].stim.onset[ii]) &
                                    (od_ts.time <= (rec[idx_subj][idx_file].stim.onset[ii] + np.floor(rec[idx_subj][idx_file].stim.duration[ii]/dt)*dt + 1e-4 )) # this dt stuff is to ensure same lengths for each trial_type
                                    )[0] )
                                )

            X_parcel_ts_tmp[trial_type] = X_parcel_ts.isel(time=t_indices_tmp.astype(int)) #.expand_dims('trial_type').assign_coords(trial_type=[trial_type])
            X_parcel_ts_tmp[trial_type].samples.values = np.arange(0, len(X_parcel_ts_tmp[trial_type].time))
            X_parcel_ts_tmp[trial_type] = X_parcel_ts_tmp[trial_type].assign_coords(time=X_parcel_ts_tmp[trial_type].time.values - X_parcel_ts_tmp[trial_type].time.values[0])


        #     if idx_trial_type == 0:
        #         t_indices = np.zeros((2,len(t_indices_tmp)), dtype=int)
        #         t_indices[0,:] = t_indices_tmp
        #     else:
        #         t_indices[idx_trial_type,:] = t_indices_tmp
        #     idx_trial_type = idx_trial_type + 1

        # # FIXME: this hard codes each trial type to be active or passive. Fix to use unique_trial_types from above
        # X_parcel_ts_tmp = X_parcel_ts.isel(time=t_indices[0,:]).expand_dims('trial_type').assign_coords(trial_type=['active'])
        # X_parcel_ts_tmp.samples.values = np.arange(0, len(X_parcel_ts_tmp.time))
        # X_parcel_ts_tmp = X_parcel_ts_tmp.assign_coords(time=X_parcel_ts_tmp.time.values - X_parcel_ts_tmp.time.values[0])

        # X_parcel_ts_tmp2 = X_parcel_ts.isel(time=t_indices[1,:]).expand_dims('trial_type').assign_coords(trial_type=['passive'])
        # X_parcel_ts_tmp2.samples.values = np.arange(0, len(X_parcel_ts_tmp2.time))
        # X_parcel_ts_tmp2 = X_parcel_ts_tmp2.assign_coords(time=X_parcel_ts_tmp2.time.values - X_parcel_ts_tmp2.time.values[0])

        # X_parcel_ts_trialtype = xr.concat( [X_parcel_ts_tmp, X_parcel_ts_tmp2], dim='trial_type' )


        # concatentate runs
        if idx_file==0:
            X_parcel_ts_all_runs = X_parcel_ts
            X_parcel_ts_trialtype_all_runs = X_parcel_ts_tmp #X_parcel_ts_trialtype
        else:
            X_parcel_ts_all_runs = xr.concat([X_parcel_ts_all_runs, X_parcel_ts], dim='time')
            for trial_type in unique_trial_types:
                X_parcel_ts_trialtype_all_runs[trial_type] = xr.concat([X_parcel_ts_trialtype_all_runs[trial_type], X_parcel_ts_tmp[trial_type]], dim='time')


        # Get the parcel correlation matrix for each run
#        foo = np.corrcoef(X_parcel_ts_gms.sel(chromo='HbO').values, rowvar=True)
        if idx_file==0:
            r_files = np.zeros( [len(cfg_dataset['file_ids']), X_parcel_ts.parcel.size**2] )
            foo = np.corrcoef(X_parcel_ts.sel(chromo='HbO').values, rowvar=True)
            r_files[idx_file,:] = foo.reshape(-1)
            corr_matrix_subj = xr.DataArray(foo, 
                                       dims=['parcel_a', 'parcel_b'], 
                                       coords={'parcel_a': X_parcel_ts.parcel.values, 'parcel_b': X_parcel_ts.parcel.values})
        else:
            foo = np.corrcoef(X_parcel_ts.sel(chromo='HbO').values, rowvar=True)
            r_files[idx_file,:] = foo.reshape(-1)
            corr_matrix_subj = xr.concat([corr_matrix_subj, xr.DataArray(foo, 
                                       dims=['parcel_a', 'parcel_b'], 
                                       coords={'parcel_a': X_parcel_ts.parcel.values, 'parcel_b': X_parcel_ts.parcel.values})], dim='run')
    # end of run loop

    # Get the average correlation matrix
    if idx_subj==0:
        r_subjs = np.zeros( [len(cfg_dataset['subj_ids']), X_parcel_ts.parcel.size**2] )
        foo = np.corrcoef( X_parcel_ts_all_runs.sel(chromo='HbO').values, rowvar=True)
        r_subjs[idx_subj,:] = foo.reshape(-1)
        corr_matrix_grp = xr.DataArray(foo,
                                       dims=['parcel_a', 'parcel_b'],
                                       coords={'parcel_a': X_parcel_ts_all_runs.parcel.values, 'parcel_b': X_parcel_ts_all_runs.parcel.values})

        r_subjs_trialtype = np.zeros( [len(cfg_dataset['subj_ids']), len(unique_trial_types), X_parcel_ts_trialtype_all_runs[unique_trial_types[0]].parcel.size**2] )
        foo_trialtype = np.zeros( [len(unique_trial_types), X_parcel_ts_trialtype_all_runs[unique_trial_types[0]].parcel.size, X_parcel_ts_trialtype_all_runs[unique_trial_types[0]].parcel.size] )
        for ii in range(len(unique_trial_types)):
            foo_trialtype[ii,:,:] = np.corrcoef( X_parcel_ts_trialtype_all_runs[unique_trial_types[ii]].sel(chromo='HbO').values, rowvar=True)
            r_subjs_trialtype[idx_subj,ii,:] = foo_trialtype[ii,:,:].reshape(-1)
        corr_matrix_grp_trialtype = xr.DataArray(foo_trialtype,
                                       dims=['trial_type', 'parcel_a', 'parcel_b'],
                                       coords={'trial_type': unique_trial_types, 'parcel_a': X_parcel_ts_trialtype_all_runs[unique_trial_types[0]].parcel.values, 'parcel_b': X_parcel_ts_trialtype_all_runs[unique_trial_types[0]].parcel.values})
    else:
        foo = np.corrcoef( X_parcel_ts_all_runs.sel(chromo='HbO').values, rowvar=True)
        r_subjs[idx_subj,:] = foo.reshape(-1)
        foo_xr = xr.DataArray(foo,
                                dims=['parcel_a', 'parcel_b'],
                                coords={'parcel_a': X_parcel_ts_all_runs.parcel.values, 'parcel_b': X_parcel_ts_all_runs.parcel.values}) 
        corr_matrix_grp = xr.concat([corr_matrix_grp, foo_xr], dim='subj')

        for ii in range(len(unique_trial_types)):
            foo_trialtype[ii,:,:] = np.corrcoef( X_parcel_ts_trialtype_all_runs[unique_trial_types[ii]].sel(chromo='HbO').values, rowvar=True)
            r_subjs_trialtype[idx_subj,ii,:] = foo_trialtype[ii,:,:].reshape(-1)
        foo_xr = xr.DataArray(foo_trialtype,
                                dims=['trial_type', 'parcel_a', 'parcel_b'],
                                coords={'trial_type': unique_trial_types, 'parcel_a': X_parcel_ts_trialtype_all_runs[unique_trial_types[0]].parcel.values, 'parcel_b': X_parcel_ts_trialtype_all_runs[unique_trial_types[0]].parcel.values})
        corr_matrix_grp_trialtype = xr.concat([corr_matrix_grp_trialtype, foo_xr], dim='subj')

    # repeatability within a subject
    foo = np.corrcoef(r_files,rowvar=True)
    if idx_subj==0:
        repeatability_subj_mean = foo[np.triu_indices(foo.shape[0], k=1)].mean()
        repeatability_subj_std = foo[np.triu_indices(foo.shape[0], k=1)].std()
    else:
        repeatability_subj_mean = np.vstack([repeatability_subj_mean, foo[np.triu_indices(foo.shape[0], k=1)].mean()])
        repeatability_subj_std = np.vstack([repeatability_subj_std, foo[np.triu_indices(foo.shape[0], k=1)].std()])
# end of subject loop

# reliability across subjects
reliability_grp = np.corrcoef(r_subjs,rowvar=True)
reliability_grp_mean = np.nanmean(reliability_grp[np.triu_indices(reliability_grp.shape[0], k=1)])
reliability_grp_std = np.nanstd(reliability_grp[np.triu_indices(reliability_grp.shape[0], k=1)])

if not flag_use_17_networks and not flag_use_parcel_networks:
    corr_matrix_grp_orig = corr_matrix_grp.copy()

# %%
# Network repeatability and reliability

for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):
    print(f'Subject {curr_subj} has mean repeatability of {repeatability_subj_mean[idx_subj][0]:.3f} +/- {repeatability_subj_std[idx_subj][0]:.3f}')

print(f'\nThe mean subject repeatability is {np.nanmean(repeatability_subj_mean):.3f} +/- {np.nanstd(repeatability_subj_mean):.3f}')
print(f'\nThe group reliability is {reliability_grp_mean:.3f} +/- {reliability_grp_std:.3f}')

# %%
# Plot the mean correlation matrix across subjects and the t-stat
from scipy.stats import t
p_value = 0.05
df = 7-1  # 8 subjects but 1 with NaN
t_crit = t.ppf(1 - p_value/2, df)  # For two-tailed test

# get the mean correlation matrix across subjects and the t-stat
corr_matrix_grp_mean = corr_matrix_grp.mean('subj',skipna=True)
corr_matrix_grp_std = corr_matrix_grp.std('subj',skipna=True) / np.sqrt(7-1) # 8 subjects but 1 with NaN 
tstat_grp = corr_matrix_grp_mean / corr_matrix_grp_std

# plot the mean and tstat
fig, axs = p.subplots(1, 1, figsize=(8, 8))
foo = corr_matrix_grp_mean.values
ax1 = axs
ax1.imshow(foo, cmap='jet', vmin=-1, vmax=1)
ax1.set_xticks(range(len(corr_matrix_grp_mean.parcel_a.values)), corr_matrix_grp_mean.parcel_a.values, rotation=90)
ax1.set_yticks(range(len(corr_matrix_grp_mean.parcel_a.values)), corr_matrix_grp_mean.parcel_a.values)
ax1.set_title('Mean correlation matrix')
cbar = fig.colorbar(ax1.imshow(foo, cmap='jet', vmin=-1, vmax=1), ax=ax1)
cbar.set_label('t-stat')
p.show()

fig, axs = p.subplots(1, 1, figsize=(8, 8))
foo = tstat_grp.values
foo = np.where(np.abs(foo)<t_crit, 0, foo) # replace abs(foo)<t_crit with 0
ax2 = axs
ax2.imshow(foo, cmap='jet', vmin=-5, vmax=5)
ax2.set_xticks(range(len(corr_matrix_grp_mean.parcel_a.values)), corr_matrix_grp_mean.parcel_a.values, rotation=90)
ax2.set_yticks(range(len(corr_matrix_grp_mean.parcel_a.values)), corr_matrix_grp_mean.parcel_a.values)
ax2.set_title(f't-stat (t-crit={t_crit:.2f})')
cbar = fig.colorbar(ax2.imshow(foo, cmap='jet', vmin=-5, vmax=5), ax=ax2)
cbar.set_label('t-stat')
p.show()


# %%
# by trial type
# Plot the mean correlation matrix across subjects for each trial type and the t-stat

# get the mean correlation matrix across subjects and the t-stat
corr_matrix_grp_trialtype_mean = corr_matrix_grp_trialtype.mean('subj',skipna=True)
corr_matrix_grp_trialtype_std = corr_matrix_grp_trialtype.std('subj',skipna=True) / np.sqrt(7-1) # 8 subjects but 1 with NaN 
tstat_grp_trialtype = corr_matrix_grp_trialtype_mean / corr_matrix_grp_trialtype_std

# plot the mean and tstat
fig, axs = p.subplots(1, 3, figsize=(20,6))
#foo = corr_matrix_grp_trialtype_mean.sel(trial_type='active').values
foo = tstat_grp_trialtype.sel(trial_type='active').values
foo = np.where(np.abs(foo)<t_crit, 0, foo) # replace abs(foo)<t_crit with 0
ax1 = axs[0]
ax1.imshow(foo, cmap='jet')
#ax1.set_title('Mean correlation matrix: active')
ax1.set_title(f'Tstat (t-crit={t_crit:.2f}): active')
cbar = fig.colorbar(ax1.imshow(foo, cmap='jet', vmin=-5, vmax=5), ax=ax1)

#foo = corr_matrix_grp_trialtype_mean.sel(trial_type='active').values - corr_matrix_grp_trialtype_mean.sel(trial_type='passive').values
foo = tstat_grp_trialtype.sel(trial_type='passive').values
foo = np.where(np.abs(foo)<t_crit, 0, foo) # replace abs(foo)<t_crit with 0
ax1 = axs[1]
ax1.imshow(foo, cmap='jet')
#ax1.set_title('Mean correlation matrix: active - passive')
ax1.set_title(f'Tstat (t-crit={t_crit:.2f}): passive')
cbar = fig.colorbar(ax1.imshow(foo, cmap='jet', vmin=-5, vmax=5), ax=ax1)

if 0: # independent t-test
    foo = corr_matrix_grp_trialtype_mean.sel(trial_type='active').values - corr_matrix_grp_trialtype_mean.sel(trial_type='passive').values
    foo = foo / np.sqrt(corr_matrix_grp_trialtype_std.sel(trial_type='active').values**2 + corr_matrix_grp_trialtype_std.sel(trial_type='passive').values**2)
else: # pairwise t-test
    foo = corr_matrix_grp_trialtype.sel(trial_type='active') - corr_matrix_grp_trialtype.sel(trial_type='passive')
    foo_mean = foo.mean('subj',skipna=True)
    foo_std = foo.std('subj',skipna=True) / np.sqrt(7-1) # 8 subjects but 1 with NaN
    foo = foo_mean / foo_std
    foo = foo.values
foo = np.where(np.abs(foo)<t_crit, 0, foo) # replace abs(foo)<t_crit with 0
ax1 = axs[2]
ax1.imshow(foo, cmap='jet')
ax1.set_title(f'Tstat (t-crit={t_crit:.2f}): active - passive')
cbar = fig.colorbar(ax1.imshow(foo, cmap='jet', vmin=-5, vmax=5), ax=ax1)

p.show()

# %%
fig, axs = p.subplots(1, 1, figsize=(30, 8))

foo = corr_matrix_grp_trialtype.sel(trial_type='active') - corr_matrix_grp_trialtype.sel(trial_type='passive')
foo_mean = foo.mean('subj',skipna=True)
foo_std = foo.std('subj',skipna=True) / np.sqrt(7-1) # 8 subjects but 1 with NaN
foo = foo_mean / foo_std
foo = foo.values
foo = np.where(np.abs(foo)<t_crit, 0, foo) # replace abs(foo)<t_crit with 0

ax2 = axs
ax2.imshow(foo, cmap='jet', vmin=-5, vmax=5)

if 0:
    ax2.set_xticks(range(0, len(corr_matrix_grp_trialtype_mean.parcel_a.values), 10))
    ax2.set_xticklabels(range(0, len(corr_matrix_grp_trialtype_mean.parcel_a.values), 10), rotation=90)
    ax2.set_yticks(range(0, len(corr_matrix_grp_trialtype_mean.parcel_a.values), 10))
else:
    ax2.set_xticks(range(len(corr_matrix_grp_mean.parcel_a.values)), corr_matrix_grp_mean.parcel_a.values, rotation=90)
    ax2.set_yticks(range(len(corr_matrix_grp_mean.parcel_a.values)), corr_matrix_grp_mean.parcel_a.values)

# set the xlim and ylim
#ax2.set_xlim([405, 468])
#ax2.set_xlim([350, 370])
#ax2.set_xlim([405, 580])
ax2.set_xlim([70, 150])

#ax2.set_ylim([120, 140])
#ax2.set_ylim([70, 95])
ax2.set_ylim([172, 212])

ax2.set_title(f't-stat (t-crit={t_crit:.2f})')
if 0:
    cbar = fig.colorbar(ax2.imshow(foo, cmap='jet', vmin=-5, vmax=5), ax=ax2)
    cbar.set_label('t-stat')

p.show()


# %%
# FC stats for groups of parcels in the sub-networks of the 17 networks. 
# I am finding 55 lev2 networks per hemisphere vs the 17 lev1

# get the mean correlation matrix across subjects and the t-stat
corr_matrix_grp_trialtype_mean = corr_matrix_grp_trialtype.mean('subj',skipna=True)
corr_matrix_grp_trialtype_std = corr_matrix_grp_trialtype.std('subj',skipna=True) / np.sqrt(7-1) # 8 subjects but 1 with NaN 
tstat_grp_trialtype = corr_matrix_grp_trialtype_mean / corr_matrix_grp_trialtype_std

corr_matrix_grp_trialtype_diff_mean = (corr_matrix_grp_trialtype.sel(trial_type='active') - corr_matrix_grp_trialtype.sel(trial_type='passive')).mean('subj',skipna=True)
corr_matrix_grp_trialtype_diff_std = (corr_matrix_grp_trialtype.sel(trial_type='active') - corr_matrix_grp_trialtype.sel(trial_type='passive')).std('subj',skipna=True) / np.sqrt(7-1) # 8 subjects but 1 with NaN
tstat_grp_trialtype_diff = corr_matrix_grp_trialtype_diff_mean / corr_matrix_grp_trialtype_diff_std

from scipy.stats import t
p_value = 5e-2
df = 7-1  # 8 subjects but 1 with NaN
t_crit = t.ppf(1 - p_value/2, df)  # For two-tailed test


lev2_R = np.zeros( [len(unique_trial_types), len(unique_parcels_lev2), len(unique_parcels_lev2)] )
lev2_tstat = np.zeros( [len(unique_trial_types), len(unique_parcels_lev2), len(unique_parcels_lev2)] )

lev2_R_diff = np.zeros( [len(unique_parcels_lev2), len(unique_parcels_lev2)] )
lev2_tstat_diff = np.zeros( [len(unique_parcels_lev2), len(unique_parcels_lev2)] )
lev2_num_tcrit = np.zeros( [len(unique_parcels_lev2), len(unique_parcels_lev2)] )
lev2_num_tot = np.zeros( [len(unique_parcels_lev2), len(unique_parcels_lev2)] )


for ii in range( 0, len(unique_parcels_lev2)-1 ):
    idx1 = [i for i, x in enumerate(parcel_list_lev2) if x == unique_parcels_lev2.tolist()[ii]]
    # print(f'Network {unique_parcels_lev2[ii]} has the following parcels: {idx}')

    for jj in range( ii+1, len(unique_parcels_lev2) ):
        idx2 = [i for i, x in enumerate(parcel_list_lev2) if x == unique_parcels_lev2.tolist()[jj]]

        for kk in range( len(unique_trial_types) ):
            foo1 = tstat_grp_trialtype.sel(trial_type=unique_trial_types[kk]).values[idx1,:][:,idx2]
            foo2 = corr_matrix_grp_trialtype_mean.sel(trial_type=unique_trial_types[kk]).values[idx1,:][:,idx2]

            foo2 = np.where(np.abs(foo1)<t_crit, np.nan, foo2)
            foo1 = np.where(np.abs(foo1)<t_crit, np.nan, foo1)

            lev2_R[kk,ii,jj] = np.nanmean(foo2)
            lev2_tstat[kk,ii,jj] = np.nanmean(foo1)
            lev2_R[kk,jj,ii] = np.nanmean(foo2)
            lev2_tstat[kk,jj,ii] = np.nanmean(foo1)

        foo1 = tstat_grp_trialtype_diff.values[idx1,:][:,idx2]
        foo2 = corr_matrix_grp_trialtype_diff_mean.values[idx1,:][:,idx2]

        lev2_num_tcrit[ii,jj] = len( np.where(np.abs(foo1.reshape([-1]))>=t_crit)[0] )
        lev2_num_tot[ii,jj] = len(foo1.reshape([-1]))
        lev2_num_tcrit[jj,ii] = len( np.where(np.abs(foo1.reshape([-1]))>=t_crit)[0] )
        lev2_num_tot[jj,ii] = len(foo1.reshape([-1]))

        foo2 = np.where(np.abs(foo1)<t_crit, np.nan, foo2)
        foo1 = np.where(np.abs(foo1)<t_crit, np.nan, foo1)

        lev2_R_diff[ii,jj] = np.nanmean(foo2)
        lev2_tstat_diff[ii,jj] = np.nanmean(foo1)
        lev2_R_diff[jj,ii] = np.nanmean(foo2)
        lev2_tstat_diff[jj,ii] = np.nanmean(foo1)

        # replace nan with 0
        lev2_R = np.nan_to_num(lev2_R, nan=0)
        lev2_tstat = np.nan_to_num(lev2_tstat, nan=0)
        lev2_R_diff = np.nan_to_num(lev2_R_diff, nan=0)
        lev2_tstat_diff = np.nan_to_num(lev2_tstat_diff, nan=0)



# %%
# let's look at results for specific networks

network1 = 'ContC'
network2 = 'SalVentAttnB'


# get sentivity to each parcel
Adot_sum_parcel = Adot.sum('channel').groupby('parcel').sum('vertex')
# remove scalp and Background+Freesurfer... parcels
Adot_sum_parcel = Adot_sum_parcel.sel(parcel=Adot_sum_parcel.parcel != 'scalp')
Adot_sum_parcel = Adot_sum_parcel.sel(parcel=Adot_sum_parcel.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
Adot_sum_parcel = Adot_sum_parcel.sel(parcel=Adot_sum_parcel.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')

# list 

# remove LH and RH from unique_parcels
parcel_list_NH = [x.split('_')[0] for x in parcel_list]
unique_parcels_NH = np.unique([x.split('_')[0] for x in unique_parcels])
unique_parcels_lev2_NH = np.unique([x.split('_')[0] for x in unique_parcels_lev2])

parcels = X_parcel_ts.parcel.values

idx1 = [i for i, x in enumerate(parcel_list_NH) if x == network1]
idx2 = [i for i, x in enumerate(parcel_list_NH) if x == network2]

# split LH and RH
if 1:
    idx1a = []
    idx1b = []
    for ii in range( 0, len(idx1) ):
        if parcels[idx1[ii]].split('_')[-1] == 'LH':
            idx1a.append(idx1[ii])
        else:
            idx1b.append(idx1[ii])
    idx1 = idx1a + idx1b

    idx2a = []
    idx2b = []
    for ii in range( 0, len(idx2) ):
        if parcels[idx2[ii]].split('_')[-1] == 'LH':
            idx2a.append(idx2[ii])
        else:
            idx2b.append(idx2[ii])
    idx2 = idx2a + idx2b



fig, axs = p.subplots(1, 2, figsize=(25, 10))

# plot corr_matrix_grp_trialtype_diff_mean over idx1 and idx2 for active
foo = corr_matrix_grp_trialtype_mean.sel(trial_type='active').values[idx1,:][:,idx2]
ax1 = axs[0]
ax1.imshow(foo, cmap='jet', vmin=-1, vmax=1)
ax1.set_xticks(range(len(idx2)), [parcels[x] for x in idx2], rotation=90)
ax1.set_yticks(range(len(idx1)), [parcels[x] for x in idx1])
ax1.set_title(f'corr_matrix_grp_trialtype_diff_mean - active')
cbar = fig.colorbar(ax1.imshow(foo, cmap='jet', vmin=-1, vmax=1), ax=ax1)

# plot tstat_grp_trialtype_diff over idx1 and idx2
foo = tstat_grp_trialtype_diff.values[idx1,:][:,idx2]
foo = np.where(np.abs(foo)<t_crit, 0, foo) # replace abs(foo)<t_crit with 0
ax1 = axs[1]
ax1.imshow(foo, cmap='jet', vmin=-5, vmax=5)
ax1.set_xticks(range(len(idx2)), [parcels[x] for x in idx2], rotation=90)
ax1.set_yticks(range(len(idx1)), [parcels[x] for x in idx1])
ax1.set_title(f't-stat (t-crit={t_crit:.2f})')
cbar = fig.colorbar(ax1.imshow(foo, cmap='jet', vmin=-5, vmax=5), ax=ax1)

p.show()



# %%
# print the lev2 networks with the highest t-stat
foos = np.sort(lev2_tstat_diff.reshape([-1]))

# get the negative lev2 networks with the highest t-stat
# sort by num_tcrit in descending order
foos_neg = foos[0:np.where(foos > -t_crit)[0][0]:2]

jj = 0
neg_parcela = {}
neg_parcelb = {}
neg_tstat = np.zeros( [len(foos_neg)] )
neg_R_diff = np.zeros( [len(foos_neg)] )
neg_num_tcrit = np.zeros( [len(foos_neg)] )
neg_num_tot = np.zeros( [len(foos_neg)] )
for ii in range(0,len(foos_neg)):
    idx1 = np.where(lev2_tstat_diff == foos_neg[ii])
    neg_parcela[jj] = unique_parcels_lev2[idx1[0][0]]
    neg_parcelb[jj] = unique_parcels_lev2[idx1[1][0]]
    neg_tstat[jj] = foos_neg[ii]
    neg_R_diff[jj] = lev2_R_diff[idx1[0][0],idx1[1][0]]
    neg_num_tcrit[jj] = lev2_num_tcrit[idx1[0][0],idx1[1][0]]
    neg_num_tot[jj] = lev2_num_tot[idx1[0][0],idx1[1][0]]
    jj += 1

# sort by neg_num_tcrit in descending order
neg_idx_sort = np.argsort(neg_num_tcrit)[::-1]
for ii in range(0,len(neg_idx_sort)):
    print(f'Network {neg_parcela[neg_idx_sort[ii]]} vs {neg_parcelb[neg_idx_sort[ii]]} has')
    print(f'   t-stat of {neg_tstat[neg_idx_sort[ii]]:.2f}')
    print(f'   R_diff of {neg_R_diff[neg_idx_sort[ii]]:.2f}')
    print(f'   {100*neg_num_tcrit[neg_idx_sort[ii]]/neg_num_tot[neg_idx_sort[ii]]:.1f}% with {neg_num_tcrit[neg_idx_sort[ii]]} out of {neg_num_tot[neg_idx_sort[ii]]} parcels-pairs above t_crit')


# %%
# previous printing of t-stats
for ii in range(0,2000,2):
    idx1 = np.where(lev2_tstat_diff == foos[ii])
    numer = lev2_num_tcrit[idx1[0][0],idx1[1][0]]
    if numer > 3:
        print(f'Network {unique_parcels_lev2[idx1[0][0]]} vs {unique_parcels_lev2[idx1[1][0]]} has')
        print(f'   t-stat of {foos[ii]:.2f}')
        print(f'   R_diff of {lev2_R_diff[idx1[0][0],idx1[1][0]]:.2f}')
        numer = lev2_num_tcrit[idx1[0][0],idx1[1][0]]
        denom = lev2_num_tot[idx1[0][0],idx1[1][0]]
        print(f'   {100*numer/denom:.1f}% with {numer} out of {denom} parcels-pairs above t_crit')




# %%
# plot the results
fig, axs = p.subplots(2, 3, figsize=(23, 15))

ax2 = axs[0][0]
ax2.imshow(lev2_R[0,:,:], cmap='jet', vmin=-1, vmax=1)

ax2 = axs[0][1]
ax2.imshow(lev2_R[1,:,:], cmap='jet', vmin=-1, vmax=1)

ax2 = axs[0][2]
ax2.imshow(lev2_R_diff[:,:], cmap='jet', vmin=-0.2, vmax=0.2)

ax2 = axs[1][0]
ax2.imshow(lev2_tstat[0,:,:], cmap='jet', vmin=-5, vmax=5)

ax2 = axs[1][1]
ax2.imshow(lev2_tstat[1,:,:], cmap='jet', vmin=-5, vmax=5)

ax2 = axs[1][2]
ax2.imshow(lev2_tstat_diff[:,:], cmap='jet', vmin=-5, vmax=5)

p.show()






# %%
# Let's do clustering
# perform agglomerative clustering on the correlation matrix

from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

cluster_threshold = 0.6

# calculate the distance matrix
#dist_matrix = 1 - np.abs(corr_matrix)
#dist_matrix = 1 - corr_matrix_grp_mean.values
dist_matrix = 1 - corr_matrix_grp.mean('subj',skipna=True).values

# Fill NaN values with a specific value (e.g., 2)
dist_matrix = np.nan_to_num(dist_matrix, nan=2)

# Ensure the distance matrix is symmetric
dist_matrix = (dist_matrix + dist_matrix.T) / 2

# Ensure the diagonal of the distance matrix is zero
np.fill_diagonal(dist_matrix, 0)

# calculate the linkage matrix
linkage_matrix = linkage(squareform(dist_matrix), method="average")



# plot the dendrogram
f,ax = p.subplots(1,1,figsize=(10,10))
dendrogram(linkage_matrix, labels=corr_matrix_grp_mean.parcel_a.values, orientation="right", ax=ax, color_threshold=cluster_threshold )
p.tight_layout()
p.show()


# %%
# Plot the reordered correlation matrix
# Extract the order of the channels from the dendrogram
ordered_channels = leaves_list(linkage_matrix)

# Reorder the correlation matrix based on the dendrogram
reordered_corr_matrix = corr_matrix_grp_mean.values[ordered_channels, :][:, ordered_channels]

f, ax = p.subplots(1, 1, figsize=(10, 8))
m = ax.pcolormesh(np.arange(len(reordered_corr_matrix)), np.arange(len(reordered_corr_matrix)), reordered_corr_matrix, shading="nearest", cmap='jet', vmin=-1, vmax=1)
cb = p.colorbar(m, ax=ax)
p.tight_layout()
ax.yaxis.set_ticks(np.arange(len(reordered_corr_matrix)))
ax.yaxis.set_ticklabels(corr_matrix_grp_mean.parcel_a.values[ordered_channels])
ax.xaxis.set_ticks(np.arange(len(reordered_corr_matrix)))
ax.xaxis.set_ticklabels(corr_matrix_grp_mean.parcel_a.values[ordered_channels])
ax.invert_yaxis()
p.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
p.title('Reordered Correlation Matrix')
p.xlabel('Channel')
p.ylabel('Channel')
p.show()

# %%
# plot the number of clusters versus threshold
f,ax = p.subplots(1,1,figsize=(7,5))
fcluster(linkage_matrix, t=cluster_threshold, criterion="distance")
fcluster(linkage_matrix, t=cluster_threshold, criterion="distance").max()
n_clusters = np.array([fcluster(linkage_matrix, t=threshold, criterion="distance").max() for threshold in np.linspace(0, 1.2, 100)])
#ax.plot(np.linspace(0, 1.2, 100), n_clusters-n_clusters[-1]+1)
ax.semilogy(np.linspace(0, 1.2, 100), n_clusters-n_clusters[-1]+1)
ax.set_xlabel("Threshold")
ax.set_ylabel("Number of Clusters")
p.tight_layout()
p.show()



# %%
# Visualize the clusters
from math import ceil

if 0: # set the cluster_threshold
    cluster_threshold2 = cluster_threshold
    cluster_threshold2 = 1.
else: # set the number of clusters     
    # get first index of n_clusters <= 10
    idx = np.where(n_clusters <= 90+(n_clusters[-1]-1))[0][0]
    cluster_threshold2 = ceil(np.linspace(0, 1.2, 100)[idx] * 100) / 100

# Assign cluster labels to each channel based on the specified threshold
cluster_labels = fcluster(linkage_matrix, cluster_threshold2, criterion='distance')

# Create a list of the cluster numbers for each channel
# channel_clusters = list(zip(corr_matrix_xr.channel.values, cluster_labels))

# Get the maximum value of the second column (cluster labels)
# This is the number of clusters
max_cluster_label = max(cluster_labels) - (n_clusters[-1]-1)

# cluster index for each cluster
parcelClusterIdx = xr.DataArray(cluster_labels, dims='parcel', coords={'parcel': corr_matrix_grp_mean.parcel_a.values})

# %%
# print the parcels for each cluster
idx_cluster = 54
print(f'Cluster {idx_cluster} has the following parcels:')
print(corr_matrix_grp_mean.parcel_a.values[np.where(cluster_labels == idx_cluster)[0]])

# %%
# bin reordered correlation matrix by cluster labels
reordered_corr_matrix_binned = np.zeros([max_cluster_label, max_cluster_label])
for ii in range(max_cluster_label):
    for jj in range(max_cluster_label):
        idx1 = np.where(cluster_labels == ii+1)[0]
        idx2 = np.where(cluster_labels == jj+1)[0]
        reordered_corr_matrix_binned[ii,jj] = np.mean(reordered_corr_matrix[idx1,:][:,idx2])

# plot the reordered correlation matrix binned by cluster labels
f, ax = p.subplots(1, 1, figsize=(10, 8))
m = ax.pcolormesh(np.arange(len(reordered_corr_matrix_binned)), np.arange(len(reordered_corr_matrix_binned)), reordered_corr_matrix_binned, shading="nearest", cmap='jet', vmin=-1, vmax=1)
cb = p.colorbar(m, ax=ax)
p.tight_layout()
ax.yaxis.set_ticks(np.arange(len(reordered_corr_matrix_binned)))
ax.yaxis.set_ticklabels(np.arange(1,max_cluster_label+1))
ax.xaxis.set_ticks(np.arange(len(reordered_corr_matrix_binned)))
ax.xaxis.set_ticklabels(np.arange(1,max_cluster_label+1))
ax.invert_yaxis()
p.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
p.title('Reordered Correlation Matrix Binned by Cluster Labels')
p.xlabel('Cluster')
p.ylabel('Cluster')
p.show()

foo = np.zeros( [X_ts.vertex.size, 2])
# loop over parcels and assign the cluster label to each vertex
for idx_parcel, curr_parcel in enumerate(corr_matrix_grp_mean.parcel_a.values):
    idx = np.where(X_ts.parcel == curr_parcel)[0]
    foo[idx,0] = parcelClusterIdx[idx_parcel]
    foo[idx,1] = parcelClusterIdx[idx_parcel]

X_ClusterIdx = xr.DataArray(foo,
                            dims=['vertex', 'chromo'],
                            coords={'chromo': ['HbO', 'HbR']})
# assign parcel and is_brain coords
X_ClusterIdx = X_ClusterIdx.assign_coords(parcel=('vertex', X_ts.parcel.values))
X_ClusterIdx = X_ClusterIdx.assign_coords(is_brain=('vertex', X_ts.is_brain.values))

# %%
import importlib
importlib.reload(pfDAB_img)

foo_img = X_ClusterIdx
clim = (0,np.max(cluster_labels))
title_str = 'foo'

Save = True

p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,1), clim, 'hbo_brain', 'scale_bar',
                            None, title_str, off_screen=Save )
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,0), clim, 'hbo_brain', 'left', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,1), clim, 'hbo_brain', 'superior', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,2), clim, 'hbo_brain', 'right', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,0), clim, 'hbo_brain', 'anterior', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,2), clim, 'hbo_brain', 'posterior', p0)

if SAVE:
    filname = f'IMG_clusters.png'
    p0.screenshot( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'image_recon', filname) )
    p0.close()
else:
    p0.show()


# %%
#plot the correlation matrix for each subject
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 3, figsize=(12, 12))
for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):
    ax1 = axs.flatten()[idx_subj]
    foo = corr_matrix_grp.isel(subj=idx_subj).values
    ax1.imshow(foo, cmap='coolwarm', vmin=-1, vmax=1)
    if idx_subj<6:
        ax1.set_xticks(range(len(corr_matrix_subj.parcel_a.values)))
        ax1.set_xticklabels(range(len(corr_matrix_subj.parcel_a.values)), rotation=90)
    else:
        ax1.set_xticks(range(len(corr_matrix_subj.parcel_a.values)), corr_matrix_subj.parcel_a.values, rotation=90)
    if idx_subj%3==0:
        ax1.set_yticks(range(len(corr_matrix_subj.parcel_b.values)), corr_matrix_subj.parcel_b.values)
    else:
        ax1.set_yticks(range(len(corr_matrix_subj.parcel_b.values)))
    # set font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(6)


plt.show()









# %% OLD
##############################################################################

import importlib
importlib.reload(pfDAB_img)


trial_type_img = 'active_o_tddr' #'STS-o' # 'DT', 'DT-ica', 'ST', 'ST-ica'
t_win = (10, 20)

file_save = True
flag_Cmeas = True # if True make sure you are using the correct y_stderr_weighted below -- covariance

BRAIN_ONLY = False
SB = False  # spatial basis

sb_cfg = {
    'mask_threshold': -2,
    'threshold_brain': 5*units.mm,      # threshold_brain / threshold_scalp: Defines spatial limits for brain vs. scalp contributions.
    'threshold_scalp': 20*units.mm,
    'sigma_brain': 5*units.mm,      # sigma_brain / sigma_scalp: Controls smoothing or spatial regularization strength.
    'sigma_scalp': 20*units.mm,
    'lambda1': 0.01,        # regularization params
    'lambda2': 0.1
}

alpha_meas_list = [1e0] #[1e-2, 1e-3, 1e-5] #[1e-3]
alpha_spatial_list = [1e-1]#[1e-2, 1e-4, 1e-5, 1e-3, 1e-1] #[1e-3]


file_path0 = cfg_dataset['root_dir'] + 'derivatives/processed_data/'
wavelength = rec[0][0]['amp'].wavelength.values
spectrum = 'prahl'


#
# Get the group average image
#

if 'chromo' in blockaverage_all.dims:
    # get the group average HRF over a time window
    hrf_conc_mag = blockaverage_all.sel(trial_type=trial_type_img).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime')
    hrf_conc_ts = blockaverage_all.sel(trial_type=trial_type_img)

    # convert back to OD
    E = cedalion.nirs.get_extinction_coefficients(spectrum, wavelength)
    hrf_od_mag = xr.dot(E, hrf_conc_mag * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
    hrf_od_ts = xr.dot(E, hrf_conc_ts * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
else:
    hrf_od_mag = blockaverage_all.sel(trial_type=trial_type_img).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime')
    hrf_od_ts = blockaverage_all.sel(trial_type=trial_type_img)


if not flag_Cmeas:    
    X_grp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, None, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img)
else:
    trial_type_img_split = trial_type_img.split('-')
    C_meas = y_stderr_weighted.sel(trial_type=trial_type_img_split[0]).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime') # FIXME: what is the correct error estimate?
    C_meas = C_meas.pint.dequantify()     # remove units
    C_meas = C_meas**2  # get variance
    C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')  # !!! assumes y_stderr_weighted is in  OD - FIX
    X_grp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img)

print('Done with Image Reconstruction')




# %% Calculate the image noise and image CNR
##############################################################################

# scale columns of W by y_stderr_weighted**2
cov_img_tmp = W * np.sqrt(C_meas.values) # W is pseudo inverse  --- diagonal (faster than W C W.T)
cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)

nV = X_grp.shape[0]
cov_img_diag = np.reshape( cov_img_diag, (2,nV) ).T

# image noise
X_noise = X_grp.copy()
X_noise.values = np.sqrt(cov_img_diag)

filepath = os.path.join(file_path0, f'X_noise_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
print(f'   Saving to X_noise_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps([X_noise, alpha_meas_list[-1], alpha_spatial_list[-1]]))
file.close()  

# image t-stat (i.e. CNR)
X_tstat = X_grp / np.sqrt(cov_img_diag)

X_tstat[ np.where(cov_img_diag[:,0]==0)[0], 0 ] = 0
X_tstat[ np.where(cov_img_diag[:,1]==0)[0], 1 ] = 0

filepath = os.path.join(file_path0, f'X_tstat_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
print(f'   Saving to X_tstat_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps([X_tstat, alpha_meas_list[-1], alpha_spatial_list[-1]]))
file.close()     


# %% Get image for each subject and do weighted average
##############################################################################
import importlib
importlib.reload(pfDAB_img)


file_save = False
trial_type_img = 'DT' # 'STS', 'DT-ica', 'ST', 'ST-ica'
t_win = (10, 20)

BRAIN_ONLY = False
SB = False

sb_cfg = {
    'mask_threshold': -2,
    'threshold_brain': 5*units.mm,
    'threshold_scalp': 20*units.mm,
    'sigma_brain': 5*units.mm,
    'sigma_scalp': 20*units.mm,
    'lambda1': 0.01,
    'lambda2': 0.1
}

alpha_meas_list = [1e0] #[1e-2, 1e-3, 1e-5] #[1e-3]
alpha_spatial_list = [1e-1]#[1e-2, 1e-4, 1e-5, 1e-3, 1e-1] #[1e-3]


file_path0 = cfg_dataset['root_dir'] + 'derivatives/processed_data/'
wavelength = rec[0][0]['amp'].wavelength.values
spectrum = 'prahl'


X_hrf_mag_subj = None
C = None # spatial regularization 
D = None

# !!! go thru each trial type (outside function)
    # 
for idx_subj in range(n_subjects):

# !!!vstart trial type loop
    hrf_od_mag = y_subj.sel(subj=cfg_dataset['subj_ids'][idx_subj]).sel(trial_type=trial_type_img).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime')
    # hrf_od_ts = blockaverage_all.sel(trial_type=trial_type_img)

    # get the image
    trial_type_img_split = trial_type_img.split('-')
    C_meas = y_mse_subj.sel(subj=cfg_dataset['subj_ids'][idx_subj]).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime').mean('trial_type') # FIXME: handle more than one trial_type -- he was getting rid of trial type dim

#    trial_type_img_split = trial_type_img.split('-')
    C_meas = y_mse_subj.sel(subj=cfg_dataset['subj_ids'][idx_subj]).sel(trial_type=trial_type_img).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime') # FIXME: handle more than one trial_type

    C_meas = C_meas.pint.dequantify()
    C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
    if C is None or D is None:
        X_hrf_mag_tmp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img) 
    else:
        X_hrf_mag_tmp, W, _, _ = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img, None, C, D)

    # get image noise
    cov_img_tmp = W * np.sqrt(C_meas.values) # get diag of image covariance
    cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)

    nV = X_hrf_mag_tmp.vertex.size
    cov_img_diag = np.reshape( cov_img_diag, (2,nV) ).T

    X_mse = X_hrf_mag_tmp.copy() 
    X_mse.values = cov_img_diag
    
    # !!! end trial type loop

    # weighted average -- same as chan space - but now is vertex space
    if X_hrf_mag_subj is None:
        X_hrf_mag_subj = X_hrf_mag_tmp
        X_hrf_mag_subj = X_hrf_mag_subj.expand_dims('subj')
        X_hrf_mag_subj = X_hrf_mag_subj.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])

        X_mse_subj = X_mse.copy()
        X_mse_subj = X_mse_subj.expand_dims('subj')
        X_mse_subj = X_mse_subj.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])

        X_hrf_mag_weighted = X_hrf_mag_tmp / X_mse
        X_mse_inv_weighted = 1 / X_mse
        X_mse_inv_weighted_max = 1 / X_mse
    elif cfg_dataset['subj_ids'][idx_subj] not in subj_id_exclude:
        X_hrf_mag_subj_tmp = X_hrf_mag_tmp.expand_dims('subj') # !!! will need to expand dims to get back trial type -- can do in function 
        X_hrf_mag_subj_tmp = X_hrf_mag_subj_tmp.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])

        X_mse_subj_tmp = X_mse.copy().expand_dims('subj')
        X_mse_subj_tmp = X_mse_subj_tmp.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])

        X_hrf_mag_subj = xr.concat([X_hrf_mag_subj, X_hrf_mag_subj_tmp], dim='subj')
        X_mse_subj = xr.concat([X_mse_subj, X_mse_subj_tmp], dim='subj')

        X_hrf_mag_weighted = X_hrf_mag_weighted + X_hrf_mag_tmp / X_mse
        X_mse_inv_weighted = X_mse_inv_weighted + 1 / X_mse
        X_mse_inv_weighted_max = np.maximum(X_mse_inv_weighted_max, 1 / X_mse)
    else:
        print(f"   Subject {cfg_dataset['subj_ids'][idx_subj]} excluded from group average")


# %%

# get the average
X_hrf_mag_mean = X_hrf_mag_subj.mean('subj')
X_hrf_mag_mean_weighted = X_hrf_mag_weighted / X_mse_inv_weighted

X_mse_mean_within_subject = 1 / X_mse_inv_weighted

X_mse_subj_tmp = X_mse_subj.copy()
X_mse_subj_tmp = xr.where(X_mse_subj_tmp < 1e-6, 1e-6, X_mse_subj_tmp)
X_mse_weighted_between_subjects_tmp = (X_hrf_mag_subj - X_hrf_mag_mean)**2 / X_mse_subj_tmp # X_mse_subj_tmp is weights for each sub
X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp.mean('subj')
X_mse_weighted_between_subjects = X_mse_weighted_between_subjects / (X_mse_subj**-1).mean('subj')

X_stderr_weighted = np.sqrt( X_mse_mean_within_subject + X_mse_weighted_between_subjects )

X_tstat = X_hrf_mag_mean_weighted / X_stderr_weighted

X_weight_sum = X_mse_inv_weighted / X_mse_inv_weighted_max 
# FIXME: I am trying to get something like number of subjects per vertex...
# maybe I need to change X_mse_inv_weighted_max to be some typical value 
# because when all subjects have a really low value, then it won't scale the way I want

#    blockaverage_stderr_weighted = blockaverage_stderr_weighted.assign_coords(trial_type=blockaverage_mean_weighted.trial_type)

filepath = os.path.join(file_path0, f'Xs_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
print(f'   Saving to Xs_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps([X_weight_sum, alpha_meas_list[-1], alpha_spatial_list[-1]]))
file.close()     


# %%
threshold = -2 # log10 absolute
wl_idx = 1
M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)


# %% Plot the images
##############################################################################
import importlib
importlib.reload(pfDAB_img)

flag_hbo = True
flag_brain = True
flag_img = 'mag' # 'tstat', 'mag', 'noise'
flag_condition = 'active_o_tddr' # 'ST', 'DT', 'STS'

# results_img_grp = {'X_grp_all_trial': all_trial_X_grp,
#            'X_noise_grp_all_trial': all_trial_X_noise,
#            'X_tstat_grp_all_trial': all_trial_X_tstat
#            }

if flag_hbo:
    title_str = 'HbO'
    hbx_brain_scalp = 'hbo'
else:
    title_str = 'HbR'
    hbx_brain_scalp = 'hbr'

if flag_brain:
    title_str = title_str + ' brain'
    hbx_brain_scalp = hbx_brain_scalp + '_brain'
else:
    title_str = title_str + ' scalp'
    hbx_brain_scalp = hbx_brain_scalp + '_scalp'

if flag_img == 'tstat':
    foo_img = X_tstat.copy()
    title_str = title_str + ' t-stat'
elif flag_img == 'mag':
#    foo_img = X_hrf_mag_mean_weighted.copy()
    foo_img = results_img_grp['X_grp_all_trial'].sel(trial_type=flag_condition)
    title_str = title_str + ' magnitude'
elif flag_img == 'noise':
    foo_img = X_stderr_weighted.copy()
    title_str = title_str + ' noise'
#    foo_img.values = np.log10(foo_img.values)+7

title_str = title_str + ' ' + flag_condition

# title_str = 'HbR'
# hbx_brain_scalp = 'hbr_brain'
# foo_img = X_hrf_mag_mean_weighted

# title_str = 'HbR t-stat'
# hbx_brain_scalp = 'hbr_brain'
# foo_img = X_tstat




foo_img[~M] = np.nan
# foo_img = xr.where(np.abs(foo_img) < 1.86, np.nan, foo_img) # one-tail is 1.86 and two tail is 2.3

clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())

p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,1), clim, hbx_brain_scalp, 'scale_bar',
                            None, title_str, off_screen=True )
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,0), clim, hbx_brain_scalp, 'left', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,1), clim, hbx_brain_scalp, 'superior', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,2), clim, hbx_brain_scalp, 'right', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,0), clim, hbx_brain_scalp, 'anterior', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,2), clim, hbx_brain_scalp, 'posterior', p0)

# p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,1), clim, hbx_brain_scalp, 'scale_bar', None, title_str)
# p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,0), hbx_brain_scalp, 'left', p0)
# p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,1), hbx_brain_scalp, 'superior', p0)
# p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,2), hbx_brain_scalp, 'right', p0)
# p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,0), hbx_brain_scalp, 'anterior', p0)
# p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,2), hbx_brain_scalp, 'posterior', p0)

p0.screenshot( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', f'IMG.png') )
p0.close()


# %%
X_foo = X_tstat.copy()
X_foo[:,0] = 0

# select parcels
# parcels with '_LH' at the end
parcels = np.unique(X_grp['parcel'].values)
parcels_LH = [x for x in parcels if x.endswith('_LH')]

parcels_sel = [x for x in parcels_LH if 'DefaultB_PFCv' in x]

X_foo[np.isin(X_foo['parcel'].values, parcels_sel), 0] = 1


p0 = pfDAB_img.plot_image_recon(X_foo, head, 'hbo_brain', 'left')



# %% MNI coordinates
head_ras = head.apply_transform(head.t_ijk2ras)

# brain indices
idx_brain = np.where(Adot.is_brain)[0]

# make an xarray associating parcels with MNI coordinates
parcels_mni_xr = xr.DataArray(
    head_ras.brain.mesh.vertices[idx_brain,:],
    dims = ('vertex', 'coord'),
    coords = {'parcel': ('vertex', Adot.coords['parcel'].values[idx_brain])},
)

# get MNI coordinates of a specific parcel 'VisCent_ExStr_11_LH'
parcel_specific = parcels_mni_xr.where(parcels_mni_xr['parcel'] == 'VisCent_ExStr_11_LH', drop=True)

# find the parcel closest to a specific MNI coordinate
mni_coord = np.array([[ -27.1, -100.1 ,    9.4]])
dist = np.linalg.norm(parcels_mni_xr.values - mni_coord, axis=1)
parcel_closest = parcels_mni_xr[np.argmin(dist)]
print(f'Parcel closest to {mni_coord} is {parcel_closest["parcel"].values} with MNI coordinates {parcel_closest.values}')
print(f'Distance is {np.min(dist):0.2f} mm')

# %% Parcels
##############################################################################
# list unique parcels in X
parcels = np.unique(X_grp['parcel'].values)

# parcels with '_LH' at the end
parcels_LH = [x for x in parcels if x.endswith('_LH')]

# select parcels with a specific name
parcels_sel = [x for x in parcels_LH if 'DefaultB_PFCv' in x]



Xo = X_tstat.sel(chromo='HbO')

# Create a mapping from vertex to parcel
vertex_to_parcel = Xo['parcel'].values

# Add the parcel information as a coordinate to the DataArray/Dataset
Xo = Xo.assign_coords(parcel=('vertex', vertex_to_parcel))

# Group by the parcel coordinate and calculate the mean over the vertex dimension
Xo_parcel = Xo.groupby('parcel').mean(dim='vertex')


if 0: # find Xo_parcel values > 2 and from parcels_LH
    Xo_parcel_2 = Xo_parcel.where(np.abs(Xo_parcel) > 1).dropna('parcel').where(Xo_parcel['parcel'].isin(parcels_LH)).dropna('parcel')
else: # find Xo_parcel values > 2 and from parcels_sel
    Xo_parcel_2 = Xo_parcel.where(np.abs(Xo_parcel) > 1).dropna('parcel').where(Xo_parcel['parcel'].isin(parcels_sel)).dropna('parcel')

X_foo = X_tstat.copy()
X_foo[:,0] = 0
X_foo[np.isin(X_foo['parcel'].values, np.unique(Xo_parcel_2['parcel'].values) ), 0] = 1



od_ts = hrf_od_ts.stack(measurement=('channel', 'wavelength')).sortby('wavelength').T
X_grp_ts = W @ od_ts.values

split = len(X_grp_ts)//2
X_grp_ts = X_grp_ts.reshape([2, split, X_grp_ts.shape[1]])
X_grp_ts = X_grp_ts.transpose(1,0,2)

X_grp_ts = xr.DataArray(X_grp_ts,
                    dims = ('vertex', 'chromo', 'reltime'),
                    coords = {'chromo': ['HbO', 'HbR'],
                            'parcel': ('vertex',Adot.coords['parcel'].values),
                            'is_brain':('vertex', Adot.coords['is_brain'].values),
                            'reltime': od_ts.reltime.values},
                    )
X_grp_ts = X_grp_ts.set_xindex("parcel")



# get the time series for the parcels
Xo_ts = X_grp_ts #.sel(chromo='HbO')
vertex_to_parcel = Xo_ts['parcel'].values
Xo_ts = Xo_ts.assign_coords(parcel=('vertex', vertex_to_parcel))
Xo_ts_parcel = Xo_ts.groupby('parcel').mean(dim='vertex')

# plot the significant parcels
foo = Xo_ts_parcel.sel(parcel=Xo_parcel_2.parcel.values)

f, ax = p.subplots(1, 1, figsize=(7, 5))
for i in range(foo.sizes['parcel']):
    line, = ax.plot(foo['reltime'], foo.sel(parcel=foo['parcel'][i], chromo='HbO'), label=foo['parcel'][i].values)
    ax.plot(foo['reltime'], foo.sel(parcel=foo['parcel'][i], chromo='HbR'), linestyle='--', color=line.get_color())
ax.set_title('Significant parcels')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (M)')
ax.legend()
p.show()

p0 = pfDAB_img.plot_image_recon(X_foo, head, 'hbo_brain', 'left')









# %% Old Code
##############################################################################
##############################################################################

