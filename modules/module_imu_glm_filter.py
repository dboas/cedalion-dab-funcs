# -*- coding: utf-8 -*-
"""
Walking Filter 

Use short time GLM to filter walking data 
-Runs ICA on accelerometer data to separate gait and cardiac on 
    walking portion of IWHD study and apply splineSG to walking and standing 
    separately
Process walking data

Created on Fri Jan 24 14:37:57 2025

@author: shank
"""

import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils
import numpy as np
import xarray as xr
import pint
import matplotlib.pyplot as plt
import cedalion.vis as plots
from cedalion import units
import scipy.signal
import os.path
import pandas as pd
from cedalion.vis.misc import plot_probe_gui as vpp
from cedalion.vis.misc import time_series_gui as vts
#from circle_probe_cedalion import plot_circle_probe
from sklearn.linear_model import LinearRegression

#import pywt
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA
from scipy.signal import butter, sosfilt


import pdb


#%%

def filterWalking(rec, rec_str, cfg_imu_glm, filenm = None, cfg_dataset = None):
    ''' Filter Walking portion of the data in dOD space 
        inputs: 
            rec - xarray 
            rec_str - timeseries name you want to filter
            cfg_imu_glm - params 
            filenm - 
            filepath - 
        
        output:
            dod_filtered - xarray containing filtered dOD data
    '''
    
    dod = rec[rec_str]
    
    # vals needd
    t = np.array(rec["amp"].time)
    hWin = cfg_imu_glm['hWin']
    statesPerDataFrame = cfg_imu_glm['statesPerDataFrame']
    
    # -------------------------
    # Get IMU data
    # -------------------------
    accel_x = rec.aux_ts["ACCEL_X"]
    accel_y = rec.aux_ts["ACCEL_Y"]
    accel_z = rec.aux_ts["ACCEL_Z"]
    accel = xr.concat([accel_x, accel_y, accel_z], dim='axis')
    accel = accel.assign_coords(axis=['x', 'y', 'z'])

    accel_np = np.array(accel).squeeze().T
    accel_mag = np.linalg.norm(accel_np, axis=1)

    gyro_x = rec.aux_ts["GYRO_X"]
    gyro_y = rec.aux_ts["GYRO_Y"]
    gyro_z = rec.aux_ts["GYRO_Z"]
    gyro = xr.concat([gyro_x, gyro_y, gyro_z], dim='axis')
    gyro = gyro.assign_coords(axis=['x', 'y', 'z'])

    gyro_np = np.array(gyro).squeeze().T
    gyro_mag = np.linalg.norm(gyro_np, axis=1)

    accel_t = accel_x["time"] # time
    gyro_t = gyro_x["time"]
    
    # -------------------------
    # ID Walking period 
    # -------------------------
    stim = rec.stim 
    lstWalk, lstStand = id_walking(dod, stim) 
    
    # -------------------------
    # Get IMU regressors w/ ICA
    # -------------------------
    print('Get IMU regressors by ICA')
    # make acc and gyr data numpy arrays
    accel_mag_np = accel_mag.reshape(-1,1) # reshape to have 2 dimensions
    accel_t_np = np.array(accel_t)
    gyr_t_np = np.array(gyro_t)

    # Fast ICA
    icaACC = FastICA( cfg_imu_glm['n_components'][1]) #, random_state=0)
    #pdb.set_trace()
    zAcc = icaACC.fit_transform(np.hstack([accel_np, accel_mag_np]))
    icaGyr = FastICA( n_components = cfg_imu_glm['n_components'][0])
    zGyr = icaGyr.fit_transform(gyro_np)
    
    # -------------------------
    # Downsample regressors (z) to match fnirs data
    # -------------------------
    z = np.hstack((zAcc, zGyr))
    z_resamp = downsample_IMU(z, t, accel_t_np, statesPerDataFrame)
    
    # -------------------------
    # Create GLM design matrix
    # -------------------------
    lstWalktmp = lstWalk[(hWin[-1]):(len(lstWalk) + hWin[0])] # adjust lstWalk for time window
    
    A, AA = GLM_designMat(z_resamp, lstWalk, hWin, lstWalktmp) # get 2D and 3D design matrix
    
    # -------------------------
    # High pass filter the dod for GLM
    # -------------------------
    Fs = 1/(t[1]-t[0]) # samp freq (Hz)
    Fc = cfg_imu_glm['Fc']  # cutoff freq (Hz)
    Wn = Fc / (Fs/2) # normalizing cutoff freq
    order = cfg_imu_glm['butter_order']  # butterworth filter order
    
    sos = butter(order, Wn, btype='high', output='sos') # Design the Butterworth high-pass filter
    # Apply the high-pass filter to dod along the time axis
    dodHP = sosfilt(sos, dod, axis=-1) # dod is chanXwavXtime - want to do filtering along time

    
    # -------------------------
    # GLM regression to remove gait artifact
    # -------------------------
    dodHP = dodHP.T # transpose for math purp

    # calc gait ratio b4 and after GLM 
    gaitRatio_b4 = (np.std(dodHP[lstWalk,:], axis=0, ddof=1)) / (np.std(dodHP[lstStand,:], axis=0, ddof=1))

    # Do GLM on every channel
    # A = predictor, h = corresponding weights --- A*h = component contribution to the modelled signal
    #dodHPfilt = dodHP
    dodHPfiltnew = np.zeros_like(dodHP)

    gaitRatio_af = np.zeros_like(gaitRatio_b4)
    varExp = np.zeros((dodHP.shape[1], dodHP.shape[2], z_resamp.shape[1]))

    dodT = dod.T
    for iw in range(dod.shape[1]): # for loop thru each wavelength
        dodHPtmp = dodHP[:,iw,:] # grab chans from curr wavelength
        dodHPfilt = dodHPtmp.copy()
        
        for iM in range(dod.shape[0]): # loop thru each channel
            h = np.linalg.inv(AA.T @ AA) @ AA.T @ dodHPtmp[lstWalktmp, iM] # calc glm weights - mins squared error btwn model & AA*h w/ ordinary least squares
            
            # remove artifacts modelled by AA @ h
            dodHPfilt[lstWalktmp, iM] = dodHPtmp[lstWalktmp, iM] - AA @ h  # AA@h is so close to 0 that subtracting it here is negligable
            
            # Calc explained variance
            for ic in range(z_resamp.shape[1]): # loop thru each component
                # calc correlation coeff btwn observed signal at curr channel and the modeled component
                foo = np.corrcoef( dodHPtmp[lstWalktmp, iM], A[:,:,ic] @ ( h[0:len(hWin)] + len(hWin)*(ic)) )
                varExp[iw, iM, ic] = foo[0,1]**2
                
            # calc gait ratio after
            gaitRatio_af[iw, iM] = (gaitRatio_b4[iw, iM] * np.std( dodHPfilt[lstWalktmp, iM])) / np.std( dodHPtmp[lstWalktmp, iM])
                
        # Store the filtered result for the current wavelength
        dodHPfiltnew[:, iw, :] = dodHPfilt
        
    # Filtered data
    dod2 = dod - dodHP.T + dodHPfiltnew.T # add back in low freq dod data
    
    # Plot gait artifact ratio before and after correction & variance explained
    plotGaitRatio(rec, dod, gaitRatio_b4, gaitRatio_af, filenm, cfg_dataset)
    plotVarExp(rec, dod, z_resamp, varExp, filenm, cfg_dataset)
        
    # -------------------------
    # Create new xarray with filtered dOD data
    # -------------------------
    #assert dod2.shape == rec["od_pruned"].shape, "Shape mismatch between dod2 and rec['od_pruned']!"

    # Create a new xarray with the same structure as rec["od_pruned"], but with new values
    dod_filt = xr.DataArray(
        data=dod2,
        dims=dod.dims,         # Use the same dimensions as the original
        coords=dod.coords,     # Use the same coordinates as the original
        attrs=dod.attrs        # Optionally copy attributes from the original
    )
    
    return dod_filt
         
    
    
def id_walking(dod, stim ):
    '''Function that identifies the walking period based on stim
    
    ** double check that this is right for all subs **
    
    inputs:
        dod - rec["od"]
        stim - rec.stim
    output: 
        lsWalk and lstStand - arrays that have indices for walking and standing
    '''
    # DOUBLE CHECK when the person starts walking/ standing again after trial start
    #dod = np.array(rec["od_pruned"]) # -- shape = chans x wavelength x time 

    # df = stim
    # startWalk = df[df['trial_type'] == 'DT']['onset'].min() - 20
    # endWalk = df[df['trial_type'] == 'DT']['onset'].max() + 21
    # endStand = df[df['trial_type'] == 'ST']['onset'].max() + 20
    
    startWalk = stim[stim['trial_type'] == 'start_walk']['onset'].iloc[0]
    endWalk = stim[stim['trial_type'] == 'end_walk']['onset'].iloc[0]
    startStand = stim[stim['trial_type'] == 'start_stand']['onset'].iloc[0]
    endStand = stim[stim['trial_type'] == 'end_stand']['onset'].iloc[0]
    
    
    t_index = dod.get_index('time')
    closest_start = t_index[np.abs(t_index - startWalk).argmin()]
    start_idx = t_index.get_loc(closest_start) # start walk index
    closest_stop = t_index[np.abs(t_index - endWalk).argmin()]
    stop_idx = t_index.get_loc(closest_stop) # end walking period index
    closestEnd = t_index[np.abs(t_index - endStand).argmin()]
    stopStand_idx = t_index.get_loc(closestEnd)

    lstWalk = np.arange(start_idx, stop_idx+1,1) # time indices for walking portion
    lstStand = np.concatenate((np.arange(0,start_idx-1,1), np.arange(stop_idx+2, stopStand_idx,1)))
    
    return lstWalk, lstStand
    

def downsample_IMU(z, t, accel_t_np, statesPerDataFrame):
    ''' Downsample IMU regressor data to match fnirs sampling rate
    '''
    z_resamp = np.zeros((len(t), z.shape[1]))

    # mean subtract to ensure mean centered around 0 (remove DC offset)
    #z = z - np.ones((z.shape[0],1))*np.mean(z,axis=0) # mean subtract 

    z = z - np.mean(z, axis=0, keepdims=True)  # Center data by subtracting column-wise mean

    # loop thru each col of z (ica components)
    for iz in range(z.shape[1]):
        # low pass filter data w/ zero-phase filter data to smooth the signal
        z[:,iz] = signal.filtfilt(np.ones((statesPerDataFrame)), statesPerDataFrame, z[:,iz])
        
        # resample and align signal
        interp_mdl = interp1d(accel_t_np, z[:,iz], kind='linear', bounds_error=False, fill_value='extrapolate')
        z_resamp[:,iz] = interp_mdl(t)
    
    return z_resamp


def GLM_designMat(z_resamp, lstWalk, hWin, lstWalktmp):
    ''' Create GLM design matrix 
        inputs:
            z_resamp - downsampled imu regressor data
            lstWalk - array of indices that indicate walk period
            hWin - array of time shifts
        output:
            A - 3D GLM design matrix
            AA - 2D GLM deisgn matrix
        
    '''
    # Initialize 3D design matrix
    A = np.zeros((len(lstWalk)-len(hWin)+1, len(hWin), z_resamp.shape[1]))

    # construct 3D matrix 
    for ih in range(len(hWin)): # loop over each time lag or lead
        for ic in range(z_resamp.shape[1]): # loop over each predictor/ component / regressor
            A[:,ih,ic] = z_resamp[lstWalktmp - hWin[ih], ic] 
            # ^ extracts vals from z_resamp corresponding to time-shifted indices for each component
            # creates time lagged predictors for each component

        
    AA = np.hstack([A[:, :, ic] for ic in range(z_resamp.shape[1])]) # concat predictors w/ time-lagged versions along col axis
    
    return A, AA

def plotGaitRatio(rec, dod, gaitRatio_b4, gaitRatio_af, filenm = None, cfg_dataset = None):
    # If save path for plot does not exist, create it
    der_dir = os.path.join(cfg_dataset["root_dir"], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', 'walking_filter')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
        
    # Plot gait artifact ratio before and after correction
    foo = gaitRatio_b4[0,:]

    f, ax = plt.subplots(2, 1, figsize=(10, 10))
    plots.scalp_plot( 
            dod,
            rec.geo3d,
            foo,
            ax[0],
            cmap='jet',
            vmin=1,
            vmax=3,
            optode_labels=False,
            title="Gait Ratio Before",
            optode_size=6
        )
        
    foo2 = gaitRatio_af[0,:]
    plots.scalp_plot( 
            dod,
            rec.geo3d,
            foo2,
            ax[1],
            cmap='jet',
            vmin=1,
            vmax=3,
            optode_labels=False,
            title="Gait Ratio After",
            optode_size=6
        )    
    
    plt.suptitle(filenm)

    plt.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots','DQR', 'walking_filter', filenm + "_imu_glm_gaitRatio.png") )
    plt.close()
    
def plotVarExp(rec, dod, z_resamp, varExp, filenm = None, cfg_dataset = None):
    '''  Plot variance explained by each ICA component '''
    f, ax = plt.subplots(5, 1, figsize=(8, 15))

    for ic in range(z_resamp.shape[1]):
        foo = varExp[0,:,ic]*100
        plots.scalp_plot( 
                dod,
                rec.geo3d,
                foo,
                ax[ic],
                cmap='jet',
                vmin=0,
                vmax=30,
                optode_labels=False,
                title=f"Var Explained by ICA # {ic}",
                optode_size=1
            ) 
    
    plt.suptitle(filenm)

    plt.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots','DQR', 'walking_filter', filenm + "_imu_glm_varExp.png") )
    plt.close()