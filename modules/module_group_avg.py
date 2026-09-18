# group_avg_GLM()
# group_avg_block()


import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality

import cedalion.models.glm as glm
import cedalion.vis as plots

from cedalion import units
import numpy as np
import xarray as xr

import matplotlib.pyplot as p

import pdb



def run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg ):
    
    subj_ids_new = [s for s in cfg_dataset['subj_ids'] if s not in cfg_dataset['subj_id_exclude']]
    
    # Build a new list excluding the subjects in subj_id_exclude:
    excluded = set(cfg_dataset['subj_id_exclude'])
    new_filenm_lst = [
        run_list for run_list in cfg_dataset['filenm_lst']
        if run_list[0].split('_')[0].replace('sub-', '') not in excluded
    ]
    
    # choose correct mse values based on if blockaveraging od or conc
    if 'chromo' in rec[0][0][rec_str].dims:
        cfg_mse = cfg_blockavg['cfg_mse_conc']
    else:
        cfg_mse = cfg_blockavg['cfg_mse_od']
        

    subj_ids = cfg_dataset['subj_ids']
    n_subjects = len(rec)
    n_files_per_subject = len(rec[0])

    print(f"Running group block average for trial_type = '{rec_str}'")

    # loop over subjects and files
    blockaverage_subj = None
    for subj_idx in range( n_subjects ):
        for file_idx in range( n_files_per_subject ):
            
            filenm = new_filenm_lst[subj_idx][file_idx]
            print(f'Running {subj_idx+1} of {n_subjects} subjects not excluded : {filenm} ')
            
            # Check if rec_str exists for current subject
            if rec_str not in rec[subj_idx][file_idx].timeseries:
                print(f"{rec_str} does not exist for subject {subj_idx+1} : {filenm}. Skipping this subject/file.")
                continue  # if rec_str does not exist, skip 
            else:
                ts = rec[subj_idx][file_idx][rec_str].copy()
            
            # select the stim for the given file
            stim = rec[subj_idx][file_idx].stim.copy()
                
            # get the epochs
            # check if ts has dimenstion chromo
            if 'chromo' in ts.dims:
                ts = ts.transpose('chromo', 'channel', 'time')
            else:
                ts = ts.transpose('wavelength', 'channel', 'time')
            ts = ts.assign_coords(samples=('time', np.arange(len(ts.time))))
            ts['time'] = ts.time.pint.quantify(units.s)     
            
            #
            # block average
            #
            epochs_tmp = ts.cd.to_epochs(
                                        stim,  # stimulus dataframe
                                        set(stim[stim.trial_type.isin(cfg_blockavg['cfg_hrf']['stim_lst'])].trial_type), # select events
                                        before = cfg_blockavg['cfg_hrf']['t_pre'],  # seconds before stimulus
                                        after = cfg_blockavg['cfg_hrf']['t_post'],  # seconds after stimulus
                                    )
            
            # concatenate the different epochs from each file for each subject
            if cfg_blockavg['flag_save_each_subj']:
                epochs_tmp = epochs_tmp.assign_coords(trial_type=('epoch', [x + '-' + subj_ids_new[subj_idx] for x in epochs_tmp.trial_type.values]))

            if file_idx == 0:
                epochs_all = epochs_tmp
            else:
                epochs_all = xr.concat([epochs_all, epochs_tmp], dim='epoch')


            # DONE LOOP OVER FILES

        # Block Average
        baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
        epochs = epochs_all - baseline
        blockaverage = epochs.groupby('trial_type').mean('epoch') # mean across all epochs


        # get MSE for weighting across subjects
        

        blockaverage_weighted = blockaverage.copy()
        n_epochs = len(epochs.epoch)
        n_chs = len(epochs.channel)

        mse_t_lst = []
        mse_t_o_lst = []
        for idxt, trial_type in enumerate(blockaverage.trial_type.values): 
    
            epochs_zeromean = epochs.where(epochs.trial_type == trial_type, drop=True) - blockaverage_weighted.sel(trial_type=trial_type) # zero mean data
    
            if 'chromo' in ts.dims:
                foo_t = epochs_zeromean.stack(measurement=['channel','chromo']).sortby('chromo')
            else:
                foo_t = epochs_zeromean.stack(measurement=['channel','wavelength']).sortby('wavelength')
            foo_t = foo_t.transpose('measurement', 'reltime', 'epoch')  # !!! this does not have trial type?
            mse_t = (foo_t**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean
    
            # set bad values in mse_t to the bad value threshold
            amp = rec[subj_idx][file_idx]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths # !!! always grabbing last file
            idx_amp = np.where(amp < cfg_mse['mse_amp_thresh'])[0]
            idx_sat = np.where(chs_pruned_subjs[subj_idx][file_idx] == 0.92)[0]  # sat chans = 0.92 in chs_pruned
            idx_bad = np.where(mse_t == 0)[0]
            idx_bad1 = idx_bad[idx_bad<n_chs]
            idx_bad2 = idx_bad[idx_bad>=n_chs] - n_chs
            
            mse_t[idx_amp,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[idx_amp+n_chs,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[idx_sat,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[idx_sat+n_chs,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[idx_bad] = cfg_mse['mse_val_for_bad_data']
            
            
            channels = blockaverage_weighted.channel
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_amp).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_sat).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad1).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad2).channel.data)] = cfg_mse['blockaverage_val']
            
            blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_amp).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_sat).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad1).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad2).channel.data)] = cfg_mse['blockaverage_val']
    
            # set the minimum value of mse_t
            mse_t = xr.where(mse_t < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], mse_t)


            if 'chromo' in ts.dims:
                mse_t = mse_t.unstack('measurement').transpose('chromo','channel','reltime')  
            else:
                if 'reltime' in mse_t.dims:
                    mse_t = mse_t.unstack('measurement').transpose('wavelength','channel','reltime')
                else:
                    mse_t = mse_t.unstack('measurement').transpose('wavelength','channel')
                
            
            mse_t = mse_t.expand_dims('trial_type')
            source_coord = blockaverage_weighted['source']
            mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
            detector_coord = blockaverage_weighted['detector']
            mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))
            
            # making channels with very small variance across epochs "have less variance" 
            # mse_t = xr.where(mse_t < mse_min_thresh, mse_min_thresh, mse_t) # where true, yeild min_thres, otherwise yield orig val in mse_t
            
            mse_t = mse_t.assign_coords(trial_type = [trial_type]) # assign coords to match curr trial type
            mse_t_lst.append(mse_t) # append mse_t for curr trial type to list

            # DONE LOOP OVER TRIAL TYPES
        
        mse_t_tmp = xr.concat(mse_t_lst, dim='trial_type') # concat the 2 trial types
        mse_t = mse_t_tmp # reassign the newly appended mse_t with both trial types to mse_t 


        # gather the blockaverage across subjects
        if blockaverage_subj is None: 
            blockaverage_subj = blockaverage
            
            # add a subject dimension and coordinate
            blockaverage_subj = blockaverage_subj.expand_dims('subj')
            blockaverage_subj = blockaverage_subj.assign_coords(subj=[subj_ids_new[subj_idx]])

            blockaverage_mse_subj = mse_t.expand_dims('subj') # mse of blockaverage for each sub
            blockaverage_mse_subj = blockaverage_mse_subj.assign_coords(subj=[subj_ids_new[subj_idx]])
            
            blockaverage_mean_weighted = blockaverage_weighted / mse_t

            blockaverage_mse_inv_mean_weighted = 1 / mse_t
            
        else:   
            blockaverage_subj_tmp = blockaverage_weighted
            blockaverage_subj_tmp = blockaverage_subj_tmp.expand_dims('subj')
            blockaverage_subj_tmp = blockaverage_subj_tmp.assign_coords(subj=[subj_ids_new[subj_idx]])
            blockaverage_subj = xr.concat([blockaverage_subj, blockaverage_subj_tmp], dim='subj')

            blockaverage_mse_subj_tmp = mse_t.expand_dims('subj')
            
            blockaverage_mse_subj_tmp = blockaverage_mse_subj_tmp.assign_coords(subj=[subj_ids_new[subj_idx]])
            blockaverage_mse_subj = xr.concat([blockaverage_mse_subj, blockaverage_mse_subj_tmp], dim='subj') # !!! this does not have trial types

            blockaverage_mean_weighted = blockaverage_mean_weighted +  blockaverage_weighted / mse_t
            blockaverage_mse_inv_mean_weighted = blockaverage_mse_inv_mean_weighted + 1/mse_t 

        
        # DONE LOOP OVER SUBJECTS

    # get the unweighted average
    blockaverage_mean = blockaverage_subj.mean('subj')
    
    # get the weighted average  (old)
    #blockaverage_mean_weighted = blockaverage_mean_weighted / blockaverage_mse_inv_mean_weighted
    
    # get the mean mse within subjects
    mse_mean_within_subject = 1 / blockaverage_mse_inv_mean_weighted

    # get the mse between subjects
    mse_weighted_between_subjects_tmp = (blockaverage_subj - blockaverage_mean)**2 / blockaverage_mse_subj_tmp   # was -blockaverage_mean_weighted
    mse_weighted_between_subjects = mse_weighted_between_subjects_tmp.mean('subj')
    mse_weighted_between_subjects = mse_weighted_between_subjects * mse_mean_within_subject
 
    # get the weighted average
    mse_btw_within_sum_subj = blockaverage_mse_subj + mse_weighted_between_subjects
    denom = (1/mse_btw_within_sum_subj).sum('subj')
    
    blockaverage_mean_weighted = (blockaverage_subj / mse_btw_within_sum_subj).sum('subj')
    blockaverage_mean_weighted = blockaverage_mean_weighted / denom
    
    mse_total = 1/denom
    
    total_stderr_blockaverage = np.sqrt( mse_total )
    total_stderr_blockaverage = total_stderr_blockaverage.assign_coords(trial_type=blockaverage_mean_weighted.trial_type)

    # Plot scalp plot of mean, tstat,rsme + Plot mse hist
    for idxt, trial_type in enumerate(blockaverage_mean_weighted.trial_type.values):         
        plot_mean_stderr(rec, rec_str, trial_type, cfg_dataset, cfg_blockavg, blockaverage_mean_weighted, 
                         total_stderr_blockaverage, mse_mean_within_subject, mse_weighted_between_subjects)
        plot_mse_hist(rec, rec_str, trial_type, cfg_dataset, blockaverage_mse_subj, cfg_mse['mse_val_for_bad_data'], cfg_mse['mse_min_thresh'])  # !!! not sure if these r working correctly tbh
    
    return blockaverage_mean, blockaverage_mean_weighted, total_stderr_blockaverage, blockaverage_subj, blockaverage_mse_subj


#%% Plotting func
    
def plot_mean_stderr(rec, rec_str, trial_type, cfg_dataset, cfg_blockavg, blockaverage_mean_weighted, blockaverage_stderr_weighted, mse_mean_within_subject, mse_weighted_between_subjects):
    # scalp_plot the mean, stderr and t-stat
    #######################################################
    
    blockaverage_mean_weighted_t = blockaverage_mean_weighted.sel(trial_type=trial_type)
    blockaverage_stderr_weighted_t = blockaverage_stderr_weighted.sel(trial_type=trial_type)
    mse_mean_within_subject_t = mse_mean_within_subject.sel(trial_type=trial_type)
    mse_weighted_between_subjects_t = mse_weighted_between_subjects.sel(trial_type=trial_type)
    
    if 'chromo' in blockaverage_mean_weighted_t.dims:
        n_wav_chromo = blockaverage_mean_weighted_t.chromo.size
    else:
        n_wav_chromo = blockaverage_mean_weighted_t.wavelength.size

    for i_wav_chromo in range(n_wav_chromo):
        f,ax = p.subplots(2,2,figsize=(10,10))

        ax1 = ax[0,0]
        if 'reltime' in blockaverage_mean_weighted_t.dims:
            foo_da = blockaverage_mean_weighted_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        else:
            foo_da = blockaverage_mean_weighted_t
        #foo_da = foo_da[0,:,:]
        title_str = 'Mean_' + rec_str + '_' + trial_type
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        max_val = np.nanmax(np.abs(foo_da_tmp.values))
        plots.scalp_plot(
                rec[0][0][rec_str],
                rec[0][0].geo3d,
                foo_da_tmp,
                ax1,
                cmap='jet',
                vmin=-max_val,
                vmax=max_val,
                optode_labels=False,
                title=title_str,
                optode_size=6
            )

        ax1 = ax[0,1]
        if 'reltime' in blockaverage_mean_weighted_t.dims:
            foo_numer = blockaverage_mean_weighted_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
            foo_denom = blockaverage_stderr_weighted_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
            foo_da = foo_numer / foo_denom
        else:
            foo_da = blockaverage_mean_weighted_t / blockaverage_stderr_weighted_t
        #foo_da = foo_da[0,:,:]
        title_str = 'T-Stat_'+ rec_str + '_' + trial_type
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        max_val = np.nanmax(np.abs(foo_da_tmp.values))
        plots.scalp_plot(
                rec[0][0][rec_str],
                rec[0][0].geo3d,
                foo_da_tmp,
                ax1,
                cmap='jet',
                vmin=-max_val,
                vmax=max_val,
                optode_labels=False,
                title=title_str,
                optode_size=6
            )
        
        ax1 = ax[1,0]
        if 'reltime' in blockaverage_mean_weighted_t.dims:
            foo_da = mse_mean_within_subject_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        else:
            foo_da = mse_mean_within_subject_t
        #foo_da = foo_da[0,:,:]
        foo_da = foo_da**0.5
        title_str = 'log10(RMSE) within subjects ' + rec_str + ' ' + trial_type
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
            foo_da_tmp = foo_da_tmp.pint.dequantify()
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        foo_da_tmp = np.log10(foo_da_tmp)
        max_val = np.nanmax(foo_da_tmp.values)
        min_val = np.nanmin(foo_da_tmp.values)
        plots.scalp_plot(
                rec[0][0][rec_str],
                rec[0][0].geo3d,
                foo_da_tmp,
                ax1,
                cmap='jet',
                vmin=min_val,
                vmax=max_val,
                optode_labels=False,
                title=title_str,
                optode_size=6
            )

        ax1 = ax[1,1]
        if 'reltime' in blockaverage_mean_weighted_t.dims:
            foo_da = mse_weighted_between_subjects_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        else:
            foo_da = mse_weighted_between_subjects_t
        #foo_da = foo_da[0,:,:]
        foo_da = foo_da**0.5
        title_str = 'log10(RMSE) between subjects ' + rec_str + ' ' + trial_type 
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
            foo_da_tmp = foo_da_tmp.pint.dequantify()
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        foo_da_tmp = np.log10(foo_da_tmp)
        max_val = np.nanmax(foo_da_tmp.values)
        min_val = np.nanmin(foo_da_tmp.values)
        plots.scalp_plot(
                rec[0][0][rec_str],
                rec[0][0].geo3d,
                foo_da_tmp,
                ax1,
                cmap='jet',
                vmin=min_val,
                vmax=max_val,
                optode_labels=False,
                title=title_str,
                optode_size=6
            )
                
        # give a title to the figure and save it
        dirnm = os.path.basename(os.path.normpath(cfg_dataset['root_dir']))
        if 'chromo' in foo_da.dims:
            title_str = f"{dirnm} - {rec_str} {trial_type} {foo_da.chromo.values[i_wav_chromo]} ({cfg_blockavg['trange_hrf_stat'][0]} to {cfg_blockavg['trange_hrf_stat'][1]} s)"
        else:
            title_str = f"{dirnm} - {rec_str} {trial_type} {foo_da.wavelength.values[i_wav_chromo]:.0f}nm ({cfg_blockavg['trange_hrf_stat'][0]} to {cfg_blockavg['trange_hrf_stat'][1]} s)"
        p.suptitle(title_str)

        if 'chromo' in foo_da.dims:
            p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', f'DQR_group_weighted_avg_{rec_str}_{trial_type}_{foo_da.chromo.values[i_wav_chromo]}.png') )
        else:
            p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', f'DQR_group_weighted_avg_{rec_str}_{trial_type}_{foo_da.wavelength.values[i_wav_chromo]:.0f}nm.png') )
        p.close()


def plot_mse_hist(rec, rec_str, trial_type, cfg_dataset, blockaverage_mse_subj, mse_val_for_bad_data, mse_min_thresh):
    # plot the MSE histogram
    ########################################################

    blockaverage_mse_subj_t = blockaverage_mse_subj.sel(trial_type = trial_type)
    
    f,ax = p.subplots(2,1,figsize=(6,10))

    # plot the diagonals for all subjects
    ax1 = ax[0]
    if 'reltime' in blockaverage_mse_subj_t.dims:
        foo = blockaverage_mse_subj_t.mean('reltime')
    else:
        foo = blockaverage_mse_subj_t
    
    if 'chromo' in rec[0][0][rec_str].dims:
        foo = foo.stack(measurement=['channel','chromo']).sortby('chromo')
    else:
        foo = foo.stack(measurement=['channel','wavelength']).sortby('wavelength')

    n_subjects = foo.shape[0]  

    for i in range(n_subjects):
        ax1.semilogy(foo[i,:], linewidth=0.5,alpha=0.5)
    ax1.set_title('variance in the mean for all subjects')
    ax1.set_xlabel('channel')
    ax1.legend()

    # histogram the diagonals
    ax1 = ax[1]
    foo1 = np.concatenate([foo[i] for i in range(n_subjects)]) # FIXME: need to loop over files too   # was foo[i][0] not sure what [0] was for, maybe trial type?
    # check if mse_val_for_bad_data has units
    if 'chromo' in rec[0][0][rec_str].dims:
        foo1 = np.where(foo1 == 0, mse_val_for_bad_data.magnitude, foo1) # some bad data gets through. amp=1e-6, but it is missed by the check above. Only 2 channels in 9 subjects. Seems to be channel 271
    else:
        foo1 = np.where(foo1 == 0, mse_val_for_bad_data, foo1)
    ax1.hist(np.log10(foo1), bins=100)
    if 'chromo' in rec[0][0][rec_str].dims:
        ax1.axvline(np.log10(mse_min_thresh.magnitude), color='r', linestyle='--', label=f'cov_min_thresh={mse_min_thresh.magnitude:.2e}')
    else:
        ax1.axvline(np.log10(mse_min_thresh), color='r', linestyle='--', label=f'cov_min_thresh={mse_min_thresh:.2e}')
    ax1.legend()
    ax1.set_title(f'{rec_str} {trial_type} - histogram for all subjects of variance in the mean')
    ax1.set_xlabel('log10(cov_diag)')

    # give a title to the figure and save it
    dirnm = os.path.basename(os.path.normpath(cfg_dataset['root_dir']))
    p.suptitle(f'Data set - {dirnm}')

    p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', f'DQR_group_mse_histogram_{rec_str}_{trial_type}.png') )
    p.close()


#%% Old funcs -- not using


def block_average_od( od_filt, stim, geo3d, cfg_blockavg ):

    # get the epochs
    od_tmp = od_filt.transpose('wavelength', 'channel', 'time')
    od_tmp = od_tmp.assign_coords(samples=('time', np.arange(len(od_tmp.time))))
    od_tmp['time'] = od_tmp.time.pint.quantify(units.s)     
    od_tmp['source'] = od_filt.source
    od_tmp['detector'] = od_filt.detector

    od_epochs = od_tmp.cd.to_epochs(
                                stim,  # stimulus dataframe
                                set(stim[stim.trial_type.isin(cfg_blockavg['cfg_hrf']['stim_lst'])].trial_type), # select events
#                                set(stim.trial_type),  # select events
                                before=cfg_blockavg['trange_hrf'][0],  # seconds before stimulus
                                after=cfg_blockavg['trange_hrf'][1],  # seconds after stimulus
                            )
        
    return od_epochs


def block_average( ts, stim, geo3d, glm_basis_func_param, glm_drift_order, flag_do_GLM, ssr_rho_thresh, cfg_blockavg ):

    # Do GLM or Block Average
    # Right now the GLM doesn't handle NaN's, but pruned channels have NaN's.. curious.
    if not flag_do_GLM:
        # do block average
        pred_hrf = ts

    else:
        # do the GLM

        # get the short separation channels
        ts_long, ts_short = cedalion.nirs.split_long_short_channels(
            ts, geo3d, distance_threshold=ssr_rho_thresh
        )

        # if stim has a column named 'amplitude' then rename it to 'value'
        # I noticed that some events files have 'amplitude' and some have 'value' and some have neither
        if 'amplitude' in stim.columns:
            stim = stim.rename(columns={"amplitude": "value"})
        elif 'amplitude' not in stim.columns and 'value' not in stim.columns:
            stim['value'] = 1

        # make the design matrix
        # ’closest’: Use the closest short channel 
        # ‘max_corr’: Use the short channel with the highest correlation 
        # ‘mean’: Use the average of all short channels.
        dm, channel_wise_regressors = glm.make_design_matrix(
            ts,
            ts_short,
            stim[stim.trial_type.isin(cfg_blockavg['stim_lst_hrf'])],
            geo3d,
            basis_function = glm.GaussianKernels(cfg_blockavg['trange_hrf'][0], cfg_blockavg['trange_hrf'][1], t_delta=glm_basis_func_param, t_std=glm_basis_func_param), 
            drift_order = glm_drift_order, 
            short_channel_method='mean'
        )

        # fit the GLM model
        betas = glm.fit(ts, dm, channel_wise_regressors, noise_model="ols")

        # prediction of all HRF regressors, i.e. all regressors that start with 'HRF '
        pred_hrf = glm.predict(
            ts,
            betas.sel(regressor=betas.regressor.str.startswith("HRF ")),
            dm,
            channel_wise_regressors
        )

    # get the HRF prediction 
    # This is a simple way to get HRF stats by getting epochs and then simple block average, 
    # but it will not work for event related designs with overlapping epochs.
    # In the future we will have to update glm.predict to provide HRF stats.
    pred_hrf = pred_hrf.transpose('chromo', 'channel', 'time')
    pred_hrf = pred_hrf.assign_coords(samples=('time', np.arange(len(pred_hrf.time))))
    pred_hrf['time'] = pred_hrf.time.pint.quantify(units.s)     
    pred_hrf['source'] = ts.source
    pred_hrf['detector'] = ts.detector

    epochs_tmp = pred_hrf.cd.to_epochs(
                                stim,  # stimulus dataframe
                                set(stim[stim.trial_type.isin(cfg_blockavg['stim_lst_hrf'])].trial_type), # select events
#                                set(stim.trial_type),  # select events
                                before=cfg_blockavg['trange_hrf'][0],  # seconds before stimulus
                                after=cfg_blockavg['trange_hrf'][1],  # seconds after stimulus
                            )
        
    return epochs_tmp




def y_mean_to_conc( y_mean_tmp, geo3d, wavelength, source, cov_mean_weighted, cfg_blockavg ):

    n_chs = y_mean_tmp.shape[0] // 2

    foo = y_mean_tmp.unstack()
    foo = foo.transpose('channel', 'wavelength', 'reltime')
    foo = foo.rename({'reltime':'time'})
    foo['time'].attrs['units'] = 's'

    dpf = xr.DataArray(
        [1, 1],
        dims="wavelength",
        coords={"wavelength": wavelength},
    )
    foo_conc = cedalion.nirs.od2conc(foo, geo3d, dpf, spectrum="prahl")
    foo_conc = foo_conc.rename({'time':'reltime'})
    foo_conc = foo_conc.assign_coords(source=source)
    foo_conc = foo_conc.expand_dims('trial_type')
    foo_conc = foo_conc.assign_coords(trial_type=cfg_blockavg['stim_lst_hrf'])

    # baseline subtract
    foo_conc = foo_conc - foo_conc.sel(reltime=slice(-cfg_blockavg['trange_hrf'][0].magnitude, 0)).mean('reltime')

    # set to NaN the noisy channels for viewing purposes
    cov_mean_weighted_diag = cov_mean_weighted.diagonal()
    idx_cov = np.where(cov_mean_weighted_diag > 1e-3)[0]
    idx_cov1 = idx_cov[idx_cov<n_chs]
    idx_cov2 = idx_cov[idx_cov>=n_chs] - n_chs
    idx_cov = np.union1d(idx_cov1, idx_cov2)

    foo_conc_tmp = foo_conc.copy()
    foo_conc_tmp[:,:,idx_cov,:] = np.nan * np.ones(foo_conc[:,:,idx_cov,:].shape) * units.micromolar

    return foo_conc, foo_conc_tmp




def GLM_extract_estimated_hrf( ts, geo3d, stim, glm_basis_func_param, betas, cfg_blockavg ):

    pred_hrf = ts

    epochs_tmp = pred_hrf.cd.to_epochs(
                                stim,  # stimulus dataframe
                                set(stim.trial_type),  # select events
                                before=cfg_blockavg['trange_hrf'][0],  # seconds before stimulus
                                after=cfg_blockavg['trange_hrf'][1],  # seconds after stimulus
                            )

    blockaverage = epochs_tmp.groupby("trial_type").mean("epoch")

    # Get GLM prediction for a single trial
    conc_hrf = blockaverage.copy() # I take this from blockaverage because I want timing from 0 as done in to_epochs()
    conc_hrf = conc_hrf.rename({"reltime": "time"})
    conc_hrf = conc_hrf.sel(trial_type=conc_hrf.trial_type[0])
    conc_hrf = conc_hrf.drop("trial_type")
    conc_hrf.values = np.zeros_like(conc_hrf.values)
    conc_hrf = conc_hrf.assign_coords(samples=("time", np.arange(len(conc_hrf.time))))
    conc_hrf['time'] = conc_hrf.time.pint.quantify(units.s)

    stim_fake = stim.copy()
    stim_fake = stim_fake.iloc[0:1]
    stim_fake.onset = 0
    stim_fake.trial_type = 'mnt'

    dm_fake, _ = glm.make_design_matrix(
        conc_hrf,
        None,
        stim_fake,
        geo3d,
        basis_function = glm.GaussianKernels(cfg_blockavg['trange_hrf'][0], cfg_blockavg['trange_hrf'][1], t_delta=glm_basis_func_param, t_std=glm_basis_func_param), 
        drift_order = None,
        short_channel_method=None,
    )

    pred_hrf = glm.predict(
        conc_hrf,
        betas.sel(regressor=betas.regressor.str.startswith("HRF ")),
        dm_fake,
        None
    )

    pred_hrf = pred_hrf.transpose('chromo','channel','time')
    pred_hrf = pred_hrf.rename({"time": "reltime"})
    pred_hrf = pred_hrf.expand_dims('trial_type')
    pred_hrf = pred_hrf.assign_coords(trial_type=('trial_type',['mnt']))
    pred_hrf = pred_hrf.assign_coords(source=('channel',blockaverage['source'].data))
    pred_hrf = pred_hrf.assign_coords(detector=('channel',blockaverage['detector'].data))

    return pred_hrf


