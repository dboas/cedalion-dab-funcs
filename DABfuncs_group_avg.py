# group_avg_GLM()
# group_avg_block()


import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality

import cedalion.models.glm as glm
import cedalion.plots as plots

from cedalion import units
import numpy as np
import xarray as xr

import matplotlib.pyplot as p

import pdb





def get_group_avg_for_diff_conds(rec, rec_str_lst, flag_save_weighted, chs_pruned_subjs, cfg_dataset, cfg_blockavg ):
    '''
    This function takes in a list of strings (trial_types/ conditions)
    and loops through each rec_str and blockaverages each.
    
    Inputs: 
        rec : list of recording objects for each subject
        rec_str_lst : list of strings identifying data arrays in rec you want to average
        flag_save_weighted : list (same size as rec_str_lst) that indicates 
            whether to save the weighted or unweighted for each data array in rec_str_lst
    '''
    
    if not rec_str_lst:
        print('No list of recording objects given. Cannot compute block average.')
        return None, None, None, None
    if any('od' in rec_str for rec_str in rec_str_lst) & any('conc' in rec_str for rec_str in rec_str_lst):
        print('rec_str_lst includes both conc and od. Cannot compute blockaverage.')
    

    for idx,rec_str in enumerate(rec_str_lst):
        
        # Change trial type name to match rec_str (for concatination purposes)
        if 'conc' in rec_str:   
            new_name = rec_str[4:]
        else: 
            new_name = rec_str[2:]
            
        # Run block average func
        y_mean, y_mean_weighted, y_stderr_weighted, y_subj, y_mse_subj = run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg)
        
        # Save weighted or unweighted
        if flag_save_weighted[idx]:
            blockaverage_mean_tmp = y_mean_weighted.assign_coords(trial_type=[x + new_name for x in y_mean_weighted.trial_type.values])
        else:
            blockaverage_mean_tmp = y_mean.assign_coords(trial_type=[x + new_name for x in y_mean.trial_type.values])
            
        # save other stuff
        blockaverage_stderr_tmp = y_stderr_weighted.assign_coords(trial_type=[x + new_name for x in y_stderr_weighted.trial_type.values]) # also save stderr weighted
        blockaverage_subj_tmp = y_subj.assign_coords(trial_type=[x + new_name for x in y_subj.trial_type.values])
        blockaverage_mse_subj_tmp = y_mse_subj.assign_coords(trial_type=[x + new_name for x in y_mse_subj.trial_type.values])
        
        
        # If first loop, then initiate blockaverage var
        if idx == 0:
            blockaverage_mean = blockaverage_mean_tmp
            blockaverage_subj = blockaverage_subj_tmp
            blockaverage_mse_subj = blockaverage_mse_subj_tmp
            blockaverage_stderr = blockaverage_stderr_tmp
                
        else:
            blockaverage_mean = xr.concat([blockaverage_mean, blockaverage_mean_tmp],dim='trial_type')
            blockaverage_subj = xr.concat([blockaverage_subj, blockaverage_subj_tmp],dim='trial_type')
            blockaverage_mse_subj = xr.concat([blockaverage_mse_subj, blockaverage_mse_subj_tmp],dim='trial_type')
            blockaverage_stderr = xr.concat([blockaverage_stderr, blockaverage_stderr_tmp], dim='trial_type')
            
            
    return blockaverage_mean, blockaverage_stderr, blockaverage_subj, blockaverage_mse_subj
        




def run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg ):
    
    # choose correct mse values 
    if 'chromo' in rec[0][0][rec_str]:
        cfg_mse = cfg_blockavg['cfg_mse_conc']
    else:
        cfg_mse = cfg_blockavg['cfg_mse_od']
        
    cfg_GLM = cfg_blockavg['cfg_GLM']
    
    mse_val_for_bad_data = cfg_mse['mse_val_for_bad_data']
    mse_amp_thresh = cfg_mse['mse_amp_thresh']
    mse_min_thresh = cfg_mse['mse_min_thresh']
    blockaverage_val = cfg_mse['blockaverage_val']

    subj_ids = cfg_dataset['subj_ids']
    n_subjects = len(rec)
    n_files_per_subject = len(rec[0])

    print(f"Running group block average for trial_type = '{rec_str}'")

    # loop over subjects and files
    blockaverage_subj = None
    for subj_idx in range( n_subjects ):
        for file_idx in range( n_files_per_subject ):

            filenm = cfg_dataset['filenm_lst'][subj_idx][file_idx]
            print( f"   Running {subj_idx+1} of {n_subjects} subjects : {filenm}" )

            # do the block average on the data in rec[subj_idx][file_idx][rec_str]
            # rec_str could be 'conc_tddr', 'conc_tddr_ica' or 'od' or even conc_splineSG
            
            # Check if rec_str exists for current subject
            if rec_str not in rec[subj_idx][file_idx].timeseries:
                continue  # if rec_str does not exist, skip 
            else:
                conc_filt = rec[subj_idx][file_idx][rec_str].copy()
            
            # select the stim for the given file
            stim = rec[subj_idx][file_idx].stim.copy()
            
            # GLM filtering step
            if 'chromo' in conc_filt.dims and cfg_blockavg['flag_do_GLM_filter']:
                rec_glm = GLM(rec[subj_idx][file_idx], rec_str, cfg_blockavg['trange_hrf'][0], cfg_blockavg['trange_hrf'][1], cfg_GLM)
                conc_filt = rec_glm[rec_str]
                
            # else no GLM filtering:
            else:
                
                # get the epochs
                # check if conc_filt has dimenstion chromo
                if 'chromo' in conc_filt.dims:
                    conc_filt = conc_filt.transpose('chromo', 'channel', 'time')
                else:
                    conc_filt = conc_filt.transpose('wavelength', 'channel', 'time')
                conc_filt = conc_filt.assign_coords(samples=('time', np.arange(len(conc_filt.time))))
                conc_filt['time'] = conc_filt.time.pint.quantify(units.s)     

            #
            # block average
            #
            conc_epochs_tmp = conc_filt.cd.to_epochs(
                                        stim,  # stimulus dataframe
                                        set(stim[stim.trial_type.isin(cfg_blockavg['stim_lst_hrf'])].trial_type), # select events
                                        before=cfg_blockavg['trange_hrf'][0],  # seconds before stimulus
                                        after=cfg_blockavg['trange_hrf'][1],  # seconds after stimulus
                                    )
            
            # concatenate the different epochs from each file for each subject
            if cfg_blockavg['flag_save_each_subj']:
                conc_epochs_tmp = conc_epochs_tmp.assign_coords(trial_type=('epoch', [x + '-' + subj_ids[subj_idx] for x in conc_epochs_tmp.trial_type.values]))

            if file_idx == 0:
                conc_epochs_all = conc_epochs_tmp
            else:
                conc_epochs_all = xr.concat([conc_epochs_all, conc_epochs_tmp], dim='epoch')


            # DONE LOOP OVER FILES

        # Block Average
        baseline_conc = conc_epochs_all.sel(reltime=(conc_epochs_all.reltime < 0)).mean('reltime')
        conc_epochs = conc_epochs_all - baseline_conc
        blockaverage = conc_epochs.groupby('trial_type').mean('epoch')


        # get MSE for weighting across subjects
        

        blockaverage_weighted = blockaverage.copy()
        n_epochs = conc_epochs.shape[0]
        n_chs = conc_epochs.shape[2]
        
        mse_t_lst = []
        mse_t_o_lst = []
        for idxt, trial_type in enumerate(blockaverage.trial_type.values): # make sure if incl walk times that its not looping thru that
            #foo = conc_epochs - blockaverage_weighted.sel(trial_type=trial_type) # FIXME: assuming one trial_type
    
            foo = conc_epochs.where(conc_epochs.trial_type == trial_type, drop=True) - blockaverage_weighted.sel(trial_type=trial_type) 
    
            if 'chromo' in conc_filt.dims:
                foo_t = foo.stack(measurement=['channel','chromo']).sortby('chromo')
            else:
                foo_t = foo.stack(measurement=['channel','wavelength']).sortby('wavelength')
            foo_t = foo_t.transpose('measurement', 'reltime', 'epoch')
            mse_t = (foo_t**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean
    
            # list of channel elements in mse corresponding to channels with amp < mse_amp_thresh
            amp = rec[subj_idx][file_idx]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
            idx_amp = np.where(amp < mse_amp_thresh)[0]
            mse_t[idx_amp,:] = mse_val_for_bad_data
            mse_t[idx_amp + n_chs,:] = mse_val_for_bad_data

            # blockaverage_weighted[0,0,idx_amp,:] = blockaverage_val 
            # blockaverage_weighted[0,1,idx_amp,:] = blockaverage_val 
            
            # Update bad data with predetermined value
            bad_vals = blockaverage_weighted.isel(channel=idx_amp)
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_vals.channel.data)] = blockaverage_val
            
            #%%
            # look at saturated channels
            idx_sat = np.where(chs_pruned_subjs[subj_idx][file_idx] == 0.0)[0] 
            mse_t[idx_sat,:] = mse_val_for_bad_data
            mse_t[idx_sat + n_chs,:] = mse_val_for_bad_data
            # blockaverage_weighted[0,0,idx_sat,:] = blockaverage_val 
            # blockaverage_weighted[0,1,idx_sat,:] = blockaverage_val 
            
            bad_vals = blockaverage_weighted.isel(channel=idx_sat)
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel= bad_vals.channel.data)] = blockaverage_val
            
    
            
            #%%
            # where mse_t is 0, set it to mse_val_for_bad_data
            # I am trying to handle those rare cases where the mse is 0 for some subjects and then it corrupts 1/mse
            # FIXME: why does this happen sometimes?
            idx_bad = np.where(mse_t == 0)[0]
            idx_bad1 = idx_bad[idx_bad<n_chs]
            idx_bad2 = idx_bad[idx_bad>=n_chs] - n_chs
            mse_t[idx_bad] = mse_val_for_bad_data
            #blockaverage_weighted[0,0,idx_bad1,:] = blockaverage_val
            #blockaverage_weighted[0,1,idx_bad2,:] = blockaverage_val
            
            bad_vals = blockaverage_weighted.isel(channel=idx_bad1)
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_vals.channel.data)] = blockaverage_val
            bad_vals = blockaverage_weighted.isel(channel=idx_bad2)
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_vals.channel.data)] = blockaverage_val
            
            

            # FIXME: do I set blockaverage_weighted too?
    
            # set the minimum value of mse_t
            if 'chromo' in conc_filt.dims:
                mse_t = mse_t.unstack('measurement').transpose('chromo','channel','reltime')
            else:
                mse_t = mse_t.unstack('measurement').transpose('wavelength','channel','reltime')
            mse_t = mse_t.expand_dims('trial_type')
            source_coord = blockaverage['source']
            mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
            detector_coord = blockaverage['detector']
            mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))
            
            mse_t_o = mse_t.copy()
            mse_t = xr.where(mse_t < mse_min_thresh, mse_min_thresh, mse_t) # where true, yeild min_thres, otherwise yield orig val in mse_t
            
            mse_t = mse_t.assign_coords(trial_type = [trial_type]) # assign coords to match curr trial type
            mse_t_o = mse_t_o.assign_coords(trial_type = [trial_type]) 
            
            mse_t_lst.append(mse_t) # append mse_t for curr trial type to list
            mse_t_o_lst.append(mse_t_o)
            
            # DONE LOOP OVER TRIAL TYPES
            
        mse_t_tmp = xr.concat(mse_t_lst, dim='trial_type') # concat the 2 trial types
        mse_t = mse_t_tmp # reassign the newly appended mse_t with both trial types to mse_t 
        mse_t_o_tmp = xr.concat(mse_t_o_lst, dim='trial_type') 
        mse_t_o = mse_t_o_tmp 


        
        # gather the blockaverage across subjects
        if blockaverage_subj is None and subj_ids[subj_idx] not in cfg_dataset['subj_id_exclude']:
            blockaverage_subj = blockaverage
            # add a subject dimension and coordinate
            blockaverage_subj = blockaverage_subj.expand_dims('subj')
            blockaverage_subj = blockaverage_subj.assign_coords(subj=[subj_ids[subj_idx]])

            blockaverage_mse_subj = mse_t_o.expand_dims('subj') 
            blockaverage_mse_subj = blockaverage_mse_subj.assign_coords(subj=[subj_ids[subj_idx]])
            
            blockaverage_mean_weighted = blockaverage_weighted / mse_t

            
            blockaverage_mse_inv_mean_weighted = 1 / mse_t

        elif subj_ids[subj_idx] not in cfg_dataset['subj_id_exclude']:
            blockaverage_subj_tmp = blockaverage
            blockaverage_subj_tmp = blockaverage_subj_tmp.expand_dims('subj')
            blockaverage_subj_tmp = blockaverage_subj_tmp.assign_coords(subj=[subj_ids[subj_idx]])
            blockaverage_subj = xr.concat([blockaverage_subj, blockaverage_subj_tmp], dim='subj')

            blockaverage_mse_subj_tmp = mse_t_o.expand_dims('subj')
            
            blockaverage_mse_subj_tmp = blockaverage_mse_subj_tmp.assign_coords(subj=[subj_ids[subj_idx]])
            blockaverage_mse_subj = xr.concat([blockaverage_mse_subj, blockaverage_mse_subj_tmp], dim='subj') # !!! this does not have trial types


            blockaverage_mean_weighted += blockaverage_weighted / mse_t

            blockaverage_mse_inv_mean_weighted = blockaverage_mse_inv_mean_weighted + 1 / mse_t # !!! is this supposed to be  += ???

        else:
            print(f"   Subject {subj_ids[subj_idx]} excluded from group average")
        
        # DONE LOOP OVER SUBJECTS

        #pdb.set_trace()
        
    # get the average
    blockaverage_mean = blockaverage_subj.mean('subj')
    
    # get the weighted average
    blockaverage_mean_weighted = blockaverage_mean_weighted / blockaverage_mse_inv_mean_weighted

    mse_mean_within_subject = 1 / blockaverage_mse_inv_mean_weighted
    
    blockaverage_mse_subj_tmp = blockaverage_mse_subj.copy()
    blockaverage_mse_subj_tmp = xr.where(blockaverage_mse_subj_tmp < mse_min_thresh, mse_min_thresh, blockaverage_mse_subj_tmp)

    mse_weighted_between_subjects_tmp = (blockaverage_subj - blockaverage_mean_weighted)**2 / blockaverage_mse_subj_tmp
    mse_weighted_between_subjects = mse_weighted_between_subjects_tmp.mean('subj')
    mse_weighted_between_subjects = mse_weighted_between_subjects * mse_mean_within_subject
    # FIXME: is it an issue that mse_mean_within_subject comes from mse_t and blockaverage_mse_subj_tmp comes from mse_t_o?
 
    # blockaverage_stderr_weighted = np.sqrt(1 / blockaverage_mse_inv_mean_weighted)
    blockaverage_stderr_weighted = np.sqrt( mse_mean_within_subject + mse_weighted_between_subjects )
    blockaverage_stderr_weighted = blockaverage_stderr_weighted.assign_coords(trial_type=blockaverage_mean_weighted.trial_type)


    # scalp_plot the mean, stderr and t-stat
    ########################################################
    # FIXME: I will want to loop over trial types and save as separate files
    if 'chromo' in blockaverage_mean_weighted.dims:
        n_wav_chromo = blockaverage_mean_weighted.chromo.size
    else:
        n_wav_chromo = blockaverage_mean_weighted.wavelength.size

    for i_wav_chromo in range(n_wav_chromo):
        f,ax = p.subplots(2,2,figsize=(10,10))

        ax1 = ax[0,0]
        foo_da = blockaverage_mean_weighted.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        foo_da = foo_da[0,:,:]
        title_str = 'Mean'
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
        foo_numer = blockaverage_mean_weighted.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        foo_denom = blockaverage_stderr_weighted.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        foo_da = foo_numer / foo_denom
        foo_da = foo_da[0,:,:]
        title_str = 'T-Stat'
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
        foo_da = mse_mean_within_subject.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        foo_da = foo_da[0,:,:]
        foo_da = foo_da**0.5
        title_str = 'log10(RMSE) within subjects'
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
        foo_da = mse_weighted_between_subjects.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        foo_da = foo_da[0,:,:]
        foo_da = foo_da**0.5
        title_str = 'log10(RMSE) between subjects'
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
            title_str = f"{dirnm} - {rec_str} {foo_da.chromo.values[i_wav_chromo]} ({cfg_blockavg['trange_hrf_stat'][0]} to {cfg_blockavg['trange_hrf_stat'][1]} s)"
        else:
            title_str = f"{dirnm} - {rec_str} {foo_da.wavelength.values[i_wav_chromo]:.0f}nm ({cfg_blockavg['trange_hrf_stat'][0]} to {cfg_blockavg['trange_hrf_stat'][1]} s)"
        p.suptitle(title_str)

        if 'chromo' in foo_da.dims:
            p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', f'DQR_group_weighted_avg_{rec_str}_{foo_da.chromo.values[i_wav_chromo]}.png') )
        else:
            p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', f'DQR_group_weighted_avg_{rec_str}_{foo_da.wavelength.values[i_wav_chromo]:.0f}nm.png') )
        p.close()



    # plot the MSE histogram
    ########################################################
    n_subjects = len(rec)-len(cfg_dataset['subj_id_exclude']) # reassign n_subjects to exclude subs 
    f,ax = p.subplots(2,1,figsize=(6,10))

    # plot the diagonals for all subjects
    ax1 = ax[0]
    foo = blockaverage_mse_subj.mean('reltime')
    if 'chromo' in conc_filt.dims:
        foo = foo.stack(measurement=['channel','chromo']).sortby('chromo')
    else:
        foo = foo.stack(measurement=['channel','wavelength']).sortby('wavelength')
    for i in range(n_subjects):
        ax1.semilogy(foo[i,0,:], linewidth=0.5,alpha=0.5)
    ax1.set_title('variance in the mean for all subjects')
    ax1.set_xlabel('channel')
    ax1.legend()

    # histogram the diagonals
    ax1 = ax[1]
    foo1 = np.concatenate([foo[i][0] for i in range(n_subjects)]) # FIXME: need to loop over files too
    # check if mse_val_for_bad_data has units
    if 'chromo' in conc_filt.dims:
        foo1 = np.where(foo1 == 0, mse_val_for_bad_data.magnitude, foo1) # some bad data gets through. amp=1e-6, but it is missed by the check above. Only 2 channels in 9 subjects. Seems to be channel 271
    else:
        foo1 = np.where(foo1 == 0, mse_val_for_bad_data, foo1)
    ax1.hist(np.log10(foo1), bins=100)
    if 'chromo' in conc_filt.dims:
        ax1.axvline(np.log10(mse_min_thresh.magnitude), color='r', linestyle='--', label=f'cov_min_thresh={mse_min_thresh.magnitude:.2e}')
    else:
        ax1.axvline(np.log10(mse_min_thresh), color='r', linestyle='--', label=f'cov_min_thresh={mse_min_thresh:.2e}')
    ax1.legend()
    ax1.set_title('histogram for all subjects of variance in the mean')
    ax1.set_xlabel('log10(cov_diag)')

    # give a title to the figure and save it
    dirnm = os.path.basename(os.path.normpath(cfg_dataset['root_dir']))
    p.suptitle(f'Data set - {dirnm}')

    p.savefig( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', f'DQR_group_mse_histogram_{rec_str}.png') )
    p.close()


    return blockaverage_mean, blockaverage_mean_weighted, blockaverage_stderr_weighted, blockaverage_subj, blockaverage_mse_subj




def GLM(rec, rec_str, t_pre, t_post, cfg_GLM):
    
    #### build design matrix
    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        rec[rec_str], rec.geo3d, distance_threshold= cfg_GLM['distance_threshold']
    )
    
    # build regressors
    dm, channel_wise_regressors = glm.make_design_matrix(
        rec[rec_str],
        #ts_long,
        ts_short,
        rec.stim,
        rec.geo3d,
        basis_function = glm.GaussianKernels(t_pre, t_post, cfg_GLM['t_delta'], cfg_GLM['t_std']),
        drift_order = cfg_GLM['drift_order'],
        short_channel_method = cfg_GLM['short_channel_method']
    )
    
    #### fit the model 
    betas = glm.fit(rec[rec_str], dm, channel_wise_regressors, noise_model=cfg_GLM['noise_model'])
    
    pred_all = glm.predict(rec[rec_str], betas, dm, channel_wise_regressors)
    pred_all = pred_all.pint.quantify('micromolar')
    
    residual = rec[rec_str] - pred_all
    
    # prediction of all HRF regressors, i.e. all regressors that start with 'HRF '
    pred_hrf = glm.predict(
                            rec[rec_str],
                            betas.sel(regressor=betas.regressor.str.startswith("HRF ")),
                            dm,
                            channel_wise_regressors
                        )
    
    pred_hrf = pred_hrf.pint.quantify('micromolar')

    rec['pred_hrf'] = pred_hrf + residual 
    
    #### get average HRF prediction 
    rec['pred_hrf'] = rec['pred_hrf'].transpose('chromo', 'channel', 'time')
    rec['pred_hrf'] = rec['pred_hrf'].assign_coords(samples=("time", np.arange(len(rec['pred_hrf'].time))))
    rec['pred_hrf']['time'] = rec['pred_hrf'].time.pint.quantify(units.s) 
    

    return rec

    



#%% Old funcs


def block_average_od( od_filt, stim, geo3d, cfg_blockavg ):

    # get the epochs
    od_tmp = od_filt.transpose('wavelength', 'channel', 'time')
    od_tmp = od_tmp.assign_coords(samples=('time', np.arange(len(od_tmp.time))))
    od_tmp['time'] = od_tmp.time.pint.quantify(units.s)     
    od_tmp['source'] = od_filt.source
    od_tmp['detector'] = od_filt.detector

    od_epochs = od_tmp.cd.to_epochs(
                                stim,  # stimulus dataframe
                                set(stim[stim.trial_type.isin(cfg_blockavg['stim_lst_hrf'])].trial_type), # select events
#                                set(stim.trial_type),  # select events
                                before=cfg_blockavg['trange_hrf'][0],  # seconds before stimulus
                                after=cfg_blockavg['trange_hrf'][1],  # seconds after stimulus
                            )
        
    return od_epochs


def block_average( conc_filt, stim, geo3d, glm_basis_func_param, glm_drift_order, flag_do_GLM, ssr_rho_thresh, cfg_blockavg ):

    # Do GLM or Block Average
    # Right now the GLM doesn't handle NaN's, but pruned channels have NaN's.. curious.
    if not flag_do_GLM:
        # do block average
        pred_hrf = conc_filt

    else:
        # do the GLM

        # get the short separation channels
        ts_long, ts_short = cedalion.nirs.split_long_short_channels(
            conc_filt, geo3d, distance_threshold=ssr_rho_thresh
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
            conc_filt,
            ts_short,
            stim[stim.trial_type.isin(cfg_blockavg['stim_lst_hrf'])],
            geo3d,
            basis_function = glm.GaussianKernels(cfg_blockavg['trange_hrf'][0], cfg_blockavg['trange_hrf'][1], t_delta=glm_basis_func_param, t_std=glm_basis_func_param), 
            drift_order = glm_drift_order, 
            short_channel_method='mean'
        )

        # fit the GLM model
        betas = glm.fit(conc_filt, dm, channel_wise_regressors, noise_model="ols")

        # prediction of all HRF regressors, i.e. all regressors that start with 'HRF '
        pred_hrf = glm.predict(
            conc_filt,
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
    pred_hrf['source'] = conc_filt.source
    pred_hrf['detector'] = conc_filt.detector

    conc_epochs_tmp = pred_hrf.cd.to_epochs(
                                stim,  # stimulus dataframe
                                set(stim[stim.trial_type.isin(cfg_blockavg['stim_lst_hrf'])].trial_type), # select events
#                                set(stim.trial_type),  # select events
                                before=cfg_blockavg['trange_hrf'][0],  # seconds before stimulus
                                after=cfg_blockavg['trange_hrf'][1],  # seconds after stimulus
                            )
        
    return conc_epochs_tmp




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




def GLM_extract_estimated_hrf( conc_filt, geo3d, stim, glm_basis_func_param, betas, cfg_blockavg ):

    pred_hrf = conc_filt

    conc_epochs_tmp = pred_hrf.cd.to_epochs(
                                stim,  # stimulus dataframe
                                set(stim.trial_type),  # select events
                                before=cfg_blockavg['trange_hrf'][0],  # seconds before stimulus
                                after=cfg_blockavg['trange_hrf'][1],  # seconds after stimulus
                            )

    blockaverage = conc_epochs_tmp.groupby("trial_type").mean("epoch")

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


