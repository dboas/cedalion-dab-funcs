import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.xrutils as xrutils
#import cedalion.datasets as datasets
import xarray as xr
import matplotlib.pyplot as p
import cedalion.vis as plots
from cedalion import units
import numpy as np

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

import module_image_recon as img_recon 
import module_spatial_basis_funs_ced as sbf 

from cedalion.sigproc.physio import global_component_subtract
#from cedalion.sigproc.quality import measurement_variance

from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import TukeyBiweight


def getCorrMatrix( conc_hbo = None, flag_GMS = "all" ):

        # global mean subtract conc_hbo
    if flag_GMS == "all": # do GMS with all channels or SS channels
        # select all channels
        gms = conc_hbo.mean(dim='channel')

        # fit GMS to the channel data
        numerator = (conc_hbo * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator

        conc_hbo_gms = conc_hbo - scl*gms

    elif flag_GMS == "ss": # do GMS with SS channels
        # select SS channels only
        gms = conc_hbo.sel(channel=conc_hbo.channel.str.contains("D31|D32|D33|D34|D35|D36|D37|D38")).mean(dim='channel')
        # FIX ME. Quality.sd_dist should be used to determine which channels are SS and LS

        # fit GMS to the channel data
        numerator = (conc_hbo * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator

        conc_hbo_gms = conc_hbo - scl*gms

    elif flag_GMS == "ss_ls": # do GMS with SS channels and then LS channels
        # SS channels
        gms = conc_hbo.sel(channel=conc_hbo.channel.str.contains("D31|D32|D33|D34|D35|D36|D37|D38")).mean(dim='channel')

        numerator = (conc_hbo * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator
        conc_hbo_gms = conc_hbo - scl*gms

        # LS channels
        gms = conc_hbo_gms.sel(channel=~conc_hbo.channel.str.contains("D31|D32|D33|D34|D35|D36|D37|D38")).mean(dim='channel')

        numerator = (conc_hbo_gms * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator
        conc_hbo_gms = conc_hbo_gms - scl*gms

    else:
        conc_hbo_gms = conc_hbo
        
    # calculate the correlation matrix of conc_hbo
    conc_hbo_array = conc_hbo_gms.values

    if 1: # correlation matrix
        corr_matrix = np.corrcoef(conc_hbo_array, rowvar=True)
        corr_max = 1

    else: # partial correlation matrix
        regularization = 1e2
        # Step 1: Calculate the covariance matrix
        cov_matrix = np.cov(conc_hbo_array, rowvar=True)    
        # Step 2: Regularize by adding a small value to the diagonal
        cov_matrix += np.eye(cov_matrix.shape[0]) * regularization    
        # Step 3: Invert the covariance matrix to get the precision matrix
        precision_matrix = np.linalg.inv(cov_matrix)
        # Step 4: Convert the precision matrix to the partial correlation matrix
        partial_corr = precision_matrix / np.sqrt(np.outer(np.diag(precision_matrix), np.diag(precision_matrix)))
        # Step 5: Ensure the diagonal is all zeros
        np.fill_diagonal(partial_corr, 0)

        corr_matrix = partial_corr
#        corr_max = np.max(np.abs(corr_matrix))

        

    # Ensure corr_matrix_xr has two dimensions, both of which are channel
    corr_matrix_xr = xr.DataArray(
        corr_matrix,
    #    cov_matrix,
        dims=["channel", "channel"],
        coords={"channel": conc_hbo.channel.values, "channel": conc_hbo.channel.values}
    )

    return corr_matrix_xr, conc_hbo_gms


def calc_dFC( conc_hbo_array_clusters = None, t = None, window_size_s = 20 ):

    fs = 1 / np.mean(np.diff(t))
    window_size = window_size_s * fs
    window_size = int(window_size.round())

    max_cluster_label = conc_hbo_array_clusters.shape[0]

    # initialize the correlation matrix
    corr_time_clusters = np.zeros([max_cluster_label, max_cluster_label, conc_hbo_array_clusters.shape[1] - window_size])

    # get the tcorr vector
    # tcorr = np.zeros(corr_time_clusters.shape[2])
    # for i in range(0, corr_time_clusters.shape[2], 1):
    #     tcorr[i] = t[i + window_size // 2]
    # Assuming tcorr and t are NumPy arrays and window_size is defined
    tcorr = t[window_size // 2 : corr_time_clusters.shape[2] + window_size // 2]

    # For loop over c1 
    for c1 in range(0, max_cluster_label):
        for c2 in range(c1, max_cluster_label):

            if c1 == c2:
                corr_time_clusters[c1,c2,:] = conc_hbo_array_clusters[c1,window_size//2:-window_size//2]
            else:
                # calculate the correlation between the two clusters over a sliding window
                y1 = conc_hbo_array_clusters[c1,:]
                y2 = conc_hbo_array_clusters[c2,:]

                # calculate the correlation between the two channels over a sliding window
                for i in range(0, y1.shape[0] - window_size, 1):
                    corr_time_clusters[c1,c2,i] = np.corrcoef(y1[i:i + window_size], y2[i:i + window_size])[0, 1]

    return corr_time_clusters, tcorr


def block_average_clusters( corr_time_clusters, tcorr, stim, events_str, t_before = 2, t_after = 20 ):
    
    max_cluster_label = corr_time_clusters.shape[0]

    if len(events_str) > 2:
        print("Only two events are allowed, but you can fix the code")
        return
    
    duration = np.zeros(2)
    stim_trial_type = stim.loc[stim.trial_type == events_str[0]]
    duration[0] = stim_trial_type.duration.mean()
    stim_trial_type = stim.loc[stim.trial_type == events_str[1]]
    duration[1] = stim_trial_type.duration.mean()

    if duration[0] != duration[1]:
        print("The duration of the two events must be the same, but you can fix the code")
        return

    # put corr_time_clusters into an xarray
    corr_time_clusters_xr = xr.DataArray(
        corr_time_clusters,
        dims=["cluster1", "cluster2", "time"],
        coords={"cluster1": np.arange(1, max_cluster_label+1), "cluster2": np.arange(1, max_cluster_label+1), "time": tcorr}
    )
    corr_time_clusters_xr = corr_time_clusters_xr.assign_coords(samples=("time",np.arange(corr_time_clusters_xr.shape[2])))

    # get the blocks
    corr_time_clusters_xr_epochs = corr_time_clusters_xr.cd.to_epochs(
        stim,  # stimulus dataframe
        events_str,  # select events
        t_before,  # seconds before stimulus
        t_after,  # seconds after stimulus
    )

    # block average
    blockaverage_clusters = corr_time_clusters_xr_epochs.groupby("trial_type").mean("epoch")

    # get t-stat
    # FIX ME - this assumes stim duration is the same for all trials
    foo = corr_time_clusters_xr_epochs.sel(reltime=(corr_time_clusters_xr_epochs.reltime > 0) & (corr_time_clusters_xr_epochs.reltime < stim.duration.mean())).mean("reltime")
    blockaverage_clusters_mean = foo.groupby("trial_type").mean("epoch")
    blockaverage_clusters_std = foo.groupby("trial_type").std("epoch")
    blockaverage_clusters_sem = foo.groupby("trial_type").std("epoch") / np.sqrt(foo.epoch.size)
    # FIX ME - CHECK that this is dividing by the correct number of epochs 

    # FIX ME - this assumes only two trials types
    # calculate the t-statistic between the two trial_types
    blockaverage_clusters_tstat = ( blockaverage_clusters_mean[0,:,:] - blockaverage_clusters_mean[1,:,:] ) / np.sqrt( blockaverage_clusters_sem[0,:,:]**2 + blockaverage_clusters_sem[1,:,:]**2 )

    return blockaverage_clusters, blockaverage_clusters_tstat, blockaverage_clusters_mean, blockaverage_clusters_std, blockaverage_clusters_sem



def corr_cluster( corr_matrix_xr, cluster_threshold ):
    
    # calculate the distance matrix
    #dist_matrix = 1 - np.abs(corr_matrix)
    dist_matrix = 1 - corr_matrix_xr.values

    # Fill NaN values with a specific value (e.g., 0)
    dist_matrix = np.nan_to_num(dist_matrix, nan=2)

    # Ensure the distance matrix is symmetric
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Ensure the diagonal of the distance matrix is zero
    np.fill_diagonal(dist_matrix, 0)

    # calculate the linkage matrix
    linkage_matrix = linkage(squareform(dist_matrix), method="average")

    return linkage_matrix



def global_preweight(innov_matrix, kappa=4.685):
    """
    Compute Tukey-bisquare weights across all channels jointly.
    innov_matrix: shape (n_channels, n_time)
    Returns: weights w_t of shape (n_time,)
    """
    # geometric length per timepoint
    r = np.sqrt(np.sum(innov_matrix**2, axis=0))
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    sigma = 1.4826 * mad
    if sigma == 0:
        return np.ones_like(r)
    w = np.where(np.abs(r) < kappa * sigma,
                 (1 - (r / (kappa * sigma))**2)**2,
                 0.0)
    return w

def robust_slope(x, y, weights=None):
    """
    Compute robust slope of y ~ x using Tukey bisquare weighting.
    Optionally apply global preweights (multiplicative).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # combine with global preweights
    if weights is not None:
        w = weights
    else:
        w = np.ones_like(x)
    # Add constant term
    X = np.column_stack([np.ones_like(x), x])
    # Apply global weights
    model = RLM(y, X, M=TukeyBiweight(), weights=w)
    res = model.fit(maxiter=50)
    return res.params[1]  # slope only

def robust_corr_pair(x, y, global_weights=None):
    """Compute symmetric robust correlation coefficient for one pair."""
    beta_xy = robust_slope(x, y, weights=global_weights)
    beta_yx = robust_slope(y, x, weights=global_weights)
    sign = np.sign(beta_xy + beta_yx)  # correct sign convention
    r_robust = sign * np.sqrt(np.abs(beta_xy * beta_yx))
    return r_robust

def robust_pearson_corrcoef(data, global_w):
    """
    Robust correlation following Lanka et al. (2022).
    data: xarray.DataArray or np.ndarray, shape (n_channels, n_time)
    Returns: np.ndarray (n_channels, n_channels)
    """
    if isinstance(data, xr.DataArray):
        arr = data.values
    else:
        arr = np.asarray(data)
    n_channels, n_time = arr.shape

    # Pairwise robust correlations
    R = np.eye(n_channels)
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            r = robust_corr_pair(arr[i], arr[j], global_weights=global_w)
            R[i, j] = R[j, i] = r
    return R




def preprocess_dataset( rec, chs_pruned_subjs, cfg_dataset, cfg_blockavg, unique_trial_types, 
                       Adot_parcels_lev1_xr, Adot_parcels_lev2_xr,
                       cfg_img_recon = None, head = None, Adot = None,
                       flag_do_image_recon = False, flag_do_bp_filter_on_conc = True, flag_do_AR_filter_on_conc = 0, 
                       flag_do_gms_chromo = True, flag_channels_to_parcels = False, flag_parcels_use_lev1 = False, flag_do_robust = False):
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import arma_order_select_ic

    n_files = len(cfg_dataset['file_ids'])

    F = None # initializing F, D, G for image reconstruction
    D = None
    G = None

    # Loop over subjects
    for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):

        # if idx_subj > 0:
        #     break

        print(f'Preprocess SUBJECT {curr_subj}')

        conc_ts_files = {}
        conc_var_files = {}

        # Initialize the time series for each subject 
        if idx_subj == 0:
            conc_ts_subjs = {}
            conc_var_subjs = {}
        conc_ts_subjs[idx_subj] = {}
        conc_var_subjs[idx_subj] = {}

        # corr_hbo_files = {}
        # corr_hbr_files = {}

        # Loop over runs
        for idx_file, curr_file in enumerate(cfg_dataset['file_ids']):

            # get the OD time series
            od_ts = rec[idx_subj][idx_file]['od_corrected'].copy()

            # get list of channels that we don't trust for estimating the variance
            amp = rec[idx_subj][idx_file]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
            idx_amp = np.where(amp < cfg_blockavg['cfg_mse_od']['mse_amp_thresh'])[0]
            idx_sat = np.where(chs_pruned_subjs[idx_subj][idx_file] == 0.0)[0] 
            idx_bad_channels = np.concatenate((idx_amp, idx_sat))
            idx_bad_channels = np.sort(idx_bad_channels)
            idx_bad_channels = rec[idx_subj][idx_file]['amp'].channel[idx_bad_channels].values                    

            # convert to conc in channel space or image space
            if not flag_do_image_recon:
                dpf = xr.DataArray(
                    [1, 1],
                    dims="wavelength",
                    coords={"wavelength": od_ts.wavelength},
                )
                conc_ts = cedalion.nirs.od2conc(od_ts, rec[idx_subj][idx_file].geo3d, dpf, spectrum="prahl")

                # get the variance, correcting channels we don't trust (saturated, low amp, and low var)
                conc_var = quality.measurement_variance(
                        conc_ts,
                        list_bad_channels = idx_bad_channels,
                        bad_rel_var = 1e6, # If bad_abs_var is none then it uses this value relative to maximum variance
                        bad_abs_var = None, #cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data'],
                        calc_covariance = False
                    )
                conc_var = conc_var + cfg_blockavg['cfg_mse_conc']['mse_min_thresh'] # set a small value to avoid dominance for low variance channels
                conc_ts.loc[dict(channel=conc_ts.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']
                conc_ts.loc[dict(channel=conc_ts.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']
                #     # FIXME can delete once I trust new code
                    # conc_var = conc_ts.var('time') + cfg_blockavg['cfg_mse_conc']['mse_min_thresh'] # set a small value to avoid dominance for low variance channels
 
                    # # correct for bad data
                    # amp = rec[idx_subj][idx_file]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
                    # idx_amp = np.where(amp < cfg_blockavg['cfg_mse_od']['mse_amp_thresh'])[0]
                    # conc_var.loc[dict(channel=conc_ts.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data']
                    # conc_ts.loc[dict(channel=conc_ts.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']

                    # idx_sat = np.where(chs_pruned_subjs[idx_subj][idx_file] == 0.0)[0] 
                    # conc_var.loc[dict(channel=conc_ts.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data']
                    # conc_ts.loc[dict(channel=conc_ts.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']

            else:
                # get the variance, correcting channels we don't trust (saturated, low amp, and low var)
                C_meas = quality.measurement_variance(
                        od_ts,
                        list_bad_channels = idx_bad_channels,
                        bad_rel_var = 1e6, # If bad_abs_var is none then it uses this value
                        bad_abs_var = cfg_blockavg['cfg_mse_od']['mse_val_for_bad_data'],
                        calc_covariance = False
                    )
                C_meas = C_meas + cfg_blockavg['cfg_mse_od']['mse_min_thresh'] # set a small value to avoid dominance for low variance channels
                od_ts.loc[dict(channel=od_ts.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_od']['blockaverage_val']
                od_ts.loc[dict(channel=od_ts.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_od']['blockaverage_val']
                # # FIXME can delete once I trust new code above
                # C_meas = od_ts.var('time') + cfg_blockavg['cfg_mse_od']['mse_min_thresh'] # set a small value to avoid dominance for low variance channels
                # # correct for bad data
                # amp = rec[idx_subj][idx_file]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
                # idx_amp = np.where(amp < cfg_blockavg['cfg_mse_od']['mse_amp_thresh'])[0]
                # C_meas.loc[dict(channel=amp.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data']
                # od_ts.loc[dict(channel=amp.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']

                # idx_sat = np.where(chs_pruned_subjs[idx_subj][idx_file] == 0.0)[0] 
                # C_meas.loc[dict(channel=amp.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data']
                # od_ts.loc[dict(channel=amp.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']

                # for od_ts stack the wavelength and channel dimension to measurement dimension
                od_ts_stacked = od_ts.stack(measurement=("wavelength", "channel")).reset_index("measurement")
                od_ts_stacked = od_ts_stacked.transpose("measurement", "time")  # transpose to have measurement as the first dimension
                
                X_ts, W, D, F, G = img_recon.do_image_recon(od_ts_stacked, head = head, Adot = Adot, C_meas_flag = cfg_img_recon['flag_Cmeas'], C_meas = C_meas, 
                                            wavelength = [760,850], BRAIN_ONLY = cfg_img_recon['BRAIN_ONLY'], DIRECT = cfg_img_recon['DIRECT'], SB = cfg_img_recon['SB'], 
                                            cfg_sbf = cfg_img_recon['cfg_sb'], alpha_spatial = cfg_img_recon['alpha_spatial'], alpha_meas = cfg_img_recon['alpha_meas'],
                                            F = F, D = D, G = G)
                conc_ts = X_ts.transpose('chromo', 'vertex', 'time')
                conc_ts = conc_ts.assign_coords(samples=("time", np.arange(conc_ts.shape[2])))
                conc_ts = conc_ts.assign_coords(time=od_ts.time) # need to get the seconds attribute back

                # get conc_var from the image reconstruction
                conc_var = img_recon.get_image_noise(C_meas, X_ts, W, DIRECT = cfg_img_recon['DIRECT'], SB= cfg_img_recon['SB'], G=G)
                # FIXME I want to use cfg_blockavg['cfg_mse_od']['mse_min_thresh'], but this is micromolar and the conc_var seems to be molar
                conc_var = xr.where(conc_var == 0, 1e6, conc_var)  # replace 0 with a large value because the data is bad

            # projecting vertices to parcels
            # but only do this before BP, AR and GMS if also doing image recon
            # otherwise projecting channels to parcels is done after GMS
            if flag_do_image_recon and flag_channels_to_parcels: # project channels to parcel_lev2 by weighted average over channels
                # weighted average with 1/var of each vertex as the weights
                # conc_ts = conc_ts.groupby('parcel').mean('vertex') # this is not weighted average
                w = 1 / conc_var
                tsw = w * conc_ts
                conc_ts = tsw.groupby('parcel').sum('vertex') / w.groupby('parcel').sum('vertex')
                conc_var = 1 / w.groupby('parcel').sum('vertex')

                # remove the 3 non-brain parcels
                conc_ts = conc_ts.sel(parcel=~conc_ts.parcel.isin([
                    'scalp',
                    'Background+FreeSurfer_Defined_Medial_Wall_LH',
                    'Background+FreeSurfer_Defined_Medial_Wall_RH'
                ])) 
                conc_var = conc_var.sel(parcel=~conc_var.parcel.isin([
                    'scalp',
                    'Background+FreeSurfer_Defined_Medial_Wall_LH',
                    'Background+FreeSurfer_Defined_Medial_Wall_RH'
                ])) 

                # weighted average to level 1 or level 2 parcels                
                if flag_parcels_use_lev1:
                    parcel_list_lev = []
                    for parcel in conc_ts.parcel.values:
                        parcel_list_lev.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )                
                else:
                    parcel_list_lev = []
                    for parcel in conc_ts.parcel.values:
                        if parcel.split('_')[1].isdigit():
                            parcel_list_lev.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )
                        else:
                            parcel_list_lev.append( parcel.split('_')[0] + '_' + parcel.split('_')[1] + '_' + parcel.split('_')[-1] )
                conc_ts = conc_ts.assign_coords(parcel=("parcel", parcel_list_lev))
                conc_var = conc_var.assign_coords(parcel=("parcel", parcel_list_lev))
                w = 1 / conc_var
                tsw = w * conc_ts
                if 1: # this was losing chromo coordinate... curious as it used to work
                    # conc_ts = tsw.groupby('parcel').sum('parcel') / w.groupby('parcel').sum('parcel')
                    # conc_var = 1 / w.groupby('parcel').sum('parcel')
                    w_sum   = w.groupby('parcel').sum()
                    tsw_sum = tsw.groupby('parcel').sum()
                    # FIX: restore chromo coordinate
                    tsw_sum = tsw_sum.assign_coords(chromo=tsw.chromo)
                    w_sum   = w_sum.assign_coords(chromo=w.chromo)

                    conc_ts  = tsw_sum / w_sum
                    conc_var = 1.0 / w_sum
                else:
                    conc_ts = tsw.sum(dim='parcel') / w.sum(dim='parcel')
                    conc_var = 1 / w.sum('parcel')
                    # conc_ts = tsw.groupby(['chromo', 'parcel']).sum('parcel') / \
                    #         w.groupby(['chromo', 'parcel']).sum('parcel')                    
                    # conc_var = 1 / w.groupby(['chromo', 'parcel']).sum('parcel')

            # bandpass filter the time series
            if flag_do_bp_filter_on_conc:
                fmin = 0.01 * units.Hz
                fmax = 0.2 * units.Hz
                conc_ts = cedalion.sigproc.frequency.freq_filter(conc_ts, fmin, fmax)


            # Do the AR filtering
            # FIXME use from cedalion.math import ar_filter instead of the following code
            #       But I notice that that function returns time x channel x chromo/wavelength
            #       I am not thrilled about that. Also, it doesn't presently handle parcels or vertices
            if flag_do_AR_filter_on_conc > 0:
                ar_order = flag_do_AR_filter_on_conc
                conc_ts_tmp = conc_ts.isel(time=slice(ar_order, None)).copy()  # remove the first ar_order points
                for idx_chromo, chromo in enumerate(conc_ts.chromo):
                    if hasattr(conc_ts, "channel"):
                        for idx_ch, ch in enumerate(conc_ts.channel):
                            y = conc_ts.sel(chromo=chromo, channel=ch).values
                            # Fit AR model if y has no NaNs or Infs
                            if np.all(np.isfinite(y)):
                                model = sm.tsa.AutoReg(y, lags=ar_order, old_names=False)
                                result = model.fit()
                                residuals = result.resid * units.micromolar  # convert back to micromolar
                                # Store the residuals back into the time series
                                conc_ts_tmp.loc[dict(chromo=chromo, channel=ch)] = xr.DataArray(residuals, dims='time', coords={'time': conc_ts_tmp.time})
                    elif hasattr(conc_ts, "parcel"):
                        for idx_parcel, parcel in enumerate(conc_ts.parcel):
                            y = conc_ts.sel(chromo=chromo, parcel=parcel).values
                            # Fit AR model if y has no NaNs or Infs
                            if np.all(np.isfinite(y)):
                                model = sm.tsa.AutoReg(y, lags=ar_order, old_names=False)
                                result = model.fit()
                                residuals = result.resid * units.micromolar
                                # Store the residuals back into the time series
                                conc_ts_tmp.loc[dict(chromo=chromo, parcel=parcel)] = xr.DataArray(residuals, dims='time', coords={'time': conc_ts_tmp.time})
                    elif hasattr(conc_ts, "vertex"):
                        for idx_vertex, vertex in enumerate(conc_ts.vertex):
                            y = conc_ts.sel(chromo=chromo, vertex=vertex).values
                            # Fit AR model if y has no NaNs or Infs
                            if np.all(np.isfinite(y)):
                                model = sm.tsa.AutoReg(y, lags=ar_order, old_names=False)
                                result = model.fit()
                                residuals = result.resid * units.micromolar
                                # Store the residuals back into the time series
                                conc_ts_tmp.loc[dict(chromo=chromo, vertex=vertex)] = xr.DataArray(residuals, dims='time', coords={'time': conc_ts_tmp.time})
                conc_ts = conc_ts_tmp.copy()


            # GLOBAL PRE-WEIGHTING
            global_w_hbo = global_preweight(conc_ts.sel(chromo='HbO').values )
            global_w_hbr = global_preweight(conc_ts.sel(chromo='HbR').values )
            global_w_xr = xr.DataArray(
                np.vstack([global_w_hbo, global_w_hbr]),
                dims=["chromo", "time"],
                coords={"chromo": conc_ts.chromo, "time": conc_ts.time}
            )


            # Global mean subtraction for each chromo
            # FIXME handle if parcel or channel
            if flag_do_gms_chromo:
                if 'parcel' in conc_ts.dims:
                    ts_dim = 'parcel'
                elif 'channel' in conc_ts.dims:
                    ts_dim = 'channel'
                elif 'vertex' in conc_ts.dims:
                    ts_dim = 'vertex'
                else:
                    raise ValueError("Input data must have dimension 'parcel', 'channel', or 'vertex'")

                if 1: # FIXME: can't use this until global_component_subtract is updated to ts_weights with a temporal dimension for global_w_xr
                      # Note that the fix is only needed for PCA regression, not GMS
                    conc_ts, _ = global_component_subtract(
                            conc_ts,
                            channel_weights = 1/conc_var,
                            temporal_weights = global_w_xr,
                            spatial_dim = ts_dim,
                            k = 1 # k is the number of PCA components to remove, 0 means do GMS
                        )
                else:
                    # FIXME this is my old way of doing it. Remove this once comfortable with new cedalion code
                    # get the weighted mean by variance
                    gms = (conc_ts / conc_var).mean('channel') / (1/conc_var).mean('channel')

                    # fit GMS to the channel data and subtract it
                    numerator = (conc_ts * gms).sum(dim="time")
                    denominator = (gms * gms).sum(dim="time")
                    scl = numerator / denominator
                    conc_ts = conc_ts - scl*gms


            # project channels to parcels
            # we do this here for channel space data (i.e. no image recon)
            # FIXME: I should probably just remove this. OR fix it because of the PRE-WEIGHTING I do above from Huppert 2022
            if not flag_do_image_recon and flag_channels_to_parcels: # project channels to parcel_lev2 by weighted average over channels
                w = 1 / conc_var
                # get the normalized weighted averaging kernel
                if flag_parcels_use_lev1:
                    Adot_parcels_weighted_xr = w * Adot_parcels_lev1_xr
                else:
                    Adot_parcels_weighted_xr = w * Adot_parcels_lev2_xr
                Adot_parcels_weighted_xr = Adot_parcels_weighted_xr / Adot_parcels_weighted_xr.sum(dim='channel')
                # do the inner product over channel between conc_ts and Adot_parcels_weighted_xr
                foo_hbo = Adot_parcels_weighted_xr.sel(chromo='HbO').T @ conc_ts.sel(chromo='HbO')
                foo_hbr = Adot_parcels_weighted_xr.sel(chromo='HbR').T @ conc_ts.sel(chromo='HbR')
                conc_ts = xr.concat([foo_hbo, foo_hbr], dim='chromo')
                # do the same with conc_var
                foo_hbo = Adot_parcels_weighted_xr.sel(chromo='HbO').T**2 @ conc_var.sel(chromo='HbO')
                foo_hbr = Adot_parcels_weighted_xr.sel(chromo='HbR').T**2 @ conc_var.sel(chromo='HbR')
                conc_var = xr.concat([foo_hbo, foo_hbr], dim='chromo')

                n_ch_or_parcel = conc_ts.parcel.size
                correlation_coord = conc_ts.parcel.values
            elif not flag_do_image_recon and not flag_channels_to_parcels:
                n_ch_or_parcel = conc_ts.channel.size
                correlation_coord = conc_ts.channel.values
            elif flag_do_image_recon and flag_channels_to_parcels: 
                n_ch_or_parcel = conc_ts.parcel.size
                correlation_coord = conc_ts.parcel.values
            elif flag_do_image_recon and not flag_channels_to_parcels:
                n_ch_or_parcel = conc_ts.vertex.size
                correlation_coord = conc_ts.vertex.values


            # loop over trial types
            # get the time series for each trial type and the correlation matrix for repeatability
            for idx_trial_type, trial_type in enumerate(unique_trial_types):
                if trial_type != 'full_ts':
                    idx = np.where(rec[idx_subj][idx_file].stim.trial_type==trial_type)[0]
                    t_indices_tmp = np.array([])
                    for ii in idx:
                        t_indices_tmp = np.concatenate( (t_indices_tmp, np.where( 
                                            (conc_ts.time >  rec[idx_subj][idx_file].stim.onset[ii]) &
                                            (conc_ts.time <= (rec[idx_subj][idx_file].stim.onset[ii] + rec[idx_subj][idx_file].stim.duration[ii] )) 
                                            )[0] )
                                        )
                    foo_ts = conc_ts.isel(time=t_indices_tmp.astype(int)).copy() 
                else:
                    # get the full time series
                    foo_ts = conc_ts.copy()

                # get the correlation matrix for assessing repeatability
                if not flag_do_robust: # not Robust
                    corr_hbo = np.corrcoef( foo_ts.sel(chromo='HbO').values, rowvar=True )
                    corr_hbr = np.corrcoef( foo_ts.sel(chromo='HbR').values, rowvar=True )
                else: # Robust
                    corr_hbo = robust_pearson_corrcoef( foo_ts.sel(chromo='HbO').values, global_w_hbo)
                    corr_hbr = robust_pearson_corrcoef( foo_ts.sel(chromo='HbR').values, global_w_hbr)

                # store results 
                # concatenate the time series and variance for each trial type for subsequent corrcoef in next cell
                # store corrcoef for each file for repeatability
                # FIXME: undo PRE-WEIGHTING 
                if idx_file == 0:
                    conc_ts_files[trial_type] = foo_ts
                    conc_var_files[trial_type] = conc_var.copy()
                else:
                    conc_ts_files[trial_type] = xr.concat([conc_ts_files[trial_type], foo_ts], dim='time', coords='minimal', compat='override') # ensure no reordering since times overlap
                    conc_var_files[trial_type] = xr.concat([conc_var_files[trial_type], conc_var], dim='file')
                
                da_hbo = xr.DataArray(
                        corr_hbo.reshape(1,1,1,1,n_ch_or_parcel,n_ch_or_parcel),
                        dims=["subj","file", "trial_type", "chromo", "correlation_A", "correlation_B"],
                        coords={"chromo": ["HbO"], "trial_type": [trial_type], 
                            "file": [idx_file], "subj": [curr_subj], 
                            "correlation_A": correlation_coord, 
                            "correlation_B": correlation_coord}
                    )
                da_hbr = xr.DataArray(
                        corr_hbr.reshape(1,1,1,1,n_ch_or_parcel,n_ch_or_parcel),
                        dims=["subj","file", "trial_type", "chromo", "correlation_A", "correlation_B"],
                        coords={"chromo": ["HbR"], "trial_type": [trial_type], "file": [idx_file], "subj": [curr_subj]}
                    )
                if idx_trial_type == 0:
                    corr_trial_type = xr.concat([da_hbo, da_hbr], dim="chromo")
                else:
                    corr_trial_type_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                    corr_trial_type = xr.concat([corr_trial_type, corr_trial_type_tmp], dim="trial_type")
            # end of trial type loop

            if idx_file == 0:
                corr_files = corr_trial_type.copy()
            else:
                corr_files = xr.concat([corr_files, corr_trial_type], dim="file")
        # end of file loop

        # store the time series for each subject for each trial_type
        for idx_trial_type, trial_type in enumerate(unique_trial_types):
            conc_ts_subjs[idx_subj][trial_type] = conc_ts_files[trial_type].copy()
            conc_var_subjs[idx_subj][trial_type] = conc_var_files[trial_type].mean('file') 

        if idx_subj == 0:
            corr_subj_files = corr_files.copy()
        else:
            corr_subj_files = xr.concat([corr_subj_files, corr_files], dim="subj")


        # get the repeatability
        for idx_trial_type, trial_type in enumerate(unique_trial_types):
            foo_hbo = np.corrcoef(np.nan_to_num(corr_files.sel(chromo='HbO',trial_type=trial_type).values).reshape(n_files,-1), rowvar=True)
            foo_hbr = np.corrcoef(np.nan_to_num(corr_files.sel(chromo='HbR',trial_type=trial_type).values).reshape(n_files,-1), rowvar=True)
            # foo_hbo = np.corrcoef(np.nan_to_num(corr_hbo_files[trial_type]), rowvar=True)
            # foo_hbr = np.corrcoef(np.nan_to_num(corr_hbr_files[trial_type]), rowvar=True)

            # compute the mean repeatability for HbO and HbR
            mean_hbo = foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].mean()
            mean_hbr = foo_hbr[np.triu_indices(foo_hbr.shape[0], k=1)].mean()

            da_hbo = xr.DataArray(mean_hbo).expand_dims(
                subj=[curr_subj],
                trial_type=[trial_type],
                chromo=["HbO"]
            ).transpose("subj", "trial_type", "chromo")

            da_hbr = xr.DataArray(mean_hbr).expand_dims(
                subj=[curr_subj],
                trial_type=[trial_type],
                chromo=["HbR"]
            ).transpose("subj", "trial_type", "chromo")            # da_hbo = xr.DataArray(
            #     mean_hbo,
            #     dims=["subj","trial_type","chromo"],
            #     coords={"trial_type": [trial_type], "chromo": ["HbO"], "subj": [curr_subj]}
            # )
            # da_hbr = xr.DataArray(
            #     mean_hbr,
            #     dims=["subj","trial_type","chromo"],
            #     coords={"trial_type": [trial_type], "chromo": ["HbR"], "subj": [curr_subj]}
            # )
            if idx_trial_type == 0:
                repeatability_trial_type_mean = xr.concat([da_hbo, da_hbr], dim="chromo")
            else:
                repeatability_trial_type_mean_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                repeatability_trial_type_mean = xr.concat([repeatability_trial_type_mean, repeatability_trial_type_mean_tmp], dim="trial_type")                

            # compute the std repeatability for HbO and HbR
            mean_hbo = foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].std()
            mean_hbr = foo_hbr[np.triu_indices(foo_hbr.shape[0], k=1)].std()

            da_hbo = xr.DataArray(mean_hbo).expand_dims(
                subj=[curr_subj],
                trial_type=[trial_type],
                chromo=["HbO"]
            ).transpose("subj", "trial_type", "chromo")

            da_hbr = xr.DataArray(mean_hbr).expand_dims(
                subj=[curr_subj],
                trial_type=[trial_type],
                chromo=["HbR"]
            ).transpose("subj", "trial_type", "chromo")            
            # da_hbo = xr.DataArray(
            #     mean_hbo,
            #     dims=["subj","trial_type","chromo"],
            #     coords={"trial_type": [trial_type], "chromo": ["HbO"], "subj": [curr_subj]}
            # )
            # da_hbr = xr.DataArray(
            #     mean_hbr,
            #     dims=["subj","trial_type","chromo"],
            #     coords={"trial_type": [trial_type], "chromo": ["HbR"], "subj": [curr_subj]}
            # )
            if idx_trial_type == 0:
                repeatability_trial_type_std = xr.concat([da_hbo, da_hbr], dim="chromo")
            else:
                repeatability_trial_type_std_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                repeatability_trial_type_std = xr.concat([repeatability_trial_type_std, repeatability_trial_type_std_tmp], dim="trial_type")                

        if idx_subj == 0:
            repeatability_subj_mean = repeatability_trial_type_mean.copy()
            repeatability_subj_std = repeatability_trial_type_std.copy()
        else:
            repeatability_subj_mean = xr.concat([repeatability_subj_mean, repeatability_trial_type_mean], dim="subj")
            repeatability_subj_std = xr.concat([repeatability_subj_std, repeatability_trial_type_std], dim="subj")
    # end of subject loop

    return conc_ts_subjs, conc_var_subjs, corr_subj_files, repeatability_subj_mean, repeatability_subj_std



def get_correlation_matrices( conc_ts_subjs, conc_var_subjs, cfg_dataset, unique_trial_types, flag_do_robust = False):

    corr_subj = {}
    corr_subj_var = {}

    # Check that the data has dimension 'parcel'
    if 'parcel' in conc_ts_subjs[0][unique_trial_types[0]].dims:
        n_ch_or_parcel = conc_ts_subjs[0][unique_trial_types[0]].parcel.size
        correlation_coord = conc_ts_subjs[0][unique_trial_types[0]].parcel.values
    elif 'channel' in conc_ts_subjs[0][unique_trial_types[0]].dims:
        n_ch_or_parcel = conc_ts_subjs[0][unique_trial_types[0]].channel.size
        correlation_coord = conc_ts_subjs[0][unique_trial_types[0]].channel.values
    elif 'vertex' in conc_ts_subjs[0][unique_trial_types[0]].dims:
        n_ch_or_parcel = conc_ts_subjs[0][unique_trial_types[0]].vertex.size
        correlation_coord = conc_ts_subjs[0][unique_trial_types[0]].vertex.values
    else:
        raise ValueError("Input data must have dimension 'parcel', 'channel', or 'vertex'")

    # Loop over subjects
    for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):

        print(f'Correlation Matrices for SUBJECT {curr_subj}')

        # loop over trial types
        for idx_trial_type, trial_type in enumerate(unique_trial_types):

            # GLOBAL PRE-WEIGHTING
            global_w_hbo = global_preweight(conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO').values )
            global_w_hbr = global_preweight(conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR').values )
            # global_w_xr = xr.DataArray(
            #     np.vstack([global_w_hbo, global_w_hbr]),
            #     dims=["chromo", "time"],
            #     coords={"chromo": conc_ts_subjs[idx_subj][trial_type].chromo, "time": conc_ts_subjs[idx_subj][trial_type].time}
            # )

            # get the correlation matrix
            if not flag_do_robust: # not Robust
                corr_hbo = np.corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO').values, rowvar=True )
                corr_hbr = np.corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR').values, rowvar=True )
            else: # Robust
                corr_hbo = robust_pearson_corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO').values, global_w_hbo)
                corr_hbr = robust_pearson_corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR').values, global_w_hbr)

            # # get the correlation matrix
            # if 1:
            #     corr_hbo = np.corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO').values, rowvar=True )
            #     corr_hbr = np.corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR').values, rowvar=True )
            # else:
            #     corr_hbo = np.corrcoef( np.nan_to_num(conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO').values,nan=0.0, posinf=0.0, neginf=0.0), rowvar=True )
            #     corr_hbr = np.corrcoef( np.nan_to_num(conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR').values,nan=0.0, posinf=0.0, neginf=0.0), rowvar=True )


            # get the correlation matrix for each subject
            da_hbo = xr.DataArray(
                    corr_hbo.reshape(1,1,1,n_ch_or_parcel,n_ch_or_parcel),
                    dims=["subj", "trial_type", "chromo", "correlation_A", "correlation_B"],
                    coords={"chromo": ["HbO"], "trial_type": [trial_type], 
                        "subj": [curr_subj], 
                        "correlation_A": correlation_coord, 
                        "correlation_B": correlation_coord}
                )
            da_hbr = xr.DataArray(
                    corr_hbr.reshape(1,1,1,n_ch_or_parcel,n_ch_or_parcel),
                    dims=["subj", "trial_type", "chromo", "correlation_A", "correlation_B"],
                    coords={"chromo": ["HbR"], "trial_type": [trial_type], 
                        "subj": [curr_subj], 
                        "correlation_A": correlation_coord, 
                        "correlation_B": correlation_coord}
                )
            # da_hbo = xr.DataArray(
            #     corr_hbo.reshape(1,1,1,-1),
            #     dims=["subj", "trial_type", "chromo", "correlation"],
            #     coords={"chromo": ["HbO"], "trial_type": [trial_type], "subj": [curr_subj]}
            # )
            # da_hbr = xr.DataArray(
            #     corr_hbr.reshape(1,1,1,-1),
            #     dims=["subj", "trial_type", "chromo", "correlation"],
            #     coords={"chromo": ["HbR"], "trial_type": [trial_type], "subj": [curr_subj]}
            # )
            if idx_trial_type == 0:
                da_chromo = xr.concat([da_hbo, da_hbr], dim="chromo")
            else:
                da_chromo_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                da_chromo = xr.concat([da_chromo, da_chromo_tmp], dim="trial_type")

            # get the variance of each element in the correlation matrix
            da_hbo = xr.DataArray(
                    (conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO').values[:, np.newaxis] + conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO').values[:, np.newaxis].T).reshape(1,1,1,n_ch_or_parcel,n_ch_or_parcel),
                    dims=["subj", "trial_type", "chromo", "correlation_A", "correlation_B"],
                    coords={"chromo": ["HbO"], "trial_type": [trial_type], 
                        "subj": [curr_subj], 
                        "correlation_A": correlation_coord, 
                        "correlation_B": correlation_coord}
                )
            da_hbr = xr.DataArray(
                    (conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR').values[:, np.newaxis] + conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR').values[:, np.newaxis].T).reshape(1,1,1,n_ch_or_parcel,n_ch_or_parcel),
                    dims=["subj", "trial_type", "chromo", "correlation_A", "correlation_B"],
                    coords={"chromo": ["HbR"], "trial_type": [trial_type], 
                        "subj": [curr_subj], 
                        "correlation_A": correlation_coord, 
                        "correlation_B": correlation_coord}
                )
            # da_hbo = xr.DataArray(
            #     (conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO').values[:, np.newaxis] + conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO').values[:, np.newaxis].T).reshape(1,1,1,-1),
            #     dims=["subj", "trial_type", "chromo", "correlation"],
            #     coords={"chromo": ["HbO"], "trial_type": [trial_type], "subj": [curr_subj]}
            # )
            # da_hbr = xr.DataArray(
            #     (conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR').values[:, np.newaxis] + conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR').values[:, np.newaxis].T).reshape(1,1,1,-1),
            #     dims=["subj", "trial_type", "chromo", "correlation"],
            #     coords={"chromo": ["HbR"], "trial_type": [trial_type], "subj": [curr_subj]}
            # )
            if idx_trial_type == 0:
                da_chromo_var = xr.concat([da_hbo, da_hbr], dim="chromo")
            else:
                da_chromo_var_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                da_chromo_var = xr.concat([da_chromo_var, da_chromo_var_tmp], dim="trial_type")
        # end of trial type loop

        if idx_subj == 0:
            corr_subj = da_chromo.copy()
            corr_subj_var = da_chromo_var.copy()
        else:
            corr_subj = xr.concat([corr_subj, da_chromo], dim="subj")
            corr_subj_var = xr.concat([corr_subj_var, da_chromo_var], dim="subj")
    # end of subject loop

    return corr_subj, corr_subj_var


def get_reliability(corr_subj, unique_trial_types, flag_do_robust = False):

    # get the reliability
    n_subjects = len(corr_subj.subj)
    reliability_mean = {}
    reliability_std = {}

    for idx_trial_type, trial_type in enumerate(unique_trial_types):
        
        foo_hbo = np.corrcoef(np.nan_to_num(corr_subj.sel(chromo='HbO',trial_type=trial_type).values.reshape(n_subjects,-1)), rowvar=True)
        foo_hbr = np.corrcoef(np.nan_to_num(corr_subj.sel(chromo='HbR',trial_type=trial_type).values.reshape(n_subjects,-1)), rowvar=True)

        # get the mean reliability of the correlation matrix
        da_hbo = xr.DataArray(
            foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].mean().reshape(1,1),
            dims=["trial_type","chromo"],
            coords={"trial_type":[trial_type],"chromo": ["HbO"]}
        )
        da_hbr = xr.DataArray(
            foo_hbr[np.triu_indices(foo_hbr.shape[0], k=1)].mean().reshape(1,1),
            dims=["trial_type","chromo"],
            coords={"trial_type":[trial_type],"chromo": ["HbR"]}
        )
        if idx_trial_type == 0:
            reliability_mean = xr.concat([da_hbo, da_hbr], dim="chromo")
        else:
            da_chromo_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
            reliability_mean = xr.concat([reliability_mean, da_chromo_tmp], dim="trial_type")

        # get the std reliability of the correlation matrix
        da_hbo = xr.DataArray(
            foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].std().reshape(1,1),
            dims=["trial_type","chromo"],
            coords={"trial_type":[trial_type],"chromo": ["HbO"]}
        )
        da_hbr = xr.DataArray(
            foo_hbr[np.triu_indices(foo_hbr.shape[0], k=1)].std().reshape(1,1),
            dims=["trial_type","chromo"],
            coords={"trial_type":[trial_type],"chromo": ["HbR"]}
        )
        if idx_trial_type == 0:
            reliability_std = xr.concat([da_hbo, da_hbr], dim="chromo")
        else:
            da_chromo_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
            reliability_std = xr.concat([reliability_std, da_chromo_tmp], dim="trial_type")

    return reliability_mean, reliability_std


def group_correlation_matrices(corr_subj, corr_subj_var, unique_trial_types):
    # This is old... better to use the boot strapping method below
    # FIXME we can delete this function... just here for now for reference

    # z-transform the correlation matrix across subjects using Fisher's z-transform
    corrz_hbo_subj_mean = {}
    corrz_hbr_subj_mean = {}
    corrz_hbo_subj_std = {}
    corrz_hbr_subj_std = {}
    corrz_hbo_subj_meanweighted = {}
    corrz_hbr_subj_meanweighted = {}
    corr_hbo_subj_mean = {}
    corr_hbr_subj_mean = {}
    corr_hbo_subj_std = {}
    corr_hbr_subj_std = {}
    corr_hbo_subj_meanweighted = {}
    corr_hbr_subj_meanweighted = {}

    for trial_type in unique_trial_types:
        corrz_hbo_subj = np.arctanh(corr_subj.sel(chromo='HbO',trial_type=trial_type).values)
        corrz_hbr_subj = np.arctanh(corr_subj.sel(chromo='HbR',trial_type=trial_type).values)

        # get the mean and std of the correlation matrix across subjects
        corrz_hbo_subj_mean[trial_type] = np.nanmean(corrz_hbo_subj, axis=0)
        corrz_hbr_subj_mean[trial_type] = np.nanmean(corrz_hbr_subj, axis=0)
        corrz_hbo_subj_std[trial_type] = np.nanstd(corrz_hbo_subj, axis=0)
        corrz_hbr_subj_std[trial_type] = np.nanstd(corrz_hbr_subj, axis=0)

        corrz_hbo_subj_meanweighted[trial_type] = np.nansum(corrz_hbo_subj * corr_subj_var.sel(chromo='HbO',trial_type=trial_type), axis=0) / np.nansum( corr_subj_var.sel(chromo='HbO',trial_type=trial_type), axis=0)
        corrz_hbr_subj_meanweighted[trial_type] = np.nansum(corrz_hbr_subj * corr_subj_var.sel(chromo='HbR',trial_type=trial_type), axis=0) / np.nansum( corr_subj_var.sel(chromo='HbR',trial_type=trial_type), axis=0)

        # get the mean and std of the correlation matrix across subjects
        corr_hbo_subj_mean[trial_type] = np.tanh(corrz_hbo_subj_mean[trial_type])
        corr_hbr_subj_mean[trial_type] = np.tanh(corrz_hbr_subj_mean[trial_type])
        corr_hbo_subj_std[trial_type] = np.tanh(corrz_hbo_subj_std[trial_type])
        corr_hbr_subj_std[trial_type] = np.tanh(corrz_hbr_subj_std[trial_type])

        corr_hbo_subj_meanweighted[trial_type] = np.tanh(corrz_hbo_subj_meanweighted[trial_type])
        corr_hbr_subj_meanweighted[trial_type] = np.tanh(corrz_hbr_subj_meanweighted[trial_type])

    return corr_hbo_subj_mean, corr_hbr_subj_mean, corr_hbo_subj_std, corr_hbr_subj_std, corr_hbo_subj_meanweighted, corr_hbr_subj_meanweighted



def boot_strap_corr(corr_subj, corr_subj_var,trial_type_list, trial_type_diff=None):

    if trial_type_diff is not None:
        if len(trial_type_diff) != 2:
            raise ValueError("trial_type_diff must contain exactly two trial types for difference calculation.")
            return        
        trial_type_diff_str = trial_type_diff[0] + '-' + trial_type_diff[1]
        trial_type_list = np.append( trial_type_list, trial_type_diff_str )
    else:
        trial_type_diff_str = 'none-none'

    for trial_type in trial_type_list:

        print(f"Bootstrapping for trial type '{trial_type}'")

        n_boot = 1000  # number of bootstrap samples

        n_subjects = len(corr_subj.subj)
        n_channels = int(len(corr_subj.correlation_A))

        z_boot_samples_hbo = np.zeros((n_boot, n_channels, n_channels))
        z_boot_samples_hbr = np.zeros((n_boot, n_channels, n_channels))

        for b in range(n_boot):
            # Resample subject indices with replacement
            boot_indices = np.random.choice(n_subjects, size=n_subjects, replace=True)

            z_weighted_sum_hbo = np.zeros((n_channels, n_channels))
            weight_sum_hbo = np.zeros((n_channels, n_channels))

            z_weighted_sum_hbr = np.zeros((n_channels, n_channels))
            weight_sum_hbr = np.zeros((n_channels, n_channels))

            for idx in boot_indices:
                if trial_type != trial_type_diff_str:
                    R = corr_subj.sel(chromo='HbO', trial_type=trial_type, subj=corr_subj.subj[idx]).values
                    # R = corr_hbo_subj[trial_type][idx]
                    z = np.arctanh(R)  # Fisher z-transform
                    w = 1 / corr_subj_var.sel(chromo='HbO', trial_type=trial_type, subj=corr_subj.subj[idx]).values
                    # w = 1 / corr_hbo_subj_var[trial_type][idx]
                else:
                    z = np.arctanh(corr_subj.sel(chromo='HbO', trial_type=trial_type_diff[0], subj=corr_subj.subj[idx]).values \
                        - corr_subj.sel(chromo='HbO', trial_type=trial_type_diff[1], subj=corr_subj.subj[idx]).values)  # Fisher z-transform
                    w = 1 / (corr_subj_var.sel(chromo='HbO', trial_type=trial_type_diff[0], subj=corr_subj.subj[idx]).values \
                        + corr_subj_var.sel(chromo='HbO', trial_type=trial_type_diff[1], subj=corr_subj.subj[idx]).values)
                    # z = np.arctanh(corr_hbo_subj[trial_type_diff[0]][idx] - corr_hbo_subj[trial_type_diff[1]][idx])  # Fisher z-transform
                    # w = 1 / (corr_hbo_subj_var[trial_type_diff[0]][idx] + corr_hbo_subj_var[trial_type_diff[1]][idx])
                z_weighted_sum_hbo += np.nan_to_num(z / w) # turn nan to 0
                weight_sum_hbo += np.nan_to_num(1 / w) # turn nan to 0

                if trial_type != trial_type_diff_str:
                    R = corr_subj.sel(chromo='HbR', trial_type=trial_type, subj=corr_subj.subj[idx]).values
                    # R = corr_hbr_subj[trial_type][idx]
                    z = np.arctanh(R)  # Fisher z-transform
                    w = 1 / corr_subj_var.sel(chromo='HbR', trial_type=trial_type, subj=corr_subj.subj[idx]).values
                    # w = 1 / corr_hbr_subj_var[trial_type][idx]
                else:
                    z = np.arctanh(corr_subj.sel(chromo='HbR', trial_type=trial_type_diff[0], subj=corr_subj.subj[idx]).values \
                        - corr_subj.sel(chromo='HbR', trial_type=trial_type_diff[1], subj=corr_subj.subj[idx]).values)
                    w = 1 / (corr_subj_var.sel(chromo='HbR', trial_type=trial_type_diff[0], subj=corr_subj.subj[idx]).values \
                        + corr_subj_var.sel(chromo='HbR', trial_type=trial_type_diff[1], subj=corr_subj.subj[idx]).values)
                    # z = np.arctanh(corr_hbr_subj[trial_type_diff[0]][idx] - corr_hbr_subj[trial_type_diff[1]][idx])                z_weighted_sum_hbr += np.nan_to_num(z / w) # turn nan to 0
                    # w = 1 / (corr_hbr_subj_var[trial_type_diff[0]][idx] + corr_hbr_subj_var[trial_type_diff[1]][idx])
                z_weighted_sum_hbr += np.nan_to_num(z / w)
                weight_sum_hbr += np.nan_to_num(1 / w) # turn nan to 0

            z_boot_samples_hbo[b] = z_weighted_sum_hbo / weight_sum_hbo
            z_boot_samples_hbr[b] = z_weighted_sum_hbr / weight_sum_hbr

        # Mean of bootstrap samples
        # Store mean of bootstrap samples in a single xarray.DataArray named z_boot_mean
        if trial_type == trial_type_list[0]:
            z_boot_mean = xr.DataArray(
                np.stack([
                    np.mean(z_boot_samples_hbo, axis=0),
                    np.mean(z_boot_samples_hbr, axis=0)
                ]),
                dims=["chromo", "correlation_A", "correlation_B"],
                coords={"chromo": ["HbO", "HbR"], "correlation_A": corr_subj.correlation_A.values, "correlation_B": corr_subj.correlation_B.values}
            )
            z_boot_mean = z_boot_mean.expand_dims(dim={"trial_type": [trial_type]})
        else:
            new = xr.DataArray(
                np.stack([
                    np.mean(z_boot_samples_hbo, axis=0),
                    np.mean(z_boot_samples_hbr, axis=0)
                ]),
                dims=["chromo", "correlation_A", "correlation_B"],
                coords={"chromo": ["HbO", "HbR"], "correlation_A": corr_subj.correlation_A.values, "correlation_B": corr_subj.correlation_B.values}
            )
            new = new.expand_dims(dim={"trial_type": [trial_type]})
            z_boot_mean = xr.concat([z_boot_mean, new], dim="trial_type")

        # Standard error (SE)
        # Standard error (SE) as xarray.DataArray with trial_type, chromo, correlation dims
        if trial_type == trial_type_list[0]:
            z_boot_se = xr.DataArray(
                np.stack([
                    np.std(z_boot_samples_hbo, axis=0),
                    np.std(z_boot_samples_hbr, axis=0)
                ]),
                dims=["chromo", "correlation_A", "correlation_B"],
                coords={"chromo": ["HbO", "HbR"], "correlation_A": corr_subj.correlation_A.values, "correlation_B": corr_subj.correlation_B.values}
            )
            z_boot_se = z_boot_se.expand_dims(dim={"trial_type": [trial_type]})
        else:
            new_se = xr.DataArray(
                np.stack([
                    np.std(z_boot_samples_hbo, axis=0),
                    np.std(z_boot_samples_hbr, axis=0)
                ]),
                dims=["chromo", "correlation_A", "correlation_B"],
                coords={"chromo": ["HbO", "HbR"], "correlation_A": corr_subj.correlation_A.values, "correlation_B": corr_subj.correlation_B.values}
            )
            new_se = new_se.expand_dims(dim={"trial_type": [trial_type]})
            z_boot_se = xr.concat([z_boot_se, new_se], dim="trial_type")

        # Confidence intervals (e.g. 95%)
        # z_ci_lower_hbo = np.percentile(z_boot_samples_hbo, 2.5, axis=0)
        # z_ci_upper_hbo = np.percentile(z_boot_samples_hbo, 97.5, axis=0)
        # z_ci_lower_hbr = np.percentile(z_boot_samples_hbr, 2.5, axis=0)
        # z_ci_upper_hbr = np.percentile(z_boot_samples_hbr, 97.5, axis=0)

        # Convert back to r-space if needed
        r_boot_mean = np.tanh(z_boot_mean)
        # r_boot_mean_hbr[trial_type] = np.tanh(z_boot_mean_hbr[trial_type])
        # r_ci_lower_hbo = np.tanh(z_ci_lower_hbo)
        # r_ci_upper_hbo = np.tanh(z_ci_upper_hbo)
        # r_ci_lower_hbr = np.tanh(z_ci_lower_hbr)
        # r_ci_upper_hbr = np.tanh(z_ci_upper_hbr)

    return z_boot_mean, z_boot_se, r_boot_mean


def plot_correlation_matrix( r_boot_mean, z_boot_mean, z_boot_se, t_crit, trial_type, vminmax, corr_subj=None, flag_show_which_axis=0 ):

    if corr_subj is None and flag_show_which_axis==0: # weighted mean across subjects
        f, axs = p.subplots(2,2,figsize=(12,8))

        ax1 = axs[0][0]
        foo1 = r_boot_mean.sel(chromo='HbO', trial_type=trial_type).values
        ax1.imshow( foo1, cmap='jet', vmin=-vminmax, vmax=vminmax)
        ax1.set_title('HbO Correlation Matrix')

        ax1 = axs[0][1]
        foo1 = r_boot_mean.sel(chromo='HbR', trial_type=trial_type).values
        ax1.imshow( foo1, cmap='jet', vmin=-vminmax, vmax=vminmax)
        ax1.set_title('HbR Correlation Matrix')

        ax1 = axs[1][0]
        foo = z_boot_mean.sel(chromo='HbO', trial_type=trial_type).values / \
            z_boot_se.sel(chromo='HbO', trial_type=trial_type).values
        foo1 = r_boot_mean.sel(chromo='HbO', trial_type=trial_type).copy().values
        foo1[np.abs(foo) < t_crit] = np.nan # remove non-significant correlations
        ax1.imshow( foo1, cmap='jet', vmin=-vminmax, vmax=vminmax)

        ax1 = axs[1][1]
        foo = z_boot_mean.sel(chromo='HbR', trial_type=trial_type).values / \
            z_boot_se.sel(chromo='HbR', trial_type=trial_type).values
        foo1 = r_boot_mean.sel(chromo='HbR', trial_type=trial_type).copy().values
        foo1[np.abs(foo) < t_crit] = np.nan # remove non-significant correlations
        ax1.imshow( foo1, cmap='jet', vmin=-vminmax, vmax=vminmax)

    if corr_subj is None and flag_show_which_axis>0: # each subject
        f, axs = p.subplots(1,1,figsize=(24,16))

        if flag_show_which_axis == 1:  # show HbO
            foo1 = r_boot_mean.sel(chromo='HbO', trial_type=trial_type).values
            axs.imshow(foo1, cmap='jet', vmin=-vminmax, vmax=vminmax)
            axs.set_title('HbO Correlation Matrix')

        elif flag_show_which_axis == 2:  # show HbR
            foo1 = r_boot_mean.sel(chromo='HbR', trial_type=trial_type).values
            axs.imshow(foo1, cmap='jet', vmin=-vminmax, vmax=vminmax)
            axs.set_title('HbR Correlation Matrix')

        elif flag_show_which_axis == 3:  # show HbO with t-stat threshold
            foo = z_boot_mean.sel(chromo='HbO', trial_type=trial_type).values / \
                z_boot_se.sel(chromo='HbO', trial_type=trial_type).values
            foo1 = r_boot_mean.sel(chromo='HbO', trial_type=trial_type).copy().values
            foo1[np.abs(foo) < t_crit] = np.nan  # remove non-significant correlations
            axs.imshow(foo1, cmap='jet', vmin=-vminmax, vmax=vminmax)
            axs.set_title('HbO Correlation Matrix')

        elif flag_show_which_axis == 4:  # show HbR with t-stat threshold
            foo = z_boot_mean.sel(chromo='HbR', trial_type=trial_type).values / \
                z_boot_se.sel(chromo='HbR', trial_type=trial_type).values
            foo1 = r_boot_mean.sel(chromo='HbR', trial_type=trial_type).copy().values
            foo1[np.abs(foo) < t_crit] = np.nan  # remove non-significant correlations
            axs.imshow(foo1, cmap='jet', vmin=-vminmax, vmax=vminmax)
            axs.set_title('HbR Correlation Matrix')

        # set X tick labels to corr_subj.correlation_A.values
        axs.set_xticks(np.arange(len(r_boot_mean.correlation_A)))
        axs.set_xticklabels(r_boot_mean.correlation_A.values, rotation=90)
        # set Y tick labels to corr_subj.correlation_B.values
        axs.set_yticks(np.arange(len(r_boot_mean.correlation_B)))
        axs.set_yticklabels(r_boot_mean.correlation_B.values)


    if corr_subj is not None: # each subject
        n_subjects = corr_subj.subj.size
        n_ch_or_parcel = int(np.sqrt(len(corr_subj.correlation)))

        f, axs = p.subplots(n_subjects,2,figsize=(12,6*n_subjects))

        for ii in range(0, n_subjects):
            ax1 = axs[ii][0]
            ax1.imshow( corr_subj.sel(chromo='HbO',trial_type=trial_type,subj=corr_subj.subj[ii]) \
                .values.reshape((n_ch_or_parcel,n_ch_or_parcel)), cmap='jet', vmin=-1, vmax=1)
            ax1.set_title(f'HbO Correlation Matrix,  Subj {corr_subj.subj[ii].values}')

            ax1 = axs[ii][1]
            ax1.imshow( corr_subj.sel(chromo='HbR',trial_type=trial_type,subj=corr_subj.subj[ii]) \
                .values.reshape((n_ch_or_parcel,n_ch_or_parcel)), cmap='jet', vmin=-1, vmax=1)
            ax1.set_title(f'HbR Correlation Matrix,  Subj {corr_subj.subj[ii].values}')



def plot_connectivity_circle( r_boot_mean, z_boot_mean, z_boot_se, t_crit, trial_type, vminmax, flag_show_hbo, flag_parcels_use_lev1, unique_parcels_lev1, unique_parcels_lev2, n_lines=100, colormap='jet', flag_show_specific_parcels=False, parcels_to_show=None, flag_show_tstat=False ):

    from mne_connectivity.viz import plot_connectivity_circle
    from mne.viz import circular_layout

    # FIXME: can I get y pos of parcels and order by that?
    # FIXME: would be nice to color code the 17 networks and visualize them on the brain. Maybe consider colors in examples at https://mne.tools/mne-connectivity/dev/generated/mne_connectivity.viz.plot_connectivity_circle.html

    n_channels = int(np.sqrt(r_boot_mean.shape[2]))

    # First, we reorder the labels based on their location in the left hemi
    #label_names = [label.name for label in labels]
    if flag_parcels_use_lev1:
        lh_labels = [name for name in unique_parcels_lev1 if name.endswith("LH")]
        rh_labels = [name for name in unique_parcels_lev1 if name.endswith("RH")] #[label[:-2] + "rh" for label in lh_labels]
        unique_parcels_list = unique_parcels_lev1.tolist()
    else:
        lh_labels = [name for name in unique_parcels_lev2 if name.endswith("LH")]
        rh_labels = [name for name in unique_parcels_lev2 if name.endswith("RH")] #[label[:-2] + "rh" for label in lh_labels]
        unique_parcels_list = unique_parcels_lev2.tolist()

    # Get the y-location of the label
    # label_ypos = list()
    # for name in lh_labels:
    #     idx = label_names.index(name)
    #     ypos = np.mean(labels[idx].pos[:, 1])
    #     label_ypos.append(ypos)

    # Reorder the labels based on their location
    #lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]


    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels)  # reverse the order
    node_order.extend(rh_labels[::-1])

    # node_angles = circular_layout(
    #     label_names, node_order, start_pos=90, group_boundaries=[0, len(label_names) / 2]
    # )


    if flag_show_hbo:
        # foo = z_boot_mean_hbo[trial_type] / z_boot_se_hbo[trial_type]
        # foo1 = r_boot_mean_hbo[trial_type].copy()
        foo = z_boot_mean.sel(chromo='HbO', trial_type=trial_type).values / \
            z_boot_se.sel(chromo='HbO', trial_type=trial_type).values
        foo1 = r_boot_mean.sel(chromo='HbO', trial_type=trial_type).copy().values
    else:
        # foo = z_boot_mean_hbr[trial_type] / z_boot_se_hbr[trial_type]
        # foo1 = r_boot_mean_hbr[trial_type].copy()
        foo = z_boot_mean.sel(chromo='HbR', trial_type=trial_type).values / \
            z_boot_se.sel(chromo='HbR', trial_type=trial_type).values
        foo1 = r_boot_mean.sel(chromo='HbR', trial_type=trial_type).copy().values


    foo1[np.abs(foo) < t_crit] = 0 #np.nan # remove non-significant correlations
    # foo1 = foo1.reshape((n_channels,n_channels))
    # foo = foo.reshape((n_channels,n_channels))



    node_angles = circular_layout(
        unique_parcels_list, node_order, start_pos=90, group_boundaries=[0, len(unique_parcels_list) / 2]
    )

    # colormap for 17networks


    # make label_colors from the FreeSurfer parcellation
    label_colors = np.zeros((len(unique_parcels_list), 4))
    label_colors[:, 0] = 1.0
    label_colors[:, 1] = 0.0
    label_colors[:, 2] = 0.0
    label_colors[:, 3] = 1.0
    for ii in range(0, len(unique_parcels_list)):
        if unique_parcels_list[ii].endswith('LH'):
            label_colors[ii, 0] = 0.0
            label_colors[ii, 1] = 1.0
            label_colors[ii, 2] = 0.0
        elif unique_parcels_list[ii].endswith('RH'):
            label_colors[ii, 0] = 0.0
            label_colors[ii, 1] = 0.0
            label_colors[ii, 2] = 1.0


    if flag_show_specific_parcels:
        for ii in range(0, len(parcels_to_show)):
            idx1 = [i for i, x in enumerate(unique_parcels_list) if x == parcels_to_show[ii]]
            idx2 = np.where(np.abs(foo1[:,idx1])>0)[0]
            idx2 = idx2[ np.where(idx2 != idx1[0])[0] ]# remove the diagonal
            if ii == 0:
                if flag_show_tstat:
                    foo2 = foo[idx2,idx1]
                else:
                    foo2 = foo1[idx2,idx1]
                indices1 = idx1 * np.ones((len(idx2))).astype(int)
                indices2 = idx2
            else:
                if flag_show_tstat:
                    foo2 = np.concatenate((foo2, foo[idx2,idx1]), axis=0)
                else:
                    foo2 = np.concatenate((foo2, foo1[idx2,idx1]), axis=0)
                indices1 = np.concatenate((indices1, idx1 * np.ones((len(idx2))).astype(int) ), axis=0)
                indices2 = np.concatenate((indices2, idx2), axis=0)
        indices = np.array([indices1, indices2])
    else:
        if flag_show_tstat:
            foo2 = foo
        else:
            foo2 = foo1


    # Plot the graph using node colors from the FreeSurfer parcellation. We only
    # show the 300 strongest connections.
    fig, ax = p.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
    if flag_show_specific_parcels:
        plot_connectivity_circle(
            foo2,
            unique_parcels_list,
            indices=indices,
            n_lines=n_lines,
            node_angles=node_angles,
            node_colors=label_colors,
        #    title="All-to-All Connectivity left-Auditory " "Condition (PLI)",
            ax=ax,
            colormap=colormap,
            vmin=vminmax[0],
            vmax=vminmax[1],
        )
    else:
        plot_connectivity_circle(
            foo2,
            unique_parcels_list,
            n_lines=n_lines,
            node_angles=node_angles,
            node_colors=label_colors,
        #    title="All-to-All Connectivity left-Auditory " "Condition (PLI)",
            ax=ax,
            colormap=colormap,
            vmin=vminmax[0],
            vmax=vminmax[1],
        )
    fig.tight_layout()
