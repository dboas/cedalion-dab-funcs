import cedalion
import cedalion_parcellation.datasets as datasets
import cedalion_parcellation.imagereco.forward_model as fw
import cedalion.io as io
import cedalion.nirs as nirs
import xarray as xr
from cedalion import units
import cedalion.dataclasses as cdc 
import numpy as np
import os.path
import pickle
from cedalion.imagereco.solver import pseudo_inverse_stacked

import matplotlib.pyplot as p
import pyvista as pv
from matplotlib.colors import ListedColormap

import gzip

import tkinter as tk
from tkinter import filedialog

import sys

import spatial_basis_funs_ced as sbf 



#
# load in head model and sensitivity profile 
#
def load_Adot( path_to_dataset = None, head_model = 'ICBM152' ):

    # Load the sensitivity profile
    # with open(os.path.join(path_to_dataset, 'derivatives', 'fw',  head_model, 'Adot_wParcels.pkl'), 'rb') as f:
    #     Adot = pickle.load(f)
    
    file_path = os.path.join(path_to_dataset, head_model, 'Adot_wParcels.pkl')
    with open(file_path, 'rb') as f:
        Adot = pickle.load(f) 
        
    #% LOAD HEAD MODEL 
    if head_model == 'ICBM152':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_icbm152_segmentation()
    elif head_model == 'colin27':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_colin27_segmentation()()

    masks, t_ijk2ras = io.read_segmentation_masks(SEG_DATADIR, mask_files)

    head = fw.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=SEG_DATADIR,
        mask_files = mask_files,
        brain_surface_file= os.path.join(SEG_DATADIR, "mask_brain.obj"),
        scalp_surface_file= os.path.join(SEG_DATADIR, "mask_scalp.obj"),
        landmarks_ras_file=landmarks_file,
        smoothing=0.5,
        fill_holes=True,
    ) 
    head.scalp.units = units.mm
    head.brain.units = units.mm

    return Adot, head




def do_image_recon( hrf_od = None, head = None, Adot = None, C_meas = None, wavelength = [760,850], cfg_img_recon = None, trial_type_img = None, save_path = None, W = None, C = None, D = None  ):
    
    cfg_sb = cfg_img_recon['cfg_sb']
    
    print( 'Starting Image Reconstruction')

    #
    # prune the data and sensitivity profile
    #

    # FIXME: I am checking both wavelengths since I have to prune both if one is null to get consistency between A_pruned and od_mag_pruned
    #        We don't have to technically do this, but it is easier. The alternative requires have Adot_pruned for each wavelengths and checking rest of code
    wav = hrf_od.wavelength.values
    if len(hrf_od.dims) == 2: # not a time series else it is a time series
        pruning_mask = ~(hrf_od.sel(wavelength=wav[0]).isnull() | hrf_od.sel(wavelength=wav[1]).isnull())
    elif 'reltime' in hrf_od.dims:
        pruning_mask = ~(hrf_od.sel(wavelength=wav[0], reltime=0).isnull() | hrf_od.sel(wavelength=wav[1], reltime=0).isnull())
    else:
        pruning_mask = ~(hrf_od.sel(wavelength=wav[0]).mean('time').isnull() | hrf_od.sel(wavelength=wav[1]).mean('time').isnull())

    if C_meas is None:
        if cfg_img_recon['BRAIN_ONLY']:
            Adot_pruned = Adot[pruning_mask.values, Adot.is_brain.values, :] 
        else:
            Adot_pruned = Adot[pruning_mask.values, :, :]
            
        # !!! fixed the assumtion that hrf_od was always a time a series.... is this ok tho
        if len(hrf_od.dims) == 2: # if nto a time series
            od_mag_pruned = hrf_od[:,pruning_mask.values].stack(measurement=('channel', 'wavelength')).sortby('wavelength')
        else:   # it is a time series
            od_mag_pruned = hrf_od[:,pruning_mask.values,:].stack(measurement=('channel', 'wavelength')).sortby('wavelength')   
        
        
        # od_mag = hrf_od.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
        # od_mag_pruned = od_mag.dropna('measurement')
        
    else: # don't prune anything if C_meas is not None as we use C_meas to essentially prune
          # but we make sure the corresponding elements of C_meas are set to BAD values
        if cfg_img_recon['BRAIN_ONLY']:
            Adot_pruned = Adot[:, Adot.is_brain.values, :] 
        else:
            Adot_pruned = Adot
            
        od_mag_pruned = hrf_od.stack(measurement=('channel', 'wavelength')).sortby('wavelength')    
        n_chs = hrf_od.channel.size
        if od_mag_pruned.dims == 2:
            od_mag_pruned[:,np.where(~pruning_mask.values)[0]] = 0
            od_mag_pruned[:,np.where(~pruning_mask.values)[0]+n_chs] = 0
        else:
            od_mag_pruned[np.where(~pruning_mask.values)[0]] = 0
            od_mag_pruned[np.where(~pruning_mask.values)[0]+n_chs] = 0

        mse_val_for_bad_data = 1e1  # FIXME: this should be passed here nad to group_avg
        # FIXME: I assume C_meas is 1D. If it is 2D then I need to do this to the columns and rows
        C_meas[np.where(~pruning_mask.values)[0]] = mse_val_for_bad_data
        C_meas[np.where(~pruning_mask.values)[0] + n_chs] = mse_val_for_bad_data

    #
    # create the sensitivity matrix for HbO and HbR
    #
    ec = nirs.get_extinction_coefficients("prahl", wavelength)

    nchannel = Adot_pruned.shape[0]
    nvertices = Adot_pruned.shape[1]
    n_brain = sum(Adot.is_brain.values)

    A = np.zeros((2 * nchannel, 2 * nvertices))

    wl1, wl2 = wavelength
    A[:nchannel, :nvertices] = ec.sel(chromo="HbO", wavelength=wl1).values * Adot_pruned.sel(wavelength=wl1) # noqa: E501
    A[:nchannel, nvertices:] = ec.sel(chromo="HbR", wavelength=wl1).values * Adot_pruned.sel(wavelength=wl1) # noqa: E501
    A[nchannel:, :nvertices] = ec.sel(chromo="HbO", wavelength=wl2).values * Adot_pruned.sel(wavelength=wl2) # noqa: E501
    A[nchannel:, nvertices:] = ec.sel(chromo="HbR", wavelength=wl2).values * Adot_pruned.sel(wavelength=wl2) # noqa: E501

    A = xr.DataArray(A, dims=("flat_channel", "flat_vertex"))
    A = A.assign_coords({"parcel" : ("flat_vertex", np.concatenate((Adot_pruned.coords['parcel'].values, Adot_pruned.coords['parcel'].values)))})


    #
    # spatial basis functions
    #
    if cfg_img_recon['SB']:
        M = sbf.get_sensitivity_mask(Adot_pruned, cfg_sb['mask_threshold'])

        G = sbf.get_G_matrix(head,     # spatial basis functions
                                M,
                                cfg_sb['threshold_brain'], 
                                cfg_sb['threshold_scalp'], 
                                cfg_sb['sigma_brain'], 
                                cfg_sb['sigma_scalp']
                                )
        
        nbrain = Adot_pruned.is_brain.sum().values
        nscalp = Adot.shape[1] - nbrain 
        
        nkernels_brain = G['G_brain'].kernel.shape[0]
        nkernels_scalp = G['G_scalp'].kernel.shape[0]

        nkernels = nkernels_brain + nkernels_scalp

        H = np.zeros((2 * nchannel, 2 * nkernels))

        A_hbo_brain = A[:, :nbrain]
        A_hbr_brain = A[:, nbrain+nscalp:2*nbrain+nscalp]
        
        A_hbo_scalp = A[:, nbrain:nscalp+nbrain]
        A_hbr_scalp = A[:, 2*nbrain+nscalp:]
        
        H[:,:nkernels_brain] = A_hbo_brain.values @ G['G_brain'].values.T
        H[:, nkernels_brain+nkernels_scalp:2*nkernels_brain+nkernels_scalp] = A_hbr_brain.values @ G['G_brain'].values.T
        
        H[:,nkernels_brain:nkernels_brain+nkernels_scalp] = A_hbo_scalp.values @ G['G_scalp'].values.T   # H projects the sensitivity matrix into the spatial basis space
        H[:,2*nkernels_brain+nkernels_scalp:] = A_hbr_scalp.values @ G['G_scalp'].values.T

        H = xr.DataArray(H, dims=("channel", "kernel"))

        A = H.copy()


    #
    # Do the Image Reconstruction
    #

    # Ensure A is a numpy array
    A = np.array(A)

    for alpha_spatial in cfg_img_recon['alpha_spatial_list']:
                        
        if not cfg_img_recon['BRAIN_ONLY'] and W is None and C is None and D is None:

            print( f'   Doing spatial regularization with alpha_spatial = {alpha_spatial}')
            # GET A_HAT
            B = np.sum((A ** 2), axis=0)
            b = B.max()

            lambda_spatial = alpha_spatial * b
            
            L = np.sqrt(B + lambda_spatial)
            Linv = 1/L
            Linv = np.diag(Linv)
            
            A_hat = A @ Linv
            
            #% GET W
            F = A_hat @ A_hat.T
            f = max(np.diag(F)) 
            print(f'   f = {f}')
            
            C = F #A @ (Linv ** 2) @ A.T
            D = Linv**2 @ A.T
        else:
            f = max(np.diag(C))
            
        for alpha_meas in cfg_img_recon['alpha_meas_list']:
            
            print(f'   Doing image recon with alpha_meas = {alpha_meas}')
            

            if cfg_img_recon['BRAIN_ONLY'] and W is None:
                Adot_stacked = xr.DataArray(A, dims=("flat_channel", "flat_vertex"))
                W = pseudo_inverse_stacked(Adot_stacked, alpha=alpha_meas)
                W = W.assign_coords({"chromo" : ("flat_vertex", ["HbO"]*nvertices  + ["HbR"]* nvertices)})
                W = W.set_xindex("chromo")
            elif W is None:
                if C_meas is None:
                    lambda_meas = alpha_meas * f 
                    W = D @ np.linalg.inv(C  + lambda_meas * np.eye(A.shape[0]) )
                else:
                    lambda_meas = alpha_meas * f
                    # check if C_meas has 2 dimensions
                    if len(C_meas.shape) == 2:
                        W = D @ np.linalg.inv(C + lambda_meas * C_meas)
                    else:
                        W = D @ np.linalg.inv(C + lambda_meas * np.diag(C_meas))
            nvertices = W.shape[0]//2
        
            #% GENERATE IMAGES FOR DIFFERENT IMAGE PARAMETERS AND ALSO FOR THE FULL TIMESERIES
            X = W @ od_mag_pruned.values.T
            
            split = len(X)//2

            if cfg_img_recon['BRAIN_ONLY']:
                if len(hrf_od.dims) == 2: # not a time series else it is a time series
                    X = xr.DataArray(X, 
                                    dims = ('vertex'),
                                    coords = {'parcel':("vertex",np.concatenate((Adot.coords['parcel'].values, Adot.coords['parcel'].values)))},
                                    )
                else:
                    # FIXME: check if it is 'reltime' or 'time' and assign appropriately
                    if 'reltime' in hrf_od.dims:
                        X = xr.DataArray(X, 
                                        dims = ('vertex', 'reltime'),
                                        coords = {'parcel':("vertex",np.concatenate((Adot.coords['parcel'].values, Adot.coords['parcel'].values))),
                                                'reltime': od_mag_pruned.reltime.values},
                                        )
                    else:
                        X = xr.DataArray(X, 
                                        dims = ('vertex', 'time'),
                                        coords = {'parcel':("vertex",np.concatenate((Adot.coords['parcel'].values, Adot.coords['parcel'].values))),
                                                'time': od_mag_pruned.time.values},
                                        )
                
            else:
                if cfg_img_recon['SB']:
                    X_hbo = X[:split]
                    X_hbr = X[split:]
                    sb_X_brain_hbo = X_hbo[:nkernels_brain]
                    sb_X_brain_hbr = X_hbr[:nkernels_brain]
                    
                    sb_X_scalp_hbo = X_hbo[nkernels_brain:]
                    sb_X_scalp_hbr = X_hbr[nkernels_brain:]
                    
                    #% PROJECT BACK TO SURFACE SPACE 
                    X_hbo_brain = G['G_brain'].values.T @ sb_X_brain_hbo
                    X_hbo_scalp = G['G_scalp'].values.T @ sb_X_scalp_hbo
                    
                    X_hbr_brain = G['G_brain'].values.T @ sb_X_brain_hbr
                    X_hbr_scalp = G['G_scalp'].values.T @ sb_X_scalp_hbr
                    
                    # concatenate them back together
                    if len(hrf_od.dims) == 2: # not a time series else it is a time series
                        X = np.stack([np.concatenate([X_hbo_brain, X_hbo_scalp]),np.concatenate([ X_hbr_brain, X_hbr_scalp])], axis=1)
                    else:
                        X = np.stack([np.vstack([X_hbo_brain, X_hbo_scalp]), np.vstack([X_hbr_brain, X_hbr_scalp])], axis =2)
                    
                else:
                    if len(hrf_od.dims) == 2: # not a time series else it is a time series
                        X = X.reshape([2, split]).T
                    else:
                        X = X.reshape([2, split, X.shape[1]])
                        X = X.transpose(1,2,0)
                    
                if len(hrf_od.dims) == 2: # not a time series else it is a time series
                    X = xr.DataArray(X, 
                                    dims = ('vertex', 'chromo'),
                                    coords = {'chromo': ['HbO', 'HbR'],
                                            'parcel': ('vertex',Adot.coords['parcel'].values),
                                            'is_brain':('vertex', Adot.coords['is_brain'].values)},
                                    )
                    X = X.set_xindex('parcel')
                elif 'reltime' in hrf_od.dims:
                    X = xr.DataArray(X,
                                        dims = ('vertex', 'reltime', 'chromo'),
                                        coords = {'chromo': ['HbO', 'HbR'],
                                                'parcel': ('vertex',Adot.coords['parcel'].values),
                                                'is_brain':('vertex', Adot.coords['is_brain'].values),
                                                'reltime': od_mag_pruned.reltime.values},
                                        )
                    X = X.set_xindex("parcel")
                else:
                    X = xr.DataArray(X,
                                        dims = ('vertex', 'time', 'chromo'),
                                        coords = {'chromo': ['HbO', 'HbR'],
                                                'parcel': ('vertex',Adot.coords['parcel'].values),
                                                'is_brain':('vertex', Adot.coords['is_brain'].values),
                                                'time': od_mag_pruned.time.values},
                                        )
                    X = X.set_xindex("parcel")

            
            # !!! SHOULD we also save W, C, D, C_meas?????
            # save the results
            if cfg_img_recon['flag_save_img_results']:
                if C_meas is None:
                    filepath = os.path.join(save_path, f'X_{trial_type_img.values}_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}.pkl.gz')
                    print(f'   Saving to X_{trial_type_img.values}_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}.pkl.gz \n')
                else:
                    filepath = os.path.join(save_path, f'X_{trial_type_img.values}_cov_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}.pkl.gz')
                    print(f'   Saving to X_{trial_type_img.values}_cov_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}.pkl.gz \n')
                file = gzip.GzipFile(filepath, 'wb')
                file.write(pickle.dumps([X, alpha_meas, alpha_spatial]))
                file.close()     

            # end loop over alpha_meas
        # end loop over alpha_spatial

    return X, W, C, D


def img_noise_tstat(X_grp, W, C_meas):
    ''' Calculate tstat and image noise of X_grp.
    
    Inputs:
        X_grp : image result of group average done in channel space
        W : pseudo inverse matrix
        Cmeas : variance (y_stderr_weighted**2)
    
    Outputs:
        X_noise : image noise
        X_tstat : iamge t-stat (i.e. CNR)
    '''
    
    # scale columns of W by y_stderr_weighted**2
    cov_img_tmp = W * np.sqrt(C_meas.values) # W is pseudo inverse  --- diagonal (faster than W C W.T)
    cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)

    nV = X_grp.shape[0]
    cov_img_diag = np.reshape( cov_img_diag, (2,nV) ).T

    # image noise
    X_noise = X_grp.copy()
    X_noise.values = np.sqrt(cov_img_diag)
    
    
    # image t-stat (i.e. CNR)
    X_tstat = X_grp / np.sqrt(cov_img_diag)

    X_tstat[ np.where(cov_img_diag[:,0]==0)[0], 0 ] = 0
    X_tstat[ np.where(cov_img_diag[:,1]==0)[0], 1 ] = 0
    
    return X_noise, X_tstat



def save_image_results(X_matrix, X_matrix_name, save_path, trial_type_img, cfg_img_recon):
    '''Save image result matrices.
    Inputs:
        X_matrix : resulat mat you wanna save (i.e. X_noise)
        X_matrix_name (str) : nam eof matric you are saving
        
    '''
    # !!! NOTE: only saves last elem in alpha lists... what if the list is > 1?
    filepath = os.path.join(save_path, f'{X_matrix_name}_{trial_type_img.values}_cov_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
    print(f'   Saving to {X_matrix_name}_{trial_type_img.values}_cov_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz \n')
    file = gzip.GzipFile(filepath, 'wb')
    file.write(pickle.dumps([X_matrix, cfg_img_recon["alpha_meas_list"][-1], cfg_img_recon["alpha_spatial_list"][-1]]))
    file.close()  
    


def plot_image_recon( X, head, shape, iax,clim=(0,1), flag_hbx='hbo_brain', view_position='superior', p0 = None, title_str = None, off_screen= True ):
    # pos_names = ['superior', 'left']

    #
    # Plot the results
    #
    cmap = p.get_cmap("jet", 256)
    new_cmap_colors = np.vstack((cmap(np.linspace(0, 1, 256))))
    custom_cmap = ListedColormap(new_cmap_colors)

    X_hbo_brain = X[X.is_brain.values, 0]
    X_hbr_brain = X[X.is_brain.values, 1]

    X_hbo_scalp = X[~X.is_brain.values, 0]
    X_hbr_scalp = X[~X.is_brain.values, 1]

    pos_names = ['superior', 'left', 'right', 'anterior', 'posterior','scale_bar']
    positions = [ 'xy',
        [(-400., 96., 130.),
        (96., 115., 165.),
        (0,0,1)],
        [(600, 96., 130.),
        (96., 115., 165.),
        (0,0,1)],
        [(100, 500, 200),
        (96., 115., 165.),
        (0,0,1)],
        [(100, -300, 300),
        (96., 115., 165.),
        (0,0,1)],
        [(100, -300, 300),
        (96., 115., 165.),
        (0,0,1)]
    ]
    #clim=(-X_hbo_brain.max(), X_hbo_brain.max())

    # get index of pos_names that matches view_position
    idx = [i for i, s in enumerate(pos_names) if view_position in s]

    pos = positions[idx[0]]

    if p0 is None:
        p0 = pv.Plotter(shape=(shape[0],shape[1]), window_size = [2000, 1500], off_screen=off_screen)
#        p.add_text(f"Group average with alpha_meas = {alpha_meas} and alpha_spatial = {alpha_spatial}", position='upper_left', font_size=12, viewport=True)

    p0.subplot(iax[0], iax[1])

    show_scalar_bar = False

    if flag_hbx == 'hbo_brain': # hbo brain 
        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
        surf = pv.wrap(surf.mesh)
        clim=(-X_hbo_brain.max(), X_hbo_brain.max())
        p0.add_mesh(surf, scalars=X_hbo_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=True )
        p0.camera_position = pos

    elif flag_hbx == 'hbr_brain': # hbr brain
        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
        surf = pv.wrap(surf.mesh)   
        clim=(-X_hbr_brain.max(), X_hbr_brain.max())
        p0.add_mesh(surf, scalars=X_hbr_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=True )
        p0.camera_position = pos

    elif flag_hbx == 'hbo_scalp': # hbo scalp
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
        surf = pv.wrap(surf.mesh)
        clim=(-X_hbo_brain.max(), X_hbo_brain.max())
        p0.add_mesh(surf, scalars=X_hbo_scalp, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=True )
        p0.camera_position = pos

    elif flag_hbx == 'hbr_scalp': # hbr scalp
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
        surf = pv.wrap(surf.mesh)
        clim=(-X_hbr_brain.max(), X_hbr_brain.max())
        p0.add_mesh(surf, scalars=X_hbr_scalp, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=True )
        p0.camera_position = pos

    if iax[0] == 1 and iax[1] == 1:
        p0.clear_actors()
        p0.add_scalar_bar(title=title_str, vertical=False, position_x=0.1, position_y=0.5,
                          height=0.1, width=0.8, fmt='%.1e',
                          label_font_size=24, title_font_size=32 )  # Add it separately
    else:
        p0.add_text(view_position, position='lower_left', font_size=10)

    # save pyvista figure
    # p0.screenshot( os.path.join(root_dir, 'derivatives', 'plots', f'IMG.png') )
    # p0.close()

    return p0

