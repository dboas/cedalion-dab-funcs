import cedalion
import cedalion.datasets as datasets
import cedalion.imagereco.forward_model as fw
import cedalion.io as io
import cedalion.nirs as nirs
import xarray as xr
from cedalion import units
import cedalion.dataclasses as cdc 
import numpy as np
import os.path
import pickle
from cedalion.imagereco.solver import pseudo_inverse_stacked
import cedalion.xrutils as xrutils

import matplotlib.pyplot as p
import pyvista as pv
from matplotlib.colors import ListedColormap

import gzip

import sys

import module_spatial_basis_funs_ced as sbf 
import pdb

#%% DATA LOADING

def load_head_model(head_model='ICBM152', with_parcels=True):
    
    if head_model == 'ICBM152':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_icbm152_segmentation()
        if with_parcels:
            PARCEL_DIR = datasets.get_icbm152_parcel_file()
        else :
            PARCEL_DIR = None
            
    elif head_model == 'colin27':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_colin27_segmentation()()
        if with_parcels:
            PARCEL_DIR = datasets.get_colin27_parcel_file()
        else :
            PARCEL_DIR = None
            
    masks, t_ijk2ras = io.read_segmentation_masks(SEG_DATADIR, mask_files)

    
    head = fw.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=SEG_DATADIR,
        mask_files = mask_files,
        brain_surface_file= os.path.join(SEG_DATADIR, "mask_brain.obj"),
        scalp_surface_file= os.path.join(SEG_DATADIR, "mask_scalp.obj"),
        landmarks_ras_file=landmarks_file,
        smoothing=0.5,
        fill_holes=True,
        parcel_file=PARCEL_DIR
    ) 
    head.scalp.units = units.mm
    # head.scalp.vertices = head.scalp.vertices * units.mm
    head.brain.units = units.mm
    # head.brain.vertices = head.brain.vertices * units.mm
    
    return head, PARCEL_DIR


def load_probe(probe_path, snirf_name ='fullhead_56x144_System2.snirf', head_model='ICBM152'):
        
    with open(os.path.join(probe_path, 'fw',  head_model, 'Adot.pkl'), 'rb') as f:
        Adot = pickle.load(f)
        
    recordings = io.read_snirf(probe_path + snirf_name)
    rec = recordings[0]
    geo3d = rec.geo3d
    amp = rec['amp']
    meas_list = rec._measurement_lists['amp']

    return Adot, meas_list, geo3d, amp



#%% MATRIX CALCULATIONS

def get_Adot_scaled(Adot, wavelengths, BRAIN_ONLY=False):
    
    if BRAIN_ONLY:
        Adot = Adot[:, Adot.is_brain.values, :] 

    nchannel = Adot.shape[0]
    nvertices = Adot.shape[1]
    E = nirs.get_extinction_coefficients('prahl', Adot.wavelength)

    A = np.zeros((2 * nchannel, 2 * nvertices))
    wl1 = wavelengths[0]
    wl2 = wavelengths[1]
    A[:nchannel, :nvertices] = E.sel(chromo="HbO", wavelength=wl1).values * Adot.sel(wavelength=wl1) # noqa: E501
    A[:nchannel, nvertices:] = E.sel(chromo="HbR", wavelength=wl1).values * Adot.sel(wavelength=wl1) # noqa: E501
    A[nchannel:, :nvertices] = E.sel(chromo="HbO", wavelength=wl2).values * Adot.sel(wavelength=wl2) # noqa: E501
    A[nchannel:, nvertices:] = E.sel(chromo="HbR", wavelength=wl2).values * Adot.sel(wavelength=wl2) # noqa: E501

    A = xr.DataArray(A, dims=("measurement", "flat_vertex"))
    A = A.assign_coords({"parcel" : ("flat_vertex", np.concatenate((Adot.coords['parcel'].values, Adot.coords['parcel'].values))),
                         "is_brain" : ("flat_vertex", np.concatenate((Adot.coords['is_brain'].values, Adot.coords['is_brain'].values)))})
    
    return A

def calculate_W(A, alpha_meas=0.1, alpha_spatial=0.01, BRAIN_ONLY = False, DIRECT=True, C_meas_flag=False, C_meas=None, D=None, F=None):
    
    
    if DIRECT:
        C_meas = np.diag(C_meas)  # !!! this is not in other script
        W_xr, D, F = _calculate_W_direct(A, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial, 
                                        BRAIN_ONLY=BRAIN_ONLY, 
                                        C_meas_flag=C_meas_flag, C_meas=C_meas, D=D, F=F)
    else:
        W_xr, D, F = _calculate_W_indirect(A, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,
                                          BRAIN_ONLY=BRAIN_ONLY, 
                                          C_meas_flag=C_meas_flag, C_meas=C_meas, D=D, F=F)
        
    return W_xr, D, F

def _calculate_W_direct(A, alpha_meas=0.1, alpha_spatial=0.01, BRAIN_ONLY=False, 
                       C_meas_flag=False, C_meas=None, D=None, F=None):
    
    A_coords = A.coords
    A = A.pint.dequantify().values
                
    if BRAIN_ONLY:
    
        B = A@A.T
        b = max(np.diag(B))
        W = A.T @ np.linalg.pinv(B + (alpha_meas * b.values * np.eye(np.shape(B)[0])))
    
    else:
        if D is None and F is None:
            B = np.sum((A ** 2), axis=0)
            b = B.max()
            
            # GET A_HAT
            lambda_spatial = alpha_spatial * b
            
            L = np.sqrt(B + lambda_spatial)
            Linv = 1/L
            # Linv = np.diag(Linv)
            
            A_hat = A * Linv
            
            #% GET W
            F = A_hat @ A_hat.T
    
            D = Linv[:, np.newaxis]**2 * A.T
        else:
            D = D.values
            F = F.values
            
        lambda_meas = alpha_meas * max(np.diag(F))
        
        if C_meas_flag:  
            # !!! np.diag(Cmeas) here                                                             
            W = D @ np.linalg.inv(F  + lambda_meas * C_meas )
        else:
            W = D @ np.linalg.inv(F  + lambda_meas * np.eye(A.shape[0]) )
        
        W_xr = xr.DataArray(W, dims=("flat_vertex", "measurement"))
        D_xr = xr.DataArray(D, dims=("flat_vertex", "measurement"))

        if 'parcel' in A_coords:
            W_xr = W_xr.assign_coords({"parcel" : ("flat_vertex", A_coords['parcel'].values)})
            D_xr = D_xr.assign_coords({"parcel" : ("flat_vertex", A_coords['parcel'].values)})
        if 'is_brain' in A_coords:
            W_xr = W_xr.assign_coords({"is_brain": ("flat_vertex", A_coords['is_brain'].values)}) 
            D_xr = D_xr.assign_coords({"is_brain": ("flat_vertex", A_coords['is_brain'].values)})

        
        F_xr = xr.DataArray(F, dims=("measurement1", "measurement2"))

    return W_xr, D_xr, F_xr

def _calculate_W_indirect(A, alpha_meas=0.1, alpha_spatial=0.01, BRAIN_ONLY=False, 
                       C_meas_flag=False, C_meas=None, D=None, F=None):
    
    
    W = []
    D_lst = []
    F_lst = []
    for wavelength in A.wavelength:
        
        if C_meas_flag:
            C_meas_wl = C_meas.sel(wavelength=wavelength)
            C_meas_wl = np.diag(C_meas_wl)
        else:
            C_meas_wl = None
            
        A_wl = A.sel(wavelength=wavelength)
        if D is None and F is None:
            W_wl, D_wl, F_wl = _calculate_W_direct(A_wl, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,BRAIN_ONLY=BRAIN_ONLY,
                                      C_meas_flag=C_meas_flag, C_meas=C_meas_wl, D=D, F=F)
        else:
            W_wl, D_wl, F_wl = _calculate_W_direct(A_wl, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,BRAIN_ONLY=BRAIN_ONLY,
                                      C_meas_flag=C_meas_flag, C_meas=C_meas_wl, 
                                      D=D.sel(wavelength=wavelength), F=F.sel(wavelength=wavelength))
        
        
        W.append(W_wl)
        D_lst.append(D_wl)
        F_lst.append(F_wl)
    
    W_xr = xr.concat(W, dim='wavelength')
    W_xr = W_xr.assign_coords(wavelength=A.wavelength)
    
    D = xr.concat(D_lst, dim='wavelength')
    D = D.assign_coords(wavelength=A.wavelength)

    F = xr.concat(F_lst, dim='wavelength')
    F = F.assign_coords(wavelength=A.wavelength)

    return W_xr, D, F

#%% do image recon
def _get_image_brain_scalp_direct(y, W, A, SB=False, G=None):
    
    try:
        X = W.values @ y.values
    except:
        X = W @ y

    split = len(X)//2

    if SB:
        X = sbf.go_from_kernel_space_to_image_space_direct(X, G)
            
    else:
        if len(X.shape) == 1:
            X = X.reshape([2, split]).T
        else:
            X = X.reshape([2, split, X.shape[1]])
            X = X.transpose(1,2,0)
    
    if len(X.shape) == 2:
        X = xr.DataArray(X, 
                         dims = ('vertex', 'chromo'),
                         coords = {'chromo': ['HbO', 'HbR']}
                         )
    else:
        if 'time' in y.dims:
            t = y.time
            t_name = 'time'
        elif 'reltime' in y.dims:
            t = y.reltime
            t_name = 'reltime'
        X = xr.DataArray(X, 
                         dims = ('vertex', 'chromo', t_name),
                         coords = {'chromo': ['HbO', 'HbR'],
                                   t_name: t},
                         )
    if 'parcel' in A.coords:
        X = X.assign_coords({"parcel" : ("vertex", A.coords['parcel'].values)})
                              
    if 'is_brain' in A.coords:
        X = X.assign_coords({"is_brain": ("vertex", A.coords['is_brain'].values)}) 

    return X


def _get_image_brain_scalp_indirect(y, W, A, SB=False, G=None):
    
     split = len(y)//2
                  
     W_indirect_wl0 = W.isel(wavelength=0)
     W_indirect_wl1 = W.isel(wavelength=1)
     
     y_wl0 = y[:split]
     y_wl1 = y[split:]

     try:
        X_wl0 = W_indirect_wl0.values @ y_wl0.values
        X_wl1 = W_indirect_wl1.values @ y_wl1.values
     except:
        X_wl0 = W_indirect_wl0 @ y_wl0
        X_wl1 = W_indirect_wl1 @ y_wl1   
        
             
     if SB:
         X_wl0 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl0, G)
         X_wl1 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl1, G)
         
         
     
     if len(X_wl0.shape) == 1:
         X_od = np.vstack([X_wl0, X_wl1]).T
         X_od = xr.DataArray(X_od, 
                          dims = ('vertex', 'wavelength'),
                          coords = {'wavelength': W.wavelength}
                          )
     else:
        X_od = np.stack([X_wl0, X_wl1], axis=2)
        X_od = X_od.transpose(0,2,1)  # make it (vertex, wavelength, time)
        if 'time' in y.dims:
            t = y.time
            t_name = 'time'
        elif 'reltime' in y.dims:
            t = y.reltime
            t_name = 'reltime'
        X_od = xr.DataArray(X_od, 
                        dims = ('vertex', 'wavelength', t_name),
                        coords = {'wavelength': W.wavelength,
                                t_name: t},
                        )

     if 'parcel' in A.coords:
        X_od = X_od.assign_coords({"parcel" : ("vertex", A.coords['parcel'].values)})
                              
     if 'is_brain' in A.coords:
        X_od = X_od.assign_coords({"is_brain": ("vertex", A.coords['is_brain'].values)}) 


     # convert to concentration 
     E = nirs.get_extinction_coefficients('prahl', W.wavelength)
     einv = xrutils.pinv(E) #FIXME check units

     X = xr.dot(einv, X_od/units.mm, dims=["wavelength"])
     
     return X

def do_image_recon(od, head, Adot, C_meas_flag, C_meas, wavelength, BRAIN_ONLY, DIRECT,
                   SB, cfg_sbf, alpha_spatial, alpha_meas, D, F, G ):
    
    
    if DIRECT:
        Adot_stacked = get_Adot_scaled(Adot, wavelength)
        
        if SB:
            if G is None:
                M = sbf.get_sensitivity_mask(Adot, cfg_sbf['mask_threshold'], 1)
                G = sbf.get_G_matrix(head, M, threshold_brain=cfg_sbf['threshold_brain'],
                                             threshold_scalp = cfg_sbf['threshold_scalp'],
                                             sigma_brain=cfg_sbf['sigma_brain'],
                                             sigma_scalp=cfg_sbf['sigma_scalp'])
                
            H_stacked = sbf.get_H_stacked(G, Adot_stacked)
            Adot_stacked = H_stacked.copy()
            
        W, D, F = calculate_W(Adot_stacked, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,
                              C_meas_flag=C_meas_flag, C_meas=C_meas, DIRECT=DIRECT, BRAIN_ONLY=BRAIN_ONLY, D=D, F=F)
       
        X = _get_image_brain_scalp_direct(od, W, Adot, SB=SB, G=G)
    
            
    else:
        if SB:
            if G is None:
                M = sbf.get_sensitivity_mask(Adot, cfg_sbf['mask_threshold'], 1)
                G = sbf.get_G_matrix(head, M, threshold_brain=cfg_sbf['threshold_brain'],
                                             threshold_scalp = cfg_sbf['threshold_scalp'],
                                             sigma_brain=cfg_sbf['sigma_brain'],
                                             sigma_scalp=cfg_sbf['sigma_scalp'])
            
            H = sbf.get_H(G, Adot)
            Adot = H.copy()
            
        W, D, F = calculate_W(Adot, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,
                              C_meas_flag=C_meas_flag, C_meas=C_meas, DIRECT=DIRECT, BRAIN_ONLY=BRAIN_ONLY, D=D, F=F)
        X = _get_image_brain_scalp_indirect(od, W, Adot, SB=SB, G=G)

      
            
    return X, W, D, F, G
    

def get_image_noise(C_meas, X, W, SB=False, DIRECT=True, G=None):
    
    if DIRECT:
        cov_img_tmp = W *np.sqrt(C_meas.values) # W is pseudo inverse  --- diagonal (faster than W C W.T)
        cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
        
        if SB:
            cov_img_diag = sbf.go_from_kernel_space_to_image_space_direct(cov_img_diag, G)
        else:
            split = len(cov_img_diag)//2
            cov_img_diag =  np.reshape( cov_img_diag, (2,split) ).T 
        

    else:
        cov_img_lst = []

        for wavelength in W.wavelength:
            W_wl = W.sel(wavelength=wavelength)
            C_wl = C_meas.sel(wavelength=wavelength)
            
            cov_img_tmp = W_wl * np.sqrt(C_wl.values) # get diag of image covariance
            cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
            
            if SB:
                cov_img_diag = sbf.go_from_kernel_space_to_image_space_indirect(cov_img_diag, G)
            
            cov_img_lst.append(cov_img_diag)
            
        cov_img_diag =  np.vstack(cov_img_lst) 
        E = nirs.get_extinction_coefficients('prahl', W.wavelength)
        einv = xrutils.pinv(E)

        cov_img_diag = einv.values**2 @ cov_img_diag

    if not hasattr(X, 'time') and not hasattr(X, 'reltime'):    
        noise = X.copy()
    else:
        if hasattr(X, 'time'):
            noise = X.isel(time=0, drop=True)
        elif hasattr(X, 'reltime'):
            noise = X.isel(reltime=0, drop=True)
        else:
            raise ValueError("X must have either 'time' or 'reltime' dimension")
    noise.values = cov_img_diag

    return noise



#%%  DB funcs

#
# load in head model and sensitivity profile 
#
def load_Adot( path_to_dataset = None, head_model = 'ICBM152' ):

    # Load the sensitivity profile
    
    file_path = os.path.join(path_to_dataset, head_model, 'Adot.pkl')
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




def do_image_recon_DB( hrf_od = None, head = None, Adot = None, C_meas = None, wavelength = [760,850], 
                   cfg_img_recon = None, trial_type_img = None, save_path = None, W = None, C = None, D = None  ):
    
    cfg_sb = cfg_img_recon['cfg_sb']
    
    print( 'Starting Image Reconstruction')

    #
    # prune the data and sensitivity profile
    #
    #pdb.set_trace()
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
        
        pdb.set_trace()
        # !!! fixed the assumption that hrf_od was always a time a series.... is this ok tho
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

    A = xr.DataArray(A, dims=("measurement", "flat_vertex"))
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
    
    alpha_spatial = cfg_img_recon['alpha_spatial']
    if not cfg_img_recon['BRAIN_ONLY'] and W is None and C is None and D is None:

        print( f'   Doing spatial regularization with alpha_spatial = {alpha_spatial}')
        # GET A_HAT
        B = np.sum((A ** 2), axis=0)
        b = B.max()

        lambda_spatial = alpha_spatial * b
        
        L = np.sqrt(B + lambda_spatial)
        Linv = 1/L
        # Linv = np.diag(Linv)

        # A_hat = A @ Linv
        A_hat = A * Linv
        
        #% GET W
        F = A_hat @ A_hat.T
        f = max(np.diag(F)) 
        print(f'   f = {f}')
        
        C = F #A @ (Linv ** 2) @ A.T
        D = Linv[:, np.newaxis]**2 * A.T
    else:
        f = max(np.diag(C))

    alpha_meas = cfg_img_recon['alpha_meas']
    print(f'   Doing image recon with alpha_meas = {alpha_meas}')
    if cfg_img_recon['BRAIN_ONLY'] and W is None:
        Adot_stacked = xr.DataArray(A, dims=("measurement", "flat_vertex"))
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
                                                'time': od_mag_pruned.time.values * units.s,
                                                'samples': ("time", np.arange(len(od_mag_pruned.time.values)))},
                                        )
                    X = X.set_xindex("parcel")
                    X.time.attrs['units'] = 's'

            
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
    filepath = os.path.join(save_path, f'{X_matrix_name}_{trial_type_img.values}_cov_alpha_spatial_{cfg_img_recon["alpha_spatial"]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas"]:.0e}.pkl.gz')
    print(f'   Saving to {X_matrix_name}_{trial_type_img.values}_cov_alpha_spatial_{cfg_img_recon["alpha_spatial"]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas"]:.0e}.pkl.gz \n')
    file = gzip.GzipFile(filepath, 'wb')
    file.write(pickle.dumps([X_matrix, cfg_img_recon["alpha_meas"], cfg_img_recon["alpha_spatial"]]))
    file.close()  
    

#%%
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
        #clim=(-0.6*X_hbo_brain.max(), 0.7*X_hbo_brain.max())
        clim=(-X_hbo_brain.max(), X_hbo_brain.max())
        p0.add_mesh(surf, scalars=X_hbo_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=True )
        p0.camera_position = pos

    elif flag_hbx == 'hbr_brain': # hbr brain
        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
        surf = pv.wrap(surf.mesh)   
        #clim=(-0.6*X_hbr_brain.max(), 0.7*X_hbr_brain.max())
        clim=(-X_hbr_brain.max(), X_hbr_brain.max())
        p0.add_mesh(surf, scalars=X_hbr_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=True )
        p0.camera_position = pos

    elif flag_hbx == 'hbo_scalp': # hbo scalp
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
        surf = pv.wrap(surf.mesh)
        #clim=(-0.6*X_hbo_brain.max(), 0.7*X_hbo_brain.max())
        clim=(-X_hbo_brain.max(), X_hbo_brain.max())
        p0.add_mesh(surf, scalars=X_hbo_scalp, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=True )
        p0.camera_position = pos

    elif flag_hbx == 'hbr_scalp': # hbr scalp
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
        surf = pv.wrap(surf.mesh)
        #clim=(-0.6*X_hbr_brain.max(), 0.7*X_hbr_brain.max())
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



def get_Adot_parcels( Adot = None ):

    # reduce parcels to 17 network parcels plus 'Background+Freesurfer...'
    # get the unique 17 network parcels and remove non-brain parcels
    unique_parcels = Adot.groupby('parcel').sum('vertex').parcel
    unique_parcels = unique_parcels.sel(parcel=unique_parcels.parcel != 'scalp')
    unique_parcels = unique_parcels.sel(parcel=unique_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
    unique_parcels = unique_parcels.sel(parcel=unique_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')


    parcel_list = []
    for parcel in unique_parcels.values:
        parcel_list.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )
    unique_parcels_lev1 = np.unique(parcel_list)

    parcel_list_lev2 = []
    for parcel in unique_parcels.values:
        if parcel.split('_')[1].isdigit():
            parcel_list_lev2.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )
        else:
            parcel_list_lev2.append( parcel.split('_')[0] + '_' + parcel.split('_')[1] + '_' + parcel.split('_')[-1] )
    unique_parcels_lev2 = np.unique(parcel_list_lev2)


    Adot_parcels = Adot.isel(wavelength=0).groupby('parcel').sum('vertex')
    Adot_parcels = Adot_parcels.sel(parcel=Adot_parcels.parcel != 'scalp')
    Adot_parcels = Adot_parcels.sel(parcel=Adot_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
    Adot_parcels = Adot_parcels.sel(parcel=Adot_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')


    Adot_parcels_lev1 = np.zeros( (Adot.shape[0], len(unique_parcels_lev1)) )
    for ii in range( 0, len(unique_parcels_lev1) ):
        idx1 = [i for i, x in enumerate(parcel_list) if x == unique_parcels_lev1.tolist()[ii]]
        Adot_parcels_lev1[:,ii] = np.sum(Adot_parcels.isel(parcel=idx1).values, axis=1)

    Adot_parcels_lev1_xr = xr.DataArray(
        Adot_parcels_lev1,
        dims=['channel','parcel'],
        coords={'channel': Adot.channel, 'parcel': unique_parcels_lev1}
    )


    Adot_parcels_lev2 = np.zeros( (Adot.shape[0], len(unique_parcels_lev2)) )
    for ii in range( 0, len(unique_parcels_lev2) ):
        idx1 = [i for i, x in enumerate(parcel_list_lev2) if x == unique_parcels_lev2.tolist()[ii]]
        Adot_parcels_lev2[:,ii] = np.sum(Adot_parcels.isel(parcel=idx1).values,axis=1)

    Adot_parcels_lev2_xr = xr.DataArray(
        Adot_parcels_lev2,
        dims=['channel','parcel'],
        coords={'channel': Adot.channel, 'parcel': unique_parcels_lev2}
    )

    return Adot_parcels_lev1_xr, Adot_parcels_lev2_xr, unique_parcels_lev1, unique_parcels_lev2