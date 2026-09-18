#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:04:39 2024

@author: lauracarlton
"""

import numpy as np 
import xarray as xr
import pandas as pd
import scipy.sparse
from scipy.spatial import KDTree
import trimesh

import cedalion
import cedalion.dataclasses as cdc
import cedalion.dot.forward_model as cfm
from cedalion.geometry.registration import register_trans_rot_isoscale
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion.geometry.segmentation import surface_from_segmentation
from cedalion.dot.utils import map_segmentation_mask_to_surface

from cedalion.dot.tissue_properties import get_tissue_properties
from tqdm import tqdm 

#%% GETTING THE SPATIAL BASIS 

def get_sensitivity_mask(sensitivity: xr.DataArray, threshold: float = -2, wavelength_idx: int = 0):
    """Generate a mask indicating vertices with sensitivity above a certain threshols
 
    Args:
        sensitivity (xr.DataArray): Sensitivity matrix for each vertex and
            wavelength.
        threshold (float): threshold for sensitivity.
        wavelength_idx (int): wavelength over which to compute mask 
            if multiple wavelengths are in sensitivity.
 
    Returns:
        xr.DataArray: mask containing True when vertex has sensitivity above given threshold
    """
   
    intensity = np.log10(sensitivity[:,:,wavelength_idx].sum('channel'))
    mask = intensity > threshold
    mask = mask.drop_vars('wavelength')
    
    return mask


def downsample_mesh(mesh: xr.DataArray, 
                    mask: xr.DataArray,
                    threshold: cedalion.Quantity = 5 * cedalion.units.mm):
    """Downsample the mesh to get seeds of spatial bases.

    Args:
        mesh (xr.DataArray): mesh of either the brain or scalp surface.
        mask (xr.DataArray): mask specifying which vertices have significant sensitivity.
        threshold (Quantity): distance between vertices in downsampled mesh.
       
    Returns:
        xr.DataArray: downsampled mesh
    
    Initial Contributors:
        - Yuanyuan Gao 
        - Laura Carlton | lcarlton@bu.edu | 2024

    """
    
    mesh_units = mesh.pint.units
    threshold = threshold.to(mesh_units)
    
    mesh = mesh.rename({'label':'vertex'}).pint.dequantify()
    mesh_masked = mesh[mask,:]
    mesh_new = []

    for vv in tqdm(mesh_masked):
        if len(mesh_new) == 0: 
            mesh_new.append(vv)
            tree = KDTree(mesh_new)  # Build KDTree for the first point
            continue
        
        # Query the nearest neighbor within the threshold
        distance, _ = tree.query(vv, distance_upper_bound=threshold.magnitude)

        
        # If no point is within the threshold, append the new point
        if distance == float('inf'):
            mesh_new.append(vv)
            tree = KDTree(mesh_new)  # Rebuild the KDTree with the new point

    
    mesh_new_xr = xr.DataArray(mesh_new,
                               dims = mesh.dims,
                               coords = {'vertex':np.arange(len(mesh_new))},
                               attrs = {'units': mesh_units }
        )
    
    mesh_new_xr = mesh_new_xr.pint.quantify()
    
    return mesh_new_xr




def get_kernel_matrix(mesh_downsampled: xr.DataArray, 
                      mesh: xr.DataArray, 
                      sigma: cedalion.Quantity = 5 * cedalion.units.mm):
    
    """Get the matrix containing the spatial bases.

    Args:
        mesh_downsampled (xr.DataArray): mesh of either the downsampled brain or scalp surface.
            This is used to define the centers of the spatial bases.
        mesh (xr.DataArray): the original fully sampeld mesh of the brain or scalp. 
        sigma (Quantity): standard deviation used for defining the Gaussian kernel.
       
    Returns:
        xr.DataArray: matrix containing the spatial bases
    
    Initial Contributors:
        - Yuanyuan Gao 
        - Laura Carlton | lcarlton@bu.edu | 2024

    """
    assert mesh.pint.units == mesh_downsampled.pint.units
    
    
    mesh_units = mesh.pint.units
    sigma = sigma.to(mesh_units)
    
    # Covariance matrix
    cov_matrix = (sigma.magnitude **2) * np.eye(3)
    inv_cov = np.linalg.inv(cov_matrix)  # Inverse of Cov_matrix
    det_cov = np.linalg.det(cov_matrix)  # Determinant of Cov_matrix
    denominator = np.sqrt((2 * np.pi) ** 3 * det_cov)  # Pre-calculate denominator

    mesh_downsampled = mesh_downsampled.pint.dequantify().values
    mesh = mesh.pint.dequantify().values
    
    diffs = mesh_downsampled[:, None, :] - mesh[None, :, :]

    # Efficient matrix multiplication using np.einsum to compute (x-mu)' * inv_cov * (x-mu) for all pairs
    exponents = -0.5 * np.einsum('ijk,kl,ijl->ij', diffs, inv_cov, diffs)

    # Compute the kernel matrix
    kernel_matrix = np.exp(exponents) / denominator
    n_vertex = mesh.shape[0]
    
    dimensions = kernel_matrix.shape
    
    if dimensions[0] != n_vertex:
        dims = ["kernel", "vertex"]
        n_kernel = dimensions[0]
    else:
        dims = ["vertex", "kernel"]
        n_kernel = dimensions[1]
    
    kernel_matrix_xr = xr.DataArray(kernel_matrix, 
                                    dims = dims,
                                    coords = {'vertex': np.arange(n_vertex),
                                              'kernel': np.arange(n_kernel)}
                                    )
    
    return kernel_matrix_xr



def get_G_matrix(head: cfm.TwoSurfaceHeadModel, 
                 M: xr.DataArray,
                 threshold_brain: cedalion.Quantity = 5 * cedalion.units.mm, 
                 threshold_scalp: cedalion.Quantity = 20 * cedalion.units.mm, 
                 sigma_brain: cedalion.Quantity = 5 * cedalion.units.mm, 
                 sigma_scalp: cedalion.Quantity = 20 * cedalion.units.mm
                 ):
    
    """Get the G matrix which contains all the information of the spatial basis

    Args:
        head (cfm.TwoSurfaceHeadModel): Head model with brain and scalp surfaces.
        M (xr.DataArray): mask defining the sensitive vertices  
        threshold_brain (Quantity): distance between vertices in downsampled mesh for the brain.
        threshold_scalp (Quantity): distance between vertices in downsampled mesh for the scalp.
        sigma_brain (Quantity): standard deviation used for defining the Gaussian kernels of the brain.
        sigma_scalp (Quantity): standard deviation used for defining the Gaussian kernels of the scalp.
       
    Returns:
        xr.DataArray: matrix containing information of the spatial basis. Each column corresponds
            to the vertex representation of one kernel in the spatial basis. 
    
    Initial Contributors:
        - Yuanyuan Gao 
        - Laura Carlton | lcarlton@bu.edu | 2024

    """
    
    brain_downsampled = downsample_mesh(head.brain.vertices, M[M.is_brain], threshold_brain)
    scalp_downsampled = downsample_mesh(head.scalp.vertices, M[~M.is_brain], threshold_scalp)
    
    G_brain = get_kernel_matrix(brain_downsampled, head.brain.vertices, sigma_brain)
    G_scalp = get_kernel_matrix(scalp_downsampled, head.scalp.vertices, sigma_scalp)
    

    G = {'G_brain': G_brain, 
         'G_scalp': G_scalp
         }
    
    return G

#%% TRANSFORMING A    H = A @ G

def get_H(G, A):
    """
    get the H matrix when A is not stacked - H for each wavelength independently
    """
    n_channel = A.shape[0]
    nV_brain = A.is_brain.sum().values 

    nkernels_brain = G['G_brain'].kernel.shape[0]
    nkernels_scalp = G['G_scalp'].kernel.shape[0]

    n_kernels = nkernels_brain + nkernels_scalp

    H = np.zeros( (n_channel, n_kernels, 2))

    for w_idx, wl in enumerate(A.wavelength):
        A_wl = A.sel(wavelength=wl)
        A_wl_brain = A_wl[:,:nV_brain]
        A_wl_scalp = A_wl[:,nV_brain:]

        H[:,:nkernels_brain, w_idx] = A_wl_brain.values @ G['G_brain'].values.T
        
        H[:, nkernels_brain:, w_idx] = A_wl_scalp.values @ G['G_scalp'].values.T

    H = xr.DataArray(H, dims=("channel", "kernel", "wavelength"))
    H = H.assign_coords({'channel': A.channel,
                         'wavelength': A.wavelength})
    
    return H

    
def get_H_stacked(G, A):
    
    """
    get the H matrix as A@G for two wavelength stacked A matrix 
    """
    n_channel = A.shape[0]
    nV_brain = A.is_brain.sum().values //2
    nV_scalp = (~A.is_brain).sum().values //2

    nkernels_brain = G['G_brain'].kernel.shape[0]
    nkernels_scalp = G['G_scalp'].kernel.shape[0]

    n_kernels = nkernels_brain + nkernels_scalp

    H = np.zeros( (n_channel, 2 * n_kernels))

    A_hbo_brain = A[:, :nV_brain]
    A_hbr_brain = A[:, nV_brain+nV_scalp:2*nV_brain+nV_scalp]
    
    A_hbo_scalp = A[:, nV_brain:nV_scalp+nV_brain]
    A_hbr_scalp = A[:, 2*nV_brain+nV_scalp:]
    
    H[:,:nkernels_brain] = A_hbo_brain.values @ G['G_brain'].values.T
    H[:, nkernels_brain+nkernels_scalp:2*nkernels_brain+nkernels_scalp] = A_hbr_brain.values @ G['G_brain'].values.T
    
    H[:, nkernels_brain:nkernels_brain+nkernels_scalp] = A_hbo_scalp.values @ G['G_scalp'].values.T
    H[:, 2*nkernels_brain+nkernels_scalp:] = A_hbr_scalp.values @ G['G_scalp'].values.T

    H = xr.DataArray(H, dims=("channel", "kernel"))
    
    return H


def go_from_kernel_space_to_image_space_direct(X, G):
    
    split = len(X)//2
    nkernels_brain = G['G_brain'].kernel.shape[0]

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
    if len(X.shape) == 1:
        X = np.stack([np.concatenate([X_hbo_brain, X_hbo_scalp]),np.concatenate([ X_hbr_brain, X_hbr_scalp])], axis=1)
    else:
        X = np.stack([np.vstack([X_hbo_brain, X_hbo_scalp]), np.vstack([X_hbr_brain, X_hbr_scalp])], axis =2)

    return X

def go_from_kernel_space_to_image_space_indirect(X, G):
    
    nkernels_brain = G['G_brain'].kernel.shape[0]

    sb_X_brain = X[:nkernels_brain]
    
    sb_X_scalp = X[nkernels_brain:]
    
    #% PROJECT BACK TO SURFACE SPACE 
    X_brain = G['G_brain'].values.T @ sb_X_brain
    X_scalp = G['G_scalp'].values.T @ sb_X_scalp
    # concatenate them back together
    X = np.concatenate([X_brain, X_scalp])
    
    return X





