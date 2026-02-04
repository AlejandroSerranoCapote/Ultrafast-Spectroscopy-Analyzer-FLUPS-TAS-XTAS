# -*- coding: utf-8 -*-
import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog
from scipy import special as _special

def load_npy(parent=None, normalize_per_wl=True):
    """
    Carga el archivo .npy.
    """
    file_path, _ = QFileDialog.getOpenFileName(parent, "Select treated data file", "", "NumPy files (*.npy)")
    if not file_path:
        raise ValueError("No file selected")
    
    data = np.load(file_path, allow_pickle=True).item()
    data_c = data['data_c'].astype(float) 
    
    WL = data['WL'].flatten()
    TD = data['TD'].flatten()
    base_dir = os.path.dirname(file_path)
    
    return data_c, TD, WL, base_dir

def crop_spectrum(data_c, WL, WLmin, WLmax):
    mask = (WL >= WLmin) & (WL <= WLmax)
    return data_c[:, mask], WL[mask] 

def crop_kinetics(data_c, TD, TDmin, TDmax):
    mask = (TD >= TDmin) & (TD <= TDmax)
    return data_c[:, mask], TD[mask] 

def binning(data_c, WL, bin_size):
    numWL = len(WL) // bin_size
    datacAVG = np.zeros((numWL, data_c.shape[1]))
    WLAVG = np.zeros(numWL)
    for i in range(numWL):
        datacAVG[i, :] = np.mean(data_c[i*bin_size:(i+1)*bin_size, :], axis=0)
        WLAVG[i] = np.mean(WL[i*bin_size:(i+1)*bin_size])
    return datacAVG, WLAVG


def convolved_exp_vectorized(t, t0, taus, w):
    """
    Versión HÍBRIDA: Vectorizada pero usando 'erf' (más rápida que erfc).
    Optimized for Speed.
    """
    # t -> (N, 1)
    if t.ndim == 1:
        t = t[:, np.newaxis]
        
    # taus -> (1, M)
    taus = np.asarray(taus)
    if taus.ndim == 1:
        taus = taus[np.newaxis, :]
    
    # Constantes pequeñas para evitar división por cero
    tau_safe = np.maximum(taus, 1e-12)
    w_safe = np.maximum(w, 1e-12)
    
    # arg1 = (w^2 - 2*tau*(t-t0)) / (2*tau^2)
    # Factorizamos para reducir operaciones:
    t_diff = t - t0
    w2 = w_safe**2
    
    # Operación matricial (Broadcasting)
    arg1 = (w2 - 2 * tau_safe * t_diff) / (2 * tau_safe**2)
    arg2 = (w2 - tau_safe * t_diff) / (np.sqrt(2) * w_safe * tau_safe)

    # Mantenemos el clip en 700 que funciona bien con float64
    arg1 = np.clip(arg1, -700, 700)
    
    # Fórmula: 0.5 * exp(arg1) * (1 - erf(arg2))
    return 0.5 * np.exp(arg1) * (1 - _special.erf(arg2))



# =============================================================================
# MODEL EVALUATION FUNCTIONS
# =============================================================================

def eval_global_model(x, t, numExp, numWL, t0_choice_str):
    F = np.zeros((len(t), numWL))
    
    if t0_choice_str == 'Yes': # CHIRP MODE
        w = x[0]
        taus = x[1:1+numExp] # Array de todos los Taus
        base_idx = 1 + numExp
        
        for j in range(numWL):
            idx = base_idx + j*(numExp+1)
            t0 = x[idx]
            Amps = x[idx+1 : idx+1+numExp] # Array de Amplitudes (shape: numExp,)
            
            # 1. Calculamos TODAS las bases exponenciales de golpe para este t0
            # Devuelve matriz (Tiempo x numExp) directamente.
            bases = convolved_exp_vectorized(t, t0, taus, w)
            
            # 2. Multiplicamos matricialmente Bases @ Amplitudes
            # (N_t, N_exp) @ (N_exp,) -> (N_t,)  (Vector plano)
            F[:, j] = bases @ Amps

    else: # GLOBAL FIT (Optimización Máxima)
        w = x[0]
        t0 = x[1]
        taus = x[2:2+numExp]
        A_base = 2 + numExp
        
        # Devuelve directamente la matriz (Tiempo x numExp)
        basis_functions = convolved_exp_vectorized(t, t0, taus, w)
            
        # Extraer Amplitudes
        all_Amps = x[A_base:].reshape(numWL, numExp).T 
        
        # Multiplicación
        F = basis_functions @ all_Amps
        
    return F


def get_sequential_populations(t, t0, w, taus):
    """ 
    Calculates populations for a sequential model A -> B -> C... 
    (Kept logical loop structure as sequential dependencies are complex to fully vectorize cleanly)
    """
    k = 1.0 / np.asarray(taus) # Rates
    pops = []
    
    # Get all basic exponentials at once
    E_matrix = convolved_exp_vectorized(t, t0, taus, w)
    E = [E_matrix[:, i] for i in range(len(taus))]
    
    # --- Species 1 ---
    pops.append(E[0])
    
    # --- Species 2 ---
    if len(taus) >= 2:
        denom = k[1] - k[0]
        if abs(denom) < 1e-9: denom = 1e-9 
        factor = k[0] / denom
        p2 = factor * (E[0] - E[1])
        pops.append(p2)
        
    # --- Species 3 ---
    if len(taus) >= 3:
        k0, k1, k2 = k[0], k[1], k[2]
        d0 = (k[1]-k[0]) * (k[2]-k[0])
        d1 = (k[0]-k[1]) * (k[2]-k[1])
        d2 = (k[0]-k[2]) * (k[1]-k[2])
        # Safety for degenerate rates
        if abs(d0) < 1e-9: d0 = 1e-9
        if abs(d1) < 1e-9: d1 = 1e-9
        if abs(d2) < 1e-9: d2 = 1e-9
        
        p3 = (k0 * k1) * ( (E[0]/d0) + (E[1]/d1) + (E[2]/d2) )
        pops.append(p3)

    return pops

def eval_sequential_model(x, t, numExp, numWL, t0_choice_str):
    """
    Sequential Model (A -> B -> C...)
    """
    F = np.zeros((len(t), numWL))
    
    if t0_choice_str == 'Yes': 
        w = x[0]
        taus = x[1:1+numExp]
        base_idx = 1 + numExp
        
        for j in range(numWL):
            idx = base_idx + j*(numExp+1)
            t0 = x[idx]
            sas_coeffs = x[idx+1 : idx+1+numExp] 
            
            pops_list = get_sequential_populations(t, t0, w, taus)
            
            # Manual dot product for the single WL
            kinetics = np.zeros_like(t)
            for n in range(numExp):
                kinetics += sas_coeffs[n] * pops_list[n]
            
            F[:, j] = kinetics

    else: 
        w = x[0]
        t0 = x[1]
        taus = x[2:2+numExp]
        A_base = 2 + numExp
        
        # 1. Basis Functions (Time x Species)
        pops_list = get_sequential_populations(t, t0, w, taus)
        basis_functions = np.column_stack(pops_list) 
        
        # 2. Amplitudes / SAS (Species x WL)
        all_SAS = x[A_base:].reshape(numWL, numExp).T 
        
        # 3. Matrix Multiplication
        F = basis_functions @ all_SAS
        
    return F

def damped_oscillation(t, t0, alpha, omega, phi, w):
    """
    Calculates damped oscillation with a Soft Step (approximating IRF convolution).
    S(t) = 0.5 * (1 + erf((t-t0)/(sqrt(2)*w))) * exp(-alpha*(t-t0)) * sin(omega*(t-t0) + phi)
    """
    t_shifted = t - t0
    
    # 1. Safety Mask: Prevent exp() overflow for very negative times.
    safe_mask = t_shifted > -6 * w
    
    osc = np.zeros_like(t_shifted)
    
    # Only calculate where it is numerically safe
    ts_safe = t_shifted[safe_mask]
    
    # Smooth Step (Simulates convolution with Gaussian IRF)
    # Using erf here is standard for step smoothing
    step = 0.5 * (1 + _special.erf(ts_safe / (np.sqrt(2) * w)))
    
    # Damped Sine
    decay = np.exp(-alpha * ts_safe)
    sine = np.sin(omega * ts_safe + phi)
    
    osc[safe_mask] = step * decay * sine
    
    return osc

def eval_oscillation_model(x, t, numExp, numWL, t0_choice_str):
    """
    Model: Sum(A_i * Exp_i) + B * Oscillation
    """
    F = np.zeros((len(t), numWL))
    
    if t0_choice_str == 'Yes':
        raise NotImplementedError("Chirp not implemented for Oscillation model yet.")

    # --- Global Parameters ---
    w = x[0]
    t0 = x[1]
    taus = x[2:2+numExp]
    
    # Oscillation parameters
    alpha = x[2+numExp]
    omega = x[2+numExp+1]
    phi   = x[2+numExp+2]
    
    A_base = 2 + numExp + 3 
    
    # 1. Decay Bases (Vectorized) -> Matrix (Time x Exp)
    basis_exp = convolved_exp_vectorized(t, t0, taus, w)
        
    # 2. Oscillation Basis -> Matrix (Time x 1)
    basis_osc = damped_oscillation(t, t0, alpha, omega, phi, w).reshape(-1, 1)
    
    # 3. Stack all bases: [T x (numExp + 1)]
    all_bases = np.hstack([basis_exp, basis_osc])
    
    # 4. Extract Local Amplitudes [A1...An, B] per wavelength
    num_local_params = numExp + 1
    all_amps = x[A_base:].reshape(numWL, num_local_params).T
    
    # 5. Matrix Multiplication
    F = all_bases @ all_amps
    
    return F