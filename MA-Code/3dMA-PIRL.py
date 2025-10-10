import time
import math
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn

# Configure logging with both file and console output
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'3D_MI_PIRL_training_{timestamp}.log'
    
    # Get the root logger and clear its handlers to avoid duplicate outputs
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logging.basicConfig(
      
        level=logging.INFO,
        format='%(message)s',  # Simple format without timestamp and level
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_filename

# Setup logging at the beginning
log_filename = setup_logging()
logger = logging.getLogger(__name__)

# Device & seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.empty_cache()

#------------------------------------------------------------------------------
# Monte Carlo functions for n-dimensional Min-Call options
#------------------------------------------------------------------------------
def generate_correlated_brownian_motion(n_assets, n_simulations, n_steps, rho, dt):
    """Generate correlated Brownian motion using Cholesky decomposition"""
    L = cholesky(rho, lower=True)
    dW_indep = np.random.normal(0, np.sqrt(dt), (n_simulations, n_steps, n_assets))
    dW = np.einsum('ij,...j->...i', L, dW_indep)
    return dW

def simulate_stock_prices(S0, r, sigma, rho, T, n_steps, n_simulations):
    """Simulate correlated stock price paths"""
    n_assets = len(S0)
    dt = T / n_steps
    S = np.zeros((n_simulations, n_steps + 1, n_assets))
    S[:, 0, :] = S0
    dW = generate_correlated_brownian_motion(n_assets, n_simulations, n_steps, rho, dt)
    
    for t in range(n_steps):
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = dW[:, t, :] * sigma
        S[:, t+1, :] = S[:, t, :] * np.exp(drift + diffusion)
    return S

def calculate_min_call_payoff(S_T, K):
    """Calculate Min-Call option payoff"""
    min_vals = np.min(S_T, axis=1)
    return np.maximum(min_vals - K, 0)

def monte_carlo_min_call_option(n, sigma, rho, r, T, K, S0, n_simulations, n_steps):
    """Monte Carlo pricer for n-dimensional Min-Call option"""
    if T <= 0:
        payoff = max(np.min(S0) - K, 0.0)
        return payoff
    
    S = simulate_stock_prices(S0, r, np.array(sigma), rho, T, n_steps, n_simulations)
    payoffs = calculate_min_call_payoff(S[:, -1, :], K)
    return np.exp(-r * T) * np.mean(payoffs)

def menet_pricer_nd(t0, S0, K, T, r, sigma, rho, num_paths, num_timesteps):
    """Monte Carlo pricer for n-dimensional Min-Call option using PyTorch tensors"""
    with torch.no_grad():
        # Convert parameters to tensors if they aren't already
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, device=device, dtype=torch.float32)
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
        if not isinstance(rho, torch.Tensor):
            rho = torch.tensor(rho, device=device, dtype=torch.float32)
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, device=device, dtype=torch.float32)
        
        
        if isinstance(t0, torch.Tensor):
            if t0.dim() == 2:
                t0 = t0.squeeze(1)  
        else:
            t0 = torch.tensor(t0, device=device, dtype=torch.float32)
        
        batch_size = S0.shape[0]
        n_assets = S0.shape[1]
        
        
        if t0.dim() == 0:  
            dt = (T - t0) / num_timesteps
        else: 
            dt = (T - t0.unsqueeze(1)) / num_timesteps  # [batch_size, 1]
        
        # Expand S0 for multiple paths
        S = S0.unsqueeze(1).expand(-1, num_paths, -1)  # [batch_size, num_paths, n_assets]
        
        # Pre-calculate drift and volatility terms
        drift = (r - 0.5 * sigma**2) * dt
  
        if drift.dim() == 1:
            drift = drift.unsqueeze(0)
        if drift.dim() == 2:
            drift = drift.unsqueeze(1)  # [batch_size, 1, n_assets]
        
        # Cholesky decomposition for correlation
        L = torch.linalg.cholesky(rho)
        vol_dt = sigma * torch.sqrt(dt)
     
        if vol_dt.dim() == 1:
            vol_dt = vol_dt.unsqueeze(0)
        if vol_dt.dim() == 2:  
            vol_dt = vol_dt.unsqueeze(1)  # [batch_size, 1, n_assets]
        
        # Expand drift and vol
        drift_expanded = drift.expand(batch_size, num_paths, n_assets)
        vol_expanded = vol_dt.expand(batch_size, num_paths, n_assets)
    
        # Add antithetic variates to reduce variance without extra compute
        half_paths = num_paths // 2
        for _ in range(num_timesteps):
            # Generate independent normal random variables for half paths
            Z_indep_half = torch.randn(batch_size, half_paths, n_assets, device=device)
            # Antithetic: append negatives
            Z_indep = torch.cat([Z_indep_half, -Z_indep_half], dim=1)
            # If odd num_paths, add one more random
            if num_paths % 2 == 1:
                Z_extra = torch.randn(batch_size, 1, n_assets, device=device)
                Z_indep = torch.cat([Z_indep, Z_extra], dim=1)
           
            # Apply correlation through Cholesky factorization
            Z_corr = torch.einsum('ij,...j->...i', L, Z_indep)
            
            S = S * torch.exp(drift_expanded + vol_expanded * Z_corr)
        
        # Calculate Min-Call payoffs
        min_vals = torch.min(S, dim=2)[0]  # [batch_size, num_paths]
        
        
        if K.dim() == 0:
            K_expanded = K
        else:
            K_expanded = K.unsqueeze(1)  # [batch_size, 1]
            
        payoff = torch.relu(min_vals - K_expanded)
        expected_payoff = torch.mean(payoff, dim=1, keepdim=True)
        
        
        if t0.dim() == 0:
            discount_factor = torch.exp(-r * (T - t0))
        else:
            discount_factor = torch.exp(-r * (T - t0).unsqueeze(1))
            
        price = expected_payoff * discount_factor
  
        return price

#------------------------------------------------------------------------------
# Enhanced PIRL Network with Residual Connections
#------------------------------------------------------------------------------
class PIRL(nn.Module):
    """Physics-Informed Residual Learning model with residual architecture"""
    def __init__(self, input_dim, hidden_dim, output_dim=1, layers=6):
        super(PIRL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        
  
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers with input concatenation
        for _ in range(layers - 2):
            self.layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
       
        self.activation = nn.Tanh()
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            fan_in = m.in_features
            fan_out = m.out_features
            std = math.sqrt(2 / (fan_in + fan_out))
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
 
            m.weight.data = torch.clamp(m.weight.data, min=-2*std, max=2*std)
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        original_input = x
        x = self.layers[0](x)
        x = self.activation(x)
        
        # Residual connections with input concatenation
        for i in range(1, len(self.layers) - 1):
            x_concat = torch.cat([x, original_input], dim=1)
            update = self.layers[i](x_concat)
            update = self.activation(update)
            x = x + update  # Residual connection
        
        output = self.layers[-1](x)
        return output

#------------------------------------------------------------------------------
# MinCallPINN wrapper for n-dimensional Min-Call option pricing
#------------------------------------------------------------------------------
class MinCallPINN:
    """PINN for n-dimensional Min-Call option pricing with MI enhancement"""
    def __init__(self, n, sigma, rho, r=0.05, T=1.0, K=1.0, S_min=0.5, S_max=1.5):
        self.n = n
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rho, dtype=torch.float32, device=device)
        self.r = r
        self.T = T
        self.K = K
        self.S_min = S_min
        self.S_max = S_max
        
        # Network architecture scaling
        hidden_dim = 2 ** (n + 3)  # 2^(n+3)
        layers = n + 3
        input_dim = n + 1  # [S1,...,Sn, t]
      
        self.model = PIRL(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers).to(device)
        logger.info(f"PIRL Model: input_dim={input_dim}, hidden_dim={hidden_dim}, layers={layers}")
    
    def predict_option_price(self, S_values, t_values):
        """Predict option prices for given S and t arrays"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(S_values, np.ndarray):
                S_tensor = torch.tensor(S_values, dtype=torch.float32, device=device)
            else:
                S_tensor = S_values
            if isinstance(t_values, np.ndarray):
                t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device).reshape(-1, 1)
            else:
                t_tensor = t_values.reshape(-1, 1)
            
            St = torch.cat([S_tensor, t_tensor], dim=1)
            prices = self.model(St).cpu().numpy().flatten()
            return prices
    
    def pde_loss(self, S, t):
        """Compute PDE residual loss for n-dimensional Black-Scholes"""
        S = S.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)
        St = torch.cat([S, t], dim=1)
        V = self.model(St)
        
        # First-order derivatives
        V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                 create_graph=True, retain_graph=True)[0]
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                 create_graph=True, retain_graph=True)[0]
        
        # Second-order derivatives (Hessian matrix)
        V_SS = torch.zeros(S.shape[0], self.n, self.n, device=device, dtype=torch.float32)
        for i in range(self.n):
            V_S_i = V_S[:, i]
        
            V_SS_i = torch.autograd.grad(V_S_i, S, grad_outputs=torch.ones_like(V_S_i),
                                        create_graph=True, retain_graph=True)[0]
            V_SS[:, i, :] = V_SS_i
        
        # PDE terms
        drift_term = torch.sum(self.r * S * V_S, dim=1)
  
        diffusion_term = torch.zeros_like(drift_term)
        for i in range(self.n):
            for j in range(self.n):
                diffusion_term += 0.5 * self.rho[i, j] * self.sigma[i] * self.sigma[j] * \
                                S[:, i] * S[:, j] * V_SS[:, i, j]
        
        # PDE residual: V_t + drift + diffusion - rV = 0
        pde_residual = V_t.squeeze() + drift_term + diffusion_term - self.r * V.squeeze()
        return torch.mean(pde_residual ** 2)
    
    def bc_loss(self, S_bc, t_bc):
        """Terminal payoff condition loss"""
        St = torch.cat([S_bc, t_bc], dim=1)
        V_pred = self.model(St).squeeze()
        min_vals, _ = torch.min(S_bc, dim=1, keepdim=True)
        payoff = torch.relu(min_vals.squeeze() - self.K)
        mse = torch.mean((V_pred - payoff) ** 2)
        mae = torch.mean(torch.abs(V_pred - payoff))
        return mse, mae

def train_model(model_type, model, params):
    """Train either standard PIRL or MI-PIRL model with mini-batch processing"""
    optimizer = torch.optim.Adam(model.model.parameters(), lr=params['lr'])
    
    if params['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7)
    
    loss_history = []
    start_time = time.time()
    
    logger.info(f"\n-- start training {model_type} --")
    logger.info("Starting epoch loop...")
    
    
    import sys
    sys.stdout.flush()
    for handler in logging.getLogger().handlers:
        handler.flush()
    
    
    last_martingale_loss = 0.0
    
    for epoch in range(params['epochs']):
    
        if epoch == 0:
            logger.info(f"Starting {model_type} training loop - Epoch 0")
            sys.stdout.flush()
        
        # Dynamic sampling each epoch
        N_interior, N_bc = params['N_interior'], params['N_bc']
        S_interior = (torch.rand(N_interior, model.n, device=device) *
                    (model.S_max - model.S_min)) + model.S_min
        t_interior = torch.rand(N_interior, 1, device=device) * model.T
        S_bc = (torch.rand(N_bc, model.n, device=device) *
               (model.S_max - model.S_min)) + model.S_min
        t_bc = torch.full((N_bc, 1), model.T, device=device)
        
        # Mini-batch iteration
        num_batches = max(1, N_interior // params['batch_size'])
        perm_interior = torch.randperm(N_interior, device=device)
        perm_bc = torch.randperm(N_bc, device=device)
        
        epoch_loss = 0.0
        
        add_martingale = False
        
        # MODIFICATION: Changed from epoch % 5 to epoch % 10
        if model_type == 'MI-PIRL' and epoch % 10 == 0:
            if epoch == 0:
                logger.info("MI-PIRL: Starting first martingale calculation...")
                sys.stdout.flush()
            
            try:
                batch_martingale = 256
                rand_vals = torch.rand(batch_martingale, 1, device=device) ** 1.5
                t_martingale = model.T * (1 - rand_vals)
                S_martingale = (torch.rand(batch_martingale, model.n, device=device) *
                               (model.S_max - model.S_min)) + model.S_min
                
                # PIRL prediction for logging
                V_pinn_martingale = model.model(torch.cat([S_martingale, t_martingale], dim=1))
            
                if epoch == 0:
                    logger.info("MI-PIRL: PIRL prediction completed, starting MC calculation...")
                    sys.stdout.flush()
                
                t_martingale_1d = t_martingale.squeeze(1)  
                
                V_menet_martingale = menet_pricer_nd(t_martingale_1d, S_martingale,
                                                    model.K, model.T, model.r,
                                                   model.sigma, model.rho,
                                                   params['menet_paths_training'],  
                                                   params['menet_steps_training'])  
               
                if V_menet_martingale.shape != V_pinn_martingale.shape:
                    V_menet_martingale = V_menet_martingale.squeeze()
                    V_pinn_martingale = V_pinn_martingale.squeeze()
                
                martingale_loss_current = torch.mean((V_pinn_martingale - V_menet_martingale)**2)
                last_martingale_loss = martingale_loss_current.item()
                
                V_menet_martingale = V_menet_martingale.detach()
                
                add_martingale = True
                
                if epoch == 0:
                    logger.info(f"MI-PIRL: First martingale calculation completed, loss={last_martingale_loss:.6f}")
                    sys.stdout.flush()
                
                logger.info(f"MI-PIRL Epoch {epoch}: Martingale points={batch_martingale}, paths={params['menet_paths_training']}, steps={params['menet_steps_training']}, loss={last_martingale_loss:.6f}")
                
            except Exception as e:
                logger.warning(f"Martingale calculation failed at epoch {epoch}: {e}")
                
                if epoch == 0:
                    logger.warning(f"Debug info: t_martingale shape: {t_martingale.shape if 't_martingale' in locals() else 'not defined'}")
                    logger.warning(f"Debug info: S_martingale shape: {S_martingale.shape if 'S_martingale' in locals() else 'not defined'}")
                    logger.warning(f"Debug info: V_pinn_martingale shape: {V_pinn_martingale.shape if 'V_pinn_martingale' in locals() else 'not defined'}")
                last_martingale_loss = 0.0
                add_martingale = False
        
        for b in range(num_batches):
            
            if epoch == 0 and b == 0:
                logger.info(f"{model_type}: Starting batch processing...")
                sys.stdout.flush()
            
            start_idx = b * params['batch_size']
            end_idx = min(start_idx + params['batch_size'], N_interior)
            batch_size_actual = end_idx - start_idx
            idx_interior = perm_interior[start_idx:end_idx]
            
            start_bc = (b * params['batch_size']) % N_bc
            idx_bc = perm_bc[start_bc : start_bc + batch_size_actual]
            if len(idx_bc) < batch_size_actual:
                extra = batch_size_actual - len(idx_bc)
                idx_bc = torch.cat([idx_bc, perm_bc[:extra]])
            
        
            S_int_batch = S_interior[idx_interior]
            t_int_batch = t_interior[idx_interior]
            S_bc_batch = S_bc[idx_bc]
            t_bc_batch = t_bc[idx_bc]
            
            optimizer.zero_grad()
            
            try:
                # PDE loss
                loss_pde = model.pde_loss(S_int_batch, t_int_batch)
                
                # Boundary condition loss
                mse_bc, _ = model.bc_loss(S_bc_batch, t_bc_batch)
            
                total_loss = loss_pde + params['lambda_bc'] * mse_bc
                
                # Add martingale constraint for MI-PIRL 
                if add_martingale:
                    V_pinn = model.model(torch.cat([S_martingale, t_martingale], dim=1))
                    V_pinn_s = V_pinn.squeeze()
                    V_menet_s = V_menet_martingale.squeeze()
                    martingale_loss = torch.mean((V_pinn_s - V_menet_s)**2)
                    total_loss += params['lambda_martingale'] * martingale_loss
   
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += total_loss.item()
                
            except Exception as e:
                logger.warning(f"Batch {b} in epoch {epoch} failed: {e}")
                epoch_loss += 1e6 
  
        
        epoch_loss /= num_batches  # Average loss over batches
        loss_history.append(epoch_loss)
        
        if params['use_scheduler']:
            scheduler.step()
        
       
        output_freq = 100 if epoch < 1000 else 500
       
        if epoch % output_freq == 0 or epoch == 0:  
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]['lr']
            
            
            try:
                base_msg = f"Epoch {epoch:6d}: Loss={epoch_loss:.4e}, Time={elapsed:.2f}s"
 
                if model_type == 'MI-PIRL' and last_martingale_loss > 0:
                    full_msg = f"{base_msg}, Martingale Loss={last_martingale_loss:.4e}"
                else:
                    full_msg = base_msg
                
                logger.info(full_msg)
                
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                
                for handler in logging.getLogger().handlers:
                    handler.flush()
                
            except Exception as e:
                
                print(f"Epoch {epoch}: Loss={epoch_loss:.6f}")
                sys.stdout.flush()
        
        
        if epoch % 1000 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"-- training finished, training time: {training_time:.2f} --")
    
    # Final metrics on last boundary points
    try:
        V_pred_bc_final = model.model(torch.cat([S_bc, t_bc], dim=1))
        min_vals_final, _ = torch.min(S_bc, dim=1, keepdim=True)
        V_true_bc_final = torch.relu(min_vals_final - model.K)
     
        mse_bc_final = torch.mean((V_pred_bc_final - V_true_bc_final)**2)
        mae_bc_final = torch.mean(torch.abs(V_pred_bc_final - V_true_bc_final))
        logger.info(f"Final Metrics: MSE={mse_bc_final.item():.4e}, MAE={mae_bc_final.item():.4e}")
    except Exception as e:
        logger.warning(f"Final metrics calculation failed: {e}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return loss_history, training_time

def create_3d_comparison_plots(pirl_model, mi_pirl_model, n, sigma, rho, r, T, K):
    """Create comparison plots for 3D case with both PIRL and MI-PIRL models"""
 
    if n != 3:
        logger.info(f"Visualization currently only supports n=3, got n={n}")
        return None, None
    
    logger.info("\n-- Creating 3D Comparison Plots --")
    
    
    N_test = 200  
    np.random.seed(123)
    S_test = np.random.uniform(0.7, 1.3, [N_test, n])
    t_test = np.random.uniform(0.1, T, N_test)
    tau_test = T - t_test
    
    # PIRL predictions (fast)
 
    logger.info("Computing PIRL predictions...")
    start_pirl = time.time()
    pirl_prices = pirl_model.predict_option_price(S_test, t_test)
    pirl_time = time.time() - start_pirl
    
    # MI-PIRL predictions (fast)
    logger.info("Computing MI-PIRL predictions...")
    start_mi_pirl = time.time()
    mi_pirl_prices = mi_pirl_model.predict_option_price(S_test, t_test)
    mi_pirl_time = time.time() - start_mi_pirl
    
    
    logger.info("Computing Monte Carlo predictions...")
    start_mc = time.time()
    mc_sims = 1000  
    mc_steps = 100 
  
    mc_prices = np.array([
        monte_carlo_min_call_option(n, sigma, rho, r, tau_test[i], K, S_test[i], mc_sims, mc_steps)
        for i in range(N_test)
    ])
    mc_time = time.time() - start_mc
    
    # Calculate errors
    abs_errors_pirl = np.abs(pirl_prices - mc_prices)
    abs_errors_mi_pirl = np.abs(mi_pirl_prices - mc_prices)
    
    # Create the comparison plots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Option Price vs Time (fixing S1, S2, S3)
    plt.subplot(2, 3, 1)
    t_plot = np.linspace(0.01, T, 30)  
    S_fixed = [[0.95, 1.05, 0.90], [1.10, 1.00, 1.15]]  # Two different asset combinations
    colors = ['blue', 'red']
    
    for i, S_vals in enumerate(S_fixed):
        # PIRL curve (smooth)
        S_curve = np.tile(S_vals, (len(t_plot), 1))
        pirl_curve = pirl_model.predict_option_price(S_curve, t_plot)
      
        plt.plot(t_plot, pirl_curve, color=colors[i], linestyle='--', linewidth=2,
                label=f'PIRL S={S_vals}', marker='o', markevery=10)
        
        # MI-PIRL curve (smooth)
        mi_pirl_curve = mi_pirl_model.predict_option_price(S_curve, t_plot)
        plt.plot(t_plot, mi_pirl_curve, color=colors[i], linestyle='-.', linewidth=2,
                label=f'MI-PIRL S={S_vals}', marker='s', markevery=10)
        
      
        # MC points (sparse due to computational cost)
        t_mc_sparse = np.linspace(0.1, T, 6)  
        tau_mc_sparse = T - t_mc_sparse
        S_mc_sparse = np.tile(S_vals, (len(t_mc_sparse), 1))
        mc_curve = np.array([
            monte_carlo_min_call_option(n, sigma, rho, r, tau_mc_sparse[j], K, S_vals, mc_sims//2, mc_steps//2)
            for j in range(len(tau_mc_sparse))
        ])
        plt.scatter(t_mc_sparse, mc_curve, color=colors[i], s=30,
                   label=f'MC S={S_vals}', marker='^')
    
    plt.xlabel('Time t')
    plt.ylabel('Option Price')
    plt.title('PIRL vs MI-PIRL vs MC: Option Price vs Time (3D)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    
    
    # Print statistics
    mae_pirl = np.mean(abs_errors_pirl)
    mse_pirl = np.mean(abs_errors_pirl**2)
    mae_mi_pirl = np.mean(abs_errors_mi_pirl)
    mse_mi_pirl = np.mean(abs_errors_mi_pirl**2)
    
    logger.info("\n-- Comparison Statistics --")
    logger.info(f"PIRL - Mean Absolute Error (MAE): {mae_pirl:.6f}")
    logger.info(f"PIRL - Mean Squared Error (MSE): {mse_pirl:.6f}")
    logger.info(f"MI-PIRL - Mean Absolute Error (MAE): {mae_mi_pirl:.6f}")
    logger.info(f"MI-PIRL - Mean Squared Error (MSE): {mse_mi_pirl:.6f}")
    logger.info(f"PIRL prediction time: {pirl_time:.4f} seconds")
    logger.info(f"MI-PIRL prediction time: {mi_pirl_time:.4f} seconds")
    logger.info(f"MC prediction time: {mc_time:.4f} seconds")
    
    return mae_pirl, mae_mi_pirl, mse_pirl, mse_mi_pirl

def main():
    """Main function to run 3D MI-PIRL comparison"""
    
    # Parameters 
    n = 3  # Number of assets (3D problem)
    # MODIFICATION: Updated shared_params dictionary
    shared_params = {
        'K': 1.0, 'T': 1.0, 'r': 0.05,
        'sigma': [0.2, 0.2, 0.2],  # Volatilities for each asset
        'S_min': 0.5, 'S_max': 1.5,
        'lr': 2e-3,
        'epochs': 5000,  # Increased for better convergence
        'batch_size': 256,
        'use_scheduler': True,
        'lambda_bc': 1.0,
        'lambda_martingale': 10.0,  # Increased to strengthen martingale supervision
        
        'N_interior': 2000 * n,
        'N_bc': 400 * n,
        'N_martingale': 200 * n,
        
        'menet_paths_training': 8000,   # Increased for better accuracy
        'menet_steps_training': 100,
        
        'menet_paths': 1000,
        'menet_steps': 100
    }
    
    # Correlation matrix for 3D problem
    rho = np.array([[1.0, 0.5, 0.3],
                    [0.5, 1.0, 0.4],
                    [0.3, 0.4, 1.0]])
    
    logger.info("Starting 3D MI-PIRL Training and Evaluation (Ultra-Optimized)")
    logger.info(f"Training MC Parameters: paths={shared_params['menet_paths_training']}, steps={shared_params['menet_steps_training']}")
    logger.info(f"Evaluation MC Parameters: paths={shared_params['menet_paths']}, steps={shared_params['menet_steps']}")
    
    logger.info("="*80)
    logger.info("3D MI-PIRL vs 3D PIRL COMPARISON (ULTRA-OPTIMIZED)")
    logger.info("="*80)
    logger.info(f"Parameters: r={shared_params['r']}, sigma={shared_params['sigma']}, T={shared_params['T']}, K={shared_params['K']}")
    logger.info(f"Network: hidden_dim=2^(n+3)={2**(n+3)}, layers={n+3}")
    logger.info(f"Training epochs: {shared_params['epochs']}")
    logger.info("-" * 60)
    
    try:
        # Create PIRL models
        logger.info("Creating PIRL models...")
        pirl_model = MinCallPINN(n, shared_params['sigma'], rho,
                                 shared_params['r'], shared_params['T'], shared_params['K'])
        mi_pirl_model = MinCallPINN(n, shared_params['sigma'], rho,
                                   shared_params['r'], shared_params['T'], shared_params['K'])
        
        # Train both models
        logger.info("Starting PIRL training...")
        pirl_loss_hist, pirl_time = train_model('PIRL', pirl_model, shared_params)
        
        logger.info("Starting MI-PIRL training...")
        mi_pirl_loss_hist, mi_pirl_time = train_model('MI-PIRL', mi_pirl_model, shared_params)
        
        logger.info("\n-- Evaluation --")
        
        # Create comparison plots
        mae_pirl, mae_mi_pirl, mse_pirl, mse_mi_pirl = create_3d_comparison_plots(
            pirl_model, mi_pirl_model, n, shared_params['sigma'], rho,
            shared_params['r'], shared_params['T'], shared_params['K'])
        
        # Calculate L2 relative errors using Monte Carlo as reference
        logger.info("\n-- Computing L2 Relative Errors --")
        N_test = 200  
        np.random.seed(789)
        S_test = np.random.uniform(0.7, 1.3, [N_test, n])
        t_test = np.random.uniform(0.1, shared_params['T'], N_test)
        tau_test = shared_params['T'] - t_test
        
        # Get predictions
        pirl_prices = pirl_model.predict_option_price(S_test, t_test)
        mi_pirl_prices = mi_pirl_model.predict_option_price(S_test, t_test)
        
        # Get MC reference
        logger.info("Computing Monte Carlo reference for L2 error calculation...")
        mc_prices = np.array([
            monte_carlo_min_call_option(n, shared_params['sigma'], rho,
                                      shared_params['r'], tau_test[i], shared_params['K'],
                                      S_test[i], shared_params['menet_paths'], shared_params['menet_steps'])
            for i in range(N_test)
        ])
        
        # Calculate L2 relative errors
        l2_error_pirl = np.linalg.norm(pirl_prices - mc_prices) / np.linalg.norm(mc_prices)
        l2_error_mi_pirl = np.linalg.norm(mi_pirl_prices - mc_prices) / np.linalg.norm(mc_prices)
        
        # Output comparison table
        log_msg_header = "\n" + "="*50
        log_msg_header += "\n" + " "*15 + "Comparison"
        log_msg_header += "\n" + "="*50
        log_msg_header += f"\n{'Metric':<25} | {'PIRL':<15} | {'MI-PIRL':<15}"
        log_msg_header += "\n" + "-"*50
        logger.info(log_msg_header)
        
        log_msg_body = f"{'Training Time (s)':<25} | {pirl_time:<15.2f} | {mi_pirl_time:<15.2f}"
        log_msg_body += f"\n{'L2 Relative Error':<25} | {l2_error_pirl:<15.4%} | {l2_error_mi_pirl:<15.4%}"
        log_msg_body += f"\n{'MAE (Mean Absolute Error)':<25} | {mae_pirl:<15.4e} | {mae_mi_pirl:<15.4e}"
        log_msg_body += f"\n{'MSE (Mean Squared Error)':<25} | {mse_pirl:<15.4e} | {mse_mi_pirl:<15.4e}"
        log_msg_body += "\n" + "="*50 + "\n"
        logger.info(log_msg_body)
        
        # Create loss convergence comparison plot
        fig = plt.figure(figsize=(10, 6))
        
        # Loss comparison
        plt.plot(pirl_loss_hist, label='PIRL Loss', color='blue', linewidth=2)
        plt.plot(mi_pirl_loss_hist, label='MI-PIRL Loss', color='red', linewidth=2)
    
        plt.yscale('log')
        plt.title('Loss Convergence Comparison (3D Min-Call Option - Ultra-Optimized)')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss (log scale)')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('3D_MI_PIRL_loss_comparison_ultra_optimized.png', dpi=300)
        logger.info("Loss comparison figure saved as: 3D_MI_PIRL_loss_comparison_ultra_optimized.png")
     
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        if log_filename:
            logger.info(f"Log file saved as: {log_filename}")
       
            # Force flush all handlers
            for handler in logging.getLogger().handlers:
                handler.flush()
        else:
            logger.warning("Log file could not be created.")

if __name__ == '__main__':
    main()
