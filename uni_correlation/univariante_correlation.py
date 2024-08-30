import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.stats import kendalltau

def plot_scatter_from_excel(path, rows, cols, n, labels, name):
    # Read the Excel file
    df = pd.read_excel(path)
    
    # Get x and y column names
    x_cols = df.columns[1:-1]
    y_col = df.columns[-1]

    # Create subplot
    fig, axs = plt.subplots(rows, cols, figsize=(10, n))
    axs = axs.flatten()

    # Define a list of colors
    colors = plt.get_cmap('tab20').colors  # Use 'tab20' colormap for distinct colors

    # Ensure we have enough labels
    if len(labels) < len(x_cols):
        raise ValueError("Not enough x_labels provided for the number of x columns")
    
    plt.rcParams['font.family'] = 'Helvetica'

    # Plot data
    for i, (x_col, color) in enumerate(zip(x_cols, colors)):
        if i >= len(axs):
            break
        axs[i].scatter(df[x_col], df[y_col], color=color, alpha=0.4, edgecolor=color, marker='o')
        tau, _ = kendalltau(df[x_col], df[y_col])
        tau = round(tau, 3)
        axs[i].set_title(f'KTC: {tau}', fontweight='bold')
        axs[i].set_xlabel(labels[i], fontweight='bold', fontsize=12)  # Use predefined labels
        axs[i].set_ylabel(labels[-1], fontweight='bold', fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    plt.savefig(f"{name}.png",dpi=300)
    plt.show()

# Example predefined label
labels_sub = [r'$\mathbf{N}$', r'$\mathbf{R_A}$', r'$\mathbf{R_B}$', 
              r'$\mathbf{m_A}$', r'$\mathbf{m_B}$', r'$\mathbf{\chi_{A}}$', 
              r'$\mathbf{\chi_{B}}$',r'$\mathbf{T_m}$', r'$\mathbf{Sublimation\ Enthalpies}$']

labels_ad = [r'$\mathbf{AN}$', r'$\mathbf{AM}$', r'$\mathbf{G}$', 
              r'$\mathbf{P}$', r'$\mathbf{R}$', r'$\mathbf{\chi}$', 
              r'$\mathbf{T_m}$', r'$\mathbf{T_B}$', r'$\mathbf{\Delta_{fus}}$',
              r'$\mathbf{\rho}$',r'$\mathbf{IE}$',r'$\mathbf{SUE}$',r'$\mathbf{Adsorption\ Energies}$']

path1="sub.xlsx"
path2="adsorption.xlsx"
plot_scatter_from_excel(path1, 4, 2, 16, labels_sub,'sub_ent')
plot_scatter_from_excel(path2, 4, 3, 12, labels_ad,'adsorption')