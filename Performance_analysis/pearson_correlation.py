#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:26:11 2020

@author: amos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def r_squared_Lumen (y_ref, y_pred):
    
    # calculate the Pearson's correlation: R2
    r_squared, _ = pearsonr(y_ref, y_pred)
    print("pearson's correlation:" +str(r_squared))    
    # calculate spearman's correlation
    Spearman_corr, _ = spearmanr(y_ref, y_pred)
    
    P_Value= 0.05 
    
    plt.scatter(y_ref, y_pred)
    plt.xlabel('Expert: Area (sq mm)', fontsize=15)
    plt.ylabel('CNN: Area (sq mm)', fontsize=15)
    plt.title('Lumen Area \nCNN vs Expert')
    plt.plot(np.unique(y_ref), np.poly1d(np.polyfit(y_ref, y_pred, 1))(np.unique(y_ref)), 'r')
    plt.text(5, 15, 'R-squared = %0.5f' % r_squared, fontsize=15)
    plt.text(5, 14, 'Spearman rho = %0.5f' % Spearman_corr, fontsize=13)
    plt.text(5, 13, 'P-Value< ' +str(P_Value), fontsize=12)
    plt.savefig('r_squared_Lumen.png')
    plt.show()
    

def r_squared_vesselWall (y_ref, y_pred):
    
    # calculate the Pearson's correlation: R2
    r_squared, _ = pearsonr(y_ref, y_pred)
    print(r_squared)
    # calculate spearman's correlation
    Spearman_corr, _ = spearmanr(y_ref, y_pred)
    
    P_Value= 0.05 
    
    plt.scatter(y_ref, y_pred)
    plt.xlabel('Expert: Area (sq mm)', fontsize=15)
    plt.ylabel('CNN: Area (sq mm)', fontsize=15)
    plt.title('Wall Area \nCNN vs Expert')
    plt.plot(np.unique(y_ref), np.poly1d(np.polyfit(y_ref, y_pred, 1))(np.unique(y_ref)), 'r')
    plt.text(8, 17, 'R-squared = %0.5f' % r_squared, fontsize=15)
    plt.text(8, 16, 'Spearman rho = %0.5f' % Spearman_corr, fontsize=13)
    plt.text(8, 15, 'P-Value< ' +str(P_Value), fontsize=12)
    plt.savefig('r_squared_vesselWall.png')
    plt.show()   
