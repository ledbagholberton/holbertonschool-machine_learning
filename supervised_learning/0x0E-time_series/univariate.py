#!/usr/bin/env python3
"""
Univariate data
"""
import pandas as pd
import numpy as np

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    """
    Univariate function        
    """
    
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)