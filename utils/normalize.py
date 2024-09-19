import numpy as np
from sklearn.preprocessing import StandardScaler

# Flatten the data for normalization
def normalize_pend(u, f):
    scalers = []
    u_normalized_list = []
    source_term_normalized_list = []
    
    # Normalize each sample individually
    for i in range(u.shape[0]):
        # Concat u_1, u_2, and source_term
        data_concat = np.vstack((u[i, 0], u[i, 1], f[i]))
    
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_concat.T).T
        scalers.append(scaler)
        
        # Split the normalized data back into u_1, u_2, and source_term
        u1_normalized = data_normalized[0]
        u2_normalized = data_normalized[1]
        source_term_normalized = data_normalized[2]
    
        # Append normalized components to lists
        u_normalized_list.append(np.vstack((u1_normalized, u2_normalized)))
        source_term_normalized_list.append(source_term_normalized)
        
    # Stack normalized data back into arrays
    u_normalized = np.stack(u_normalized_list, axis=0)
    source_term_normalized = np.stack(source_term_normalized_list, axis=0)
    #print(u_normalized.shape)
    #print(source_term_normalized.shape)
    return u_normalized, source_term_normalized, scalers

def unnormalize_pend(u1_pred, u2_pred, f, scalers):
    # Rescale the normalized predictions back to the original scale
    u1_pred_original_list = []
    u2_pred_original_list = []
    #print(u1_pred.shape)
    #print(f.shape)

    for i in range(u1_pred.shape[0]): # for each sample
        # Concatenate predicted u_1, u_2, and source_term
        data_pred_normalized = np.vstack((u1_pred[i], u2_pred[i], f[i])) #we'll ignore f

        # Inverse transform the concatenated data
        data_pred_original = scalers[i].inverse_transform(data_pred_normalized.T).T
        
        # Split the original data back into u_1, u_2, and source_term
        u1_pred_original = data_pred_original[0]
        u2_pred_original = data_pred_original[1]
    
        u1_pred_original_list.append(u1_pred_original)
        u2_pred_original_list.append(u2_pred_original)
    
    # Combine the rescaled predictions back into the original shape
    u1_pred_original = np.stack(u1_pred_original_list)
    #print(u1_pred_original)
    u2_pred_original = np.stack(u2_pred_original_list)
    u_pred_original = np.stack((u1_pred_original, u2_pred_original), axis=1)
    return u_pred_original