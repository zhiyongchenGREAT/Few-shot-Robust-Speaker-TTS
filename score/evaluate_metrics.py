import numpy as np
from .my_scorer import score_me

def calculate_eer_metrics(pred_k, labels, pred_u):
    num_samples = pred_k.shape[0]  
    num_classes = pred_k.shape[1]
    score_list = []
    label_list = []

    for i in range(num_samples):
        logits = pred_k[i]  
        true_label = labels[i]  
        for cls in range(num_classes):
            score_list.append(logits[cls]) 
            label_list.append(1 if cls == true_label else 0) 
            
    num_out_samples = pred_u.shape[0]  
    num_out_classes = pred_u.shape[1]

    for i in range(num_out_samples):
        logits = pred_u[i]  
        for cls in range(num_out_classes):
            score_list.append(logits[cls])  
            label_list.append(0) 

    score_list = np.array(score_list)
    label_list = np.array(label_list)

    configuration = {
        'p_target': [0.1],  
        'c_miss': 1,          
        'c_fa': 1                
    }
    
    eer, min_c, act_c = score_me(score_list, label_list, configuration)
    
    return {
        'EER': eer * 100,
        'Min_C': min_c,
        'Act_C': act_c
    }

def print_metrics(metrics):
    print(f"EER: {metrics['EER']:.2f}%")
    print(f"Min_C: {metrics['Min_C']:.3f}")