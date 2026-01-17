import numpy as np
import pandas as pd
import os

labels_path = 'Submission/labelsX.npy'
dataset_path = 'Datasets/datasetTest.csv'

try:
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        print(f"labelsX.npy is readable.")
        print(f"labelsX.npy shape: {labels.shape}")
    else:
        print(f"labelsX.npy not found at {labels_path}")
        labels = None

    if os.path.exists(dataset_path):
        df_test = pd.read_csv(dataset_path, header=None)
        print(f"datasetTest.csv shape: {df_test.shape}")
    else:
        print(f"datasetTest.csv not found at {dataset_path}")
        df_test = None

    if labels is not None and df_test is not None:
        if labels.shape[0] == df_test.shape[0]:
            print("labelsX.npy Row counts MATCH.")
        else:
            print(f"labelsX.npy Row counts DO NOT MATCH. ({labels.shape[0]} vs {df_test.shape[0]})")

    grandmaster_path = 'PartD/outputs/labelsX_grandmaster.npy'
    if os.path.exists(grandmaster_path):
        labels_gm = np.load(grandmaster_path)
        print(f"labelsX_grandmaster.npy is readable.")
        print(f"labelsX_grandmaster.npy shape: {labels_gm.shape}")
        
        if df_test is not None:
            if labels_gm.shape[0] == df_test.shape[0]:
                 print("labelsX_grandmaster.npy Row counts MATCH.")
            else:
                 print(f"labelsX_grandmaster.npy Row counts DO NOT MATCH. ({labels_gm.shape[0]} vs {df_test.shape[0]})")
    else:
        print(f"labelsX_grandmaster.npy not found at {grandmaster_path}")

except Exception as e:
    print(f"Error: {e}")
