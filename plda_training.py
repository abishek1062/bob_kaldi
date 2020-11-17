import bob.kaldi
import bob.io.audio
import numpy as np
import os
from feature_extraction import *


def get_min_array_dim(MFCC_plda_feats):
  '''
  Returns minimum dimesional array within MFCC_iVec_feats

  Parameters:
  MFCC_plda_feats(np.array) - numpy array containing MFCCs all speakers; dim (100,xxx,39)

  Returs:
  minarray_dim(int) - min dimensional array
  '''

  min_array_dim = 0

  for i,np_array in enumerate(MFCC_plda_feats):
    if i == 0:
      min_array_dim = np_array.shape[1]
        
    else:
        if np_array.shape[1] < min_array_dim:
            min_array_dim = np_array.shape[1]
  return min_array_dim

#driver code
config = configuration()

voxceleb1_mfcc_path = config['dirs']['mfcc_path'].get()
MFCC_plda_feats = [np.load(os.path.join(voxceleb1_mfcc_path,npy_file)) for npy_file in os.listdir(voxceleb1_mfcc_path)]

# altering dimensions of MFCC_plda_feats for PLDA training
MFCC_plda_feats = [np.expand_dims(np_array,axis=0) for np_array in MFCC_plda_feats]

min_array_dim = get_min_array_dim(MFCC_plda_feats)

MFCC_plda_feats = [np_array[:,0:min_array_dim,:] for np_array in MFCC_plda_feats]
MFCC_plda_feats = np.vstack(MFCC_plda_feats)

plda_model_file = config['model_files']['trained_plda'].get()
global_plda_mean_file = config['model_files']['plda_mean'].get()

print('training plda model ...')
plda_model = bob.kaldi.plda_train(MFCC_plda_feats, plda_model_file, global_plda_mean_file)

print(f'trained PLDA model at {plda_model_file}')
