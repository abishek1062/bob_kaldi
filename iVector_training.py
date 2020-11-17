import bob.kaldi
import bob.io.audio
import numpy as np
import glob
import os
import librosa
from pyAudioAnalysis import audioSegmentation as aS
import soundfile as sf
import math
from collections import defaultdict
import tempfile
from scipy.spatial import distance
from feature_extraction import *

# i-Vector training
# list comprehension of mfccs in npy files of voxceleb1's 100 speakers
# MFCC_iVec_feats  - features for multiple speakers are 3-dimensional in the format 
# (<no. of speakers>,<no. of frames>,< dimension of mfccs>)

def get_min_array_dim(MFCC_iVec_feats):
  '''
  Returns minimum dimesional array within MFCC_iVec_feats

  Parameters:
  MFCC_iVec_feats(np.array) - numpy array containing MFCCs all speakers; dim (100,xxx,39)

  Returs:
  minarray_dim(int) - min dimensional array
  '''

  min_array_dim = 0

  for i,np_array in enumerate(MFCC_iVec_feats):
    if i == 0:
      min_array_dim = np_array.shape[1]
        
    else:
        if np_array.shape[1] < min_array_dim:
            min_array_dim = np_array.shape[1]
  return min_array_dim


#driver code

config = configuration()

voxceleb1_mfcc_path = config['dirs']['mfcc_path'].get()
MFCC_iVec_feats = [np.load(os.path.join(voxceleb1_mfcc_path,npy_file)) for npy_file in os.listdir(voxceleb1_mfcc_path)]

# altering dimensions of MFCC_iVec_feats for iVector training
MFCC_iVec_feats = [np.expand_dims(np_array,axis=0) for np_array in MFCC_iVec_feats]

min_array_dim = get_min_array_dim(MFCC_iVec_feats)

MFCC_iVec_feats = [np_array[:,0:min_array_dim,:] for np_array in MFCC_iVec_feats]
MFCC_iVec_feats = np.vstack(MFCC_iVec_feats)



full_GMM_UBM_file = config['model_files']['full_GMM-UBM'].get()
with open(full_GMM_UBM_file,'r') as file:
  full_GMM_UBM_model = file.read()

ivector_file = config['model_files']['iVector'].get()

print('training i-Vector model ...')
ivector_model = bob.kaldi.ivector_train(MFCC_iVec_feats, 
                                        full_GMM_UBM_model, 
                                        ivector_file, 
                                        num_gselect = config['iVector_params']['num_gselect'].get(), 
                                        ivector_dim = config['iVector_params']['ivector_dim'].get(),
                                        use_weights = config['iVector_params']['use_weights'].get(),
                                        num_iters = config['iVector_params']['num_iters'].get(),
                                        min_post = config['iVector_params']['min_post'].get(),
                                        num_samples_for_weights = config['iVector_params']['num_samples_for_weights'].get(),
                                        posterior_scale = config['iVector_params']['posterior_scale'].get())

print(f'i-Vector model trained at {ivector_file}')