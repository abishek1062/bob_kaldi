import bob.kaldi
import bob.io.audio
import numpy as np
import os
from pyAudioAnalysis import audioSegmentation as aS
from feature_extraction import *

config = configuration()

for model_file in config['model_files']:
	model_path = config['model_files'][model_file].get()

	if not(os.path.exists(model_path)): # only create empty model file if the file does not already exist 
		open(model_path,'w').close()

mfcc_path = config['dirs']['mfcc_path'].get()

# import pdb; pdb.set_trace()
if not(os.path.isdir(mfcc_path)):
	os.makedirs(mfcc_path)

voxceleb1_path = config['dirs']['voxceleb1_path'].get()

# saving mfccs of speaker in mfcc_path
save_speaker_MFCC(voxceleb1_path)


# loading all MFCC npy files of speakers in 1 np.array
MFCC_feats = [np.load(config['dirs']['mfcc_path'].get()+np_file) for np_file in os.listdir(config['dirs']['mfcc_path'].get())]
MFCC_feats = np.vstack(MFCC_feats)

#UBM training
diag_GMM_UBM_file = config['model_files']['diag_GMM-UBM'].get()

print('training diagonal GMM UBM ...')
diag_GMM_UBM_model = bob.kaldi.ubm_train(MFCC_feats, 
                                         diag_GMM_UBM_file,
                                         num_threads = config['diag_gmm_ubm_params']['num_threads'].get(),
                                         min_gaussian_weight = config['diag_gmm_ubm_params']['min_gaussian_weight'].get(),
                                         num_gauss = config['diag_gmm_ubm_params']['num_gauss'].get(),
                                         num_gauss_init = config['diag_gmm_ubm_params']['num_gauss_init'].get(),
                                         num_gselect = config['diag_gmm_ubm_params']['num_gselect'].get(),
                                         num_iters_init = config['diag_gmm_ubm_params']['num_iters_init'].get(), 
                                         num_iters = config['diag_gmm_ubm_params']['num_iters'].get(), 
                                         remove_low_count_gaussians = config['diag_gmm_ubm_params']['remove_low_count_gaussians'].get())

print('training full GMM UBM ...')
full_GMM_UBM_file = config['model_files']['full_GMM-UBM'].get()

full_GMM_UBM_model = bob.kaldi.ubm_full_train(MFCC_feats, 
                                              diag_GMM_UBM_model, 
                                              full_GMM_UBM_file, 
                                              num_gselect=config['full_gmm_ubm_params']['num_gselect'].get(), 
                                              num_iters=config['full_gmm_ubm_params']['num_iters'].get(), 
                                              min_gaussian_weight=config['full_gmm_ubm_params']['min_gaussian_weight'].get())

print(f'diag GMM-UBM model trained at {diag_GMM_UBM_file}')
print(f'full GMM-UBM model trained at {full_GMM_UBM_file}')