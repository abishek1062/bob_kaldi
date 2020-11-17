import bob.kaldi
import bob.io.audio
import numpy as np
import os
from pyAudioAnalysis import audioSegmentation as aS
from collections import defaultdict
import confuse

def configuration():
    config = confuse.Configuration(os.getcwd(),__name__)

    return config


def silence_detection(speech):
    '''
    Returns list of lists containing the start and end time stamps of speech segments
    e.g.: [[0.05,0.07],[0.09,0.012]] --> speech exists in the intervals 0.05-0.07 and 
                                         0.09-1.12   
    Utlises a pretained Hidden Markov Model in pyAudioanalysis that determines speech 
    activity.

    Parameters: 
    data (reader): reader returned by bob.io.audio.reader  
  
    Returns: 
    intervals(list): list of lists containing intervals where speech segments exist
    '''
    config = configuration()['speech_params']
    intervals = aS.silence_removal(speech,
                                   config['sample_rate'].get(),
                                   config['frame_length'].get(),
                                   config['frame_overlap'].get(),
                                   smooth_window = config['smooth_window'].get(),
                                   weight = config['sil_rem_wt'].get(),
                                   plot = config['sil_rem_plt'].get())

    for i in range(len(intervals)):
        intervals[i] = [int(stamp*config['sample_rate'].get()) for stamp in intervals[i]]

    return intervals

def utterance_selection(spkr_path,max_utterances = 16):
    '''
    given a speaker's directory of utterances, the function returns a list of selected utterances 
    (=max_utterances) that maximize the channel variability in UBM training

    Parameters:
    spkr_path(string) : path of directory of speaker(voxceleb1) containing audio files
    max_utterances(int) : maximum no. of utterances that will be selected from speakers path

    Returns:
    selected_utterances(list) : list containing path of specific utterances thay together will 
                                maximize channel variablility
    '''
    utter_dict = defaultdict(list) # {'channel_name' : [list of utterances in that channel]}
    
    
    for utterance in os.listdir(spkr_path):
        if utterance[-4:] == '.npy':
            utter_dict[utterance.split('_')[0]].append(utterance)

    utter_count = 0
    channel_index = 0
    selected_utterances = []

    while utter_count < max_utterances:
        for channel in utter_dict.keys():
            if utter_count < max_utterances:
                try:
                    selected_utterances.append(os.path.join(spkr_path,utter_dict[channel][channel_index]))
                    utter_count += 1
                except:# some channels may have more utterances than others
                    continue
            else:
                break
        channel_index += 1

    return selected_utterances

def save_speaker_MFCC(dataset_path,max_utter_duration=5):
    '''
    function builds a big MFCC np.array by extracting mfccs from several speakers' utterances where
    each utterance is trimmed till ~5 seconds of speech activity

    Parameters:
    dataset_path(string) : path to the dataset(voxceleb1)
    max_utter_duration(int) : maximum duration of speech activity selected from utterance

    Returns:
    (int) : integer 1
    '''
    config = configuration()
    for spkr_index,spkr in enumerate(os.listdir(dataset_path)):
        MFCC_feats = []
        if spkr_index >= config['diag_gmm_ubm_params']['no_of_spkrs'].get(): # UBM built on only 100 speakers worth of data; break when limit reached
            break
            
        else:
            print('--------------------------------------------------------------------------------------')
            print(f'analyzing utterances from speaker {spkr}')
            spkr_path = os.path.join(dataset_path,spkr)
            
            # utterances selected based on maximing channel variability
            selected_utterances = utterance_selection(spkr_path)
            
            for utterance in selected_utterances:
                wav = utterance.split('/')[-1]
                print(f'  utterance {wav}')

                # data = bob.io.audio.reader(utterance)
                speech = np.load(utterance)
                norm_speech = speech/2**(config['speech_params']['bits_per_sample'].get()-1) # normalizing speech utterance
                
                speech_segs = silence_detection(speech)
                
                utter_duration = 0
                
                for seg in speech_segs:
                    if utter_duration <= max_utter_duration:
                        seg_duration = (seg[1] - seg[0])/config['speech_params']['sample_rate'].get()
                        
                        if utter_duration + seg_duration > max_utter_duration:
                            upper_bound = int(5*config['speech_params']['sample_rate'].get() + seg[0])

                        elif utter_duration + seg_duration <= max_utter_duration:
                            upper_bound = seg[1]
                            
                        target_speech_seg = norm_speech[seg[0]:upper_bound+1]

                        mfcc_seg = bob.kaldi.mfcc(target_speech_seg)
                        MFCC_feats.append(mfcc_seg)

                        utter_duration += (upper_bound - seg[0])/config['speech_params']['sample_rate'].get()
                        
        print(f' saving {spkr}.npy')
        np.save(config['dirs']['mfcc_path'].get()+f'{spkr}.npy',np.vstack(MFCC_feats))
                        
    return 1