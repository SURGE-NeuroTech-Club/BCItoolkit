'''

Requirements
------------
numpy
pandas
pylsl
mvlearn
'''


import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_stream
from mvlearn.embed import CCA

numberpad_string =\
'''
-------
|7 8 9|
|4 5 6|
|1 2 3|
|* 0 #|
-------
'''

character_classes = ['0','1','2','3','4','5','6','7','8','9','*','#']

Fs = 256
channel_to_label = ["G2", "G4", "F32", "G8", "G12", "G5", "G7", "G9"]
target_to_freq = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
Ns = 1025

data_csv = pd.read_csv('Subject1_AllTrials.csv')
marker_indices = data_csv[data_csv['Marker']>0]
marker_times = marker_indices['Time']

harmonics = np.arange(1,4)
Yn_list = []

for freq_idx in range(len(target_to_freq)):
    Yn = None
    freq = target_to_freq[int(freq_idx)]
    time_tmp = np.linspace(1/Fs, Ns/Fs, Ns)

    for harmon in harmonics:
        if Yn is None:
            Yn = np.vstack((np.sin(2*np.pi*harmon*freq*time_tmp), np.cos(2*np.pi*harmon*freq*time_tmp)))
        else:
            Yn = np.vstack((Yn, np.sin(2*np.pi*harmon*freq*time_tmp), np.cos(2*np.pi*harmon*freq*time_tmp)))
    Yn_list.append(Yn)



def classify(X_trial, cca_object):
    X_len = X_trial.shape[0]
    corrs_tmp = []
    for Yn in Yn_list:
        Xs_scores = cca_object.fit_transform([X_trial, Yn.T[-X_len:,:]])
        corrs_tmp.append(cca_object.canon_corrs(Xs_scores)[0])

    pred = np.argmax(corrs_tmp) + 1
    return pred, np.max(corrs_tmp)


if __name__=='__main__':
    cca = CCA(n_components=1)
    print("\n\nLooking for EEG streams...")
    streams = resolve_stream('type', 'Control')
    print("\n\nStream found.")
    inlet = StreamInlet(streams[0])
    print("\n\nInlet created. Now streaming data...")


    # while True:
    #     sample, timestamp = inlet.pull_sample()
    #     print(timestamp, sample)

    all_trials = []
    all_labels = []

    first_time = 0

    trial_counter = 0

    while trial_counter < 20:

        print('-'*30)

        print(numberpad_string)

        trial_data = []
        trial_times = []

        chunks = 0

        current_time = 0
        this_trial_time = 0

        sample = inlet.pull_sample()
        
        if len(all_trials) == 0:
            first_time = sample[1]
            first_idx = round(first_time / 4.263670037515112)
            # print('First index: {}'.format(first_idx))
            for ii in range(first_idx):
                all_trials.append(None)
                all_labels.append(None)
                trial_counter += 1

        while current_time == this_trial_time:

            current_time = round(sample[1], 5)

            if chunks == 0:
                this_trial_time = current_time

            if current_time != this_trial_time:
                break

            this_trial_idx = round(this_trial_time / 4.263670037515112)
    #         print('trial index {}'.format(this_trial_idx))

            trial_data.append(sample[0])
            trial_times.append(current_time)

            chunks+=1

            sample = inlet.pull_sample()
            
        trial_counter += 1
            
        data_np = np.array(trial_data)
        all_trials.append(data_np)
        all_labels.append(marker_indices['Marker'].iloc[this_trial_idx])
        
        pred, corr = classify(data_np, cca)
        # new_numberpad = numberpad_string.replace(str(pred),'X')
        new_numberpad = numberpad_string
        for c in character_classes:
            if c != str(pred):
                new_numberpad = new_numberpad.replace(c, ' ')
        print(new_numberpad)
        print("\n\nDecoded button: {}".format(pred))
        # print(trial_counter)

        # print('True class: {}, predicted: {}'.format(all_labels[-1], pred))