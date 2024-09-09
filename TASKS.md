# Tasks/To-Do's:

The goal of this repo/library is to make a toolkit for making SSVEP/ERP-based brain-computer interfaces accessible for those who don't have much/any experience.

**Testing:**
- [X] Split into Segmenting, Filters, Other(?)
  - [ ] Test Segmentation.py
    - Overlapping data issues? -> better way to implement segmentation? (just have `get_data` module??)
  - [ ] Test Filtering.py -> validate the results in processing_test.ipynb
- [X] Test the new `SSVEP_stim.py` for flicker frequency!!
  - Preliminary testing looks good (not sure what min/max of program (or ssvep doohikey) are)
- [ ] Test new classification module
  - [ ] CCA
  - [ ] FBCCA
- [ ] Test the entire example pipeline [`testing/Full_Pipeline_Test.ipynb`] with a cyton board!


**Feature Timeline:**
- Artifact removal  
  - Based on amplitude
  - EOG artifact removal?
- ERP_stim (large host of challenges with markers and training data requirements)
- Classification methods for ERPs


# Notes:


sim_ssvep_data.npy: simulated SSVEP data in shape (8, 15000)
8 channels, 15000 samples (60 seconds at 250 Hz Sample Rate)
Simulated SSVEP signal changes between [9.25, 11.25, 13.25, 15.25] Hz every 10 seconds

**Cyton Board**: streams data in 24 channels
- 1-8 = EEG
- 9-11 = Accelerometer Channels
- 13+ Aux Channels(?)

**SSVEP Projects w/ Cyton:**
- https://github.com/WATOLINK/mind-speech-interface-ssvep
- https://github.com/NTX-McGill/NeuroTechX-McGill-2021/tree/main

For SSVEP: Oz, O1, O2, pOz, PO3, PO4, Pz (+ reference?)
    --> https://www.researchgate.net/publication/349257316/figure/fig2/AS:990562102571009@1613179817695/a-The-10-10-electrode-placement-system-Bold-10-20-system-b-The-cerebral-lobes-and.ppm

Brainflow has it's own filtering/ML?
- https://brainflow.readthedocs.io/en/stable/UserAPI.html#brainflow-ml-model
- https://brainflow.readthedocs.io/en/stable/UserAPI.html#brainflow-data-filter


## CCA/SSVEP Python Libraries: (other than scikit-learn)
- https://github.com/jameschapman19/cca_zoo
- https://github.com/nbara/python-meegkit
- https://wiki.mentalab.com/applications/ssvep/
- MNE-LSL: https://mne.tools/mne-lsl/stable/api/index.html

- https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0140703&type=printable