# Goals:

The outline & functionality is split into categories related to different aspects of BCI's

**Signal Acquisition/Hardware Interfacing:**
- [X] Support for OpenBCI Cyton Board: *stream_data.py*
  - 
- [ ] Support for Unicorn

**Signal Processing:**
- [ ] Filters:
  - [ ] Bandpass
- [ ] Epoching/Segmenting
- [ ] 

**Classification:**
- SSVEP:
  - [ ] CCA
  - [ ] FBCCA
- ERP:
  - [ ] Training data handler
  - [ ] LDA
  - [ ] SVM

**Stimulus Presentation:**
- [X] SSVEP (working)
- [ ] Oddball(?)

**Output:**
- [ ] Output hooks
  - Easy way for the classifier output to be hooked into performing tasks/executing commands (i.e., the 'computer' part)

**Documentation + Examples:**
1. [ ] Data aquisition: *docs/stream_data.qmd*
   1. Exists, but will need to be revised upon module completion.
2. [ ] Filtering
3. [X] Stimulus Presentation: *docs/psychopy_ssvep_stim.qmd*
4. [ ] Classifier
