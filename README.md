# BCI Toolkit

This repository holds the codebase and documentation for the BCItoolkit library. The goal of this library is to provide an accesible API to those with minimal python and/or brain-computer interface (BCI) experience to get started working on making online BCI systems.

### Modules and Functionalities Overview
***Modules/*: Each module has some self-contained documentation**
- `brainflow_stream.py`: A custom class that simplifies usage of the brainflow library to connect and stream from any board supported by brainflow. 
  - Some added features: automatically finds the serial port with the attached dongle, simplifies streaming from multiple boards simultaneously, is designed to be compatible with all of [Brainflow's BoardShim attributes](https://brainflow.readthedocs.io/en/stable/UserAPI.html#brainflow-board-shim).
- `brainflow_filtering.py/filtering.py`: These modules support several filtering methods for EEG data. 
  - brainflow_filtering.py simplifies in-place usage of the brainflow library's built-in filters.
  - filtering.py uses filters from the Scipy library. 
- `ssvep_stim.py`: Creates customizable SSVEP stimuli and has a class to run them in a separate process to reduce number of required scripts without blocking analysis execution.
  - Functionality: provide intended flicker frequencies, optional names and locations. Produces flickering stimuli at frequency nearest to intended while being possible using the monitors refresh rate. Also returns the actual flicker (target) frequencies for classification purposes.
- `classification.py`: Classification module built off scikit-learn. Currently only for SSVEP & CCA (more methods to come)
  - Handles target/reference signal generation, scaling, and fit_transformation of the data.
  - **Currently Broken** --> still ironing out implementation of this with other modules.
- ~~`segmentation.py`: Creates time-based segments of data from the EEG stream for SSVEP processing~~
  - *Deprecated* - Considering implementation into brainflow_stream module; can segment via time.sleep() before retrieving new data from the brainflow board buffer.
- **Extras:**
  - `get_freqs.py`: Simple function that measures a monitors average refresh rate and returns a dictionary of all possible frequencies the monitor can accurately display using the `ssvep_stim` module. (This isn't necessary since the ssvep_stim module automatically calculates and displays the closest possible flicker frequencies to those given)
  - `psychopy_monitor_manager.py`: A simple module that allows for creation of psychopy monitors without downloading and using the psychopy GUI's 'Monitor Center'.
    - This is designed to only has to be done once to create a monitor, which can then be referenced when calling `ssvep_stim`.

## Usage:
Documentation is a work-in-progress, for examples see scripts in *Examples/*

## Road-map:
**First Release:**
- [ ] Add (offline) visualization module
  - Online 'Live' time-series visualization future goal (several existing libraries may be compatible)
- [ ] Validate CCA/SSVEP Classification (and online BCI system)
- [ ] Add documentation on current modules

**Future Releases:**
- [ ] Add modules support ERP-based BCI
  - `erp_stim` module for stimulus
  - add SVM, LDA, other common classifiers to `classification` module
    - Implement base classifier for integration of custom user classifiers
- [ ] Enhance signal processing
  - Artifact removal, ICA, wavelet transforms
- [ ] Pipeline module for more seamless integration of an online BCI system


## Links/Reference

Brainflow
- https://brainflow.readthedocs.io/en/stable/Examples.html#python
- https://brainflow.readthedocs.io/en/stable/UserAPI.html#python-api-reference