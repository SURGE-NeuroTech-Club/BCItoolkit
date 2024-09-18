# BCI Toolkit

This repository holds the codebase and documentation for the BCItoolkit library. The goal of this library is to provide an accesible API to those with minimal python and/or brain-computer interface (BCI) experience to get started working on making online BCI systems.


*Modules/*: Each file should contain documentation on classes & functions
- `brainflow_stream.py`: A custom class that simplifies usage of the brainflow library to connect and stream from any board supported by brainflow
- `segmentation.py`: Creates time-based segments of data from the EEG stream for SSVEP processing
- `brainflow_filtering.py/filtering.py`: These modules support several filtering methods for EEG data. 
  - brainflow_filtering.py simplifies in-place usage of the brainflow libraries built-in filters
- `ssvep_stim.py`: Creates customizable SSVEP stimuli and has a class to run them in a seperate process to reduce number of required scripts
- `classification.py`: Has 1 class (more to come) with several methods for classificaiton of SSVEP data

Extras:
    - `get_freqs.py`: Simple module that measures refresh rate and returns a dictionary of all possible frequencies the monitor can accurately display


## Links/Reference

Brainflow
- https://brainflow.readthedocs.io/en/stable/Examples.html#python
- https://brainflow.readthedocs.io/en/stable/UserAPI.html#python-api-reference