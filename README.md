# BCI Toolkit

This repository holds the codebase and documentation for the BCItoolkit library. The goal of this library is to provide those with minimal brain-computer interface (BCI) and/or python experience to get started working on making online BCI systems.

*Modules/*: Each file should contain documentation on classes & functions
- `stream_data.py`: A custom class that uses the brainflow library to connect and stream from the Cyton Board
- `preprocessing.py`: Class that contains functions to segment, filter, and save data
- `ssvep_handler.py`: Classes that generate harmonics and uses canonical correlation analysis (CCA) to classify SSVEP data, and functions that perform and return signal-to-noise ratio (SNR)
- `stim_pres.py`: Code related to stimulus presentation (i.e., flickering stimuli to elicit SSVEP)
- `maintenence.py`: Code related to listening for the 'esc' key and raising stop flags




## Links/Reference

LSL:
- https://docs.openbci.com/Software/CompatibleThirdPartySoftware/LSL/
- https://github.com/openbci-archive/OpenBCI_LSL

Brainflow
- https://brainflow.readthedocs.io/en/stable/Examples.html#python
- https://brainflow.readthedocs.io/en/stable/UserAPI.html#python-api-reference