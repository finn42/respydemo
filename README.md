# respy Respiration recording analysis toolbox Demo notebooks
Introduces basic functions of library for processing respiration recordings (single best chest stretch).

https://pypi.org/project/respy/0.1.1/

    pip install respy

This library provides some basic signal processing tools to extract respiration phase and related features from recordings of respiration from healthy awake adults. Initial developed to track respiration phase in music listeners (2019), the features are now being adapted to also detect respiration phase in performing musicians (2023) and similar active conditions. 

Chest stretch measurements do not perfectly capture airflow. Many interesting aspects of respiratory behaviour require more demanding sensor arrangements to accurately capture gas partial pressures and volumns of air displaced. While measured change in chest circumference is often analogous to air flow, changes in posture and respiration strategy can shift the scale. 

This library instead focuses on respiratory phase timing, which is more reliably captured with this sensor. (Though there is also some noise here as well.) This time domain analysis of a quasi-oscillatory signal uses optimised heuristics to distinguish respiratory events from sensor noise. With accurate detection of indiviual breaths, many interesting follow questions can be asked, such as the relationship of respiratory phase to concurrently presented stimuli or bodily actions. 

And after respiratory phase information has been extracted with some degree of reliability, respiratory shape and style can be assessed with much greater refinement. 

Demo respiraton recordings shared to demonstrate the phase extraction and classification of respiratory behaviour.

Packaged with respy is the activity analysis library for phase alignment assessment, within the module act.py

Developed by Finn Upham 2023 

Note: This is not yet suitable for the evaluation of respiration during high intensity exertions, or for non-human animals, or for respiration measurements taken with other types of sensors (flow meters, double belts).

This toolbox is written in python 3.9 with the following dependencies:
import time
import datetime as dt
import math
import numpy as np 
import pandas as pd
import scipy as sc 
from scipy.signal import butter,filtfilt
from scipy import interpolate
from scipy.interpolate import interp1d


## Installation
Add the package with pip with the code above like: 
> pip install respy

## Example respiration analysis

Find demo application this github account Finn42
https://github.com/finn42/respydemo

Activity analysis demo (with test package or loaded definitions)
https://github.com/finn42/aa_test_package

