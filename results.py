import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from bmtk.analyzer.spike_trains import plot_raster, plot_rates, plot_rates_boxplot, to_dataframe, spike_statistics

# raster plots of simulation output
plot_raster(config_file='config.json', group_by='pop_name')
plot_rates(config_file='config.json', group_by='pop_name')
plot_rates_boxplot(config_file='config.json', group_by='pop_name')
