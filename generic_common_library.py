# =======================
# Import Required Libraries
# =======================

# Core scientific and data analysis libraries
import numpy as np                              # For numerical operations
import pandas as pd                             # For data manipulation and analysis
import random                                   # For random number generation

# Date and time utilities
import datetime                                 # For working with dates and times
import datetime as dt                           # Duplicate alias, unnecessary but can coexist
from datetime import date, datetime, timedelta  # Specific classes for date/time manipulation
from dateutil.relativedelta import relativedelta  # For more flexible date calculations (e.g., months)

# Statistical libraries
import scipy                                     # Core scientific library
from scipy.stats import ttest_ind, t            # For t-tests and t-distribution

# System and performance utilities
from sys import stdout                           # Access to system standard output
from time import sleep                           # For time delays in execution
import itertools                                 # Efficient looping and combinatorics
import io                                        # Handling of I/O streams
import tracemalloc                               # Memory tracking
import gc                                        # Garbage collection
import multiprocessing                           # Parallel processing
import os                                        # OS-related utilities
import getpass                                   # Secure password input

# Statistical tests
from statsmodels.stats.proportion import proportions_ztest  # For z-tests on proportions

# Logging
import logging                                   # For application-level logging

# Machine learning and model tools
import matplotlib.pyplot as plt                  # For plotting/visualization
import math                                      # Math functions
import xgboost as xgb                            # Extreme Gradient Boosting library
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    TimeSeriesSplit, cross_val_score, BaseCrossValidator
)                                                # Model evaluation and cross-validation tools
import sklearn                                   # Machine learning library
import pickle                                    # Object serialization
import joblib                                    # Faster object serialization, often used with scikit-learn
import tempfile                                  # For creating temporary files and directories


# =======================
# Utility Function
# =======================

# Function to get the ISO week number of the last week of the previous year
def lastweeknumberoflastyear(date_input):
    # December 28 is always in the last ISO week of the year
    return datetime(int(date_input.year - 1), 12, 28).isocalendar()[1]
