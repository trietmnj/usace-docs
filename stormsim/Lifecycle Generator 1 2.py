import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn.linear_model
import statistics
import statsmodels.api as sm
import itertools
import plotly.express as px
#import pylab
import distfit
import fitter
import random
from datetime import datetime, timedelta

### USER INPUTS
Initialize_date=2033
lifecycle_duration=50
num_LCs=100
min_arrrival_time=[7,4] #[tropical, extratropical]
#Storm_max=12
lam=1.7 #Local storm recurrence rate
InputFile=pd.read_csv(r'C:\Users\RDCRLHPS\Documents\STORMSIM CHART\Relative_probability_bins_Atlantic.csv')


### READ PROBABILITY FILE INFO [Month, Day, Daily Probability, Cumulative Probability]
df=pd.DataFrame(InputFile) #Dataframe format of the input file
#df_array=df.to_numpy() #Array format of the input file
Cumulative_probs = df['Cumulative trop prob']
Cumulative_probs_a=Cumulative_probs.to_numpy()
Month=df['Month'].to_numpy()
Day=df['Day'].to_numpy()

### GENERATE POISSON DISTRIBUTION
samples=np.random.poisson(lam=lam, size=(lifecycle_duration,1))  #generate n=lifecycle_duration samples with Lambda=lam
samples=pd.DataFrame(samples)
f = np.nonzero(samples)[0]

### PRODUCTION LOOP
    for i in range(0, num_LCs+1):
        StormDates = []

        for ii in range(0, len(f)):
            Yr=np.array([Initialize_date + f[ii]])
            samples1=np.random.rand(1, f[ii]+1)
            samples1.sort()

            def find_first_greater_index(arr1, arr2):
                results = []
                for iii, val1 in enumerate(arr1):
                    found = False
                    for j, val2 in enumerate(arr2):
                        if any(val2 > val1):
                            results.append(j)
                            found = True
                            break  # Stop after finding the first greater element
                    if not found:
                        results.append(None)  # No element in arr2 was greater than val1
                return results
            output_indices = find_first_greater_index(samples1, Cumulative_probs_a)
            Mo = Month[output_indices]
            Da = Day[output_indices]
            Hr = np.random.rand(len(samples1[0]), 1)
            Hr = Hr * 24
            H = Hr[0]

            storm_date = pd.DataFrame({'ii': f[ii],'year': Yr, 'month': Mo, 'day': Da, 'hour': H})

            if len(samples1)>1:
                diff = storm_date[1:, 0] - storm_date[:-1, 0]
                foo = np.where(diff < min_arrrival_time[0, 0])
                while foo:
                    samples1 = np.random.rand(1, f[ii] + 1)
                    samples1.sort()

                    def find_first_greater_index(arr1, arr2):
                        results = []
                        for iii, val1 in enumerate(arr1):
                            found = False
                            for j, val2 in enumerate(arr2):
                                if any(val2 > val1):
                                    results.append(j)
                                    found = True
                                    break  # Stop after finding the first greater element
                            if not found:
                                results.append(None)  # No element in arr2 was greater than val1
                        return results

                    output_indices = find_first_greater_index(samples1, Cumulative_probs_a)
                    Mo = Month[output_indices]
                    Da = Day[output_indices]
                    Hr = np.random.rand(len(samples1[0]), 1)
                    Hr = Hr * 24
                    H = Hr[0]

                    storm_date = pd.DataFrame({'ii': f[ii],'year': Yr, 'month': Mo, 'day': Da, 'hour': H})

            else:
                print('Break')


        ### OUTPUT RESULTS IN TEXT FILE
                StormDates.append(storm_date)
                filename=f"EventDate_LC_{i}.txt"
                with open(filename, "w") as file:
                    for item in StormDates:
                        file.write(f"{item}\n")


                    print(f"Processing lifecycle: {i} Duration {ii} ")









