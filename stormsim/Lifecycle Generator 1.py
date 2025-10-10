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

        Mo=Month[output_indices]
        Da=Day[output_indices]
        Hr=np.random.rand(len(samples1[0]),1)
        Hr=Hr*24
        H=Hr[0]
        Yr=np.full(samples1.shape[0], Yr)
        storm_date=pd.DataFrame({'year':Yr, 'month': Mo, 'day': Da, 'hour':H})

            if len(samples1)>1:
                diff=storm_date[1:,0]-storm_date[:-1,0]
                foo=

            #while foo

print(f"Processing lifecycle: {i} ")





#
# ### PRODUCTION LOOP
# for i in range(1, num_LCs):
#     for ii in range(1, len(f)):
#         Yr=Initialize_date+f[ii]-1 #included -1 offset so loop starts in user specified input
#         #samples1[ii]= np.random.rand([0,samples[ii]])#Return n-samples according to n-storms in the year
#         #samples1[ii]=samples.sample(n=f[ii], replace=True, random_state=ii)
#         samples1 = np.random.rand(1, f[ii])
#         random_numbers_list=[]
#         random_numbers_list.append(samples1)
#         #samples1=samples1.sort() #Sorts by ascending values
#         # findgreat=find_first_greater(Cumulative_probs, 1)
#         # def find_first_greater(x):
#         #     return np.argmax(x)
#         # result=map(find_first_greater,samples1)
#         #result=np.where(Cumulative_probs > samples1)[0]
#         findfirst=(Cumulative_probs>x for x in samples1)
#         #results=list(findfirst)
#         num_cols=samples1.shape[1]
#         Hr=np.random.randint(0,25,size=(num_cols, 1))
#         #storm_date=pd.DataFrame({Yr,Month,Day, Hr, 0,0 })
#         storm_date = np.concatenate({Yr, Month, Day, Hr}) #, 0, 0})
#         storm_date.to_csv('stormdate_output.txt')
#             if len(samples1[0]) > 1:
#             diff=np.diff(storm_date[:,0])
#             foo=np.where(diff<min_arrrival_time[0,0])
#         #         while foo:
#         #         print(f"Redistributing dates due to minimum arrival time threshold exceedance, lifecycle year = {Yr}")
#         #         #clear
#         #         samples1 = np.random.randint(1, samples(ii))  # Return n-samples according to n-storms in the year
#         #         #samples1 = samples1.sort()  # Sorts by ascending values
#         #        # def find_first_greater(x):
#         #            # return np.argmax(x)
#         #         #result = list(map(find_first_greater, samples1))
#         #         num_cols = samples1.shape[1]
#         #         Hr = np.random.randit(0, 25, size=(num_cols, 1))
#         #         storm_date = pd.DataFrame({Yr,Month,Day, Hr, 0,0 })
#         #         storm_date.to_csv('stormdate_output.txt')
#         #         diff = np.diff(storm_date[:, 0])
#         #         foo = np.where(diff < min_arrrival_time[1,1])
#         #         storm_date=pd.Dataframe({Yr,Month,Day, Hr, 0,0 })
#         #         storm_date.to_csv('stormdate_output.txt')
#         # print(storm_date.to_csv('stormdate_output.txt'))
#     print(storm_date.to_csv('stormdate_output.txt'))
# print(f"Processing lifecycle: {i} ")




