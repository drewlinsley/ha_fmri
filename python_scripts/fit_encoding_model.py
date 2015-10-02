
# coding: utf-8

# In[1]:

import os
from multiprocessing import Pool
import numpy as np
import scipy as sp
import nilearn
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVR
import pylab
import matplotlib
import nitime
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer


# In[2]:

def extract_1d_voxels (voxel_volume):
    #Convert Voxel volume [x/y/time] into timepoint X nonan voxels
    num_vox = np.sum(~np.isnan(voxel_volume[:,:,1]))
    vox_it = voxel_volume.shape[2]
    voxel_array = np.zeros((vox_it,num_vox))
    for idx in range(0,vox_it-1):
        curr_vox = np.isnan(voxel_volume[:,:,idx])
        this_data = voxel_volume[:,:,idx]
        #voxel_array[idx,:] = this_data[~curr_vox].T
        voxel_array[idx,:] = this_data.T
        #print 'iteration %d' % (idx)
    return voxel_array
        
def shift_voxels (voxel_timecourse, run_ids, hdr_est, model, trim_amount):
    #Shift each voxel's timecourse by hdr_est backwards at each run_id
    cutoff = np.inf
    #run_splits = np.where(run_ids==1)
    num_events = run_ids.shape[0]
    print run_ids.shape
    max_num = voxel_timecourse.shape[0]
    num_vox = voxel_timecourse.shape[1]
    #lagged_voxels = np.zeros((voxel_timecourse.shape[0],voxel_timecourse.shape[1]))
    lagged_voxels = []
    lagged_model = []
    for idx in range(0,num_events-1):
        start_idx = run_ids[idx]
        end_idx = run_ids[idx+1] - 1
        old_vec = voxel_timecourse[start_idx:end_idx,:]
        mean_pad = np.ones((hdr_est,num_vox)) * np.mean(old_vec) #create mean pad
        trimmed_data = old_vec[hdr_est:num_vox,:]
        old_vec = np.vstack((trimmed_data,mean_pad)) #push neural data hdr_est back in time (since it's delayed relative the movie)
        time_model = model[start_idx:end_idx,:]
        (old_vec, time_model) = remove_front_scans(old_vec,time_model,trim_amount)
        old_vec = zscore_voxels(old_vec,cutoff)
        lagged_voxels.append(old_vec)
        lagged_model.append(time_model)
        #lagged_voxels[start_idx:end_idx,:] = old_vec
        #lagged_voxels[start_idx:end_idx,:] = zscore_voxels(old_vec,cutoff) #put the corrected data back
    lagged_voxels = np.vstack(lagged_voxels)
    lagged_model = np.vstack(lagged_model)
    return lagged_voxels, lagged_model

def zscore_voxels (voxel_timecourse, cutoff):
    #zscore run-wise and windzorize measurements that are +- cutoff sds
    voxel_timecourse = sp.stats.mstats.zscore(voxel_timecourse)
    voxel_timecourse[voxel_timecourse>cutoff] = cutoff
    voxel_timecourse[voxel_timecourse<-cutoff] = -cutoff
    return voxel_timecourse

def remove_front_scans (voxel_timecourse, model, trim_amount):
    voxel_timecourse = voxel_timecourse[trim_amount + 1 : -1,:]
    model = model[trim_amount + 1 : -1,:]
    return voxel_timecourse, model
    
def make_encoding_model (X,y1,y2,movie_idx):
    ####
    #Encoding model pipeline
    ####
    #1. Fit linear regression at each voxel of training data
    #2. Get model fit at each voxel of testing data
    #3. Find max loading PE for each voxel (assuming these are sorted) and project these to a new brain volume

    #Experimenter knobs:
    #1. Crossval is being done on run_1/run_2. Could change this to 90%/10% having averaged together 1st/2nd half.

    #Fit the voxel timecourse Y with X
    #clf = linear_model.RidgeCV(alphas=[0.001, 0.1, 1, 10],fit_intercept=True)
    #clf = linear_model.Ridge(alpha=0.1,fit_intercept=True)
    clf = SVR(kernel='linear', C=1e3, gamma=0.1)
    #clf = linear_model.Ridge(alpha=1,fit_intercept=True).fit(X,rh_1_array)
    #clf.fit(X,y1) #Fit to first half
    #coeffs = clf.coef_
    #y2_hat = clf.predict(X)
    #MSE = metrics.mean_squared_error(y2, y2_hat) #compare [predicted y2] to y2
    #r2 = clf.score(object_model, y2) #compare [predicted y2] to y2
    #return coeffs, r2, MSE, y2_hat
    ntimes = movie_idx.shape[0]
    coeffs = np.zeros((ntimes,X.shape[1]))
    r2_array = np.zeros(ntimes)
    r_array = np.zeros(ntimes)
    mse_array = np.zeros(ntimes)
    y2_hat_array = []
    for idx in range(0,ntimes - 1):
        #seperate out this run from the rest
        start_idx = movie_idx[idx]
        end_idx = movie_idx[idx + 1] - 1
        tx = X[start_idx:end_idx,:]
        ty1 = y1[start_idx:end_idx]
        ty2 = y2[start_idx:end_idx]
        clf = SVR(kernel='linear', C=10, gamma=0.1, verbose = False, max_iter = 1000)
        clf.fit(tx,ty1) #Fit to first half
        coeffs[idx,:] = clf.coef_
        y2_hat = clf.predict(tx)
        y2_hat_array.append(y2_hat)
        r2_array[idx] = clf.score(tx, ty2)
        r_array[idx] = np.corrcoef(ty2, y2_hat)[0,1]
        mse_array[idx] = metrics.mean_squared_error(ty2,y2_hat)

    #y2_hat_array = np.vstack(y2_hat_array)
    return coeffs, r2_array, mse_array, y2_hat_array, r_array



def filter_data (X,TR):
    T = TimeSeries(X, sampling_interval=TR)
    F = FilterAnalyzer(T, ub=0.1, lb=0.02)
    Y = F.fir.data
    return Y


# In[3]:

#File directory
main_dir = '/Users/drewlinsley/Downloads/mmbl_data/'
#Load .mat file
my_data = sp.io.loadmat(os.path.join(main_dir,'ha_data.mat'))
#Load .mat file
object_model = np.load(os.path.join(main_dir,'object_output.npy'))


# In[4]:

#Get data from .mat file
movie_times = my_data['movie_times']
cuts = my_data['cuts']
rh_1 = my_data['rh_1']
rh_2 = my_data['rh_2']
lh_1 = my_data['lh_1']
lh_2 = my_data['lh_2']
movie_cuts = my_data['movie_cuts'] #This needs to be validated as correct
movie_idx = np.where(movie_cuts==1)[1]
movie_idx = np.hstack((movie_idx,rh_1.shape[2]))
r_size = rh_1.shape
l_size = lh_1.shape


# In[ ]:

hdr_lag = 2 #2 TRs = 4 seconds
trim_amount = 3
X = object_model

#Convert rh_1 into timepoint X nonan voxels
rh_1_array = extract_1d_voxels(rh_1)

#Some dumb checks on the data
print r_size
print l_size
#fig, (ax1, ax2, ax3) = matplotlib.pyplot.subplots(nrows=3, figsize=(6,10))
#ax1.imshow(rh_1[:,:,1])
#matplotlib.pyplot.tight_layout()
#ax2.imshow(rh_1_array[:,1:1000])
#matplotlib.pyplot.tight_layout()
#
(rh_1_array, Xtrim) = shift_voxels(rh_1_array, movie_idx, hdr_lag, X, trim_amount)
rh_1_array = filter_data(rh_1_array, hdr_lag)
#ax3.imshow(rh_1_array[:,1:1000])
#matplotlib.pyplot.tight_layout()
#matplotlib.pyplot.show()


# In[66]:

#matplotlib.pyplot.plot(rh_1_array[:,1])
#matplotlib.pyplot.show()


# In[ ]:

#Convert rh_2 into timepoint X nonan voxels
rh_2_array = extract_1d_voxels(rh_2)
(rh_2_array,Xtoss) = shift_voxels(rh_2_array, movie_idx, hdr_lag, X, trim_amount)
rh_2_array = filter_data(rh_2_array, hdr_lag)
#Convert lh_1 into timepoint X nonan voxels
lh_1_array = extract_1d_voxels(lh_1)
(lh_1_array,Xtoss) = shift_voxels(lh_1_array, movie_idx, hdr_lag, X, trim_amount)
lh_1_array = filter_data(lh_1_array, hdr_lag)

#Convert lh_2 into timepoint X nonan voxels
lh_2_array = extract_1d_voxels(lh_2)
(lh_2_array,Xtoss) = shift_voxels(lh_2_array, movie_idx, hdr_lag, X, trim_amount)
lh_2_array = filter_data(lh_2_array, hdr_lag)


# In[184]:

#Generate random data of the expected convnet size
#X = np.random.normal(0,1,[r_size[0]*r_size[1],conv_columns])
num_vox = rh_1_array.shape[1]
conv_columns = Xtrim.shape[1]
rh_vox_r = np.zeros((num_vox,1)) #correlation between the two

#Loop through all voxels (indexed by num_vox), gather correlations
for idx in range(0,num_vox):
    it_y1 = rh_1_array[:,idx]
    it_y2 = rh_2_array[:,idx]
    #(coeffs, r2, MSE, y2_hat) = make_encoding_model(Xtrim,it_y1,it_y2)
    #rh_vox_r2[idx] = r2
    #rh_vox_MSE[idx] = MSE
    #rh_vox_PE[idx] = coeffs
    rh_vox_r[idx] =  np.corrcoef(it_y1,it_y2)[0,1]
    
    if idx % 1e1 == 0:
        print 'Working voxel %d/%d\n' % (idx,num_vox)

suprathresh = np.array(np.where(rh_vox_r > .15)[0]) #voxels must have > .15 r1 -> r2 correlation


#Store the parameter estimates, the r2, and the MSE
num_vox = suprathresh.shape[0]
rh_vox_r2 = np.zeros((num_vox,movie_idx.shape[0]))
rh_vox_r = np.zeros((num_vox,movie_idx.shape[0]))
rh_vox_MSE = np.zeros((num_vox,movie_idx.shape[0]))
rh_vox_PE = np.zeros((num_vox,conv_columns))

#Loop through all voxels (indexed by num_vox), gather correlations
for idx in range(0,num_vox):
    it_y1 = rh_1_array[:,suprathresh[idx]]
    it_y2 = rh_2_array[:,suprathresh[idx]]
    mu = np.mean(it_y1,0)
    std = np.std(it_y1,0)
    it_y1 = (it_y1 - mu) / std
    it_y2 = (it_y2 - mu) / std
    (coeffs, r2, MSE, y2_hat,r) = make_encoding_model(Xtrim,it_y1,it_y2,movie_idx)
    rh_vox_r[idx,:] = r
    rh_vox_r2[idx,:] = r2
    rh_vox_MSE[idx,:] = MSE
    #rh_vox_PE[idx] = coeffs
    #if idx % 1e1 == 0:
    print 'Working voxel %d/%d\n' % (idx,num_vox)

matplotlib.pyplot.plot(np.mean(rh_vox_r,0))
matplotlib.pyplot.show()


matplotlib.pyplot.plot(np.mean(rh_vox_r[:,0:10],1))
matplotlib.pyplot.show()

empty_array = np.zeros((rh_1.shape[0],rh_1.shape[1]))
nonan = ~np.isnan(rh_1[:,:,1])
to_add = np.mean(rh_vox_r[:,0:10],1)
for idx in range(0,suprathresh.shape[0]):

    conv_coors = ind2sub(rh_1_.shape,suprathresh[idx])
    empty_array[nonan[conv_coors[0]][conv_coors[1]]] = to_add[idx]

empty_array[[suprathresh]] = to_add
np.reshape(empty_array,rh_1.shape[0],rh_1.shape[1])


matplotlib.pyplot.imshow(empty_array)
matplotlib.pyplot.show()

(np.mean(rh_vox_r,1))
matplotlib.pyplot.show()




# y1 = rh_1_array[movie_idx[1]:movie_idx[2]-1,suprathresh[1]]
# y2 = rh_2_array[movie_idx[1]:movie_idx[2]-1,suprathresh[1]]
# tx = Xtrim[movie_idx[1]:movie_idx[2]-1,:]

# clf = SVR(kernel='linear', C=10, gamma=0.1, verbose = True, max_iter = 1000)
# #clf = linear_model.Ridge(alpha=1,fit_intercept=True).fit(X,rh_1_array)
# clf.fit(tx,y1) #Fit to first half
# coeffs = clf.coef_
# y2_hat = clf.predict(tx)
# r2 = clf.score(tx, y2)
# mse = metrics.mean_squared_error(y2,y2_hat)


# matplotlib.pyplot.plot(y1)
# matplotlib.pyplot.plot(y2)
# matplotlib.pyplot.plot(y2_hat)
# matplotlib.pyplot.legend(['y1','y2','y2_hat'], 'upper left')
# matplotlib.pyplot.show()


