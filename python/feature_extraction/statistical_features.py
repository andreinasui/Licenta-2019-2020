import numpy as np
import pandas as pd

OVERLAP = 0.5 # window overlap in seconds
# Window length is 1 second long

# Extracts statistical features
def extract_stat_features(data, classification):
	start = 0
	end = 1

	result = []

	result_f = pd.DataFrame()
	result_h = pd.DataFrame()
	result_q = pd.DataFrame()
	fw_time = data.loc[(data['Time'] >= start) & (data['Time'] < end)]

	while end <= 60:
		result_f = result_f.append(full_window_features(fw_time), sort=False)
		result_h = result_h.append(half_window_features(fw_time), sort=False)
		result_q = result_q.append(quarter_window_features(fw_time), sort=False)
		
		# Advance window
		start += OVERLAP
		end = start + 1
		fw_time = data.loc[(data['Time'] >= start) & (data['Time'] < end)]
	
	result.append(result_f)
	result.append(result_h)
	result.append(result_q)

	result = pd.concat(result,axis=1, sort=False)

	return result, [len(result_f.columns), 
		len(result_h.columns), len(result_q.columns)]


# We got 38 features from here
def full_window_features(window):
	# Remove Time columns so we have only the signal channels
	window = window.drop('Time', axis=1)

	# Extract mean
	# -> 4 features	
	fw_mean = window.mean()
	fw_mean.index = [f'fw_mean_{count}' for count,idx in enumerate(fw_mean.index)]

	# Extract standard deviation
	# -> 4 features	
	fw_std = window.std()
	fw_std.index = [f'fw_stddev_{count}' for count,idx in enumerate(fw_std.index)]

	# Extract min value 
	# -> 4 features	fw_min = window.min()
	fw_min = window.min()
	fw_min.index = [f'fw_min_{count}' for count,idx in enumerate(fw_min.index)]

	# Extract max value
	# -> 4 features	
	fw_max = window.max()
	fw_max.index = [f'fw_max_{count}' for count,idx in enumerate(fw_max.index)]

	# Extract skewness
	# -> 4 features
	fw_skewness = window.skew()
	fw_skewness.index = [f'fw_skew_{count}' for count,idx in enumerate(fw_skewness.index)]

	# Extract kurtosis
	# -> 4 features
	fw_kurtosis = window.kurtosis()
	fw_kurtosis.index = [f'fw_kurt_{count}' for count,idx in enumerate(fw_kurtosis.index)]

	# Signals covariance matrix
	fw_covar_matrix = window.cov()

	# Extract variance of signals from cov matrix
	# -> 4 features
	fw_var = pd.Series(np.diag(fw_covar_matrix), index=['variance_0','variance_1',
	'variance_2','variance_3'])	

	# Extract signals covariance with each other from cov matrix
	# -> 6 features
	fw_covar = pd.Series({'covar_0':fw_covar_matrix.iloc[0,1], 'covar_1':fw_covar_matrix.iloc[0,2], 
                          'covar_2':fw_covar_matrix.iloc[0,3], 'covar_3':fw_covar_matrix.iloc[1,2],
                          'covar_4':fw_covar_matrix.iloc[1,3], 'covar_5':fw_covar_matrix.iloc[2,3]})
	
	# Extract eigenvalues from cov matrix
	# -> 4 features
	fw_eigen = np.linalg.eigvals(fw_covar_matrix)
	fw_eigen = pd.Series({'eignval_0':fw_eigen[0],'eignval_1':fw_eigen[1],
		'eignval_2':fw_eigen[2],'eignval_3':fw_eigen[3],})

	return pd.DataFrame(data=pd.concat([fw_mean,fw_std,fw_min,fw_max,fw_skewness,
		fw_kurtosis,fw_var,fw_covar,fw_eigen], sort=False)).T


# We got 16 features from here
def half_window_features(window):
	# Remove Time columns so we have only the signal channels
	window = window.drop('Time', axis=1)

	hw = np.vsplit(window,2)

	# Compute the change in the sample means between 1st and 2nd half-window 
	# -> 1 feature per channel
	hw_d_mean = (hw[1].mean() - hw[0].mean())
	hw_d_mean.index = [f'hw_d_mean_{count}' for count,idx in enumerate(hw_d_mean.index)]

	# Compute the change in the sample std deviation between 1st and 2nd half-window
	# -> 1 feature per channel
	hw_d_stddev = (hw[1].std() - hw[0].std())
	hw_d_stddev.index = [f'hw_d_stddev_{count}' for count,idx in enumerate(hw_d_stddev.index)]
	
	# Compute the change in the sample max between 1st and 2nd half-window
	# -> 1 feature per channel
	hw_d_max = (hw[1].max() - hw[0].max())
	hw_d_max.index = [f'hw_d_max_{count}' for count,idx in enumerate(hw_d_max.index)]

	# Compute the change in the sample min between 1st and 2nd half-window
	# -> 1 feature per channel
	hw_d_min = (hw[1].min() - hw[0].min())
	hw_d_min.index = [f'hw_d_min_{count}' for count,idx in enumerate(hw_d_min.index)]


	return pd.DataFrame(data=pd.concat([hw_d_mean,hw_d_stddev,
		hw_d_max,hw_d_min], sort=False)).T


# We got 120 features from here
def quarter_window_features(window):
	# Remove Time columns so we have only the signal channels
	window = window.drop('Time', axis=1)

	qw = np.vsplit(window,4)
	# Compute mean for each quarter window
	# -> 16 features
	qw_mean = [x.mean() for x in qw]

	# Compute mean differences between each quarter window
	# -> 24 features
	qw_d_mean = []
	qw_d_mean.append((qw_mean[1] - qw_mean[0]))
	qw_d_mean.append((qw_mean[2] - qw_mean[0]))
	qw_d_mean.append((qw_mean[3] - qw_mean[0]))
	qw_d_mean.append((qw_mean[2] - qw_mean[1]))
	qw_d_mean.append((qw_mean[3] - qw_mean[1]))
	qw_d_mean.append((qw_mean[3] - qw_mean[2]))

	qw_mean = pd.DataFrame(pd.concat(qw_mean, sort=False))
	qw_mean.index = [f'qw_mean_{count}' for count,idx in enumerate(qw_mean.index)]

	qw_d_mean = pd.DataFrame(pd.concat(qw_d_mean, sort=False))
	qw_d_mean.index = [f'qw_d_mean_{count}' for count,idx in enumerate(qw_d_mean.index)]

	# Compute max and min for each quarter window
	# -> 32 features
	qw_max = [x.max() for x in qw]
	qw_min = [x.min() for x in qw]

	# Compute max and min differences between each quarter window
	# -> 48 features
	qw_d_max = []
	qw_d_min = []
	qw_d_max.append((qw_max[1] - qw_max[0]))
	qw_d_max.append((qw_max[2] - qw_max[0]))
	qw_d_max.append((qw_max[3] - qw_max[0]))
	qw_d_max.append((qw_max[2] - qw_max[1]))
	qw_d_max.append((qw_max[3] - qw_max[1]))
	qw_d_max.append((qw_max[3] - qw_max[2]))

	qw_d_min.append((qw_min[1] - qw_min[0]))
	qw_d_min.append((qw_min[2] - qw_min[0]))
	qw_d_min.append((qw_min[3] - qw_min[0]))
	qw_d_min.append((qw_min[2] - qw_min[1]))
	qw_d_min.append((qw_min[3] - qw_min[1]))
	qw_d_min.append((qw_min[3] - qw_min[2]))

	qw_max = pd.DataFrame(pd.concat(qw_max, sort=False))
	qw_max.index = [f'qw_max_{count}' for count,idx in enumerate(qw_max.index)]
	qw_min = pd.DataFrame(pd.concat(qw_min, sort=False))
	qw_min.index = [f'qw_min_{count}' for count,idx in enumerate(qw_min.index)]


	qw_d_max = pd.DataFrame(pd.concat(qw_d_max, sort=False))
	qw_d_max.index = [f'qw_d_max_{count}' for count,idx in enumerate(qw_d_max.index)]
	qw_d_min = pd.DataFrame(pd.concat(qw_d_min, sort=False))
	qw_d_min.index = [f'qw_d_min_{count}' for count,idx in enumerate(qw_d_min.index)]

	return pd.concat([qw_mean, qw_d_mean,qw_max,
		qw_min,qw_d_max,qw_d_min], sort=False).T