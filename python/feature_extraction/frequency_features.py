import pandas as pd
from scipy import signal
import numpy as np

OVERLAP = 0.5 # window overlap in seconds
# Window length is 1 second long

# Extracts frequency features
def extract_freq_features(data, classification):
	start = 0
	end = 1

	result = pd.DataFrame()
	w_time = data.loc[(data['Time'] >= start) & (data['Time'] < end)]

	while end <= 60:
		result = result.append(freq_features(w_time), sort=False)

		# Advance window
		start += OVERLAP
		end = start + 1
		w_time = data.loc[(data['Time'] >= start) & (data['Time'] < end)]


	return result, len(result.columns)


def freq_features(window):
	# Save Time column in a separate data frame
	time_window = pd.DataFrame(data=window.Time)
	# Remove Time column so we have only the signal channels
	window = window.drop('Time', axis=1)

	# Center the data
	window = window - window.mean()

	# Apply a low pass filter at 50 Hz to filter out the AC noise ;
	# Brain waves freq go up to 50 Hz, so we filter out everything above that

	Fs = 256			# sampling rate
	Fn = Fs / 2			# Nyquist frequency
	f_low = 50 / Fn 	# normalized freq for lowpass filter

	# Chebyshev I order 6 lowpass filter
	sos = signal.cheby1(6, 0.5, f_low, btype='low', output='sos')

	# Apply filter on all columns
	window.TP9 = signal.sosfilt(sos, window.TP9)
	window.AF7 = signal.sosfilt(sos, window.AF7)
	window.AF8 = signal.sosfilt(sos, window.AF8)
	window.TP10 = signal.sosfilt(sos, window.TP10)

	# Compute FFT
	freq_axis = (Fs/2)*np.linspace(0,1,len(window.index)//2 + 1)
	dataTP9 = np.fft.rfft(window.TP9)
	dataAF7 = np.fft.rfft(window.AF7)
	dataAF8 = np.fft.rfft(window.AF8)
	dataTP10 = np.fft.rfft(window.TP10)

	# Get magnitude for each signal component
	# -> 200 features
	magTP9 = pd.Series(np.asarray([abs(dataTP9[np.where(freq_axis == i+1)]) for i in range(50)])[:,0])
	magAF7 = pd.Series(np.asarray([abs(dataAF7[np.where(freq_axis == i+1)]) for i in range(50)])[:,0])
	magAF8 = pd.Series(np.asarray([abs(dataAF8[np.where(freq_axis == i+1)]) for i in range(50)])[:,0])
	magTP10 = pd.Series(np.asarray([abs(dataTP10[np.where(freq_axis == i+1)]) for i in range(50)])[:,0])

	mags = pd.concat([magTP9,magAF7,magAF8,magTP10],axis=0,sort=False)
	mags.index = [f'fft_mag_{count}' for count in range(len(mags))]

	# Get the first 10 most energetic frequencies
	# -> 40 features
	maxTP9 = pd.Series(magTP9.nlargest(10).sort_index())
	maxAF7 = pd.Series(magAF7.nlargest(10).sort_index())
	maxAF8 = pd.Series(magAF8.nlargest(10).sort_index())
	maxTP10 = pd.Series(magTP10.nlargest(10).sort_index())

	maxs = pd.concat([maxTP9,maxAF7,maxAF8,maxTP10], axis=0, sort=False)
	maxs.index = [f'fft_max_{count}' for count in range(len(maxs))]

	# Construct a DataFrame from these arrays to send to output
	return pd.DataFrame(data=pd.concat([mags,maxs], sort=False)).T
