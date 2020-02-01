import pandas as pd
import numpy as np
from statistical_features import extract_stat_features
from frequency_features import extract_freq_features
import glob
import os
from pathlib import Path

"""
Data is written as a csv file with this format

Columns: headers for all channels; ex: mean_0 - TP9, mean_1 - AF7
									   mean_2 - AF8, mean_3 - TP10

Last column represents the class each data belongs to
The class is deduced from the filename being process
Classification:
	netrual: 0
	focused: 1
	relaxed: 2

Rows: processed data for each time window from each file
	ex: file 1: all time windows ( 0 - 60 => 119 rows, with 0.5 overlap)
		file 2: all time windows ( 0 - 60 => 119 rows, with 0.5 overlap)
		...
		file n: all time windows ( 0 - 60 => 119 rows, with 0.5 overlap)

"""

if __name__ == '__main__':

    path_prefix = "../../"

    output_file = 'processed-mental-states.csv'
    output_dir = Path(path_prefix + 'dataset/processed-data/')
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = path_prefix + 'dataset/raw-data/'
    input_files = '*.csv'

    output = pd.DataFrame()

    for file in glob.glob(input_path + input_files):
        clas = [0 if "neutral" in file else 1 if "focused" in file else 2]

        print('Processing file: {}\tClass: {}'.format(file, clas))
        data = pd.read_csv(file)
        # Discard Right AUX channel as we don't need it
        try:
            data = data.drop('Right AUX', axis=1)
        except:
            pass

        # Normalize time from unix time to seconds
        # and select 60 sec of data from 65 extracted
        data['timestamps'] = data['timestamps'] - data['timestamps'].iloc[0]
        data = data.loc[(data['timestamps'] >= 3) & (data['timestamps'] < 63)]
        data['timestamps'] = data['timestamps'] - data['timestamps'].iloc[0]
        data.rename(columns={'timestamps': 'Time'}, inplace=True)
        data.reset_index(drop=True, inplace=True)

        stat_result, stat_features = extract_stat_features(data, clas)
        freq_result, freq_features = extract_freq_features(data, clas)

        temp_res = pd.concat([stat_result, freq_result], axis=1, sort=False)
        temp_res['class'] = clas * np.ones(len(temp_res.index))

        output = output.append(temp_res)

    print(f'\nFull window statistical features: {stat_features[0]}')
    print(f'Half window statistical features: {stat_features[1]}')
    print(f'Quarter window statistical features: {stat_features[2]}')
    print(f'Total statistical features: {sum(stat_features)}\n')

    print(f'Total frequency features:{freq_features}\n')

    print(f'Total features:{sum(stat_features)+freq_features}\n')

    # Shuffle the output before writing it to file
    output = output.sample(frac=1)
    output.reset_index()

    output.to_csv(output_dir / output_file, index=False)

    print(f'Finished writing data to : {output_dir.resolve() / output_file}\n')
