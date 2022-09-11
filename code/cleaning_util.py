"""
cleaning_util.py

Utilities that may be used for data cleaning
"""

# Imports
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import math
import datetime


# Sleep map
sleep_map = {
    1: 3,
    2: 3.5,
    3: 4,
    4: 4.5,
    5: 5,
    6: 5.5,
    7: 6,
    8: 6.5,
    9: 7,
    10: 7.5,
    11: 8,
    12: 8.5,
    13: 9,
    14: 9.5,
    15: 10,
    16: 10.5,
    17: 11,
    18: 11.5,
    19: 12
}


def map_epoch(timestamp):
    """
    Map quarter a timestamp to an epoch of a day

    :param timestamp: <datetime>, a datetime

    return epoch, the epoch of the day
    """
    if timestamp.hour < 6:
        return 1
    elif timestamp.hour < 12:
        return 2
    elif timestamp.hour < 18:
        return 3
    else:
        return 4


def localize_time(timedata):
    """
    Convert to EST (both datasets east coast U.S.)

    :param timedata: list<datetime>, list of UTC datetime
    
    :return: shifted time
    """
    return pd.to_datetime(list(timedata), unit="s").tz_localize("UTC").tz_convert('US/Eastern')


def fill_hours(df):
    """
    Fill hours of a DataFrame

    :param df: pd.DataFrame, the df to fill hours

    :return: pd.DataFrame, the df with filled hours 
    """
    # Get min and max hour
    min_hour, max_hour = df.hour.min(), df.hour.max()
    # Now create a df of all the hours in-between
    hours = pd.DataFrame(
        pd.date_range(min_hour, max_hour, freq='H')[1:-1], columns=['hour']
    )
    # Filter to hours not in the other df
    # hours = hours.loc[~hours.hour.isin(df.hour), :]
    hours['study_id'] = df['study_id'].unique()[0]
     # Prep for merging
    hours['datetime'] = hours['hour'].copy()
    hours['day'] = hours['hour'].dt.floor('d')
    # Merge
    return pd.merge(
        left=df, right=hours, how='outer', on=['hour', 'study_id', 'datetime', 'day']
    ).sort_values('datetime').reset_index(drop=True)


def get_good_days(dfs):
    """
    Get the days with w/ <19 hours of data.
    These are the days that should remain for analysis.

    Using the activity dfs which should have samples almost every minute.
    """
    day_dfs = []
    for f in list(dfs.keys()):
        # Organize columns
        df = dfs[f].copy()
        df['study_id'] = int(f.split('_')[1][1:3])
        df['datetime'] = localize_time(df['timestamp'])
        df['day'] = df['datetime'].dt.floor('d')
        df['hour'] = df['datetime'].dt.floor('h')

        # Group
        df_grouped = df.groupby(['study_id', 'day'], as_index=False)['hour'].nunique()
        df_grouped = df_grouped.loc[df_grouped.hour >= 19, :]
        day_dfs.append(df_grouped)

    # Concatenate and return
    return pd.concat(day_dfs).reset_index(drop=True)


def prep_studentlife_df(df_dict):
    """
    Prep StudentLife dfs

    :param df_dict: <pd.DataFrame>, dict of DFs

    :return: <pd.DataFrame> cleaned DF
    """
    # Initialize list for dfs
    dfs = []

    # Go through dfs and add study IDs
    for k in df_dict.keys():
        df = df_dict[k].copy()
        if df.shape[0] > 0:
            df['study_id'] = int(k.split('_')[1][1:3])
            df['data_type'] = k.split('_')[0]
            df['date'] = localize_time(df['resp_time'])
            df['day'] = df.date.dt.floor('d')
            # Drop "null" columns
            if 'null' in df.columns:
                df.drop(['null'], axis=1, inplace=True)
            dfs.append(df.dropna())

    return pd.concat(dfs).reset_index(drop=True)


def prep_ema_data(dfs):
    """
    Prep EMA data

    :param dfs: list<pd.DataFrame>, list of EMA dfs
    
    :return: merged_df, the df with all merged EMA data
    """
    # Create dummary var
    merged_df = None

    # GO through eeach df
    for df in dfs:
        # Group to get average over day
        cols = [c for c in df if c not in ['location', 'resp_time', 'date', 'study_id', 'day', 'data_type']]
        # Convert to float
        for c in cols:
            df.loc[df[c] == 'null', c] = None
            df[c] = df[c].astype(float)
        df_grouped = df.groupby(['study_id', 'day'], as_index=False)[cols].mean()
        
        # Rename
        data_type = df.data_type.unique()[0]
        rename_cols = ['ema_' +  data_type + '_' + c for c in cols]
        df_grouped.rename(columns=dict(zip(cols, rename_cols)), inplace=True)

        # Now merge
        if merged_df is not None:
            merged_df = pd.merge(left=merged_df, right=df_grouped, on=['study_id', 'day'], how='outer')
        else:
            merged_df = df_grouped.copy()

    return merged_df


def clean_studentlife_activity(dfs):
    """
    Clean the activity data

    :param dfs: dict<pd.DataFrame>, dictionary of activity dfs

    :return: daily activity df
    """
    activity_map = {
        0: 'act_still',
        1: 'act_walking',
        2: 'act_running',
        3: 'act_unknown'
    }

    activity_df = []

    for f in list(dfs.keys()):
        # Organize columns
        df = dfs[f].copy()
        df['study_id'] = int(f.split('_')[1][1:3])
        df['datetime'] = localize_time(df['timestamp'])
        df['day'] = df['datetime'].dt.floor('d')
        df['hour'] = df['datetime'].dt.floor('h')
        # Fill hours data
        df = fill_hours(df)
        df['act'] = df[' activity inference'].map(activity_map)
        df['epoch'] = df['datetime'].apply(map_epoch)
        df.sort_values(by='datetime', inplace=True)
        df['duration'] = list(
            (df['datetime'].diff(periods=-1) * -1).iloc[:-1].astype(int) / 1e9
        ) + [None]
        # Drop missing hours
        df = df.dropna().reset_index(drop=True)
        
        # Group by day and epoch, sum
        df_grouped_day = df.groupby(
            ['study_id', 'day', 'act'], as_index=False)['duration'].sum()
        df_grouped_epoch = df.groupby(
            ['study_id', 'day', 'epoch', 'act'], as_index=False)['duration'].sum()
        
        # Concatenate together
        df_grouped_day['epoch'] = 0
        df_grouped = pd.concat([df_grouped_day, df_grouped_epoch], sort=True).reset_index(drop=True)
        
        # Pivot
        df_pivot = pd.pivot_table(
            df_grouped, index=['study_id', 'day'], columns=['act', 'epoch'], values=['duration']).reset_index()
        df_pivot.columns = [i[0] if i[1] == '' else i[1] + '_ep_' + str(i[2]) for i in df_pivot.columns]
        df_pivot.fillna(0, inplace=True)
        
        # Append
        activity_df.append(df_pivot)

    # Concatenate
    activity_df = pd.concat(activity_df).reset_index(drop=True)

    # Add the "on foot"
    for e in range(5):
        activity_df['act_on foot_ep_' + str(e)] = \
            activity_df['act_walking_ep_' + str(e)] + activity_df['act_running_ep_' + str(e)]

    return activity_df.fillna(0)


def clean_studentlife_conversations(dfs):
    """
    Clean the conversation data

    :param dfs: dict<pd.DataFrame>, dictionary of conversations dfs

    :return: daily conversation df
    """
    # Intialize df
    conversation_df = []

    for f in list(dfs.keys()):
        df = dfs[f].copy()
        df['study_id'] = int(f.split('_')[1][1:3])

        # Calculate timing information
        df['start_datetime'] = localize_time(df['start_timestamp'])
        df['end_datetime'] = localize_time(df[' end_timestamp'])
        df['epoch'] = df['start_datetime'].apply(map_epoch)
        df['day'] = df.start_datetime.dt.floor('d')

        # Calculate duration
        df['end_datetime'].head()
        df['start_datetime'].head()
        df['audio_convo'] = (df['end_datetime'] - df['start_datetime']).dt.seconds

        # Group
        df_grouped_day = df.groupby(
            ['study_id', 'day'], as_index=False
        )['audio_convo'].agg(['sum', 'count']).reset_index()
        df_grouped_epoch = df.groupby(
            ['study_id', 'day', 'epoch'], as_index=False
        )['audio_convo'].agg(['sum', 'count']).reset_index()

        # Concatenate together
        df_grouped_day['epoch'] = 0
        df_grouped = pd.concat([df_grouped_day, df_grouped_epoch], sort=True).reset_index(drop=True)

        # Rename
        df_grouped.rename(
            columns={'sum': 'audio_convo_duration', 'count': 'audio_convo_num'}, 
            inplace=True
        )
        
        # Pivot
        df_pivot = pd.pivot_table(
            df_grouped, index=['study_id', 'day'], 
            columns=['epoch'], values=['audio_convo_duration', 'audio_convo_num']
        ).reset_index()
        df_pivot.columns = [i[0] if i[1] == '' else i[0] + '_ep_' + str(i[1]) for i in df_pivot.columns]
        df_pivot.fillna(0, inplace=True)
        
        # Add onto list
        conversation_df.append(df_pivot)
    
    # Concatenate and return
    conversation_df = pd.concat(conversation_df).reset_index(drop=True)
    return conversation_df.fillna(0)


def update_times(df, ind, indices, start_1, end_1):
    """
    Update times for unlock duration
    
    :param df: pd.DataFrame, the df with start and end times
    :param ind: <int>, a current index
    :param indices: list<int>, a list of indices
    :param start_1: datetime, the datetime of the original unlock
    :param end_1: datetime, the end datetime of the original unlock
    
    :return ind: <int>, the current index of end_1
    :return indices: list<int>, the list of updates indices
    :return start_1: datetime, the start datetime
    :reutrn end_1: datetie, the update end datetime
    """
    # Look for all start times that are greater than this index
    # But are less than the end time
    temp = df.loc[(ind + 1):, :]
    # Get the rest of the start_datetimes that are < end_1
    temp = temp.loc[(temp.start_datetime) < end_1, :]
    # Sort with the max value at the beginning
    temp.sort_values(by='datetime', ascending=False, inplace=True)
    
    # check if temp > 0
    if temp.shape[0] > 0:
        # Make sure the update is valid, new ending point should be
        # greater than old
        if temp.iloc[0, :]['datetime'] > end_1:
            end_1 = temp.iloc[0, :]['datetime']
        # Update the index list
        indices = [i for i in indices if i not in temp.index]
        # Update the current index
        ind = temp.iloc[[0], :].index[0]
        return update_times(df, ind, indices, start_1, end_1)
    else:
        return ind, indices, start_1, end_1


def fix_unlock_df_timing(df):
    """
    Fix the timing overlap in the unlock df

    :param df: pd.DataFrame, the unlock df times
    
    :return: pd.DataFrame, the cleaned df
    """
    # Setup lists for df
    start_datetime = []
    end_datetime = []

    # Setup 
    curr = 0
    indices = df.index
    end_data = df.shape[0]
    while curr < len(indices):
        # Get the current index
        ind = indices[curr]
        
        # Get the start datetimes
        start_1 = df.loc[ind, 'start_datetime']    
        # Get the end datetime
        end_1 = df.loc[ind, 'datetime']
        
        # Get the updated datetime
        ind, indices, start_1, end_1 = update_times(df, ind, indices, start_1, end_1)
        start_datetime.append(start_1)
        end_datetime.append(end_1)
            
        # Move to next index
        curr += 1
            
    # Now put together into dataframe
    df2 = pd.DataFrame({
        'start_datetime': start_datetime,
        'datetime': end_datetime,
    })
    df2['study_id'] = df.study_id.unique()[0]

    return df2


def fix_epoch_hours(df):
    """
    Fix epoch hours. Columns should be 
    study_id, start, end

    :param df: pd.DataFrame with start and end times
    
    :return: the fixed df with correct cutoffs
    """
    start = []
    end = []
    for ind in df.index:
        # Get hours in between start and end
        start_hour, end_hour = df.loc[ind, 'start'].floor('h'), df.loc[ind, 'end'].floor('h')
        # Get all hours in-between start and end
        hours = pd.date_range(start_hour, end_hour, freq='H')[1:]
        # Go through hours and append
        start.append(df.loc[ind, 'start'])
        for h in hours:
            end.append(h)
            start.append(h)
        # Now append end
        end.append(df.loc[ind, 'end'])

    # Make end df
    study_id = df.study_id.unique()[0]
    df = pd.DataFrame({'start': start, 'end': end})[['start', 'end']]
    df['study_id'] = study_id

    return df


def clean_studentlife_unlock(dfs):
    """
    Clean the unlock phone data

    :param dfs: dict<pd.DataFrame>, dictionary of unlock dfs

    :return: daily unlock df
    """
    # Intialize df
    unlock_df = []

    for f in list(dfs.keys()):
        df = dfs[f].copy()
        df['study_id'] = int(f.split('_')[1][1:3])
        
        # Calculate timing information
        df['start_datetime'] = localize_time(df['start'])
        df['datetime'] = localize_time(df['end'])
        df = df.sort_values(by='start_datetime').reset_index(drop=True)

        # Now fix times in unlock df
        df = fix_unlock_df_timing(df)

        # Shift the start time as the end time in the previous row
        df['end_datetime_2'] = None
        df.loc[0:(df.shape[0] - 2), 'end_datetime_2'] = pd.to_datetime(
            df.iloc[1:, :]['start_datetime'].values
        )
        df = df.dropna().reset_index(drop=True)
        df['end_datetime_2'] = pd.to_datetime(
            df['end_datetime_2'].values
        ).tz_localize('UTC').tz_convert("US/Eastern")
        df = df.loc[df['datetime'] < df['end_datetime_2'], :].reset_index(drop=True) 

        # Rename
        df = df[['study_id', 'datetime', 'end_datetime_2']]
        df.columns = ['study_id', 'start', 'end']
        df = df.sort_values(by=['start']).reset_index(drop=True)

        # Fill hours
        df = fix_epoch_hours(df)
        df['epoch'] = df['start'].apply(map_epoch)
        df['day'] = df['start'].dt.floor('d')

        # Calculation the unlock duration
        df['unlock_duration'] = (df['end'] - df['start']).dt.seconds   

        # Group
        df_grouped_day = df.groupby(
            ['study_id', 'day'], as_index=False
        )['unlock_duration'].sum()
        df_grouped_epoch = df.groupby(
            ['study_id', 'day', 'epoch'], as_index=False
        )['unlock_duration'].sum()

        # Concatenate together
        df_grouped_day['epoch'] = 0
        df_grouped = pd.concat([df_grouped_day, df_grouped_epoch], sort=True).reset_index(drop=True)

        # Pivot
        df_pivot = pd.pivot_table(
            df_grouped, index=['study_id', 'day'], 
            columns=['epoch'], values=['unlock_duration']
        ).reset_index()
        df_pivot.columns = [i[0] if i[1] == '' else i[0] + '_ep_' + str(i[1]) for i in df_pivot.columns]
        df_pivot.fillna(0, inplace=True)

        # Add onto list
        unlock_df.append(df_pivot)

    # Concatenate and return
    unlock_df = pd.concat(unlock_df).reset_index(drop=True)
    return unlock_df.fillna(0)


def distance(origin, destination):
    """
    Compute distance between two points

    :param origin: (<float>, <float>) lat, long location origin
    :param destination: (<float>, <float>) lat, long location destination

    :return: <float> the distance in m between points
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c

    # Return distances in meters
    return d*1000
    

def clean_studentlife_location(dfs):
    """
    Clean the GPS phone data

    :param dfs: dict<pd.DataFrame>, dictionary of GPS dfs

    :return: daily GPS df
    """
    # Intialize df
    loc_df = []

    # Initialize km's per radian calculation
    kms_per_radian = 6371.0088
    # 10 meters per cluster
    epsilon = 0.01 / kms_per_radian

    for f in list(dfs.keys()):
        df = dfs[f].copy()
        df['study_id'] = int(f.split('_')[1][1:3])

        # Fix columns
        cols = list(df.columns)
        df.reset_index(inplace=True)
        df.drop(['travelstate'], axis=1, inplace=True)
        df.columns = cols
        
        # Calculate timing information
        df['datetime'] = localize_time(df['time'])
        df['epoch'] = df['datetime'].apply(map_epoch)
        df['day'] = df.datetime.dt.floor('d')

        # Convert to radians
        df['latitude_rad'] = np.radians(df['latitude'])
        df['longitude_rad'] = np.radians(df['longitude'])

        # Cluster locations based upon 10 meters from CrossCheck paper
        df['loc_visit_num'] = DBSCAN(
            eps=epsilon, min_samples=10, metric='haversine'
        ).fit_predict(df[['latitude_rad', 'longitude_rad']])

        # Get distances
        df['loc_dist'] = 0
        for i in df.index[:-1]:
            origin = df.loc[i, 'latitude'], df.loc[i, 'longitude']
            dest = df.loc[i + 1, 'latitude'], df.loc[i + 1, 'longitude']
            df.loc[i, 'loc_dist'] = distance(origin, dest)

        # Group
        df_grouped_day = df.groupby(
            ['study_id', 'day'], as_index=False
        ).agg({'loc_dist': 'sum', 'loc_visit_num': 'nunique'})
        df_grouped_epoch = df.groupby(
            ['study_id', 'day', 'epoch'], as_index=False
        ).agg({'loc_dist': 'sum', 'loc_visit_num': 'nunique'})

        # Concatenate together
        df_grouped_day['epoch'] = 0
        df_grouped = pd.concat([df_grouped_day, df_grouped_epoch], sort=True).reset_index(drop=True)

        # Pivot
        df_pivot = pd.pivot_table(
            df_grouped, index=['study_id', 'day'], 
            columns=['epoch'], values=['loc_dist', 'loc_visit_num']
        ).reset_index()
        df_pivot.columns = [i[0] if i[1] == '' else i[0] + '_ep_' + str(i[1]) for i in df_pivot.columns]
        df_pivot.fillna(0, inplace=True)

        # Add onto list
        loc_df.append(df_pivot)

    # Concatenate and return
    loc_df = pd.concat(loc_df).reset_index(drop=True)
    return loc_df.fillna(0)


def filter_short_unlocks(df, cutoff_duration=30):
    """
    Filter out unlocks <= cutoff_duration seconds
    Will recursively call function on udpated DataFrames until
    no short unlocks exist.

    :param df: <pd.DataFrame>, DataFrame with sleep information
    :return: <pd.DataFrame>, the unlocks filtered
    """
    # If any changes occur set this to true
    callback = False

    # Append first unlock
    start_est = [df.loc[0, :]['start_est']]
    end_est = []

    # Set index to 1
    ind = 1
    while ind < df.shape[0]:
        # Need to concatenate subsequent rows
        if (df.loc[ind, 'start_est'] - df.loc[ind - 1, 'end_est']).seconds < cutoff_duration:
            end_est.append(df.loc[ind, 'end_est'])
            # Set call back to function
            callback = True
            if ind < (df.shape[0] - 2):
                start_est.append(df.loc[ind + 1, 'start_est'])
            # Skip the next index
            ind += 2
        else:
            end_est.append(df.loc[ind - 1, 'end_est'])
            if ind < (df.shape[0] - 1):
                start_est.append(df.loc[ind, 'start_est'])
            # Move to the next index
            ind += 1

    # Check to see if last end_est needs to be appended
    if start_est[-1] == df.iloc[-1, :]['start_est']:
        end_est.append(df.iloc[-1, :]['end_est'])

    # Make df
    df_filtered = pd.DataFrame({
        'start_est': start_est,
        'end_est': end_est
    })

    # Call function back of return
    if callback:
        return filter_short_unlocks(df_filtered, cutoff_duration=cutoff_duration)
    else:
        return df_filtered
    

def sleep_epochs(sleep_start):
    """
    Intake time, output 7.5 minute epochs from 8PM
    
    :param sleep_start: <datetime>, the sleep start time 
    :return: <float>, 7.5 minute
    """
    # Check if hour is less than 7 (7AM)
    if sleep_start.hour <= 7:
        start_time = sleep_start.floor('d') - datetime.timedelta(hours=4)
    else:
        start_time = sleep_start.floor('d') + datetime.timedelta(hours=20)
        
    return ((sleep_start - start_time).seconds / 3600) * 8


def get_day_for_sleep(sleep_start):
    """
    Get day for sleep

    :param sleep_start: <datetime>, the sleep start time 
    :return: <datetime>, the day
    """
    if sleep_start.hour <= 7:
        return sleep_start.floor('d')
    else:
        return sleep_start.floor('d') + datetime.timedelta(days=1)


def correct_sleep(df, ema_df, correction='mean'):
    """
    Correct sleep based upon actual sleep

    :param df: pd.DataFrame, the DataFrame
    :param ema_df: pd.DataFrame, the EMA df
    :param correction: <str>, whether to use  the mean or median corrective term
                            the mean corrects the square error, the median the absolute error

    :return: pd.DataFrame, the corrected sleep
    """
    # Map hours of sleep
    ema_df['sleep_hour'] = ema_df['ema_Sleep_hour'].map(sleep_map)
    # Shift EMA df day back by 1
    ema_df['day'] = pd.to_datetime(ema_df['day'])  # - datetime.timedelta(days=1)

    # Now merge
    sleep_df_merged = pd.merge(
        left=df, right=ema_df[['study_id', 'day', 'sleep_hour']], on=['study_id', 'day']
    ).dropna()

    # Calculate corrective term
    sleep_df_merged['err'] = sleep_df_merged['sleep_hour'] - sleep_df_merged['sleep_duration']

    if correction == 'mean':
        corrective_term = sleep_df_merged['err'].mean()
    else:
        corrective_term = sleep_df_merged['err'].median()

    df['sleep_duration'] += corrective_term

    return df


def clean_sleep_data(df_dict, cutoff_duration=30, start_time=22, ema_df=None, correction='mean'):
    """
    Create sleep duration from lock data
    
    :param df: dict<str:pd.DataFrame>, the sleep lock dataframes
    :param cutoff_duration: <int>, the time to cutoff phone unlocks
    :param start_time: <int>, the time to start looking at phone duration
    :param ema_df: pd.DataFrame, the ema_df to calculate the corrective term
    :param correction: <str>, whether to use mean/median error as correction
    :return: pd.DataFrame, the DataFrame
    """
    all_dfs = []
    for k in df_dict:
        # Get study ID
        study_id = int(k.split('_')[1][1:3])
        df = df_dict[k].copy().reset_index(drop=True)

        # 1. Convert to EST using pandas library
        df['start_est'] = localize_time(df['start'])
        df['end_est'] = localize_time(df['end'])
        
        # 2. Remove short unlocks
        df_filtered = filter_short_unlocks(df, cutoff_duration=cutoff_duration)

        # 3. Calculate lock duration
        df_filtered['lock_duration'] = df_filtered['end_est'] - df_filtered['start_est']
        
        # 4. Identify longest phone lock starting between start_time and 7AM as sleep
        # Filter to phone locks starting between sleep_start and 7AM
        df_filtered = df_filtered.loc[
            (df_filtered['start_est'].dt.hour >= start_time) | (df_filtered['start_est'].dt.hour < 7), :
        ]

        # Sort values by datetime and reset index so we can go through
        # time row by row
        df_filtered = df_filtered.sort_values(by='start_est').reset_index(drop=True)

        # Add counter to mark each period between xPM and 7AM
        curr = 0
        df_filtered['period'] = 0
        for ind in df_filtered.index[1:]:
            # Check to see if previous reading is more than start_time - 7 hours ago
            time_diff = (df_filtered.loc[ind, 'start_est'] - df_filtered.loc[ind - 1, 'start_est']).seconds // 3600
            if time_diff > (start_time - 7):
                curr += 1
            df_filtered.loc[ind, 'period'] = curr

        # Groupby the period and choose max lock duration
        idxmax = df_filtered.groupby(['period'])['lock_duration'].idxmax()
        df_filtered = df_filtered.loc[idxmax, :]

        # Clean up the file
        # Convert to epochs since 8PM
        df_filtered['sleep_start'] = df_filtered.start_est.apply(sleep_epochs)
        df_filtered['sleep_end'] = df_filtered.end_est.apply(sleep_epochs)
        df_filtered['day'] = df_filtered.start_est.apply(get_day_for_sleep)

        # Convert to hours
        df_filtered['sleep_duration'] = df_filtered['lock_duration'].dt.seconds / 3600
        df_filtered['study_id'] = study_id

        sleep_df = df_filtered[['study_id', 'day', 'sleep_start', 'sleep_end', 'sleep_duration']].reset_index(drop=True)

        # Correct
        if ema_df is not None:
            sleep_df = correct_sleep(df=sleep_df, ema_df=ema_df.loc[ema_df.study_id == study_id, :], correction=correction)

        all_dfs.append(sleep_df)
    
    return pd.concat(all_dfs).reset_index(drop=True)
