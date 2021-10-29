"""
run_cv.py

Main code to run the LOSO-CV procedure. Will default to using
multi-processing.

Command:
python run_cv.py
"""

# Imports
import pandas as pd
import numpy as np
import concurrent

# Created modules
import regression_cv

### IMPORTANT VARIABLES ###
params_dict = {
    'gbt': {
        'loss': ['huber'],
        'learning_rate': [0.001, 0.01, 0.1, 1.0],
        'n_estimators': [20, 100, 1000],
        'max_depth': [3, 7, 10],
        'random_state': [42]
    }
}
output_filename ='../res/loso_res_df_all_v11_10182021.csv.gz'


if __name__ == "__main__":
    # Open data
    crosscheck = pd.read_csv('../data/crosscheck_daily_data_cleaned_w_sameday.csv', index_col=0)
    studentlife = pd.read_csv('../data/studentlife_daily_data_cleaned_w_sameday_08282021.csv')

    # Prep

    # EMA cols
    ema_cols_crosscheck = [i for i in crosscheck.columns if 'ema' in i]
    ema_cols_studentlife = [i for i in studentlife.columns if 'ema' in i]

    # Get features
    # Behavior cols
    behavior_cols_crosscheck = [
        i for i in crosscheck.columns if i not in ['study_id', 'eureka_id', 'date'] + ema_cols_crosscheck
    ]
    behavior_cols_studentlife = [
        i for i in studentlife.columns if i not in ['study_id', 'eureka_id', 'day'] + ema_cols_studentlife
    ]

    behavior_cols = list(set(behavior_cols_crosscheck) & set(behavior_cols_studentlife))
    behavior_cols.sort()

    features = behavior_cols[:]

    crosscheck_temp = crosscheck.copy()
    crosscheck_temp[behavior_cols] = crosscheck_temp[behavior_cols].fillna(0) # Not using the columns with NAs. All 
                                                                            # ambient audio/light

    features = [f for f in features if len(crosscheck_temp[f].unique()) > 1]

    studentlife_temp = studentlife[['study_id', 'day'] + 
        behavior_cols + ['ema_Stress_level', 'ema_Sleep_rate']
    ].reset_index(drop=True).copy() # TEMP FILL

    # Fill NA
    non_sleep_loc_cols = [i for i in behavior_cols if ('loc' not in i) and ('sleep' not in i)]
    studentlife_temp[non_sleep_loc_cols] = studentlife_temp[non_sleep_loc_cols].fillna(0)

    # Fill sleep with average value for that individual
    for s in studentlife_temp.study_id.unique():
        temp = studentlife_temp.loc[studentlife_temp.study_id == s, :]
        duration_mean = temp['sleep_duration'].mean()
        start_mean = temp['sleep_start'].mean()
        end_mean = temp['sleep_end'].mean()
        ind = (studentlife_temp.study_id == s) & pd.isnull(studentlife_temp['sleep_duration'])
        studentlife_temp.loc[ind, 'sleep_duration'] = duration_mean
        studentlife_temp.loc[ind, 'sleep_start'] = start_mean
        studentlife_temp.loc[ind, 'sleep_end'] = end_mean

    # Drop days without location (14 total) and days still w/o sleep (all IDs with no sleep info)
    studentlife_temp = studentlife_temp.dropna(subset=behavior_cols).reset_index()
    # Need to map all of them from 0-3
    # Stress [1]A little stressed, [2]Definitely stressed, [3]Stressed out, [4]Feeling good, [5]Feeling great, 
    studentlife_temp['ema_STRESSED'] = studentlife_temp['ema_Stress_level'].map({
        5:0, 4:0, 1:1, 2:2, 3:3
    })
    # Map from 0 - 3
    minimum = studentlife_temp['ema_STRESSED'].min()
    maximum = studentlife_temp['ema_STRESSED'].max()
    studentlife_temp['ema_STRESSED'] =  3 * (studentlife_temp['ema_STRESSED'] - minimum) / (maximum - minimum)
    # Sleeping [1]Very good, [2]Fairly good, [3]Fairly bad, [4]Very bad, 
    # Map from 0 - 3
    studentlife_temp['ema_SLEEPING'] = 4 - studentlife_temp['ema_Sleep_rate'].copy().round()

    # Targets
    targets = ['ema_SLEEPING', 'ema_STRESSED']
    studentlife_temp['data'] = 'sl'
    crosscheck_temp['data']= 'cc'
    crosscheck_temp['day'] = pd.to_datetime(crosscheck_temp['date']).dt.tz_localize('US/Eastern')

    # Make study IDs dataset specific
    crosscheck_temp['study_id'] = 'cc' + crosscheck_temp['study_id'].astype(str)
    studentlife_temp['study_id'] = 'sl' + studentlife_temp['study_id'].astype(str)

    # Run LOSO
    # By study ID
    args = []

    for d in ['cc', 'sl', 'both']:
        if d == 'sl':
            data = studentlife_temp.copy()
        elif d == 'cc':
            data = crosscheck_temp.copy()
        else:
            data = pd.concat([crosscheck_temp, studentlife_temp], axis=0).reset_index(drop=True)
        for target in targets:
            for smote in [True, False]:
                for neighbors in [None, 5, 10, 50, 100, 500]:
                    for m in params_dict.keys():
                        for params in regression_cv.get_param_combinations(params_dict[m]):
                            print(params)
                            for t, v in zip(*regression_cv.get_loso_cv_data(
                                data=data, features=features, target=target
                            )):
                                # Each "fold" is a study ID since LOSO
                                fold = v.study_id.unique()[0]
                                args.append(
                                    [m, params, t, m, features, target, v, smote, neighbors]
                                )

    # Add counter
    total_args = len(args)
    for i in range(len(args)):
        args[i].append(total_args - i)

    # Collect outputs
    model_type_list = []
    smote_list = []
    neighbors_list = []
    data_list = []
    target_list = []
    fold_list = []
    params_list = []
    mae_list = []
    y_pred_list = []
    y_mean_list = []
    y_true_list = []
    days_list = []
    study_id_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for arg, output_val in zip(args, executor.map(
            regression_cv.train_validate_model, args)):
            model, mae, y_true, y_pred, study_id = output_val
            m, params, t, m, features, target, v, smote, neighbors, curr = arg
            orig_dataset = v.study_id.unique()[0][0:2]
            orig_study_ids = [i for i in t.study_id.unique() if orig_dataset in i]
            y_true_list.append(str(list(y_true)))
            y_pred_list.append(str(list(y_pred)))
            days_list.append(str(list(v['day'].values)))
            y_mean_list.append(
                str([t.loc[t.study_id.isin(orig_study_ids), target].mean()] * len(y_pred))
            )
            study_id_list.append(str(list(study_id)))
            if len(t['data'].unique()) > 1:
                data_list.append('both')
            else:
                data_list.append(v.study_id.unique()[0][:2])
            model_type_list.append(m)
            target_list.append(target)
            fold_list.append(v.study_id.unique()[0])
            params_list.append(str(params))
            mae_list.append(mae)
            smote_list.append(smote)
            neighbors_list.append(neighbors)
            
    loso_res_df = pd.DataFrame({
        'model_type': model_type_list,
        'smote': smote_list,
        'neighbors': neighbors_list,
        'data': data_list,
        'target': target_list,
        'fold': fold_list,
        'mae': mae_list,
        'params': params_list,
        'day': days_list,
        'y_true': y_true_list,
        'y_pred': y_pred_list,
        'y_mean': y_mean_list
    })

    # Save
    loso_res_df.to_csv(output_filename, compression='gzip', index=False)
