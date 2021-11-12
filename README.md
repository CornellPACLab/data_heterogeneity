# data_heterogeneity

Repository for paper "Machine learning for passive mental health symptom prediction: generalization across different longitudinal mobile sensing studies"

Last updated: 11/11/2021

## Questions?

If you have any questions email me: dadler@infosci.cornell.edu

## Raw Data

You can download the StudentLife and CrossCheck public datasets from the following sources:

Raw CrossCheck data: [link](https://cornell.app.box.com/s/rkx46bgv36lkmo2eu349ka95senn48gh). Download the "CrossCheck_Daily_Data.csv" file.

Raw StudentLife data: [link](https://studentlife.cs.dartmouth.edu/). Download the entire data zip file.

## Folder Structure

### data

Cleaned data, created by the notebooks/data_cleaning.ipynb notebook, is provided in the "data" folder. The cleaned data has behavior features-EMAs aligned between the datasets, by taking the average across the 3-days up to and including the day the EMA was taken. See the "Sensor-EMA Alignment Across Studies" subsection in the paper. The files are:

* CrossCheck data: crosscheck_daily_data_cleaned_w_sameday.csv
* StudentLife data: studentlife_daily_data_cleaned_w_sameday_08282021.csv

### notebooks

* data_cleaning.ipynb: Processes raw CrossCheck/StudentLife data to produce the behavior feature-EMA aligned files. You will need to change paths throughout the notebook after downlading the raw CrossCheck/StudentLife data to use this file.
* Demographics_and_Distributions.ipynb: Creates all figures/statistics in the paper to document demographic information from the cleaned data.
* simple_regression_models_analysis.ipynb: Processes model results and produces relevant tables/figures. Runs statistical tests.

### code

* cleaning_util.py: Utilities for data cleaning. You will need to change certain paths throughout the notebook 
* util.py: Various utility functions used throughout the code
* run_cv.py: Runs the regression_cv.py LOSO-CV code
* regression_cv.py: LOSO-CV code

### res

This folder is left blank, but you will see that various results files are used in the simple_regression_models_analysis.ipynb notebook. I was not able to push the output files to github because the filesize is too large.

You can produce the LOSO-CV results file, called "loso_res_df_all_v10_10182021.csv.gz" by running the following command:

```bash
python run_cv.py
```

And in the run_cv.py file setting these variables to the following:

```python
params_dict = {
    'gbt': {
        'loss': ['huber'],
        'learning_rate': [0.001, 0.01, 0.1, 1.0],
        'n_estimators': [20, 100, 1000],
        'max_depth': [3, 7, 10],
        'random_state': [42]
    }
}
output_filename ='../res/loso_res_df_all_v10_10182021.csv.gz'
```

These are the hyperparameters used for the model results in the paper.





