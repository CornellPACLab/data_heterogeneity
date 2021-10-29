"""
util.py

Utilities that may be used for multiple purposes.
"""

# Imports
import pandas as pd
import numpy as np
import pingouin as pg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import shapiro, wilcoxon, ttest_rel, normaltest, norm
from sklearn.metrics import mean_absolute_error, recall_score, precision_score, confusion_matrix
import ot
import os


def open_file(f, file_type='csv'):
    """
    Open file

    :param f: <str>, absolute path to file
    :param file_type: <str>, the type of file
    """
    # Iterate through files and upload
    if (file_type == 'csv') and ('.csv' in f):
        df = pd.read_csv(f, sep=',')
    elif (file_type == 'json') and ('.json' in f):
        df = pd.read_json(f, convert_dates=False)
    return df


def upload_directory(directory, file_type='csv'):
    """
    Upload directory

    :param directory: <str>, the absolute path to the directory
    :param file_type: <str>, the type of file

    :return: <dict<pd.DataFrame>> the data in a dict where each key is the filename
                                  and each entry is a df of that file
    """
    # Create dict to store files
    files = dict()
    for i in os.listdir(directory):
        if (file_type in i):
            files[i] = open_file(directory + i, file_type)

    return files


def grouped_sensitivity(df, y_true, y_pred):
    """
    Sensitivity, for groupby statements

    :param df: pd.DataFrame, the df
    :param y_true: <str>, the column name of y_true
    :param y_pred: <str>, the column name of y_pred

    :return: float, the sensitivity
    """
    return recall_score(y_true=df[y_true], y_pred=df[y_pred])


def grouped_specificity(df, y_true, y_pred):
    """
    Specificity, for groupby statements

    :param df: pd.DataFrame, the df
    :param y_true: <str>, the column name of y_true
    :param y_pred: <str>, the column name of y_pred

    :return: float, the sensitivity
    """
    cm = confusion_matrix(y_true=df[y_true], y_pred=df[y_pred])
    if (cm[0, 1] + cm[0, 0]) == 0:
        return 0
    return cm[0, 0] / (cm[0, 1] + cm[0, 0])
    

def grouped_ppv(df, y_true, y_pred):
    """
    PPV, for groupby statements

    :param df: pd.DataFrame, the df
    :param y_true: <str>, the column name of y_true
    :param y_pred: <str>, the column name of y_pred

    :return: float, the PPV
    """
    return precision_score(y_true=df[y_true], y_pred=df[y_pred])


def percentile(n):
    """
    Percentile outer function
    Call as in "percentile(25)" on another iterable

    :param n: <float>, the percentile to keep

    :return: function, the percentile
    """
    def percentile_(x):
        """
        Percentile inner function

        :param x: <iterable>, some iterable 1D object

        :return: function, the callable function name
        """
        x = x.dropna()
        return np.percentile(x, n, interpolation='nearest')
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def calculate_emd(d1, d2, center=False):
    """
    Calculate earth mover's distance

    :param d1: np.array, first data
    :param d2: np.array, second data
    :param center: <bool>, whether to only use the center (5 std dev)
    :return: <float>, the EMD
    """
    # Filter if center
    if center:
        d1 = d1.loc[d1.abs().max(axis=1) < 5, :]
        d2 = d2.loc[d2.abs().max(axis=1) < 5, :]

    if (d1.shape[0] == 0) or (d2.shape[0] == 0):
        return None
    a, b = np.ones((d1.shape[0],)) / d1.shape[0], np.ones((d2.shape[0],)) / d2.shape[0]
    M = ot.dist(d1.values, d2.values, metric='euclidean')
    return ot.emd2(a=a, b=b, M=M)


def grouped_paired_test(df, col1, col2):
    """
    Run a paired test but only return pvalue
    that df[col1] > df[col2]

    :param df: pd.DataFrame, the df
    :param col1: <str>, the first column
    :param col2: <str>, the second column
    """
    if shapiro(df[col1] - df[col2])[1] < .05:
        res = pg.wilcoxon(df[col1], df[col2], alternative='greater')
        res = pd.Series(
            ['W', res.loc['Wilcoxon', 'p-val'], res.loc['Wilcoxon', 'RBC']],
            index=['test', 'pval', 'effect']
        )
        return res
    else:
        res = pg.ttest(df[col1], df[col2], alternative='greater', paired=True)
        return 'T', res.loc['T-test', 'p-val'], res.loc['T-test', 'cohen-d']


def non_paired_test(x, y):
    """
    Run a non paired test. Assume we are looking at hypothesis x != y

    :param x: <list>, the first dataset
    :param y: <list>, the second dataset
    """
    all_data = x + y
    # Get diff
    s = normaltest(a=all_data, axis=None)

    # Run test
    if s[1] < 0.05:
        res = pg.mwu(x, y)
    else:
        res = pg.ttest(x, y, paired=False)
        
    return res, s


def rosner_test(df, baseline_col, treatment_col, groups_col):
    """
    Clustered Wilcoxon signed rank test using Rosner-Glynn-Lee method. 
    Paper: https://www.jstor.org/stable/3695720

    Will test the H_a baseline > treatment in paired samples.

    Tested using the crsdUnb data and the R clusrank package.
    clusrank documentation: 
    https://cran.r-project.org/web/packages/clusrank/clusrank.pdf

    Commands to test in R:
    $ data(crsdUnb)
    $ clusWilcox.test(z, cluster = id, data = crsdUnb, paired=TRUE, alternative='greater')

    # Output produced from both programs:
    $ Z = -0.44794, p-value = 0.6729

    :param df: pd.DataFrame, the df
    :param baseline_col: <str>, the name of the baseline column
    :param treatment_col: <str>, the name of the treatment column
    :param groups_col: <str>, the name of the groups column
    """
    # Compute the difference
    df['diff'] = (df[baseline_col] - df[treatment_col]).values

    # Compute the rank orders of the difference
    df.loc[df['diff'].abs().sort_values().index, 'rank'] = list(range(df.shape[0]))

    # Get the sign of the difference
    df['sign'] = (np.sign(df['diff'])).values
    df['signed_rank'] = (df['sign'] * df['rank']).values

    # Compute the average ranks per study id
    df_grouped = df.groupby(groups_col, as_index=False)['signed_rank'].agg(['mean', 'var', 'count'])
    df_merged = pd.merge(left=df[[groups_col, 'signed_rank']], right=df_grouped, on=[groups_col])

    # Compute the variance and weights
    m = df_grouped.shape[0]
    G = df_grouped['count'].sum()
    sigma_squared = np.sum((df_merged['signed_rank'] - df_merged['mean'])**2) / (G - m)
    go = (np.sum(df_grouped['count']) - (np.sum(df_grouped['count']**2) / np.sum(df_grouped['count']))) / (m - 1)
    sigma_a_squared = np.sum(df_grouped['count'] * (df_grouped['mean'] - df_grouped['mean'].mean())**2) / (m - 1)
    sigma_a_squared = max((sigma_a_squared - sigma_squared) / go, 0)
    rho_s = max(sigma_a_squared / (sigma_a_squared + sigma_squared), 0)
    rho_s_corr = rho_s * (1 + (1 - rho_s**2) / (m - 2.5))
    var_s = np.sum((df['signed_rank'] - df['signed_rank'].mean())**2) / (G - 1)
    df_grouped['weight'] = df_grouped['count'] / (var_s * (1 + (df_grouped['count'] - 1) * rho_s_corr))

    # Now compute W
    T_obs = np.sum(df_grouped['weight'] * df_grouped['mean'])
    W = T_obs / np.sqrt(np.sum(df_grouped['weight']**2 * df_grouped['mean']**2))
    
    # Lastly get p-value
    p_val = norm.sf(W)

    return W, p_val


def proxy_a_distance(d1, d2, features):
    """
    Calculate the proxy-a distance

    Proxy a distance measures the error of a classifier trained to distinguish
    between datasets.

    :param d1: pd.DataFrame, the first dataset
    :param d2: pd.DataFrame, the second dataset
    :param features: <str>, the features

    :return: <float>, the Proxy-A distance
    """
    # Split data
    d1['target'] = 0
    d2['target'] = 1

    # Combine
    d = pd.concat(
        [d1[features + ['target']], d2[features + ['target']]]
    ).reset_index(drop=True)

    # Get train test split
    train, test = train_test_split(d, test_size=0.2, random_state=42)

    # Normalize
    sc = StandardScaler()
    sc.fit(train[features])
    train_norm = pd.DataFrame(sc.transform(train[features]), columns=features)
    test_norm = pd.DataFrame(sc.transform(test[features]), columns=features)
    train_norm['target'] = train.target.values
    test_norm['target'] = test.target.values

    # Train classifier
    sv = SVC(probability=True, kernel='linear', class_weight='balanced', random_state=42)
    sv.fit(X=train_norm[features], y=train_norm['target'])
    y_pred = sv.predict_proba(test_norm[features])[:, 1]

    return 2 * (1 - 2*mean_absolute_error(test_norm['target'], y_pred))
