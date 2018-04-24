import pandas as pd
import numpy as np
import click
import sqlite3

from sklearn import linear_model

def mw(outfile, complex_threshold_factor):
    def model(df):
        y = df[['log_sec_mw']].values
        X = df[['sec_id']].values
        lm = linear_model.LinearRegression()
        lm.fit(X, y)
        return pd.Series({'coef_': lm.coef_[0][0], 'intercept_': lm.intercept_[0]})

    def lm(df, val):
        return np.exp(df['intercept_'] + df['coef_']*df[val])

    con = sqlite3.connect(outfile)
    df = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, FEATURES.bait_id as bait_id, FEATURES.prey_id as prey_id, decoy, feature_id, RT, leftWidth, rightWidth, bait_mw, prey_mw FROM FEATURES INNER JOIN (SELECT protein_id AS bait_id, protein_mw AS bait_mw FROM PROTEIN) AS bait_mw ON FEATURES.bait_id = bait_mw.bait_id INNER JOIN (SELECT protein_id AS prey_id, protein_mw AS prey_mw FROM PROTEIN) AS prey_mw ON FEATURES.prey_id = prey_mw.prey_id;', con)
    sec_mw = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, sec_id, sec_mw FROM SEC;', con)
    con.close()

    # fit model
    sec_mw['log_sec_mw'] = np.log(sec_mw['sec_mw'])
    sec_mw_model = sec_mw.groupby(['condition_id','replicate_id']).apply(model)

    df = pd.merge(df, sec_mw_model.reset_index(), on=['condition_id','replicate_id'])

    df['sec_mw'] = df.apply(lambda x: lm(x,'RT'), 1)
    df['left_sec_mw'] = df.apply(lambda x: lm(x,'leftWidth'), 1)
    df['right_sec_mw'] = df.apply(lambda x: lm(x,'rightWidth'), 1)

    # decide whether the feature is a complex or monomer feature
    df['complex'] = pd.Series((df['sec_mw'].astype(float) > complex_threshold_factor*df['bait_mw'].astype(float)) & (df['sec_mw'].astype(float) > complex_threshold_factor*df['prey_mw'].astype(float)))
    df['monomer'] = pd.Series((df['bait_id'] == df['prey_id']) & (df['left_sec_mw'].astype(float) > df['bait_mw'].astype(float)) & (df['right_sec_mw'].astype(float) < df['bait_mw'].astype(float)) & (df['sec_mw'].astype(float) <= complex_threshold_factor*df['bait_mw'].astype(float)))

    return df[['condition_id','replicate_id','bait_id','prey_id','decoy','feature_id','sec_mw','left_sec_mw','right_sec_mw','complex','monomer']]

def filter_training(outfile):
    con = sqlite3.connect(outfile)
    df = pd.read_sql('SELECT FEATURES.condition_id AS condition_id, FEATURES.replicate_id AS replicate_id, FEATURES.bait_id AS bait_id, FEATURES.prey_id AS prey_id, FEATURES.decoy AS decoy, FEATURES.feature_id AS feature_id, complex, monomer, bait_elution_model_fit, bait_log_sn, bait_xcorr_shape, bait_xcorr_coelution, main_var_xcorr_shape_score, var_xcorr_coelution_score, var_log_sn_score, var_stoichiometry_score FROM FEATURES INNER JOIN FEATURES_MW ON FEATURES.condition_id = FEATURES_MW.condition_id AND FEATURES.replicate_id = FEATURES_MW.replicate_id AND FEATURES.bait_id = FEATURES_MW.bait_id AND FEATURES.prey_id = FEATURES_MW.prey_id AND FEATURES.decoy = FEATURES_MW.decoy AND FEATURES.feature_id = FEATURES_MW.feature_id LIMIT 500000;', con)
    con.close()

    # bait-feature-level filtering
    df = df[((df['complex'] == True) | (df['monomer'] == True)) & (df['bait_xcorr_coelution'] < 3) & (df['bait_xcorr_shape'] > 0.8) & (df['bait_elution_model_fit'] > 0.9) & (df['bait_log_sn'] > 1.0)]

    # prey-feature-level filtering
    df = df.groupby(['bait_id', 'prey_id','feature_id', 'decoy'])[['main_var_xcorr_shape_score','var_xcorr_coelution_score','var_log_sn_score','var_stoichiometry_score']].quantile([0.33, 0.66]).reset_index()
    df = df.rename(index=str, columns={"level_4": "quantile"})
    print df[(df[df['quantile'] == 0.33]['main_var_xcorr_shape_score'] > 0.8) & (df[df['quantile'] == 0.66]['var_xcorr_coelution_score'] < 2.0) & (df[df['quantile'] == 0.33]['var_log_sn_score'] > 0.0) & (df[df['quantile'] == 0.66]['var_stoichiometry_score'] < 2.0) & (df[df['quantile'] == 0.33]['var_stoichiometry_score'] > 0.1)]
