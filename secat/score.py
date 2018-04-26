import pandas as pd
import numpy as np
import click
import sqlite3
import os

from sklearn import linear_model

from pyprophet.pyprophet import PyProphet
from pyprophet.report import save_report

from scipy.stats import rankdata


def filter_mw(outfile, complex_threshold_factor):
    def model(df):
        y = df[['log_sec_mw']].values
        X = df[['sec_id']].values
        lm = linear_model.LinearRegression()
        lm.fit(X, y)
        return pd.Series({'coef_': lm.coef_[0][0], 'intercept_': lm.intercept_[0]})

    def lm(df, val):
        return np.exp(df['intercept_'] + df['coef_']*df[val])

    con = sqlite3.connect(outfile)
    df = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, FEATURE.bait_id AS bait_id, FEATURE.prey_id AS prey_id, feature_id, RT, leftWidth, rightWidth, bait_mw, prey_mw FROM FEATURE INNER JOIN (SELECT protein_id AS bait_id, protein_mw AS bait_mw FROM PROTEIN) AS BAIT_MW ON FEATURE.bait_id = BAIT_MW.bait_id INNER JOIN (SELECT protein_id AS prey_id, protein_mw AS prey_mw FROM PROTEIN) AS PREY_MW ON FEATURE.prey_id = PREY_MW.prey_id;', con)
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

    return df[['feature_id','sec_mw','left_sec_mw','right_sec_mw','complex','monomer']]

def filter_training(outfile):
    con = sqlite3.connect(outfile)
    df = pd.read_sql('SELECT FEATURE.bait_feature_id AS bait_feature_id, FEATURE.bait_id AS bait_id, FEATURE.prey_id AS prey_id, FEATURE.feature_id AS feature_id, FEATURE.decoy AS decoy, complex, monomer, bait_elution_model_fit, bait_log_sn, bait_xcorr_shape, bait_xcorr_coelution, main_var_xcorr_shape_score, var_xcorr_coelution_score, var_log_sn_score, var_stoichiometry_score FROM FEATURE INNER JOIN FEATURE_MW ON FEATURE.feature_id = FEATURE_MW.feature_id;', con)
    con.close()

    # bait-feature-level filtering
    df = df[((df['complex'] == True) | (df['monomer'] == True)) & (df['bait_xcorr_coelution'] < 3) & (df['bait_xcorr_shape'] > 0.8) & (df['bait_elution_model_fit'] > 0.9) & (df['bait_log_sn'] > 1.0)]

    # prey-feature-level filtering
    df = df.groupby(['bait_feature_id'])[['main_var_xcorr_shape_score','var_xcorr_coelution_score','var_log_sn_score','var_stoichiometry_score']].quantile([0.33, 0.66]).reset_index()
    df = df.rename(index=str, columns={"level_1": "quantile"})

    df_high = (df[(df['quantile'] == 0.33) & (df['main_var_xcorr_shape_score'] > 0.8) & (df['var_log_sn_score'] > 0.0) & (df['var_stoichiometry_score'] > 0.1)][['bait_feature_id']])

    df_low = (df[(df['quantile'] == 0.66) & (df['var_xcorr_coelution_score'] < 2.0) & (df['quantile'] == 0.66) & (df['var_stoichiometry_score'] < 2.0)][['bait_feature_id']]) 

    return pd.merge(df_high, df_low, on=['bait_feature_id'])

class pyprophet:
    def __init__(self, outfile, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test):

        self.outfile = outfile
        self.xeval_fraction = xeval_fraction
        self.xeval_num_iter = xeval_num_iter
        self.ss_initial_fdr = ss_initial_fdr
        self.ss_iteration_fdr = ss_iteration_fdr
        self.ss_num_iter = ss_num_iter
        self.ss_main_score = 'main_var_xcorr_shape_score'
        self.group_id = 'feature_id'
        self.parametric = parametric
        self.pfdr = pfdr
        self.pi0_lambda = pi0_lambda
        self.pi0_method = pi0_method
        self.pi0_smooth_df = pi0_smooth_df
        self.pi0_smooth_log_pi0 = pi0_smooth_log_pi0
        self.lfdr_truncate = lfdr_truncate
        self.lfdr_monotone = lfdr_monotone
        self.lfdr_transformation = lfdr_transformation
        self.lfdr_adj = lfdr_adj
        self.lfdr_eps = lfdr_eps
        self.threads = threads
        self.test = test

        learning_data = self.read_learning()
        _, self.weights = self.learn(learning_data)
        print self.weights
        all_data = self.read_all()
        self.df = self.apply(all_data)[['feature_id','d_score','p_value','q_value','pep']]

    def read_learning(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT * FROM FEATURE INNER JOIN FEATURE_TRAINING ON FEATURE.bait_feature_id = FEATURE_TRAINING.bait_feature_id WHERE var_xcorr_coelution_score < 5 AND var_stoichiometry_score < 2;', con)
        con.close()

        return df

    def read_all(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT * FROM FEATURE WHERE var_xcorr_coelution_score < 5 AND var_stoichiometry_score < 2;', con)
        con.close()

        return df

    def learn(self, learning_data):
        (result, scorer, weights) = PyProphet(self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, False, self.threads, self.test).learn_and_apply(learning_data)
        self.plot(result, scorer.pi0, "learn")
        return result, weights

    def apply(self, all_data):
        (result, scorer, weights) = PyProphet(self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, False, self.threads, self.test).apply_weights(all_data, self.weights)
        self.plot(result, scorer.pi0, "all")
        return result.scored_tables

    def plot(self, result, pi0, tag):
        cutoffs = result.final_statistics["cutoff"].values
        svalues = result.final_statistics["svalue"].values
        qvalues = result.final_statistics["qvalue"].values

        pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
        top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
        top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

        save_report(os.path.splitext(os.path.basename(self.outfile))[0]+"_"+tag+".pdf", tag, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0)

class infer:
    def __init__(self, outfile):
        self.outfile = outfile

        peptide_data = self.read()

        self.interactions, self.proteins = self.infer_features(peptide_data)

    def read(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT condition_id, replicate_id, FEATURE.bait_id AS bait_id, FEATURE.prey_id AS prey_id, bait_feature_id, FEATURE.feature_id AS feature_id, monomer, complex, interaction_confidence AS prior, pep, decoy, prey_peptide_intensity, prey_peptide_total_intensity FROM FEATURE INNER JOIN FEATURE_SCORED ON FEATURE.feature_id = FEATURE_SCORED.feature_id INNER JOIN NETWORK ON FEATURE.bait_id = NETWORK.bait_id AND FEATURE.prey_id = NETWORK.prey_id INNER JOIN FEATURE_MW ON FEATURE.feature_id = FEATURE_MW.feature_id;', con)
        con.close()

        return df

    def compute_model_fdr(data):
        # compute model based FDR estimates from posterior error probabilities
        order = np.argsort(data)

        ranks = np.zeros(data.shape[0], dtype=np.int)
        fdr = np.zeros(data.shape[0])

        # rank data with with maximum ranks for ties
        ranks[order] = rankdata(data[order], method='max')

        # compute FDR/q-value by using cumulative sum of maximum rank for ties
        fdr[order] = data[order].cumsum()[ranks[order]-1] / ranks[order]

        return fdr


    def protein_inference(self, peptide_data):
        if (peptide_data.shape[0] >= 3):
            top_peptide_data = peptide_data.sort_values('prey_peptide_intensity', ascending=False).head(3)
            prey_intensity = top_peptide_data['prey_peptide_intensity'].mean()
            prey_total_intensity = top_peptide_data['prey_peptide_total_intensity'].mean()
            pp_score = ((1-top_peptide_data['pep']).prod()*0.5) / (((1-top_peptide_data['pep']).prod()*0.5) + ((top_peptide_data['pep']).prod()*0.5))
            return(pd.DataFrame({'pp_score': [pp_score], 'prey_intensity': [prey_intensity], 'prey_total_intensity': [prey_total_intensity]}))


    def interaction_inference(self, protein_data):
        pf_score = (protein_data['pp_score'].prod()*protein_data['bait_pp_score'].prod()*protein_data['prior'].mean()) / (protein_data['pp_score'].prod()*protein_data['bait_pp_score'].prod()*protein_data['prior'].mean() + (1-protein_data['pp_score']).prod()*(1-protein_data['bait_pp_score']).prod()*((1-protein_data['prior'].mean())/3) + (protein_data['pp_score']).prod()*(1-protein_data['bait_pp_score']).prod()*((1-protein_data['prior'].mean())/3) + (1-protein_data['pp_score']).prod()*(protein_data['bait_pp_score']).prod()*((1-protein_data['prior'].mean())/3))
        bait_intensity = protein_data['bait_intensity'].mean()
        prey_intensity = protein_data['prey_intensity'].mean()
        pf_intensity = prey_intensity / bait_intensity

        return(pd.DataFrame({'interaction_id': ['_'.join(sorted([protein_data['bait_id'].unique()[0], protein_data['prey_id'].unique()[0]]))], 'pf_score': [pf_score], 'pf_intensity': [pf_intensity], 'bait_intensity': [bait_intensity], 'prey_intensity': [prey_intensity]}))


    def infer_features(self, peptide_data):
        # protein level
        peptide_data.loc[peptide_data.decoy==1,'prior'] = 0.5
        # peptide_data['prior'] = 0.25
        protein_data = peptide_data.groupby(["bait_feature_id","bait_id", "prey_id", "decoy", "prior", "monomer"]).apply(self.protein_inference).reset_index()

        # interaction level
        bait_proteins = protein_data.ix[(protein_data.bait_id == protein_data.prey_id) & (protein_data.decoy==0)][["bait_feature_id","bait_id","pp_score","prey_intensity"]]
        bait_proteins.columns = ["bait_feature_id","bait_id","bait_pp_score","bait_intensity"]
        protein_bait_data = pd.merge(protein_data.ix[(protein_data.bait_id != protein_data.prey_id)], bait_proteins, on=["bait_feature_id","bait_id"])

        interaction_data = protein_bait_data.groupby(["bait_feature_id","bait_id", "prey_id", "decoy", "prior"]).apply(self.interaction_inference).reset_index()

        return([interaction_data, protein_data])