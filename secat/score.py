import pandas as pd
import numpy as np
import click
import sqlite3
import os
import sys

from sklearn import linear_model

from pyprophet.pyprophet import PyProphet
from pyprophet.report import save_report

from scipy.stats import rankdata


try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from scipy.stats import gaussian_kde
from numpy import linspace, concatenate, around

def prepare_filter(outfile, complex_threshold_factor):

    con = sqlite3.connect(outfile)
    df = pd.read_sql('SELECT DISTINCT condition_id, replicate_id FROM SEC;', con)
    con.close()

    experiments = []
    for idx, exp in enumerate(df.to_dict('records')):
        experiments.append({'idx': idx, 'tmpoutfile': outfile + "_tmp" + str(idx), 'outfile': outfile, 'condition_id': exp['condition_id'], 'replicate_id': exp['replicate_id'], 'complex_threshold_factor': complex_threshold_factor})

    return experiments

def filter_mw(exp):
    def model(df):
        y = df[['log_sec_mw']].values
        X = df[['sec_id']].values
        lm = linear_model.LinearRegression()
        lm.fit(X, y)
        return pd.Series({'coef_': lm.coef_[0][0], 'intercept_': lm.intercept_[0]})

    def lm(df, val):
        return np.exp(df['intercept_'] + df['coef_']*df[val])

    con = sqlite3.connect(exp['outfile'])
    df = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, FEATURE.bait_id AS bait_id, FEATURE.prey_id AS prey_id, FEATURE.feature_id AS feature_id, RT, leftWidth, rightWidth, bait_mw, prey_mw FROM FEATURE INNER JOIN FEATURE_META ON FEATURE.feature_id = FEATURE_META.feature_id INNER JOIN (SELECT protein_id AS bait_id, protein_mw AS bait_mw FROM PROTEIN) AS BAIT_MW ON FEATURE.bait_id = BAIT_MW.bait_id INNER JOIN (SELECT protein_id AS prey_id, protein_mw AS prey_mw FROM PROTEIN) AS PREY_MW ON FEATURE.prey_id = PREY_MW.prey_id WHERE condition_id = "%s" AND replicate_id = "%s";' % (exp['condition_id'], exp['replicate_id']), con)
    sec_mw = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, sec_id, sec_mw FROM SEC WHERE condition_id = "%s" AND replicate_id = "%s";' % (exp['condition_id'], exp['replicate_id']), con)
    con.close()

    # fit model
    sec_mw['log_sec_mw'] = np.log(sec_mw['sec_mw'])
    sec_mw_model = sec_mw.groupby(['condition_id','replicate_id']).apply(model)

    df = pd.merge(df, sec_mw_model.reset_index(), on=['condition_id','replicate_id'])

    df['sec_mw'] = df.apply(lambda x: lm(x,'RT'), 1)
    df['left_sec_mw'] = df.apply(lambda x: lm(x,'leftWidth'), 1)
    df['right_sec_mw'] = df.apply(lambda x: lm(x,'rightWidth'), 1)

    # decide whether the feature is a complex or monomer feature
    df['complex'] = pd.Series((df['sec_mw'].astype(float) > exp['complex_threshold_factor']*df['bait_mw'].astype(float)) & (df['sec_mw'].astype(float) > exp['complex_threshold_factor']*df['prey_mw'].astype(float)))
    df['monomer'] = pd.Series((df['bait_id'] == df['prey_id']) & (df['left_sec_mw'].astype(float) > df['bait_mw'].astype(float)) & (df['right_sec_mw'].astype(float) < df['bait_mw'].astype(float)) & (df['sec_mw'].astype(float) <= exp['complex_threshold_factor']*df['bait_mw'].astype(float)))

    return df[['feature_id','prey_id','sec_mw','left_sec_mw','right_sec_mw','complex','monomer']]

def filter_training(exp):
    con = sqlite3.connect(exp['outfile'])
    df = pd.read_sql('SELECT FEATURE_SUPER.feature_id AS feature_id, FEATURE_SUPER.prey_id AS prey_id, complex, monomer, bait_elution_model_fit, bait_log_sn, bait_xcorr_shape, bait_xcorr_coelution, main_var_xcorr_shape_score_q2, var_xcorr_coelution_score_q3, var_log_sn_score_q1, var_stoichiometry_score_q2 FROM FEATURE_SUPER INNER JOIN FEATURE_META ON FEATURE_SUPER.feature_id = FEATURE_META.feature_id INNER JOIN FEATURE_MW ON FEATURE_SUPER.feature_id = FEATURE_MW.feature_id AND FEATURE_SUPER.prey_id = FEATURE_MW.prey_id WHERE condition_id = "%s" AND replicate_id = "%s";' % (exp['condition_id'], exp['replicate_id']), con)
    con.close()

    return df[((df['complex'] == True) | (df['monomer'] == True)) & (df['bait_xcorr_coelution'] < 1.0) & (df['bait_xcorr_shape'] > 0.8) & (df['bait_elution_model_fit'] > 0.9) & (df['bait_log_sn'] > 1.0) & (df['main_var_xcorr_shape_score_q2'] > 0.8) & (df['var_xcorr_coelution_score_q3'] < 1.0) & (df['var_log_sn_score_q1'] > 0.0) & (df['var_stoichiometry_score_q2'] > 0.2)][['feature_id','prey_id']]

class pyprophet:
    def __init__(self, outfile, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test):

        self.outfile = outfile
        self.xeval_fraction = xeval_fraction
        self.xeval_num_iter = xeval_num_iter
        self.ss_initial_fdr = ss_initial_fdr
        self.ss_iteration_fdr = ss_iteration_fdr
        self.ss_num_iter = ss_num_iter
        self.ss_main_score = 'main_var_xcorr_shape_score'
        self.group_id = 'pyprophet_feature_id'
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

        # learn from high confidence data
        learning_data = self.read_learning()
        learned_data, self.weights = self.learn(learning_data)
        print self.weights

        # apply to detecting data
        detecting_data = self.read_detecting()

        self.df = detecting_data.groupby('sec_group').apply(self.apply)

    def read_learning(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT FEATURE_SUPER.*, FEATURE_SUPER.feature_id || "_" || FEATURE_SUPER.prey_id || "_" || FEATURE_SUPER.bait_id AS pyprophet_feature_id FROM FEATURE_SUPER INNER JOIN FEATURE_TRAINING ON FEATURE_SUPER.feature_id = FEATURE_TRAINING.feature_id AND FEATURE_SUPER.prey_id = FEATURE_TRAINING.prey_id LEFT OUTER JOIN NETWORK ON FEATURE_SUPER.bait_id = NETWORK.bait_id AND FEATURE_SUPER.prey_id = NETWORK.prey_id WHERE interaction_confidence > 0.9 OR decoy == 1;', con)

        con.close()

        return df

    def read_detecting(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT FEATURE_SUPER.*, FEATURE_SUPER.feature_id || "_" || FEATURE_SUPER.prey_id || "_" || FEATURE_SUPER.bait_id AS pyprophet_feature_id, ROUND(FEATURE_META.RT) AS sec_group FROM FEATURE_SUPER INNER JOIN FEATURE_TRAINING ON FEATURE_SUPER.feature_id = FEATURE_TRAINING.feature_id AND FEATURE_SUPER.prey_id = FEATURE_TRAINING.prey_id INNER JOIN FEATURE_META ON FEATURE_SUPER.FEATURE_ID = FEATURE_META.FEATURE_ID;', con)
        con.close()

        df['interaction_id'] = df.apply(lambda x: "_".join(sorted([x['bait_id'], x['prey_id']])), axis=1)

        df.loc[df['sec_group'] >= 40,'sec_group'] = 40
        df['sec_group'] = np.round(df['sec_group'] / 5.0)

        return df

    def learn(self, learning_data):
        (result, scorer, weights) = PyProphet(self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, False, self.threads, self.test).learn_and_apply(learning_data)
        self.plot(result, scorer.pi0, "learn")
        self.plot_scores(result.scored_tables, "learn")

        return result, weights

    def apply(self, detecting_data):
        (result, scorer, weights) = PyProphet(self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, False, self.threads, self.test).apply_weights(detecting_data, self.weights)
        self.plot(result, scorer.pi0, str(int(detecting_data['sec_group'].values[0])))

        df = result.scored_tables[['feature_id', 'bait_id', 'prey_id', 'interaction_id', 'decoy', 'pep', 'q_value']]
        df.columns = ['feature_id', 'bait_id', 'prey_id', 'interaction_id', 'decoy', 'pep', 'qvalue']
        return df

    def plot(self, result, pi0, tag):
        cutoffs = result.final_statistics["cutoff"].values
        svalues = result.final_statistics["svalue"].values
        qvalues = result.final_statistics["qvalue"].values

        pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
        top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
        top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

        save_report(os.path.splitext(os.path.basename(self.outfile))[0]+"_"+tag+".pdf", tag, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0)

    def plot_scores(self, df, tag):

        if plt is None:
            raise ImportError("Error: The matplotlib package is required to create a report.")

        out = os.path.splitext(os.path.basename(self.outfile))[0]+"_"+tag+"_scores.pdf"

        score_columns = ["d_score"] + [c for c in df.columns if c.startswith("main_var_")] + [c for c in df.columns if c.startswith("var_")]

        with PdfPages(out) as pdf:
            for idx in score_columns:
                top_targets = df[df["decoy"] == 0][idx]
                top_decoys = df[df["decoy"] == 1][idx]

                if not (top_targets.isnull().values.any() or top_decoys.isnull().values.any()):
                    tdensity = gaussian_kde(top_targets)
                    tdensity.covariance_factor = lambda: .25
                    tdensity._compute_covariance()
                    ddensity = gaussian_kde(top_decoys)
                    ddensity.covariance_factor = lambda: .25
                    ddensity._compute_covariance()
                    xs = linspace(min(concatenate((top_targets, top_decoys))), max(
                        concatenate((top_targets, top_decoys))), 200)

                    plt.figure(figsize=(10, 10))
                    plt.subplots_adjust(hspace=.5)

                    plt.subplot(211)
                    plt.title(idx)
                    plt.xlabel(idx)
                    plt.ylabel("# of groups")
                    plt.hist(
                        [top_targets, top_decoys], 20, color=['g', 'r'], label=['target', 'decoy'], histtype='bar')
                    plt.legend(loc=2)

                    plt.subplot(212)
                    plt.xlabel(idx)
                    plt.ylabel("density")
                    plt.plot(xs, tdensity(xs), color='g', label='target')
                    plt.plot(xs, ddensity(xs), color='r', label='decoy')
                    plt.legend(loc=2)

                    pdf.savefig()
                    plt.close()

class infer:
    def __init__(self, outfile):
        self.outfile = outfile

        peptide_data = self.read()

        self.directional, self.interactions = self.infer_features(peptide_data)

    def read(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, FEATURE_SUPER.bait_id AS bait_id, FEATURE_SUPER.prey_id AS prey_id, FEATURE_SCORED.interaction_id AS interaction_id, FEATURE_SUPER.feature_id AS feature_id, interaction_confidence AS prior, FEATURE_SCORED.pep FROM FEATURE_SUPER INNER JOIN FEATURE_META ON FEATURE_SUPER.feature_id = FEATURE_META.feature_id INNER JOIN FEATURE_SCORED ON FEATURE_SUPER.feature_id = FEATURE_SCORED.feature_id AND FEATURE_SUPER.prey_id = FEATURE_SCORED.prey_id INNER JOIN FEATURE_MW ON FEATURE_SUPER.feature_id = FEATURE_MW.feature_id AND FEATURE_SUPER.prey_id = FEATURE_MW.prey_id INNER JOIN (SELECT DISTINCT * FROM NETWORK) AS NETWORK ON FEATURE_SUPER.bait_id = NETWORK.bait_id AND FEATURE_SUPER.prey_id = NETWORK.prey_id WHERE FEATURE_SUPER.decoy == 0 AND FEATURE_MW.complex == 1;', con)
        con.close()

        return df

    def compute_model_fdr(self, data):
        # compute model based FDR estimates from posterior error probabilities
        order = np.argsort(data)

        ranks = np.zeros(data.shape[0], dtype=np.int)
        fdr = np.zeros(data.shape[0])

        # rank data with with maximum ranks for ties
        ranks[order] = rankdata(data[order], method='max')

        # compute FDR/q-value by using cumulative sum of maximum rank for ties
        fdr[order] = data[order].cumsum()[ranks[order]-1] / ranks[order]

        return fdr

    def interaction_inference(self, protein_data):
        pf_score = ((1.0-protein_data['pep']).prod()*(1.0-protein_data['bait_pep']).prod()*protein_data['prior'].mean()) / ((1.0-protein_data['pep']).prod()*(1.0-protein_data['bait_pep']).prod()*protein_data['prior'].mean() + protein_data['pep'].prod()*protein_data['bait_pep'].prod()*((1-protein_data['prior'].mean())/3) + (1.0-protein_data['pep']).prod()*protein_data['bait_pep'].prod()*((1-protein_data['prior'].mean())/3) + protein_data['pep'].prod()*(1.0-protein_data['bait_pep']).prod()*((1-protein_data['prior'].mean())/3))

        return(pd.Series({'pep': 1.0-pf_score}))


    def infer_features(self, protein_data):
        # merge baits and preys
        bait_proteins = protein_data.ix[(protein_data.bait_id == protein_data.prey_id)][["feature_id","bait_id","pep"]]
        bait_proteins.columns = ["feature_id","bait_id","bait_pep"]
        protein_bait_data = pd.merge(protein_data.ix[(protein_data.bait_id != protein_data.prey_id)], bait_proteins, on=["feature_id","bait_id"])

        # feature level
        feature_data = protein_bait_data.groupby(["condition_id", "replicate_id", "feature_id", "interaction_id", "bait_id", "prey_id"]).apply(self.interaction_inference).reset_index()

        # interaction level
        interaction_data = protein_bait_data.groupby(["condition_id", "interaction_id"]).apply(self.interaction_inference).reset_index()

        # add q-value
        feature_data['qvalue'] = self.compute_model_fdr(feature_data['pep'])
        interaction_data['qvalue'] = self.compute_model_fdr(interaction_data['pep'])

        return feature_data, interaction_data
