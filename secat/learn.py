import pandas as pd
import numpy as np
import scipy as sp
import click
import sqlite3
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from scipy.stats import gaussian_kde
from numpy import linspace, concatenate, around

from pyprophet.pyprophet import PyProphet
from pyprophet.report import save_report
from pyprophet.stats import pemp, qvalue, pi0est

class pyprophet:
    def __init__(self, outfile, minimum_monomer_delta, minimum_mass_ratio, maximum_sec_shift, minimum_peptides, maximum_peptides, cb_decoys, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test):

        self.outfile = outfile
        self.classifier = 'XGBoost'
        self.xgb_hyperparams = {'num_boost_round': 100, 'early_stopping_rounds': 10, 'test_size': 0.33}
        self.xgb_params = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'binary:logitraw', 'nthread': 1, 'eval_metric': 'error'}
        self.minimum_monomer_delta = minimum_monomer_delta
        self.minimum_mass_ratio = minimum_mass_ratio
        self.maximum_sec_shift = maximum_sec_shift
        self.minimum_peptides = minimum_peptides
        self.maximum_peptides = maximum_peptides
        self.cb_decoys = cb_decoys
        self.xeval_fraction = xeval_fraction
        self.xeval_num_iter = xeval_num_iter
        self.ss_initial_fdr = ss_initial_fdr
        self.ss_iteration_fdr = ss_iteration_fdr
        self.ss_num_iter = ss_num_iter
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

        # Read data
        global_abundance = self.read_global_abundance()
        data = self.read_data(global_abundance)

        if data[data['learning'] == 1].shape[0] > 0:
            # Learn model
            if self.cb_decoys:
                click.echo("Info: Using decoys from same confidence bin for learning.")
                self.weights = self.learn(data[(data['learning'] == 1)])
            else:
                self.weights = self.learn(data[(data['learning'] == 1) | (data['decoy'] == 1)])
            # Apply model
            self.df = data[data['learning'] == 0].groupby('confidence_bin').apply(self.apply)
        else:
            # Learn model
            self.weights = self.learn(data[data['confidence_bin'] == data['confidence_bin'].max()])
            # Apply model
            self.df = data.groupby('confidence_bin').apply(self.apply)

    def read_data(self, global_abundance):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT *, condition_id || "_" || replicate_id || "_" || bait_id || "_" || prey_id AS pyprophet_feature_id, condition_id || "_" || bait_id || "_" || prey_id AS pyprophet_metafeature_id FROM FEATURE ORDER BY pyprophet_metafeature_id;', con)
        con.close()

        # Append total mass ratio score
        dfa = pd.merge(pd.merge(df[['condition_id','replicate_id','bait_id','prey_id']], global_abundance, left_on = ['condition_id','replicate_id','bait_id'], right_on = ['condition_id','replicate_id','protein_id']), global_abundance, left_on = ['condition_id','replicate_id','prey_id'], right_on = ['condition_id','replicate_id','protein_id'])
        dfa['var_total_mass_ratio'] = dfa['peptide_intensity_x'] / dfa['peptide_intensity_y']
        dfa.loc[dfa['var_total_mass_ratio'] > 1, 'var_total_mass_ratio'] = 1 / dfa.loc[dfa['var_total_mass_ratio'] > 1, 'var_total_mass_ratio']
        dfa = dfa[['condition_id','replicate_id','bait_id','prey_id','var_total_mass_ratio']]
        df = pd.merge(df, dfa, on=['condition_id','replicate_id','bait_id','prey_id'])

        # Filter according to boundaries
        df_filter = df.groupby(["bait_id","prey_id","decoy"])[["var_monomer_delta","var_xcorr_shift","var_mass_ratio","var_total_mass_ratio"]].mean().dropna().reset_index(level=["bait_id","prey_id","decoy"])

        df_filter = df_filter[(df_filter['var_monomer_delta'] >= self.minimum_monomer_delta) & (df_filter['var_xcorr_shift'] <= self.maximum_sec_shift) & (df_filter['var_mass_ratio'] >= self.minimum_mass_ratio) & (df_filter['var_total_mass_ratio'] >= self.minimum_mass_ratio)]

        df = pd.merge(df, df_filter[["bait_id","prey_id","decoy"]], on=["bait_id","prey_id","decoy"]).dropna()

        # We need to generate a score that selects for the very best interactions heterodimers of similar size: perfect shape, co-elution and identical mass
        df['main_var_kickstart'] = (df['var_xcorr_shape'] * df['var_mass_ratio']) / (df['var_xcorr_shift'] + 1)

        return df

    def read_global_abundance(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT condition_id, replicate_id, QUANTIFICATION.protein_id, QUANTIFICATION.peptide_id, sum(peptide_intensity) as peptide_intensity FROM QUANTIFICATION INNER JOIN SEC ON QUANTIFICATION.run_id = SEC.run_id INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id WHERE peptide_count >= %s AND peptide_rank <= %s GROUP BY condition_id, replicate_id, QUANTIFICATION.protein_id, QUANTIFICATION.peptide_id;' % (self.minimum_peptides, self.maximum_peptides), con)
        con.close()

        return df.groupby(['condition_id','replicate_id','protein_id'])['peptide_intensity'].mean().reset_index()

    def learn(self, learning_data):
        (result, scorer, weights) = PyProphet(self.classifier, self.xgb_hyperparams, self.xgb_params, self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, False, self.threads, self.test).learn_and_apply(learning_data)

        self.plot(result, scorer.pi0, "learning")
        self.plot_scores(result.scored_tables, "learning")

        return weights

    def apply(self, detecting_data):
        (result, scorer, weights) = PyProphet(self.classifier, self.xgb_hyperparams, self.xgb_params, self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, False, self.threads, self.test).apply_weights(detecting_data, self.weights)

        df = result.scored_tables[['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin','d_score','p_value','q_value','pep']]
        df.columns = ['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin','score','pvalue','qvalue','pep']

        self.plot(result, scorer.pi0, "detecting_" + str(detecting_data['confidence_bin'].values[0]))
        self.plot_scores(result.scored_tables, "detecting_" + str(detecting_data['confidence_bin'].values[0]))

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

class combine:
    def __init__(self, outfile, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, pfdr):

        self.outfile = outfile
        self.pi0_lambda = pi0_lambda
        self.pi0_method = pi0_method
        self.pi0_smooth_df = pi0_smooth_df
        self.pi0_smooth_log_pi0 = pi0_smooth_log_pi0
        self.pfdr = pfdr

        scores = self.read()
        self.df = scores.groupby('confidence_bin').apply(self.combine_scores)

    def read(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT * FROM FEATURE_SCORED;', con)
        con.close()

        return df

    def combine_scores(self, scores):
        combined_scores = scores.groupby(['condition_id','bait_id','prey_id','decoy','confidence_bin'])['score'].mean().reset_index()

        combined_scores.loc[combined_scores['decoy'] == 0,'pvalue'] = pemp(combined_scores[combined_scores['decoy'] == 0]['score'], combined_scores[combined_scores['decoy'] == 1]['score'])

        pi0_combined = pi0est(combined_scores[combined_scores['decoy'] == 0]['pvalue'], self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0)['pi0']
        combined_scores.loc[combined_scores['decoy'] == 0,'qvalue'] = qvalue(combined_scores[combined_scores['decoy'] == 0]['pvalue'], pi0_combined, self.pfdr)

        click.echo("Info: Unique interactions detected before integration:")
        click.echo("%s (at q-value < 0.01)" % (scores[(scores['decoy'] == 0) & (scores['qvalue'] < 0.01)][['bait_id','prey_id']].drop_duplicates().shape[0]))
        click.echo("%s (at q-value < 0.05)" % (scores[(scores['decoy'] == 0) & (scores['qvalue'] < 0.05)][['bait_id','prey_id']].drop_duplicates().shape[0]))
        click.echo("%s (at q-value < 0.1)" % (scores[(scores['decoy'] == 0) & (scores['qvalue'] < 0.1)][['bait_id','prey_id']].drop_duplicates().shape[0]))
        click.echo("%s (at q-value < 0.2)" % (scores[(scores['decoy'] == 0) & (scores['qvalue'] < 0.2)][['bait_id','prey_id']].drop_duplicates().shape[0]))
        click.echo("%s (at q-value < 0.5)" % (scores[(scores['decoy'] == 0) & (scores['qvalue'] < 0.5)][['bait_id','prey_id']].drop_duplicates().shape[0]))

        click.echo("Info: Unique interactions detected after integration:")
        click.echo("%s (at q-value < 0.01)" % (combined_scores[(combined_scores['decoy'] == 0) & (combined_scores['qvalue'] < 0.01)][['bait_id','prey_id']].drop_duplicates().shape[0]))
        click.echo("%s (at q-value < 0.05)" % (combined_scores[(combined_scores['decoy'] == 0) & (combined_scores['qvalue'] < 0.05)][['bait_id','prey_id']].drop_duplicates().shape[0]))
        click.echo("%s (at q-value < 0.1)" % (combined_scores[(combined_scores['decoy'] == 0) & (combined_scores['qvalue'] < 0.1)][['bait_id','prey_id']].drop_duplicates().shape[0]))
        click.echo("%s (at q-value < 0.2)" % (combined_scores[(combined_scores['decoy'] == 0) & (combined_scores['qvalue'] < 0.2)][['bait_id','prey_id']].drop_duplicates().shape[0]))
        click.echo("%s (at q-value < 0.5)" % (combined_scores[(combined_scores['decoy'] == 0) & (combined_scores['qvalue'] < 0.5)][['bait_id','prey_id']].drop_duplicates().shape[0]))
        click.echo("Info: Combined pi0: %s." % pi0_combined)

        return combined_scores