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
from pyprophet.stats import qvalue, pi0est

class pyprophet:
    def __init__(self, outfile, minimum_monomer_delta, minimum_mass_ratio, maximum_sec_shift, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test):

        self.outfile = outfile
        self.classifier = 'XGBoost'
        self.xgb_hyperparams = {'num_boost_round': 100, 'early_stopping_rounds': 10, 'test_size': 0.33}
        self.xgb_params = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'binary:logitraw', 'nthread': 1, 'eval_metric': 'error'}
        self.minimum_monomer_delta = minimum_monomer_delta
        self.minimum_mass_ratio = minimum_mass_ratio
        self.maximum_sec_shift = maximum_sec_shift
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
        data = self.read_data()

        # Learn model
        self.weights = self.learn(data[data['confidence_bin'] == data['confidence_bin'].max()])

        # Apply model
        self.df = data.groupby('confidence_bin').apply(self.apply)

    def read_data(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT *, condition_id || "_" || replicate_id || "_" || bait_id || "_" || prey_id AS pyprophet_feature_id, condition_id || "_" || bait_id || "_" || prey_id AS pyprophet_metafeature_id FROM FEATURE WHERE var_monomer_delta >= %s AND var_xcorr_shift <= %s AND var_mass_ratio >= %s ORDER BY pyprophet_metafeature_id;' % (self.minimum_monomer_delta, self.maximum_sec_shift, self.minimum_mass_ratio), con)
        con.close()

        # We need to generate a score that selects for the very best interactions heterodimers of similar size: perfect shape, co-elution and identical mass
        df['main_var_kickstart'] = (df['var_xcorr_shape'] * df['var_mass_ratio']) / (df['var_xcorr_shift'] + 1)

        return df

    def learn(self, learning_data):
        (result, scorer, weights) = PyProphet(self.classifier, self.xgb_hyperparams, self.xgb_params, self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, False, self.threads, self.test).learn_and_apply(learning_data)

        self.plot(result, scorer.pi0, "learning")
        self.plot_scores(result.scored_tables, "learning")

        return weights

    def apply(self, detecting_data):
        (result, scorer, weights) = PyProphet(self.classifier, self.xgb_hyperparams, self.xgb_params, self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, False, self.threads, self.test).apply_weights(detecting_data, self.weights)

        df = result.scored_tables[['condition_id','replicate_id','bait_id','prey_id','decoy','d_score','p_value','q_value','pep']]
        df.columns = ['condition_id','replicate_id','bait_id','prey_id','decoy','score','pvalue','qvalue','pep']

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
        self.df = self.combine_scores(scores)

    def read(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT * FROM FEATURE_SCORED;', con)
        con.close()

        return df

    def combine_scores(self, scores):
        # Generate full experimental design
        interactions = scores[['bait_id','prey_id','decoy']].drop_duplicates().reset_index()
        interactions['id'] = 1
        experimental_design = scores[['condition_id','replicate_id']].drop_duplicates().reset_index()
        experimental_design['id'] = 1

        # Missing detections need to be punished. We set the p-values, q-values and pep to 1.
        scores = pd.merge(pd.merge(interactions, experimental_design, on='id')[['condition_id','replicate_id','bait_id','prey_id','decoy']], scores, on=['condition_id','replicate_id','bait_id','prey_id','decoy'], how='left')
        scores['pvalue'] = scores['pvalue'].fillna(1)
        scores['qvalue'] = scores['qvalue'].fillna(1)
        scores['pep'] = scores['pep'].fillna(1)

        combined_scores = scores.groupby(['condition_id','bait_id','prey_id','decoy'])['pvalue'].median().reset_index()

        combined_scores.loc[combined_scores['decoy'] == 0,'qvalue'] = qvalue(combined_scores[combined_scores['decoy'] == 0]['pvalue'], pi0est(combined_scores[combined_scores['decoy'] == 0]['pvalue'], self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0)['pi0'], self.pfdr)

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

        return combined_scores