import pandas as pd
import numpy as np
import scipy as sp
import click
import sqlite3
import pickle
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

from hyperopt import hp

class pyprophet:
    def __init__(self, outfile, apply_model, minimum_abundance_ratio, maximum_sec_shift, cb_decoys, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, xgb_autotune, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, plot_reports, threads, test, export_tables):

        self.outfile = outfile
        self.apply_model = apply_model
        self.classifier = 'XGBoost'
        self.xgb_autotune = xgb_autotune

        self.xgb_hyperparams = {'autotune': self.xgb_autotune, 'autotune_num_rounds': 10, 'num_boost_round': 100, 'early_stopping_rounds': 10, 'test_size': 0.33}

        self.xgb_params = {'eta': 1.0, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'colsample_bynode': 1, 'lambda': 1, 'alpha': 0, 'scale_pos_weight': 1, 'objective': 'binary:logitraw', 'nthread': 1, 'eval_metric': 'auc'}

        self.xgb_params_space = {'eta': hp.uniform('eta', 0.5, 1.0), 'gamma': hp.uniform('gamma', 0.0, 0.5), 'max_depth': hp.quniform('max_depth', 2, 8, 1), 'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1), 'subsample': hp.uniform('subsample', 0.5, 1.0), 'colsample_bytree': 1.0, 'colsample_bylevel': 1.0, 'colsample_bynode': 1.0, 'lambda': hp.uniform('lambda', 0.0, 1.0), 'alpha': hp.uniform('alpha', 0.0, 1.0), 'scale_pos_weight': 1.0, 'silent': 1, 'objective': 'binary:logitraw', 'nthread': 1, 'eval_metric': 'auc'}

        self.minimum_abundance_ratio = minimum_abundance_ratio
        self.maximum_sec_shift = maximum_sec_shift
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
        self.plot_reports = plot_reports
        self.test = test
        self.export_tables = export_tables
        self.has_learning = self.has_learning()

        # Load pretrained model if available
        if self.apply_model is not None:
            self.weights = self.load_model()

        # Learn classifier
        else:
            if self.has_learning:
                learning_data = self.read_data(learning=True)
                if self.cb_decoys:
                    click.echo("Info: Using decoys from same confidence bin for learning.")
                    self.weights = self.learn(learning_data[(learning_data['learning'] == 1)])
                else:
                    self.weights = self.learn(learning_data[(learning_data['learning'] == 1) | (learning_data['decoy'] == 1)])
            else:
                learning_data = self.read_data(learning=False)
                self.weights = self.learn(learning_data[learning_data['confidence_bin'] == learning_data['confidence_bin'].max()])
            # Store model
            self.store_model()

        # Apply classifier to full dataset
        runs = self.read_runs()

        # Apply separately for each run
        for run in runs.iterrows():
            click.echo("Info: Apply scores to condition %s and replicate %s." %(run[1]['condition_id'], run[1]['replicate_id']))
            data = self.read_data(learning=False, condition_id=run[1]['condition_id'], replicate_id=run[1]['replicate_id'])

            if self.has_learning:
                scored_data = data[data['learning'] == 0].groupby('confidence_bin', group_keys=False).apply(self.apply, condition_id=run[1]['condition_id'], replicate_id=run[1]['replicate_id'])
            else:
                scored_data = data.groupby('confidence_bin', group_keys=False).apply(self.apply, condition_id=run[1]['condition_id'], replicate_id=run[1]['replicate_id'])

            con = sqlite3.connect(outfile)
            scored_data.to_sql('FEATURE_SCORED', con, index=False, if_exists='append')
            con.close()

    def has_learning(self):
        con = sqlite3.connect(self.outfile)
        c = con.cursor()
        c.execute('SELECT count(*) FROM FEATURE WHERE learning==1;')
        if c.fetchone()[0] == 0:
            learning = False
        else:
            learning = True
        con.close()
        return learning

    def read_runs(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT DISTINCT condition_id, replicate_id FROM FEATURE;', con)
        con.close()

        return df

    def read_data(self, learning=False, condition_id=None, replicate_id=None):
        con = sqlite3.connect(self.outfile)
        if learning and condition_id is None and replicate_id is None:
            df = pd.read_sql('SELECT *, condition_id || "_" || replicate_id || "_" || bait_id || "_" || prey_id || "_" || decoy AS pyprophet_feature_id, condition_id || "_" || bait_id || "_" || prey_id || "_" || decoy AS pyprophet_metafeature_id FROM FEATURE WHERE learning==1 OR decoy==1 ORDER BY pyprophet_metafeature_id;', con)
        elif condition_id is not None and replicate_id is not None:
            df = pd.read_sql('SELECT *, condition_id || "_" || replicate_id || "_" || bait_id || "_" || prey_id || "_" || decoy AS pyprophet_feature_id, condition_id || "_" || bait_id || "_" || prey_id || "_" || decoy AS pyprophet_metafeature_id FROM FEATURE WHERE learning==0 AND condition_id=="%s" AND replicate_id=="%s" ORDER BY pyprophet_metafeature_id;' % (condition_id, replicate_id), con)
        else:
            df = pd.read_sql('SELECT *, condition_id || "_" || replicate_id || "_" || bait_id || "_" || prey_id || "_" || decoy AS pyprophet_feature_id, condition_id || "_" || bait_id || "_" || prey_id || "_" || decoy AS pyprophet_metafeature_id FROM FEATURE ORDER BY pyprophet_metafeature_id;', con)
        con.close()

        # Filter according to boundaries
        df_filter = df.groupby(["bait_id","prey_id","decoy"], group_keys=False)[["var_xcorr_shift","var_abundance_ratio","var_total_abundance_ratio"]].mean(numeric_only=True).reset_index(level=["bait_id","prey_id","decoy"])

        df_filter = df_filter[(df_filter['var_xcorr_shift'] <= self.maximum_sec_shift) & (df_filter['var_abundance_ratio'] >= self.minimum_abundance_ratio) & (df_filter['var_total_abundance_ratio'] >= self.minimum_abundance_ratio)]

        df = pd.merge(df, df_filter[["bait_id","prey_id","decoy"]], on=["bait_id","prey_id","decoy"])

        # We need to generate a kickstart score for semi-supervised learning that selects for the very best interaction heterodimers: perfect shape, co-elution and overlap
        df['main_var_kickstart'] = (df['var_xcorr_shape'] * df['var_total_abundance_ratio']) / (df['var_xcorr_shift'] + 1)

        # df = df.rename(index=str, columns={"var_xcorr_shape": "main_var_xcorr_shape", "var_sec_overlap": "main_var_sec_overlap"})

        return df

    def learn(self, learning_data):
        (result, scorer, weights) = PyProphet(
            self.classifier, 
            self.xgb_hyperparams, 
            self.xgb_params, 
            self.xgb_params_space, 
            self.xeval_fraction, 
            self.xeval_num_iter, 
            self.ss_initial_fdr, 
            self.ss_iteration_fdr, 
            self.ss_num_iter, 
            self.group_id, 
            self.parametric, 
            self.pfdr, 
            self.pi0_lambda, 
            self.pi0_method, 
            self.pi0_smooth_df, 
            self.pi0_smooth_log_pi0, 
            self.lfdr_truncate, 
            self.lfdr_monotone, 
            self.lfdr_transformation, 
            self.lfdr_adj, 
            self.lfdr_eps, 
            False, 
            self.threads, 
            self.test, 
            ss_score_filter = '', 
            color_palette='normal'
        ).learn_and_apply(learning_data)

        self.plot(result, scorer.pi0, "learning")
        self.plot_scores(result.scored_tables, "learning")

        return weights

    def store_model(self):
        con = sqlite3.connect(self.outfile)
        c = con.cursor()
        c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_XGB";')
        if c.fetchone()[0] == 1:
            c.execute('DELETE FROM PYPROPHET_XGB')
        else:
            c.execute('CREATE TABLE PYPROPHET_XGB (xgb BLOB)')

        c.execute('INSERT INTO PYPROPHET_XGB VALUES(?)', [pickle.dumps(self.weights)])
        con.commit()
        con.close()

    def load_model(self):
        try:
            con = sqlite3.connect(self.apply_model)

            data = con.execute("SELECT xgb FROM PYPROPHET_XGB").fetchone()
            con.close()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

        return pickle.loads(data[0])

    def apply(self, detecting_data, condition_id, replicate_id):
        (result, scorer, weights) = PyProphet(self.classifier, self.xgb_hyperparams, self.xgb_params, self.xgb_params_space, self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, False, self.threads, self.test, ss_score_filter = '', color_palette='normal').apply_weights(detecting_data, self.weights)

        df = result.scored_tables[['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin','d_score','p_value','q_value','pep']]
        df.columns = ['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin','score','pvalue','qvalue','pep']

        if self.export_tables:
            learning_interaction_name = "learn_int_scored.csv"
            result.scored_tables.to_csv(learning_interaction_name, index=False)

        if self.plot_reports:
            self.plot(result, scorer.pi0, condition_id + "_" + replicate_id + "_" + "detecting_" + str(detecting_data['confidence_bin'].values[0]))
            self.plot_scores(result.scored_tables, condition_id + "_" + replicate_id + "_" + "detecting_" + str(detecting_data['confidence_bin'].values[0]))

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
            plt.figure(figsize=(10, 10))
            plt.subplots_adjust(hspace=.5)
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
                    plt.clf()
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
        self.df = scores.groupby('confidence_bin', group_keys=False).apply(self.combine_scores)

    def read(self):
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT condition_id, bait_id , prey_id , decoy , confidence_bin, score, qvalue FROM FEATURE_SCORED;', con)
        con.close()

        return df

    def combine_scores(self, scores):
        combined_scores = scores.groupby(['condition_id','bait_id','prey_id','decoy','confidence_bin'], group_keys=False)['score'].mean(numeric_only=True).reset_index()

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
