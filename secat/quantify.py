import pandas as pd
import numpy as np
import click
import sqlite3
import os
import sys

from EmpiricalBrownsMethod import EmpiricalBrownsMethod
import itertools

from scipy.stats import mannwhitneyu, ttest_ind

from pyprophet.stats import pi0est, qvalue


class quantitative_matrix:
    def __init__(self, outfile):
        self.outfile = outfile

        self.interactions, self.chromatograms = self.read()
        self.complex = self.quantify()

    def read(self):
        con = sqlite3.connect(self.outfile)

        interactions = pd.read_sql('SELECT * FROM FEATURE_SCORED WHERE pvalue < 0.01 AND bait_id != prey_id;' , con)

        chromatograms = pd.read_sql('SELECT condition_id, replicate_id, sec_id, QUANTIFICATION.protein_id, QUANTIFICATION.peptide_id, peptide_intensity FROM QUANTIFICATION INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id INNER JOIN SEC ON QUANTIFICATION.RUN_ID = SEC.RUN_ID WHERE peptide_rank <= 6;', con)

        con.close()

        return interactions, chromatograms

    def quantify(self):
        def summarize(df):
            def aggregate(x):
                peptide_ix = (x['peptide_intensity']-x['peptide_intensity'].median()).abs().argsort()[:1]
                return pd.DataFrame({'peptide_intensity': x.iloc[peptide_ix]['peptide_intensity'], 'total_peptide_intensity': x.iloc[peptide_ix]['total_peptide_intensity']})

            # Summarize total peptide intensities
            peptide_total = df.groupby(['is_bait','peptide_id'])['peptide_intensity'].sum().reset_index()
            peptide_total.columns = ['is_bait','peptide_id','total_peptide_intensity']

            # Find SEC intersections
            bait_sec = df[df['is_bait']]['sec_id'].unique()
            prey_sec = df[~df['is_bait']]['sec_id'].unique()
            intersection = pd.DataFrame({'sec_id': list(set(bait_sec) & set(prey_sec))})
            
            # There needs to be at least one fraction where peptides from both proteins are measured.
            if intersection.shape[0] > 0:
                # Summarize intersection peptide intensities
                df_is = pd.merge(df, intersection, on='sec_id')
                peptide_is = df_is.groupby(['peptide_id'])['peptide_intensity'].sum().reset_index()
                peptide_is.columns = ['peptide_id','peptide_intensity']

                peptide = pd.merge(peptide_is, peptide_total, on='peptide_id')

                # At least three peptides are required for both interactors for quantification.
                if peptide[peptide['is_bait']].shape[0] >= 3 and peptide[~peptide['is_bait']].shape[0] >= 3:
                    # Select representative closest to median and aggregate
                    peptide = peptide.groupby(['is_bait']).apply(aggregate).reset_index(level=['is_bait'])

                    # Aggregate to protein level
                    protein = pd.DataFrame({'bait_intensity': peptide[peptide['is_bait']]['peptide_intensity'].values, 'total_bait_intensity': peptide[peptide['is_bait']]['total_peptide_intensity'].values, 'fractional_bait_intensity': peptide[peptide['is_bait']]['peptide_intensity'].values / peptide[peptide['is_bait']]['total_peptide_intensity'].values, 'prey_intensity': peptide[~peptide['is_bait']]['peptide_intensity'].values, 'total_prey_intensity': peptide[~peptide['is_bait']]['total_peptide_intensity'].values, 'fractional_prey_intensity': peptide[~peptide['is_bait']]['peptide_intensity'].values / peptide[~peptide['is_bait']]['total_peptide_intensity'].values})

                    protein['complex_intensity_ratio'] = protein['prey_intensity'] / protein['bait_intensity']
                    protein['complex_fraction_ratio'] = protein['fractional_prey_intensity'] / protein['fractional_bait_intensity']

                    return protein

        interactions = self.interactions

        # Generate long list
        baits = pd.merge(interactions, self.chromatograms, left_on=['condition_id','replicate_id','bait_id'], right_on=['condition_id','replicate_id','protein_id']).drop(columns=['protein_id'])
        baits['is_bait'] = True

        preys = pd.merge(interactions, self.chromatograms, left_on=['condition_id','replicate_id','prey_id'], right_on=['condition_id','replicate_id','protein_id']).drop(columns=['protein_id'])
        preys['is_bait'] = False

        data = pd.concat([baits, preys]).reset_index()

        data_agg = data.groupby(['condition_id','replicate_id','bait_id','prey_id']).apply(summarize).reset_index(level=['condition_id','replicate_id','bait_id','prey_id'])

        return data_agg

class quantitative_test:
    def __init__(self, outfile):
        self.outfile = outfile
        self.levels = ['bait_intensity','total_bait_intensity','fractional_bait_intensity','prey_intensity','total_prey_intensity','fractional_prey_intensity','complex_intensity_ratio','complex_fraction_ratio']
        self.comparisons = self.contrast()

        self.complex_qm = self.read()

        self.edge_directional = self.compare()
        self.edge_level, self.edge, self.node_level, self.node = self.integrate()

    def contrast(self):
        con = sqlite3.connect(self.outfile)
        conditions = pd.read_sql('SELECT DISTINCT condition_id FROM SEC;' , con)['condition_id'].values.tolist()
        con.close()

        if len(conditions) < 2:
            sys.exit("Error: Your experimental design is not appropriate. At least two conditions are necessary for quantification.")

        comparisons = []
        # prepare single-sample comparisons
        if 'control' in conditions:
            conditions.remove('control')
            for condition in conditions:
                comparisons.append([condition, 'control'])
        # prepare multi-sample comparisons
        else:
            comparisons = list(itertools.combinations(conditions, 2))

        return comparisons

    def read(self):
        con = sqlite3.connect(self.outfile)

        complex_qm = pd.read_sql('SELECT * FROM COMPLEX_QM;' , con)

        con.close()

        return complex_qm

    def compare(self):
        dfs = []
        for level in self.levels:
            for comparison in self.comparisons:
                for state in [self.complex_qm]:
                    if level in state.columns:
                        df = self.test(state, level, comparison[0], comparison[1])

                        # Multiple testing correction via q-value
                        df['qvalue'] = qvalue(df['pvalue'].values, pi0est(df['pvalue'].values)['pi0'])
                        dfs.append(df)

        return pd.concat(dfs).sort_values(by='pvalue', ascending=True, na_position='last')

    def test(self, df, level, condition_1, condition_2):
        def stat(x, experimental_design):
            x.set_index('condition_id')
            if condition_1 in x['condition_id'].values and condition_2 in x['condition_id'].values:
                if x['condition_id'].value_counts()[condition_1] > 0 and x['condition_id'].value_counts()[condition_2] > 0:
                    qm = pd.merge(experimental_design, x, how='left')

                    if level == 'score':
                        qm[level].fillna(0, inplace=True)

                    qmt = qm.transpose()
                    qmt.columns = "quantitative" + "_" + experimental_design["condition_id"] + "_" + experimental_design["replicate_id"]
                    # qmt['pvalue'] = mannwhitneyu(qm[qm['condition_id'] == condition_1][level].values, qm[qm['condition_id'] == condition_2][level].values)[1]
                    qmt['pvalue'] = ttest_ind(qm[qm['condition_id'] == condition_1][level].dropna().values, qm[qm['condition_id'] == condition_2][level].dropna().values, equal_var=False)[1]


                    return qmt.loc[level]

        # compute number of replicates
        experimental_design = df[['condition_id','replicate_id']].drop_duplicates()

        df_test = df.groupby(['bait_id','prey_id']).apply(lambda x: stat(x, experimental_design)).reset_index()#.dropna()
        df_test['condition_1'] = condition_1
        df_test['condition_2'] = condition_2
        df_test['level'] = level

        return df_test[['condition_1','condition_2','level','bait_id','prey_id']+[c for c in df_test.columns if c.startswith("quantitative_")]+['pvalue']]

    def integrate(self):
        def collapse(x):
            if x.shape[0] > 1:
                return pd.Series({'pvalue': EmpiricalBrownsMethod(x[[c for c in x.columns if c.startswith("quantitative_")]].values, x['pvalue'].values)})
            elif x.shape[0] == 1:
                return pd.Series({'pvalue': x['pvalue'].values[0]})

        df = self.edge_directional

        df_edge_level = df.groupby(['condition_1', 'condition_2','level']).apply(collapse).reset_index()
        df_edge = df.groupby(['condition_1', 'condition_2']).apply(collapse).reset_index()

        df_node_level = df.groupby(['condition_1', 'condition_2','level','bait_id']).apply(collapse).reset_index()
        df_node = df.groupby(['condition_1', 'condition_2','bait_id']).apply(collapse).reset_index()

        # Multiple testing correction via q-value
        df_edge_level['qvalue'] = qvalue(df_edge_level['pvalue'].values, pi0est(df_edge_level['pvalue'].values)['pi0'])
        df_edge['qvalue'] = qvalue(df_edge['pvalue'].values, pi0est(df_edge['pvalue'].values)['pi0'])

        df_node_level['qvalue'] = qvalue(df_node_level['pvalue'].values, pi0est(df_node_level['pvalue'].values)['pi0'])
        df_node['qvalue'] = qvalue(df_node['pvalue'].values, pi0est(df_node['pvalue'].values)['pi0'])

        return df_edge_level, df_edge, df_node_level, df_node
