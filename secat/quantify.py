import pandas as pd
import numpy as np
import click
import sqlite3
import os
import sys

from EmpiricalBrownsMethod import EmpiricalBrownsMethod
import itertools

from scipy.stats import ttest_ind

from pyprophet.stats import pi0est, qvalue

class quantitative_matrix:
    def __init__(self, outfile):
        self.outfile = outfile

        self.complex, self.monomer = self.read()

    def read(self):
        con = sqlite3.connect(self.outfile)

        # CREATE INDEX idx_feature_meta_feature_id ON FEATURE_META (feature_id);
        # CREATE INDEX idx_complex_feature_id_prey_id ON COMPLEX (feature_id, prey_id);
        # CREATE INDEX idx_complex_feature_id ON COMPLEX (feature_id);
        # CREATE INDEX idx_monomer_feature_id_prey_id ON MONOMER (feature_id, prey_id);
        # CREATE INDEX idx_monomer_feature_id ON MONOMER (feature_id);

        complex_data = pd.read_sql('SELECT feature_meta.condition_id, feature_meta.replicate_id, complex.feature_id, complex.bait_id, complex.prey_id, (complex.bait_intensity + complex.prey_intensity)/2 AS intensity, complex.pf_intensity_fraction AS intensity_fraction, complex.pf_score FROM complex INNER JOIN feature_meta ON complex.feature_id = feature_meta.feature_id WHERE feature_meta.decoy = 0 AND complex.decoy = 0;' , con)

        monomer_data = pd.read_sql('SELECT feature_meta.condition_id, feature_meta.replicate_id, monomer.feature_id, monomer.bait_id, monomer.prey_id, monomer.prey_intensity AS intensity, monomer.prey_intensity_fraction AS intensity_fraction, monomer.pp_score AS pf_score FROM monomer INNER JOIN feature_meta ON monomer.feature_id = feature_meta.feature_id WHERE feature_meta.decoy = 0 AND monomer.decoy = 0;' , con)

        con.close()

        # Summarize individual features
        complex_data = complex_data.groupby(['condition_id','replicate_id','bait_id','prey_id']).apply(lambda x: pd.Series({'intensity': sum(x['intensity']), 'intensity_fraction': np.mean(x['intensity_fraction'])})).reset_index()
        monomer_data = monomer_data.groupby(['condition_id','replicate_id','bait_id','prey_id']).apply(lambda x: pd.Series({'intensity': sum(x['intensity']), 'intensity_fraction': np.mean(x['intensity_fraction'])})).reset_index()

        # Log transform raw intensities
        complex_data['intensity'] = np.log(complex_data['intensity'])
        monomer_data['intensity'] = np.log(monomer_data['intensity'])

        return complex_data, monomer_data

class quantitative_test:
    def __init__(self, outfile):
        self.outfile = outfile
        self.levels = ['intensity','intensity_fraction']
        self.comparisons = self.contrast()

        self.complex, self.monomer = self.read()

        self.edges = self.compare()
        self.nodes = self.integrate()

    def contrast(self):
        con = sqlite3.connect(self.outfile)
        conditions = pd.read_sql('SELECT DISTINCT condition_id FROM SEC;' , con)['condition_id'].values.tolist()
        con.close()

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

        complex_data = pd.read_sql('SELECT * FROM complex_qm;' , con)

        monomer_data = pd.read_sql('SELECT * FROM monomer_qm;' , con)

        con.close()

        return complex_data, monomer_data

    def compare(self):
        dfs = []
        for level in self.levels:
            for comparison in self.comparisons:
                for state in [self.complex, self.monomer]:
                    df = self.test(state, level, comparison[0], comparison[1])

                    # Multiple testing correction via q-value
                    df['qvalue'] = qvalue(df['pvalue'].values, pi0est(df['pvalue'].values)['pi0'])
                    dfs.append(df)

        return pd.concat(dfs).sort_values(by='pvalue', ascending=True, na_position='last')

    def test(self, df, level, condition_1, condition_2):
        def ttest(x, experimental_design):
            x.set_index('condition_id')
            if condition_1 in x['condition_id'].values and condition_2 in x['condition_id'].values:
                if x['condition_id'].value_counts()[condition_1] > 1 and x['condition_id'].value_counts()[condition_2] > 1:
                    qm = pd.merge(experimental_design, x, how='left').transpose()
                    qm.columns = "quantitative" + "_" + experimental_design["condition_id"] + "_" + experimental_design["replicate_id"]
                    qm = qm.loc[level]
                    qm['pvalue'] = ttest_ind(x[x['condition_id'] == condition_1][level].values, x[x['condition_id'] == condition_2][level].values, equal_var=False)[1]

                    return qm

        # compute number of replicates
        experimental_design = df[['condition_id','replicate_id']].drop_duplicates()

        df_test = df.groupby(['bait_id','prey_id']).apply(lambda x: ttest(x, experimental_design)).reset_index().dropna()
        df_test['condition_1'] = condition_1
        df_test['condition_2'] = condition_2
        df_test['level'] = level

        return df_test[['condition_1','condition_2','level','bait_id','prey_id']+[c for c in df_test.columns if c.startswith("quantitative_")]+['pvalue']]

    def integrate(self):
        def empbrown(x):
            if x.shape[0] > 1:
                return pd.Series({'pvalue': EmpiricalBrownsMethod(x[[c for c in x.columns if c.startswith("quantitative_")]].values, x['pvalue'].values)})

        df = self.edges
        df_node = df.groupby(['condition_1', 'condition_2','level','bait_id']).apply(empbrown).reset_index()

        # Multiple testing correction via q-value
        df_node['qvalue'] = qvalue(df_node['pvalue'].values, pi0est(df_node['pvalue'].values)['pi0'])

        return df_node.sort_values(by='pvalue', ascending=True, na_position='last')
