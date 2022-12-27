import pandas as pd
import numpy as np
import click
import pdb
import sqlite3
import sys
from decoupler.method_viper import run_viper
import os

from .EmpiricalBrownsMethod import EmpiricalBrownsMethod
import itertools

from scipy.stats import ttest_ind, ttest_rel

from statsmodels.stats.multitest import multipletests

class quantitative_matrix:
    def __init__(self, outfile, maximum_interaction_qvalue, minimum_peptides, maximum_peptides):
        self.outfile = outfile
        self.maximum_interaction_qvalue = maximum_interaction_qvalue
        self.minimum_peptides = minimum_peptides
        self.maximum_peptides = maximum_peptides

        self.interactions, self.detections, self.chromatograms, self.peaks = self.read()
        self.monomer_peptide = self.quantify_monomers()
        self.complex_peptide = self.quantify_complexes()

    def read(self):
        con = sqlite3.connect(self.outfile)

        click.echo("Getting interations table")
        interactions = pd.read_sql(
            f"""
            SELECT DISTINCT bait_id, prey_id 
            FROM FEATURE_SCORED_COMBINED 
            WHERE qvalue <= {self.maximum_interaction_qvalue} 
            AND bait_id != prey_id 
            AND decoy == 0;
            """, 
            con
        )

        click.echo("Getting detections table")
        detections = pd.read_sql(
            f"""
            SELECT DISTINCT condition_id, replicate_id, FEATURE_SCORED.bait_id, FEATURE_SCORED.prey_id 
            FROM FEATURE_SCORED 
            INNER JOIN (
                SELECT DISTINCT bait_id, prey_id 
                FROM FEATURE_SCORED_COMBINED 
                WHERE qvalue <= {self.maximum_interaction_qvalue} 
                AND bait_id != prey_id 
                AND decoy == 0
            ) AS FEATURE_SCORED_COMBINED 
            ON FEATURE_SCORED.bait_id = FEATURE_SCORED_COMBINED.bait_id 
            AND FEATURE_SCORED.prey_id = FEATURE_SCORED_COMBINED.prey_id;
            """, 
            con
        )

        click.echo("Getting chromatograms table")
        chromatograms = pd.read_sql(
            f"""
            SELECT 
                SEC.condition_id, 
                SEC.replicate_id, 
                SEC.sec_id, 
                QUANTIFICATION.protein_id, 
                QUANTIFICATION.peptide_id, 
                peptide_intensity, 
                MONOMER.sec_id AS monomer_sec_id 
            FROM QUANTIFICATION 
            INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id 
            INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id 
            INNER JOIN SEC ON QUANTIFICATION.RUN_ID = SEC.RUN_ID 
            INNER JOIN MONOMER ON QUANTIFICATION.protein_id = MONOMER.protein_id AND SEC.condition_id = MONOMER.condition_id AND SEC.replicate_id = MONOMER.replicate_id 
            WHERE peptide_count >= {self.minimum_peptides} 
            AND peptide_rank <= {self.maximum_peptides};
            """, 
            con
        )

        # TODO: Consider replacing pd.read_sql with https://github.com/sfu-db/connector-x to improve read speeds and utilize concurrency
        click.echo("Getting peaks table")
        peaks = pd.read_sql('SELECT * FROM PROTEIN_PEAKS;', con)

        con.close()

        return interactions, detections, chromatograms, peaks

    def quantify_monomers(self):
        def sec_summarize(df):
            def aggregate(x):
                peptide_ix = (x['peptide_intensity']-x['peptide_intensity'].max()).abs().argsort()[:self.maximum_peptides]
                return pd.DataFrame({'peptide_id': x.iloc[peptide_ix]['peptide_id'], 'peptide_intensity': x.iloc[peptide_ix]['peptide_intensity'], 'total_peptide_intensity': x.iloc[peptide_ix]['total_peptide_intensity']})

            # Summarize total peptide intensities
            peptide_total = df.groupby(['peptide_id'])['peptide_intensity'].sum().reset_index()
            peptide_total.columns = ['peptide_id','total_peptide_intensity']

            # Summarize monomer peptide intensities
            if df[df['sec_id'] >= df['monomer_sec_id']].shape[0] > 0:
                peptide_mono = df[df['sec_id'] >= df['monomer_sec_id']].groupby(['peptide_id'])['peptide_intensity'].sum().reset_index()
                peptide_mono.columns = ['peptide_id','peptide_intensity']
            else:
                peptide_mono = pd.DataFrame({'peptide_id': df['peptide_id'].unique(), 'peptide_intensity': 0})

            peptide = pd.merge(peptide_mono, peptide_total, on='peptide_id')

            # Ensure that minimum peptides are present
            if peptide.shape[0] >= self.minimum_peptides:
                # Select representative closest to max and aggregate
                peptide = aggregate(peptide)

                return peptide

        def peptide_summarize(df):
            # Aggregate to peptide level
            peptide = df[['condition_id','replicate_id','bait_id','prey_id','is_bait','peptide_id']].copy()
            peptide['monomer_abundance'] = np.log2(df['peptide_intensity'].values+1)
            peptide['assembled_abundance'] = np.log2(df['total_peptide_intensity'].values-df['peptide_intensity'].values+1)
            peptide['total_abundance'] = np.log2(df['total_peptide_intensity'].values+1)

            return peptide

        # Quantify monomers
        monomers = self.chromatograms.copy()
        monomers['bait_id'] = monomers['protein_id']
        monomers['prey_id'] = monomers['protein_id']
        monomers['is_bait'] = True

        monomers_sec = monomers.groupby(['condition_id','replicate_id','bait_id','prey_id','is_bait']).apply(sec_summarize).reset_index(level=['condition_id','replicate_id','bait_id','prey_id','is_bait'])
        monomers_peptides = peptide_summarize(monomers_sec)

        return monomers_peptides

    def quantify_complexes(self):
        def sec_summarize(df):
            def aggregate(x):
                peptide_ix = (x['peptide_intensity'] - x['peptide_intensity'].max()).abs().argsort().iloc[:self.maximum_peptides]
                return pd.DataFrame({'peptide_id': x.iloc[peptide_ix]['peptide_id'], 'peptide_intensity': x.iloc[peptide_ix]['peptide_intensity']})

            # Remove monomer fractions for complex-centric quantification
            df = df[df['sec_id'] < df['monomer_sec_id']]
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

                peptide = pd.merge(df[['condition_id','replicate_id','bait_id','prey_id','is_bait','peptide_id']].drop_duplicates(), peptide_is, on='peptide_id')
                # Ensure that minimum peptides are present for both interactors for quantification.
                if peptide[peptide['is_bait']].shape[0] >= self.minimum_peptides and peptide[~peptide['is_bait']].shape[0] >= self.minimum_peptides:
                    # Select representative closest to max and aggregate
                    peptide = peptide.groupby(['is_bait'], group_keys=True).apply(aggregate).reset_index(level=['is_bait'])
                    
                    return peptide

        def peptide_summarize(df):
            # Aggregate to peptide level
            peptide = df[['condition_id','replicate_id','bait_id','prey_id','is_bait','peptide_id']].copy()
            peptide['interactor_abundance'] = np.log2(df['peptide_intensity']+1)

            return peptide

        # Restrict chromatographic data to selected peaks only
        chromatograms = pd.merge(self.chromatograms, self.peaks, on=['condition_id','replicate_id','protein_id','sec_id'])

        # Quantify interactions
        baits = pd.merge(self.interactions, chromatograms, left_on=['bait_id'], right_on=['protein_id']).drop(columns=['protein_id'])
        baits['is_bait'] = True

        preys = pd.merge(self.interactions, chromatograms, left_on=['prey_id'], right_on=['protein_id']).drop(columns=['protein_id'])
        preys['is_bait'] = False

        complexes = pd.concat([baits, preys]).reset_index()
        complexes_sec = complexes.groupby(['condition_id','replicate_id','bait_id','prey_id']).apply(sec_summarize).reset_index(level=['condition_id','replicate_id','bait_id','prey_id'])
        complexes_peptides = peptide_summarize(complexes_sec)

        return complexes_peptides

class enrichment_test:
    def __init__(self, outfile, control_condition, paired, min_abs_log2fx, missing_peptides, peptide_log2fx, threads):
        self.outfile = outfile
        self.control_condition = control_condition
        self.paired = paired
        self.min_abs_log2fx = min_abs_log2fx
        self.missing_peptides = missing_peptides
        self.peptide_log2fx = peptide_log2fx
        self.threads = threads
        self.levels = ['interactor_abundance','complex_abundance','interactor_ratio','monomer_abundance','assembled_abundance','total_abundance']
        self.comparisons = self.contrast()

        self.monomer_qm, self.complex_qm = self.read()

        self.tests = self.compare()
        self.edge_level, self.edge, self.node_level, self.node, self.protein_level = self.integrate()

    def contrast(self):
        con = sqlite3.connect(self.outfile)
        conditions = pd.read_sql('SELECT DISTINCT condition_id FROM SEC;' , con)['condition_id'].values.tolist()

        con.close()

        if len(conditions) < 2:
            sys.exit("Error: Your experimental design is not supported. At least two conditions are necessary for differential analysis.")

        comparisons = []
        # prepare single-sample comparisons
        if self.control_condition in conditions:
            conditions.remove(self.control_condition)
            for condition in conditions:
                comparisons.append([condition, self.control_condition])
        # prepare multi-sample comparisons
        elif self.control_condition == 'center':
            comparisons = list(itertools.combinations(conditions, 2))
        else:
            sys.exit("Error: Specify correct control condition identifier as reference or use 'center' to compare all against all.")

        return comparisons

    def read(self):
        con = sqlite3.connect(self.outfile)

        monomer_qm = pd.read_sql('SELECT * FROM MONOMER_QM;' , con)
        complex_qm = pd.read_sql('SELECT * FROM COMPLEX_QM;' , con)

        con.close()

        return monomer_qm, complex_qm

    def compare(self):
        # TODO: Add tqdm progress bar to better visualize compare() progress
        dfs = []
        for level in self.levels:
            for state in [self.monomer_qm, self.complex_qm]:
                if level in state.columns or (level in ['complex_abundance', 'interactor_ratio'] and 'interactor_abundance' in state.columns):
                    if level in state.columns:
                        dat = state[state[level] > 0].copy()
                    else:
                        dat = state[state['interactor_abundance'] > 0].copy()
                        dat = dat.rename(index=str, columns={"interactor_abundance": level})

                    dat['query_id'] = dat['bait_id'] + '_' + dat['prey_id']
                    dat['query_peptide_id'] = dat['bait_id'] + '_' + dat['prey_id'] + '_' + dat['peptide_id']
                    dat['quantification_id'] = 'viper_' + dat['condition_id'] + '_' + dat['replicate_id']
                    dat['run_id'] = dat['condition_id'] + '_' + dat['replicate_id']
                    qm_ids = dat[['quantification_id','condition_id','replicate_id']].drop_duplicates()

                    if self.missing_peptides == 'drop':
                        peptide_fill_value = np.nan
                    elif self.missing_peptides == 'zero':
                        peptide_fill_value = 0
                    else:
                        sys.exit("Error: Invalid parameter for 'missing_peptides' selected.")

                    # Generate matrix for fold-change
                    quant_mx = dat.pivot_table(index=['query_id','is_bait','query_peptide_id'], columns='quantification_id', values=level, fill_value=peptide_fill_value)

                    # Generate matrix for ratio-change
                    ratio_mx = dat.pivot_table(index=['query_id','is_bait','query_peptide_id'], columns='quantification_id', values=level, fill_value=peptide_fill_value)

                    # Generate matrix for VIPER
                    data_mx = dat.pivot_table(index='query_peptide_id', columns='quantification_id', values=level, fill_value=0)

                    net = pd.DataFrame(columns=["source", "target", "weight"])
                    
                    # Generate subunit set for VIPER
                    if level == 'complex_abundance':
                        # Complex abundance testing combines bait and prey peptides into a single regulon with positive tfmode sign
                        query_set = dat[['query_id','is_bait','query_peptide_id']].copy()
                        query_set['query_id'] = query_set['query_id'] + "+1"

                        net['source'] = query_set['query_id']
                        net['target'] = query_set['query_peptide_id']
                        net['weight'] = [1 for _ in range(query_set.shape[0])]
                        net = net.groupby(['source', 'target']).sum('weight').reset_index()
                    elif level == 'interactor_ratio':
                        # Complex stoichiometry testing combines bait and prey peptides into a single regulon but with different tfmode signs
                        query_set = dat[['query_id','is_bait','query_peptide_id']].copy()
                        query_set.loc[query_set['is_bait']==0,'is_bait'] = -1
                        query_set['query_id'] = query_set['query_id'] + "+1"
                        filtered_query_set = query_set.groupby(['query_id']).apply(lambda x: x.drop_duplicates(['is_bait', 'query_peptide_id'])).reset_index(drop=True)

                        net['source'] = filtered_query_set['query_id']
                        net['target'] = filtered_query_set['query_peptide_id']
                        net['weight'] = filtered_query_set['is_bait'].apply(lambda x: 1 if x == True else -1)
                        net = net.groupby(['source', 'target']).sum('weight').reset_index() # Not sure if this is needed
                    else:
                        # All other modalities are assessed on protein-level, separately for bait and prey proteins
                        query_set = dat[['query_id','is_bait','query_peptide_id']].copy()
                        query_set['query_id'] = query_set['query_id'] + "+" + query_set['is_bait'].astype(int).astype(str)

                        net['source'] = query_set['query_id']
                        net['target'] = query_set['query_peptide_id']
                        net['weight'] = [1 for _ in range(query_set.shape[0])]
                        net = net.groupby(['source', 'target']).sum('weight').reset_index()

                    results = run_viper(data_mx.T, net, verbose=False, pleiotropy=False, min_n=1)
                    results = results[0].T.reset_index()
                    results['query_id'] = results['index']
                    results.drop('index', inplace=True, axis=1)

                    results[['query_id','is_bait']] = results['query_id'].str.split("+", expand=True)
                    results['is_bait'] = results['is_bait'].astype('int')
                    results['level'] = level

                    # Append reverse information for complex_abundance and interactor_ratio levels
                    if level in ['complex_abundance', 'interactor_ratio']:
                        results_rev = results.copy()
                        results_rev['is_bait'] = 0
                        results = pd.concat([results, results_rev])

                    for comparison in self.comparisons:
                        results['condition_1'] = comparison[0]
                        results['condition_2'] = comparison[1]

                        # Compute fold-change and absolute fold-change
                        quant_mx_avg = quant_mx.groupby(['query_id','is_bait','query_peptide_id']).apply(lambda x: pd.Series({'comparison_0': np.nanmean(np.exp2(x[qm_ids[qm_ids['condition_id']==comparison[0]]['quantification_id'].values].values)), 'comparison_1': np.nanmean(np.exp2(x[qm_ids[qm_ids['condition_id']==comparison[1]]['quantification_id'].values].values))})).reset_index(level=['query_id','is_bait','query_peptide_id'])

                        if self.peptide_log2fx:
                            quant_mx_log2fx = quant_mx_avg.groupby(['query_id','is_bait','query_peptide_id']).apply(lambda x: np.log2((x['comparison_0'])/(x['comparison_1']))).reset_index(level=['query_id','is_bait','query_peptide_id'])

                            quant_mx_log2fx_prot = quant_mx_log2fx.groupby(['query_id','is_bait']).mean(numeric_only=True).reset_index()
                            quant_mx_log2fx_prot.columns = ['query_id','is_bait','log2fx']
                            quant_mx_log2fx_prot['abs_log2fx'] = np.abs(quant_mx_log2fx_prot['log2fx'])
                        else:
                            quant_mx_avg_prot = quant_mx_avg.groupby(['query_id','is_bait'])[['comparison_0','comparison_1']].mean().reset_index()
                            quant_mx_log2fx_prot = quant_mx_avg_prot.groupby(['query_id','is_bait']).apply(lambda x: np.log2((x['comparison_0'])/(x['comparison_1']))).reset_index(level=['query_id','is_bait'])

                            quant_mx_log2fx_prot.columns = ['query_id','is_bait','log2fx']
                            quant_mx_log2fx_prot['abs_log2fx'] = np.abs(quant_mx_log2fx_prot['log2fx'])

                        results = pd.merge(results, quant_mx_log2fx_prot, on=['query_id','is_bait'], how='left')

                        # Compute interactor ratio
                        if level in ['complex_abundance', 'interactor_ratio']:
                            ratio_mx_prot = ratio_mx.groupby(['query_id','is_bait'])[[c for c in ratio_mx.columns if c.startswith("viper_")]].mean().reset_index()
                            ratio_mx_prot_ratio = ratio_mx_prot.groupby('query_id').apply(lambda x: (x.loc[x['is_bait']==0].squeeze()+1) / (x.loc[x['is_bait']==1].squeeze()+1)).reset_index(level='query_id')

                            ratio_change = ratio_mx_prot_ratio.groupby('query_id').apply(lambda x: np.mean(x[qm_ids[qm_ids['condition_id']==comparison[0]]['quantification_id'].values].values) / np.mean(x[qm_ids[qm_ids['condition_id']==comparison[1]]['quantification_id'].values].values)).reset_index(level='query_id')
                            ratio_change.columns = ['query_id','interactor_ratio']
                            ratio_change.loc[ratio_change['interactor_ratio'] > 1,'interactor_ratio'] = (1 / ratio_change.loc[ratio_change['interactor_ratio'] > 1,'interactor_ratio'])
                            results = pd.merge(results, ratio_change, on=['query_id'], how='left')
                        else:
                            results['interactor_ratio'] = np.nan

                        # Conduct statistical tests
                        # Paired analysis: For example replicates 1 of conditions A & B were measured by the same SILAC experiment
                        if self.paired:
                            results_pvalue = results.groupby(['query_id','is_bait','level']).apply(lambda x: pd.Series({"pvalue": ttest_rel(x[qm_ids[qm_ids['condition_id']==comparison[0]].sort_values(by=['quantification_id'])['quantification_id'].values].values[0], x[qm_ids[qm_ids['condition_id']==comparison[1]].sort_values(by=['quantification_id'])['quantification_id'].values].values[0])[1]})).reset_index()
                        # Treat samples as independent measurements, e.g. quantification by LFQ
                        else:
                            results_pvalue = results.groupby(['query_id','is_bait','level']).apply(lambda x: pd.Series({"pvalue": ttest_ind(x[qm_ids[qm_ids['condition_id']==comparison[0]]['quantification_id'].values].values[0], x[qm_ids[qm_ids['condition_id']==comparison[1]]['quantification_id'].values].values[0], equal_var=True)[1]})).reset_index()
                        results = pd.merge(results, results_pvalue, on=['query_id','is_bait','level'])

                        # Set p-value to 1.0 if invalid
                        results.loc[np.isnan(results['pvalue']),'pvalue'] = 1.0

                        # Append meta information
                        results = pd.merge(results, dat[['query_id','bait_id','prey_id']].drop_duplicates(), on='query_id')

                        dfs.append(results[['condition_1','condition_2','level','bait_id','prey_id','is_bait','log2fx','abs_log2fx','interactor_ratio','pvalue']+[c for c in results.columns if c.startswith("viper_")]])

        return pd.concat(dfs, ignore_index=True, sort=True).sort_values(by='pvalue', ascending=True, na_position='last')

    def integrate(self):
        def collapse(x):
            if x.shape[0] > 1:
                result = pd.Series({'num_interactors': x.shape[0], 'log2fx': np.mean(x['log2fx'].values), 'abs_log2fx': np.mean(x['abs_log2fx'].values), 'interactor_ratio': np.mean(x['interactor_ratio'].values), 'pvalue': EmpiricalBrownsMethod(x[[c for c in x.columns if c.startswith("viper_")]].values, x['pvalue'].values)})
            else:
                result = pd.Series({'num_interactors': x.shape[0], 'log2fx': x['log2fx'].values[0], 'abs_log2fx': x['abs_log2fx'].values[0], 'interactor_ratio': x['interactor_ratio'].values[0], 'pvalue': x['pvalue'].values[0]})
            return(result)

        def mtcorrect(x):
            x['pvalue_adjusted'] = multipletests(pvals=x['pvalue'].values, method="fdr_bh")[1]

            return(x)

        df_edge_level = self.tests[(self.tests['bait_id'] != self.tests['prey_id']) & self.tests['is_bait']]
        df_edge_level_rev = self.tests[(self.tests['bait_id'] != self.tests['prey_id']) & ~self.tests['is_bait']]
        df_edge_level_rev = df_edge_level_rev.rename(index=str, columns={"bait_id": "prey_id", "prey_id": "bait_id"})

        df_protein_level = self.tests[self.tests['bait_id'] == self.tests['prey_id']]
        df_edge_full = pd.concat([df_protein_level, df_edge_level, df_edge_level_rev], sort=False)
        df_edge_level = pd.concat([df_edge_level, df_edge_level_rev[df_edge_level_rev['level']=='interactor_abundance']], sort=False)
        df_node_level = df_edge_full.groupby(['condition_1', 'condition_2','level','bait_id'], group_keys=False).apply(collapse).reset_index()

        # Multi-testing correction and pooling
        df_protein_level = df_protein_level.groupby(['condition_1', 'condition_2','level'], group_keys=False).apply(mtcorrect).reset_index()
        df_edge_level = df_edge_level.groupby(['condition_1', 'condition_2','level'], group_keys=False).apply(mtcorrect).reset_index()
        df_edge = df_edge_level.sort_values('pvalue').groupby(['condition_1','condition_2','bait_id','prey_id']).head(1).reset_index()
        df_node_level = df_node_level.groupby(['condition_1', 'condition_2','level'], group_keys=False).apply(mtcorrect).reset_index()

        df_node_level_filtered = df_node_level[df_node_level['abs_log2fx'] > self.min_abs_log2fx]
        df_node = df_node_level_filtered.sort_values('pvalue_adjusted').groupby(['condition_1','condition_2','bait_id'], group_keys=False).head(1).reset_index()

        click.echo("Info: Total dysregulated proteins detected:")
        click.echo("%s (at FDR < 0.01)" % (df_node[df_node['pvalue_adjusted'] < 0.01][['bait_id']].drop_duplicates().shape[0]))
        click.echo("%s (at FDR < 0.05)" % (df_node[df_node['pvalue_adjusted'] < 0.05][['bait_id']].drop_duplicates().shape[0]))
        click.echo("%s (at FDR < 0.1)" % (df_node[df_node['pvalue_adjusted'] < 0.1][['bait_id']].drop_duplicates().shape[0]))

        for level in df_node_level_filtered['level'].unique():
            click.echo("Info: Dysregulated (%s-mode) proteins detected:" % (level))
            click.echo("%s (at FDR < 0.01)" % (df_node_level_filtered[(df_node_level_filtered['level'] == level) & (df_node_level_filtered['pvalue_adjusted'] < 0.01)][['bait_id']].drop_duplicates().shape[0]))
            click.echo("%s (at FDR < 0.05)" % (df_node_level_filtered[(df_node_level_filtered['level'] == level) & (df_node_level_filtered['pvalue_adjusted'] < 0.05)][['bait_id']].drop_duplicates().shape[0]))
            click.echo("%s (at FDR < 0.1)" % (df_node_level_filtered[(df_node_level_filtered['level'] == level) & (df_node_level_filtered['pvalue_adjusted'] < 0.1)][['bait_id']].drop_duplicates().shape[0]))

        return df_edge_level[['condition_1','condition_2','level','bait_id','prey_id','log2fx','abs_log2fx','interactor_ratio','pvalue','pvalue_adjusted']+[c for c in df_edge_level.columns if c.startswith("viper_")]], df_edge[['condition_1','condition_2','level','bait_id','prey_id','log2fx','abs_log2fx','interactor_ratio','pvalue','pvalue_adjusted']], df_node_level[['condition_1','condition_2','level','bait_id','log2fx','abs_log2fx','interactor_ratio','num_interactors','pvalue','pvalue_adjusted']], df_node[['condition_1','condition_2','level','bait_id','log2fx','abs_log2fx','interactor_ratio','num_interactors','pvalue','pvalue_adjusted']], df_protein_level[['condition_1','condition_2','level','bait_id','log2fx','abs_log2fx','interactor_ratio','pvalue','pvalue_adjusted'] + [c for c in df_protein_level.columns if c.startswith("viper_")]]