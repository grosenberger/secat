import pandas as pd
import numpy as np
import click
import sqlite3
import os
import sys

from .EmpiricalBrownsMethod import EmpiricalBrownsMethod
import itertools

from scipy.stats import mannwhitneyu, ttest_ind

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

        interactions = pd.read_sql('SELECT DISTINCT bait_id, prey_id FROM FEATURE_SCORED_COMBINED WHERE qvalue <= %s AND bait_id != prey_id AND decoy == 0;' % (self.maximum_interaction_qvalue), con)

        detections = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, FEATURE_SCORED.bait_id, FEATURE_SCORED.prey_id, 1-pep AS score FROM FEATURE_SCORED INNER JOIN (SELECT DISTINCT bait_id, prey_id FROM FEATURE_SCORED_COMBINED WHERE qvalue <= %s AND bait_id != prey_id AND decoy == 0) AS FEATURE_SCORED_COMBINED ON FEATURE_SCORED.bait_id = FEATURE_SCORED_COMBINED.bait_id AND FEATURE_SCORED.prey_id = FEATURE_SCORED_COMBINED.prey_id;' % (self.maximum_interaction_qvalue), con)

        chromatograms = pd.read_sql('SELECT SEC.condition_id, SEC.replicate_id, SEC.sec_id, QUANTIFICATION.protein_id, QUANTIFICATION.peptide_id, peptide_intensity, MONOMER.sec_id AS monomer_sec_id FROM QUANTIFICATION INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id INNER JOIN SEC ON QUANTIFICATION.RUN_ID = SEC.RUN_ID INNER JOIN MONOMER ON QUANTIFICATION.protein_id = MONOMER.protein_id and SEC.condition_id = MONOMER.condition_id AND SEC.replicate_id = MONOMER.replicate_id WHERE peptide_count >= %s AND peptide_rank <= %s;' % (self.minimum_peptides, self.maximum_peptides), con)

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
            peptide['monomer_intensity'] = np.log2(df['peptide_intensity'].values+1)
            peptide['total_intensity'] = np.log2(df['total_peptide_intensity'].values+1)

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
                peptide_ix = (x['peptide_intensity']-x['peptide_intensity'].max()).abs().argsort()[:self.maximum_peptides]
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
                    peptide = peptide.groupby(['is_bait']).apply(aggregate).reset_index(level=['is_bait'])

                    return peptide

        def peptide_summarize(df):
            # Aggregate to peptide level
            peptide = df[['condition_id','replicate_id','bait_id','prey_id','is_bait','peptide_id']].copy()
            peptide['fraction_intensity'] = np.log2(df['peptide_intensity']+1)

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
    def __init__(self, outfile, control_condition, enrichment_permutations, threads):
        self.outfile = outfile
        self.control_condition = control_condition
        self.enrichment_permutations = enrichment_permutations
        self.threads = threads
        self.levels = ['fraction_intensity','monomer_intensity','total_intensity']
        self.comparisons = self.contrast()

        self.monomer_qm, self.complex_qm = self.read()

        self.tests = self.compare()
        self.edge_level, self.edge, self.node_level, self.node, self.protein_level = self.integrate()

    def contrast(self):
        con = sqlite3.connect(self.outfile)
        conditions = pd.read_sql('SELECT DISTINCT condition_id FROM SEC;' , con)['condition_id'].values.tolist()
        con.close()

        if len(conditions) < 2:
            sys.exit("Error: Your experimental design is not appropriate. At least two conditions are necessary for quantification.")

        comparisons = []
        # prepare single-sample comparisons
        if self.control_condition in conditions:
            conditions.remove(self.control_condition)
            for condition in conditions:
                comparisons.append([condition, self.control_condition])
        # prepare multi-sample comparisons
        else:
            comparisons = list(itertools.combinations(conditions, 2))

        return comparisons

    def viper(self, data_mx, subunit_set):
        from rpy2 import robjects
        from rpy2.rinterface import RRuntimeError
        from rpy2.robjects import r, pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        numpy2ri.activate()
        pandas2ri.activate()

        base = importr('base')
        try:
            vp = importr("viper")
        except RRuntimeError:
            base.source("http://www.bioconductor.org/biocLite.R")
            biocinstaller = importr("BiocInstaller")
            biocinstaller.biocLite("viper")
            vp = importr("viper")

        # Conduct VIPER analysis
        regulons = []
        for subunit in subunit_set:
            tfmode = robjects.FloatVector(np.repeat(1.0,len(subunit_set[subunit])))
            tfmode.names = robjects.StrVector(subunit_set[subunit])
            likelihood = robjects.FloatVector(np.repeat(1.0,len(subunit_set[subunit])))
            
            regulon = robjects.ListVector({'tfmode': tfmode, 'likelihood': likelihood})
            regulons.append(regulon)

        # Generate R Regulon
        r_network = robjects.ListVector(zip(subunit_set.keys(), regulons))

        # Generate R matrix
        mx_nr, mx_nc = data_mx.shape
        mx_vec = robjects.FloatVector(data_mx.values.transpose().reshape((data_mx.size)))
        r_mx = robjects.r.matrix(mx_vec, nrow=mx_nr, ncol=mx_nc)
        r_mx.rownames = robjects.StrVector(data_mx.index)
        r_mx.colnames = robjects.StrVector(data_mx.columns)
        reference_ix = [i+1 for i, s in enumerate(r_mx.colnames) if self.control_condition in s] # R index
        if len(reference_ix) < 3:
            sys.exit("Error: Only %s control samples were specified. SECAT requires at least three controls." % (len(reference_ix)))

        # Compute VIPER profile
        vpres = vp.viper(r_mx, r_network, verbose = False, minsize = 1, cores = self.threads)
        vpsig = vp.viperSignature(vpres, vpres.rx(True, robjects.IntVector(reference_ix)), per = self.enrichment_permutations, cores=self.threads).rx2('signature')
        
        pd_mx = pd.DataFrame(pandas2ri.ri2py(vpsig), columns = vpsig.colnames)
        pd_mx['query_id'] = vpsig.rownames
        return(pd_mx)

    def read(self):
        con = sqlite3.connect(self.outfile)

        monomer_qm = pd.read_sql('SELECT * FROM MONOMER_QM;' , con)
        complex_qm = pd.read_sql('SELECT * FROM COMPLEX_QM;' , con)

        con.close()

        # monomer_qm = pd.merge(monomer_qm, monomer_qm[['bait_id']].drop_duplicates().head(100), on=['bait_id'])
        # complex_qm = pd.merge(complex_qm, complex_qm[['bait_id','prey_id']].drop_duplicates().head(100), on=['bait_id','prey_id'])

        return monomer_qm, complex_qm

    def compare(self):
        def collapse(x):
            ebm = EmpiricalBrownsMethod(x[[c for c in x.columns if c.startswith("viper_")]].values, x['pvalue'].values, extra_info=True)

            result = pd.Series({'level': 'complex_intensity', 'nes': np.mean(x['nes'].values), 'anes': np.mean(x['anes'].values), 'pvalue': ebm[0], 'cfactor': ebm[2]})

            result = pd.concat([result, x[[c for c in x.columns if c.startswith("viper_")]].mean(axis=0)])

            return(result)

        dfs = []
        for level in self.levels:
            for state in [self.monomer_qm, self.complex_qm]:
                if level in state.columns:
                    # Generate matrix
                    dat = state[state[level] > 0].copy()
                    dat['query_id'] = dat['bait_id'] + '_' + dat['prey_id']
                    dat['query_peptide_id'] = dat['bait_id'] + '_' + dat['prey_id'] + '_' + dat['peptide_id']
                    dat['quantification_id'] = 'viper_' + dat['condition_id'] + '_' + dat['replicate_id']
                    dat['run_id'] = dat['condition_id'] + '_' + dat['replicate_id']
                    data_mx = dat.pivot_table(index='query_peptide_id', columns='quantification_id', values=level, fill_value=0)

                    qm_ids = dat[['quantification_id','condition_id','replicate_id']].drop_duplicates()

                    # Generate subunit set
                    query_set = dat[['query_id','is_bait','query_peptide_id']].copy()
                    query_set['query_id'] = query_set['query_id'] + "+" + query_set['is_bait'].astype(int).astype(str)
                    subunit_set = query_set.groupby(['query_id'])['query_peptide_id'].apply(lambda x: x.unique().tolist()).to_dict()

                    # Run VIPER
                    results = self.viper(data_mx, subunit_set)

                    results[['query_id','is_bait']] = results['query_id'].str.split("+", expand=True)
                    results['is_bait'] = results['is_bait'].astype('int')
                    results['level'] = level

                    for comparison in self.comparisons:
                        results['condition_1'] = comparison[0]
                        results['condition_2'] = comparison[1]

                        # Conduct statistical test
                        results_pvalue = results.groupby(['query_id','is_bait','level']).apply(lambda x: pd.Series({"nes": np.mean(x[qm_ids[qm_ids['condition_id']==comparison[0]]['quantification_id'].values].values[0]), "anes": np.mean(np.abs(x[qm_ids[qm_ids['condition_id']==comparison[0]]['quantification_id'].values].values[0])), "cfactor": 0, "pvalue": ttest_ind(x[qm_ids[qm_ids['condition_id']==comparison[0]]['quantification_id'].values].values[0], x[qm_ids[qm_ids['condition_id']==comparison[1]]['quantification_id'].values].values[0], equal_var=True)[1]})).reset_index()
                        results = pd.merge(results, results_pvalue, on=['query_id','is_bait','level'])

                        # Append aggregated bait and prey metrics per query
                        if level == "fraction_intensity":
                            results_aggregated = results.groupby(['condition_1','condition_2','query_id']).apply(collapse).reset_index(level=['condition_1','condition_2','query_id'])
                            results_aggregated['is_bait'] = True
                            result_aggregated_prey = results_aggregated.copy()
                            result_aggregated_prey['is_bait'] = False
                            results = pd.concat([results, results_aggregated, result_aggregated_prey], sort=False)

                        # Append meta information
                        results = pd.merge(results, dat[['query_id','bait_id','prey_id']].drop_duplicates(), on='query_id')

                        dfs.append(results[['condition_1','condition_2','level','bait_id','prey_id','is_bait','nes','anes','pvalue','cfactor']+[c for c in results.columns if c.startswith("viper_")]])

        return pd.concat(dfs, ignore_index=True, sort=True).sort_values(by='pvalue', ascending=True, na_position='last')

    def integrate(self):
        def collapse(x):
            if x.shape[0] > 1:
                result = pd.Series({'nes': np.mean(x['nes'].values), 'anes': np.mean(x['anes'].values), 'cfactor': np.mean(x['cfactor'].values), 'pvalue': EmpiricalBrownsMethod(x[[c for c in x.columns if c.startswith("viper_")]].values, x['pvalue'].values)})
            elif x.shape[0] == 1 and x['level'].values[0] in ['monomer_intensity','total_intensity']:
                result = pd.Series({'nes': x['nes'].values[0], 'anes': x['anes'].values[0], 'cfactor': x['cfactor'].values[0], 'pvalue': x['pvalue'].values[0]})
            else:
                result = pd.Series({'nes': x['nes'].values[0], 'anes': x['anes'].values[0], 'cfactor': x['cfactor'].values[0], 'pvalue': 1.0})
            return(result)

        def mtcorrect(x):
            x['pvalue_adjusted'] = multipletests(pvals=x['pvalue'].values, method="fdr_bh")[1]

            return(x)

        df_edge_level = self.tests[(self.tests['bait_id'] != self.tests['prey_id']) & self.tests['is_bait']]
        df_edge_level_rev = self.tests[(self.tests['bait_id'] != self.tests['prey_id']) & ~self.tests['is_bait']]
        df_edge_level_rev = df_edge_level_rev.rename(index=str, columns={"bait_id": "prey_id", "prey_id": "bait_id"})

        df_protein_level = self.tests[self.tests['bait_id'] == self.tests['prey_id']]
        df_edge_full = pd.concat([df_protein_level, df_edge_level, df_edge_level_rev], sort=False)
        df_node_level = df_edge_full.groupby(['condition_1', 'condition_2','level','bait_id']).apply(collapse).reset_index()

        # Multi-testing correction and pooling
        df_protein_level = df_protein_level.groupby(['condition_1', 'condition_2','level']).apply(mtcorrect).reset_index()
        df_edge_level = df_edge_level.groupby(['condition_1', 'condition_2','level']).apply(mtcorrect).reset_index()
        df_edge = df_edge_level.sort_values('pvalue').groupby(['condition_1','condition_2','bait_id','prey_id']).head(1).reset_index()
        df_node_level = df_node_level.groupby(['condition_1', 'condition_2','level']).apply(mtcorrect).reset_index()
        df_node = df_node_level.sort_values('pvalue_adjusted').groupby(['condition_1','condition_2','bait_id']).head(1).reset_index()

        click.echo("Info: Total dysregulated proteins detected:")
        click.echo("%s (at FDR < 0.01)" % (df_node[df_node['pvalue_adjusted'] < 0.01][['bait_id']].drop_duplicates().shape[0]))
        click.echo("%s (at FDR < 0.05)" % (df_node[df_node['pvalue_adjusted'] < 0.05][['bait_id']].drop_duplicates().shape[0]))
        click.echo("%s (at FDR < 0.1)" % (df_node[df_node['pvalue_adjusted'] < 0.1][['bait_id']].drop_duplicates().shape[0]))

        for level in df_node_level['level'].unique():
            click.echo("Info: Dysregulated (%s-mode) proteins detected:" % (level))
            click.echo("%s (at FDR < 0.01)" % (df_node_level[(df_node_level['level'] == level) & (df_node_level['pvalue_adjusted'] < 0.01)][['bait_id']].drop_duplicates().shape[0]))
            click.echo("%s (at FDR < 0.05)" % (df_node_level[(df_node_level['level'] == level) & (df_node_level['pvalue_adjusted'] < 0.05)][['bait_id']].drop_duplicates().shape[0]))
            click.echo("%s (at FDR < 0.1)" % (df_node_level[(df_node_level['level'] == level) & (df_node_level['pvalue_adjusted'] < 0.1)][['bait_id']].drop_duplicates().shape[0]))

        return df_edge_level[['condition_1','condition_2','level','bait_id','prey_id','nes','anes','cfactor','pvalue','pvalue_adjusted']+[c for c in df_edge_level.columns if c.startswith("viper_")]], df_edge[['condition_1','condition_2','level','bait_id','prey_id','nes','anes','cfactor','pvalue','pvalue_adjusted']], df_node_level[['condition_1','condition_2','level','bait_id','nes','anes','cfactor','pvalue','pvalue_adjusted']], df_node[['condition_1','condition_2','level','bait_id','nes','anes','cfactor','pvalue','pvalue_adjusted']], df_protein_level[['condition_1','condition_2','level','bait_id','nes','anes','cfactor','pvalue','pvalue_adjusted'] + [c for c in df_protein_level.columns if c.startswith("viper_")]]

