import pandas as pd
import numpy as np
import scipy as sp
import click
import sqlite3
import os
import sys

from joblib import Parallel, delayed
import multiprocessing

from minepy import cstats
from pyitlib import discrete_random_variable as drv
from sets import Set
from pyprophet.stats import pemp, pi0est, qvalue

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from scipy.stats import gaussian_kde
from scipy.signal import correlate
from numpy import linspace, concatenate, around

def split(dfm, chunk_size):
    def index_marks(nrows, chunk_size):
        return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=3)(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

class monomer:
    def __init__(self, outfile, complex_threshold_factor):
        self.outfile = outfile
        self.complex_threshold_factor = complex_threshold_factor
        self.df = self.protein_thresholds()

    def protein_thresholds(self):
        def get_sec_ids(protein_meta, sec_meta):
            return sec_meta.groupby(['condition_id','replicate_id']).apply(lambda x: x.ix[(x['sec_mw']-self.complex_threshold_factor*protein_meta['protein_mw'].values).abs().argsort()[:1]][['sec_id']]).reset_index(level=['condition_id','replicate_id'])

        con = sqlite3.connect(self.outfile)
        protein_mw = pd.read_sql('SELECT protein_id, protein_mw FROM PROTEIN;', con)
        sec_meta = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, sec_id, sec_mw FROM SEC;', con)
        con.close()

        protein_sec_thresholds = protein_mw.groupby(['protein_id','protein_mw']).apply(lambda x: get_sec_ids(x, sec_meta=sec_meta)).reset_index(level=['protein_id','protein_mw'])

        return protein_sec_thresholds[['condition_id','replicate_id','protein_id','sec_id']]

def information(df):
    def longest_stretch(arr):
        n = len(arr)
        s = Set()
        ans=0
        for ele in arr:
            s.add(ele)
        for i in range(n):
            if (arr[i]-1) not in s:
                j=arr[i]
                while(j in s):
                    j+=1
                ans=max(ans, j-arr[i])
        return ans

    def xcorr_lag(bait_peptides, prey_peptides, intersection):
        bait_peptides = bait_peptides[intersection]
        prey_peptides = prey_peptides[intersection]

        bait_xcorr = correlate(np.repeat(np.nan_to_num(bait_peptides.values), bait_peptides.shape[0], axis=0), np.repeat(np.nan_to_num(bait_peptides.values), bait_peptides.shape[0], axis=0), mode='same')
        prey_xcorr = correlate(np.repeat(np.nan_to_num(prey_peptides.values), prey_peptides.shape[0], axis=0), np.repeat(np.nan_to_num(prey_peptides.values), prey_peptides.shape[0], axis=0), mode='same')
        bait_prey_xcorr = correlate(np.repeat(np.nan_to_num(bait_peptides.values), prey_peptides.shape[0], axis=0), np.repeat(np.nan_to_num(prey_peptides.values), bait_peptides.shape[0], axis=0), mode='same')

        bait_peak = np.median(np.array(intersection)[np.apply_along_axis(np.argmax, 1, bait_xcorr)])
        prey_peak = np.median(np.array(intersection)[np.apply_along_axis(np.argmax, 1, prey_xcorr)])
        bait_prey_peak = np.median(np.array(intersection)[np.apply_along_axis(np.argmax, 1, bait_prey_xcorr)])

        return max([abs(bait_prey_peak-bait_peak), abs(bait_prey_peak-prey_peak)])

    # Workaround for parallelization
    minimum_overlap = df['minimum_overlap'].min()
    sec_boundaries = pd.DataFrame({'sec_id': range(df['lower_sec_boundaries'].min(), df['upper_sec_boundaries'].max()+1)})

    # Require minimum overlap
    intersection = list(set(df[df['is_bait']]['sec_id'].unique()) & set(df[~df['is_bait']]['sec_id'].unique()))
    sec_overlap = longest_stretch(intersection)

    # Require minimum mass similarity
    df_bait_mass_peptide = df[df['sec_id'].isin(intersection) & df['is_bait']].groupby('peptide_id')['peptide_intensity'].sum().reset_index(level='peptide_id')
    df_prey_mass_peptide = df[df['sec_id'].isin(intersection) & ~df['is_bait']].groupby('peptide_id')['peptide_intensity'].sum().reset_index(level='peptide_id')

    mass_ratio = df_bait_mass_peptide['peptide_intensity'].median() / df_prey_mass_peptide['peptide_intensity'].median()

    if mass_ratio > 1:
        mass_ratio = 1 / mass_ratio

    if sec_overlap >= minimum_overlap:
        bait_sec_boundaries = sec_boundaries
        bait_sec_boundaries['peptide_id'] = df[df['is_bait']]['peptide_id'].unique()[0]
        bait_peptides = pd.merge(df[df['is_bait']], bait_sec_boundaries, on=['peptide_id','sec_id'], how='outer').sort_values(['sec_id']).pivot(index='peptide_id', columns='sec_id', values='peptide_intensity')

        prey_sec_boundaries = sec_boundaries
        prey_sec_boundaries['peptide_id'] = df[~df['is_bait']]['peptide_id'].unique()[0]
        prey_peptides = pd.merge(df[~df['is_bait']], prey_sec_boundaries, on=['peptide_id','sec_id'], how='outer').sort_values(['sec_id']).pivot(index='peptide_id', columns='sec_id', values='peptide_intensity')

        # MIC
        stat = cstats(bait_peptides[intersection].values, prey_peptides[intersection].values, est="mic_e")

        mic = stat[0].mean(axis=0).mean() # Axis 0: summary for prey peptides / Axis 1: summary for bait peptides
        tic = stat[1].mean(axis=0).mean() # Axis 0: summary for prey peptides / Axis 1: summary for bait peptides

        # MI
        mi = np.nanmean(drv.information_mutual(bait_peptides[intersection], prey_peptides[intersection], cartesian_product=True))

        res = df[['condition_id','replicate_id','bait_id','prey_id','decoy']].drop_duplicates()
        res['mic'] = mic
        res['tic'] = tic
        res['mi'] = mi
        res['longest_stretch'] = longest_stretch(intersection)
        res['sec_lag'] = xcorr_lag(bait_peptides, prey_peptides, intersection)
        res['mass_ratio'] = mass_ratio
    else:
        res = df[['condition_id','replicate_id','bait_id','prey_id','decoy']].drop_duplicates()
        res['mic'] = np.nan
        res['tic'] = np.nan
        res['mi'] = np.nan
        res['longest_stretch'] = np.nan
        res['sec_lag'] = np.nan
        res['mass_ratio'] = np.nan

    return(res)

class scoring:
    def __init__(self, outfile, chunck_size, minimum_peptides, maximum_peptides, minimum_overlap):
        self.outfile = outfile
        self.chunck_size = chunck_size
        self.minimum_peptides = minimum_peptides
        self.maximum_peptides = maximum_peptides
        self.minimum_overlap = minimum_overlap

        self.chromatograms = self.read_chromatograms()
        self.queries = self.read_queries()
        self.sec_boundaries = self.read_sec_boundaries()

        self.df = self.compare()

    def read_chromatograms(self):
        # Read data
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT SEC.condition_id, SEC.replicate_id, SEC.sec_id, QUANTIFICATION.protein_id, QUANTIFICATION.peptide_id, peptide_intensity FROM QUANTIFICATION INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id INNER JOIN SEC ON QUANTIFICATION.RUN_ID = SEC.RUN_ID INNER JOIN MONOMER ON QUANTIFICATION.protein_id = MONOMER.protein_id and SEC.condition_id = MONOMER.condition_id AND SEC.replicate_id = MONOMER.replicate_id WHERE SEC.sec_id < MONOMER.sec_id AND peptide_count >= %s AND peptide_rank <= %s;' % (self.minimum_peptides, self.maximum_peptides), con)
        # df = pd.read_sql('SELECT SEC.condition_id, SEC.replicate_id, SEC.sec_id, QUANTIFICATION.protein_id, QUANTIFICATION.peptide_id, peptide_intensity FROM QUANTIFICATION INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id INNER JOIN SEC ON QUANTIFICATION.RUN_ID = SEC.RUN_ID WHERE peptide_count >= %s AND peptide_rank <= %s;' % (self.minimum_peptides, self.maximum_peptides), con)

        con.close()

        return df

    def read_queries(self):
        # Read data
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT * FROM QUERY;', con)
        con.close()

        return df

    def read_sec_boundaries(self):
        # Read data
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT min(sec_id) AS min_sec_id, max(sec_id) AS max_sec_id FROM SEC;', con)
        con.close()

        return pd.DataFrame({'sec_id': range(df['min_sec_id'].values[0], df['max_sec_id'].values[0]+1)})

    def compare(self):
        # Obtain experimental design
        exp_design = self.chromatograms[['condition_id','replicate_id','protein_id']].drop_duplicates().sort_values(['condition_id','replicate_id','protein_id'])
        comparisons = pd.merge(pd.merge(self.queries, exp_design, left_on='bait_id', right_on='protein_id')[['bait_id','prey_id','decoy']], exp_design, left_on='prey_id', right_on='protein_id')[['condition_id','replicate_id','bait_id','prey_id','decoy']]


        comparisons_chunks = split(comparisons, self.chunck_size)
        click.echo("Info: Total number of queries: %s. Split into %s chuncks." % (comparisons.shape[0], len(comparisons_chunks)))

        chunck_data = []
        for chunck_it, comparison_chunk in enumerate(comparisons_chunks):
            click.echo("Info: Processing chunck %s out of %s chuncks." % (chunck_it+1, len(comparisons_chunks)))
            # Generate long list
            baits = pd.merge(comparison_chunk, self.chromatograms, left_on=['condition_id','replicate_id','bait_id'], right_on=['condition_id','replicate_id','protein_id']).drop(columns=['protein_id'])
            baits['is_bait'] = True

            preys = pd.merge(comparison_chunk, self.chromatograms, left_on=['condition_id','replicate_id','prey_id'], right_on=['condition_id','replicate_id','protein_id']).drop(columns=['protein_id'])
            preys['is_bait'] = False

            data_pd = pd.concat([baits, preys]).reset_index()

            # Workaround for parallelization
            data_pd['minimum_overlap'] = self.minimum_overlap
            data_pd['lower_sec_boundaries'] = self.sec_boundaries['sec_id'].min()
            data_pd['upper_sec_boundaries'] = self.sec_boundaries['sec_id'].max()

            # Single threaded implementation
            # data = data_pd.groupby(['condition_id','replicate_id','bait_id','prey_id','decoy']).apply(information)
            # Multi threaded implementation
            data = applyParallel(data_pd.groupby(['condition_id','replicate_id','bait_id','prey_id','decoy']), information)

            # Require passing of mass ratio and SEC lag thresholds
            data = data.dropna()
            chunck_data.append(data)

        return pd.concat(chunck_data)

class significance:
    def __init__(self, outfile, minimum_mass_ratio, maximum_sec_lag):
        self.outfile = outfile
        self.minimum_mass_ratio = minimum_mass_ratio
        self.maximum_sec_lag = maximum_sec_lag

        self.features = self.read_features()
        self.plot_distributions()
        self.df = self.test_pvalue()

    def read_features(self):
        # Read data
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT * FROM FEATURE WHERE mass_ratio >= %s AND sec_lag <= %s;' % (self.minimum_mass_ratio, self.maximum_sec_lag), con)
        con.close()

        return df

    def test_pvalue(self):
        def find_closest(x, df):
            return df.iloc[(df['mi']-x).abs().argsort()[:1]][['pvalue','qvalue']]

        target_features = self.features[self.features['decoy'] == False]
        decoy_features = self.features[self.features['decoy'] == True]
        self.features['pvalue'] = pemp(self.features['mi'].values, decoy_features['mi'].values)
        self.features['qvalue'] = qvalue(self.features['pvalue'].values, pi0est(self.features['pvalue'].values)['pi0'])

        return self.features

    def plot_distributions(self):
        if plt is None:
            raise ImportError("Error: The matplotlib package is required to create a report.")

        out = os.path.splitext(os.path.basename(self.outfile))[0]+"_statistics.pdf"

        df = self.features
        score_columns = ["mic","tic","mi","sec_lag","mass_ratio"]

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
                    plt.ylabel("# of interactions")
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
