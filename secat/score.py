import pandas as pd
import numpy as np
import scipy as sp
import click
import sqlite3
import os
import sys

from joblib import Parallel, delayed

from scipy.signal import find_peaks, peak_widths
from minepy import cstats

np.seterr(divide='ignore', invalid='ignore')

def split(dfm, chunk_size):
    def index_marks(nrows, chunk_size):
        return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

def applyParallel(dfGrouped, func):
    # retLst = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=3)(delayed(func)(group) for name, group in dfGrouped)
    retLst = Parallel(n_jobs=8, verbose=3)(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

class monomer:
    def __init__(self, outfile, monomer_threshold_factor):
        self.outfile = outfile
        self.monomer_threshold_factor = monomer_threshold_factor
        self.df = self.protein_thresholds()

    def protein_thresholds(self):
        def get_sec_ids(protein_mw, sec_meta):
            def condition_replicate_sec_ids(protein_mw, run_sec_meta):
                return pd.Series({'sec_id': (run_sec_meta['sec_mw']-self.monomer_threshold_factor*protein_mw).abs().argsort()[:1].values[0]})

            return sec_meta.groupby(['condition_id','replicate_id']).apply(lambda x: condition_replicate_sec_ids(protein_mw, x[['sec_id','sec_mw']])).reset_index()

        con = sqlite3.connect(self.outfile)
        protein_mw = pd.read_sql('SELECT protein_id, protein_mw FROM PROTEIN;', con)
        sec_meta = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, sec_id, sec_mw FROM SEC;', con)
        con.close()

        # Compute expected SEC fraction
        protein_sec_thresholds = protein_mw.groupby(['protein_id']).apply(lambda x: get_sec_ids(x['protein_mw'].mean(), sec_meta=sec_meta)).reset_index(level=['protein_id'])

        return protein_sec_thresholds[['condition_id','replicate_id','protein_id','sec_id']]

def interaction(df):
    def longest_intersection(arr):
        # Compute longest continuous stretch
        n = len(arr)
        s = set()
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

    def normalized_xcorr(a, b):
        # Normalize matrices
        a = (a - np.mean(a, axis=1, keepdims=True)) / (np.std(a, axis=1, keepdims=True))
        b = (b - np.mean(b, axis=1, keepdims=True)) / (np.std(b, axis=1, keepdims=True))

        nxcorr = [] # normalized cross-correlation
        lxcorr = [] # cross-correlation lag

        if np.array_equal(a,b):
            # Compare all rows of a against other rows of a but not against themselves
            for i in range(0, len(a)):
                for j in range(i+1, len(a)):
                    nxcorr.append(np.correlate(a[i], a[j], 'valid')[0] / len(a[i])) # Normalize by length
                    lxcorr.append(np.argmax(np.correlate(a[i], a[j], 'same'))) # Peak
        else:
            # Compare all rows of a against all rows of b
            for i in range(0, len(a)):
                for j in range(0, len(b)):
                    nxcorr.append(np.correlate(a[i], b[j], 'valid')[0] / len(a[i])) # Normalize by length
                    lxcorr.append(np.argmax(np.correlate(a[i], b[j], 'same'))) # Peak

        return np.array(nxcorr), np.array(lxcorr)        

    def sec_xcorr(bm, pm):
        # Compute SEC xcorr scores
        bnx, blx = normalized_xcorr(bm, bm)
        pnx, plx = normalized_xcorr(pm, pm)
        bpnx, bplx = normalized_xcorr(bm, pm)

        xcorr_shape = np.mean(bpnx)
        xcorr_apex = np.mean(bplx)
        xcorr_shift = max([abs(xcorr_apex - np.mean(blx)), abs(xcorr_apex - np.mean(plx))])

        return xcorr_shape, xcorr_shift, xcorr_apex

    def mass_similarity(bm, pm):
        # Sum bait and prey peptides
        bpmass = np.sum(bm, axis=1, keepdims=True)
        ppmass = np.sum(pm, axis=1, keepdims=True)

        # Compute mass ratio of bait and prey protein
        mass_ratio = np.mean(bpmass) / np.mean(ppmass)
        if mass_ratio > 1:
            mass_ratio = 1 / mass_ratio

        return mass_ratio

    # Workaround for parallelization
    minimum_peptides = df['minimum_peptides'].min()
    sec_boundaries = pd.DataFrame({'sec_id': range(df['lower_sec_boundaries'].min(), df['upper_sec_boundaries'].max()+1)})

    # Require minimum overlap
    intersection = list(set(df[df['is_bait']]['sec_id'].unique()) & set(df[~df['is_bait']]['sec_id'].unique()))
    bait_overlap = len(df[df['is_bait']]['sec_id'].unique())
    prey_overlap = len(df[~df['is_bait']]['sec_id'].unique())
    delta_overlap = abs(bait_overlap-prey_overlap)
    longest_overlap = longest_intersection(intersection)

    # Requre minimum peptides
    num_bait_peptides = len(df[df['sec_id'].isin(intersection) & df['is_bait']]['peptide_id'].unique())
    num_prey_peptides = len(df[df['sec_id'].isin(intersection) & ~df['is_bait']]['peptide_id'].unique())

    if num_bait_peptides >= minimum_peptides and num_prey_peptides >= minimum_peptides:
        # Compute bait SEC boundaries
        bait_sec_boundaries = sec_boundaries.copy()
        bait_sec_boundaries['peptide_id'] = df[df['is_bait']]['peptide_id'].unique()[0]

        # Compute bait peptide matrix over intersection
        bpi = pd.merge(df[df['sec_id'].isin(intersection) & df['is_bait']], bait_sec_boundaries, on=['peptide_id','sec_id'], how='outer').sort_values(['sec_id']).pivot(index='peptide_id', columns='sec_id', values='peptide_intensity')
        bmi = np.nan_to_num(bpi.values) # Replace missing values with zeros

        # Compute prey SEC boundaries
        prey_sec_boundaries = sec_boundaries.copy()
        prey_sec_boundaries['peptide_id'] = df[~df['is_bait']]['peptide_id'].unique()[0]

        # Compute prey peptide matrix over intersection
        ppi = pd.merge(df[df['sec_id'].isin(intersection) & ~df['is_bait']], prey_sec_boundaries, on=['peptide_id','sec_id'], how='outer').sort_values(['sec_id']).pivot(index='peptide_id', columns='sec_id', values='peptide_intensity')
        pmi = np.nan_to_num(ppi.values) # Replace missing values with zeros

        # Cross-correlation scores
        xcorr_shape, xcorr_shift, xcorr_apex = sec_xcorr(bmi, pmi)

        # MIC/TIC scores
        mic_stat, tic_stat = cstats(bpi[intersection].values, ppi[intersection].values, est="mic_e")
        mic = mic_stat.mean(axis=0).mean() # Axis 0: summary for prey peptides / Axis 1: summary for bait peptides
        tic = tic_stat.mean(axis=0).mean() # Axis 0: summary for prey peptides / Axis 1: summary for bait peptides

        # Mass similarity score
        mass_ratio = mass_similarity(bmi, pmi)

        # Monomer delta score
        monomer_delta = np.min(np.array([df[df['is_bait']]['monomer_sec_id'].min() - xcorr_apex, df[~df['is_bait']]['monomer_sec_id'].min() - xcorr_apex]))

        res = df[['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin']].drop_duplicates()
        res['var_xcorr_shape'] = xcorr_shape
        res['var_xcorr_shift'] = xcorr_shift
        res['var_mic'] = mic
        res['var_tic'] = tic
        res['var_mass_ratio'] = mass_ratio
        res['var_monomer_delta'] = monomer_delta
        res['var_sec_apex'] = xcorr_apex
        res['var_sec_left'] = min(intersection)
        res['var_sec_right'] = max(intersection)
        res['var_intersection'] = longest_overlap
        res['var_delta_intersection'] = delta_overlap
        res['var_total_intersection'] = len(intersection)

    else:
        res = df[['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin']].drop_duplicates()
        res['var_xcorr_shape'] = np.nan
        res['var_xcorr_shift'] = np.nan
        res['var_mic'] = np.nan
        res['var_tic'] = np.nan
        res['var_mass_ratio'] = np.nan
        res['var_monomer_delta'] = np.nan
        res['var_sec_apex'] = np.nan
        res['var_sec_left'] = np.nan
        res['var_sec_right'] = np.nan
        res['var_intersection'] = np.nan
        res['var_delta_intersection'] = np.nan
        res['var_total_intersection'] = np.nan

    return(res)

class scoring:
    def __init__(self, outfile, chunck_size, minimum_peptides, maximum_peptides):
        self.outfile = outfile
        self.chunck_size = chunck_size
        self.minimum_peptides = minimum_peptides
        self.maximum_peptides = maximum_peptides

        self.sec_boundaries = self.read_sec_boundaries()

        click.echo("Info: Read peptide chromatograms.")
        chromatograms = self.read_chromatograms()
        click.echo("Info: Filter peptide chromatograms.")
        self.chromatograms = self.filter_peptides(chromatograms)
        self.store_filtered()
        click.echo("Info: Read queries and SEC boundaries.")
        self.queries = self.read_queries()

        click.echo("Info: Score PPI.")
        self.df = self.compare()

    def read_chromatograms(self):
        # Read data
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT SEC.condition_id, SEC.replicate_id, SEC.sec_id, QUANTIFICATION.protein_id, QUANTIFICATION.peptide_id, peptide_intensity, MONOMER.sec_id AS monomer_sec_id FROM QUANTIFICATION INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id INNER JOIN SEC ON QUANTIFICATION.RUN_ID = SEC.RUN_ID INNER JOIN MONOMER ON QUANTIFICATION.protein_id = MONOMER.protein_id and SEC.condition_id = MONOMER.condition_id AND SEC.replicate_id = MONOMER.replicate_id WHERE peptide_count >= %s AND peptide_rank <= %s;' % (self.minimum_peptides, self.maximum_peptides), con)

        con.close()

        return df

    def filter_peptides(self, df):
        def peptide_detrend(x):
            peptide_mean = np.mean(np.append(x['peptide_intensity'], np.zeros(len(self.sec_boundaries['sec_id'].unique())-x.shape[0])))
            return x[x['peptide_intensity'] > peptide_mean][['sec_id','peptide_intensity','monomer_sec_id']]

        # Report statistics before filtering
        click.echo("Info: %s unique peptides before filtering." % len(df['peptide_id'].unique()))
        click.echo("Info: %s peptide chromatograms before filtering." % df[['condition_id','replicate_id','protein_id','peptide_id']].drop_duplicates().shape[0])
        click.echo("Info: %s data points before filtering." % df.shape[0])

        # Remove constant trends from peptides
        # df = df.groupby(['condition_id','replicate_id','protein_id','peptide_id']).apply(peptide_detrend).reset_index(level=['condition_id','replicate_id','protein_id','peptide_id'])

        # Filter monomers for detrending
        df = df[df['sec_id'] <= df['monomer_sec_id']]

        # Report statistics after filtering
        click.echo("Info: %s unique peptides after filtering." % len(df['peptide_id'].unique()))
        click.echo("Info: %s peptide chromatograms after filtering." % df[['condition_id','replicate_id','protein_id','peptide_id']].drop_duplicates().shape[0])
        click.echo("Info: %s data points after filtering." % df.shape[0])

        return df

    def store_filtered(self):
        con = sqlite3.connect(self.outfile)
        self.chromatograms[['condition_id','replicate_id','protein_id','sec_id']].drop_duplicates().to_sql('PROTEIN_PEAKS', con, index=False, if_exists='replace')
        con.close()

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
        comparisons = pd.merge(pd.merge(self.queries, exp_design, left_on='bait_id', right_on='protein_id')[['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin']], exp_design, left_on=['condition_id','replicate_id','prey_id'], right_on=['condition_id','replicate_id','protein_id'])[['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin']]

        # Split data into chunks for parallel processing
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
            data_pd['minimum_peptides'] = self.minimum_peptides
            data_pd['lower_sec_boundaries'] = self.sec_boundaries['sec_id'].min()
            data_pd['upper_sec_boundaries'] = self.sec_boundaries['sec_id'].max()

            # Single threaded implementation
            # data = data_pd.groupby(['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin']).apply(interaction)
            # Multi threaded implementation
            data = applyParallel(data_pd.groupby(['condition_id','replicate_id','bait_id','prey_id','decoy','confidence_bin']), interaction)

            # Require passing of mass ratio and SEC lag thresholds
            data = data.dropna()
            chunck_data.append(data)

        return pd.concat(chunck_data)
