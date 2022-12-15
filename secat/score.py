import pandas as pd
import numpy as np
import scipy as sp
import click
import sqlite3
import os
import sys

import multiprocessing
from functools import partial
from tqdm import tqdm

from scipy.signal import find_peaks, peak_widths
from minepy import cstats

# np.seterr(divide='ignore', invalid='ignore')
# np.seterr(all='raise')

# Find Monomer Threshold
class monomer:
    def __init__(self, outfile, monomer_threshold_factor):
        self.outfile = outfile
        self.monomer_threshold_factor = monomer_threshold_factor
        self.df = self.protein_thresholds()

    def protein_thresholds(self):
        def get_sec_ids(protein_mw, sec_meta):
            def condition_replicate_sec_ids(protein_mw, run_sec_meta):
                return pd.Series({'sec_id': (run_sec_meta['sec_mw']-self.monomer_threshold_factor*protein_mw).abs().argsort().iloc[:1].values[0]})

            return sec_meta.groupby(['condition_id','replicate_id']).apply(lambda x: condition_replicate_sec_ids(protein_mw, x[['sec_id','sec_mw']])).reset_index()

        con = sqlite3.connect(self.outfile)
        protein_mw = pd.read_sql('SELECT protein_id, protein_mw FROM PROTEIN;', con)
        sec_meta = pd.read_sql('SELECT DISTINCT condition_id, replicate_id, sec_id, sec_mw FROM SEC;', con)
        con.close()

        # Compute expected SEC fraction
        protein_sec_thresholds = protein_mw.groupby(['protein_id']).apply(lambda x: get_sec_ids(x['protein_mw'].mean(), sec_meta=sec_meta)).reset_index(level=['protein_id'])

        return protein_sec_thresholds[['condition_id','replicate_id','protein_id','sec_id']]

def score_chunk(queries, qm, run):
    scores = []
    for query_ix, query in queries.iterrows():
        bait = qm.xs(query['bait_id'], level='protein_id')
        bait_monomer_sec_id = bait.iloc[0].name[1]

        prey = qm.xs(query['prey_id'], level='protein_id')
        prey_monomer_sec_id = prey.iloc[0].name[1]

        score = score_interaction(bait.values.copy(), prey.values.copy(), bait_monomer_sec_id, prey_monomer_sec_id)
        if score is not None:
            score['condition_id'] = run['condition_id']
            score['replicate_id'] = run['replicate_id']
            score = {**score, **query}
            scores.append(score)
    return(scores)

def score_interaction(bait, prey, bait_monomer_sec_id, prey_monomer_sec_id):
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
            # Compare all rows of a against all rows of a, including itself (auto-correlation)
            for i in range(0, len(a)):
                for j in range(i, len(a)):
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
        bpabundance = np.sum(bm, axis=1, keepdims=True).mean()
        ppabundance = np.sum(pm, axis=1, keepdims=True).mean()

        # Compute abundance ratio of bait and prey protein
        abundance_ratio = bpabundance / ppabundance
        if abundance_ratio > 1:
            abundance_ratio = 1 / abundance_ratio

        return abundance_ratio


   # Compute bait and prey overlap
    overlap = (np.nansum(bait, axis=0) > 0) | (np.nansum(prey, axis=0) > 0)
    total_overlap = np.count_nonzero(overlap)

    # Compute bait and prey intersection
    intersection = (np.nansum(bait, axis=0) > 0) & (np.nansum(prey, axis=0) > 0)
    total_intersection = np.count_nonzero(intersection)
    if total_intersection > 0:
        longest_intersection = longest_intersection(intersection.nonzero()[0])

        # Require at least three consecutive overlapping data points
        if longest_intersection > 2:
            # Prepare total bait and prey profiles & Replace nan with 0
            total_bait = np.nan_to_num(bait)
            total_prey = np.nan_to_num(prey)

            # Remove non-overlapping segments
            bait[:,~intersection] = np.nan
            prey[:,~intersection] = np.nan

            # Remove completely empty peptides
            bait = bait[(np.nansum(bait,axis=1) > 0),:]
            prey = prey[(np.nansum(prey,axis=1) > 0),:]

            # Replace nan with 0
            bait = np.nan_to_num(bait)
            prey = np.nan_to_num(prey)

            # Require at least one remaining peptide for bait and prey
            if (bait.shape[0] > 0) and (prey.shape[0] > 0):
                # Compute cross-correlation scores
                xcorr_shape, xcorr_shift, xcorr_apex = sec_xcorr(bait, prey)

                # Compute MIC/TIC scores
                mic_stat, tic_stat = cstats(bait[:,intersection], prey[:,intersection], est="mic_e")
                mic = mic_stat.mean(axis=0).mean() # Axis 0: summary for prey peptides / Axis 1: summary for bait peptides
                tic = tic_stat.mean(axis=0).mean() # Axis 0: summary for prey peptides / Axis 1: summary for bait peptides

                # Compute mass similarity score
                abundance_ratio = mass_similarity(bait, prey)

                # Compute total mass similarity score
                total_abundance_ratio = mass_similarity(total_bait, total_prey)

                # Compute relative intersection score
                relative_overlap = total_intersection / total_overlap

                # Compute delta monomer score
                delta_monomer = np.abs(bait_monomer_sec_id - prey_monomer_sec_id)

                # Compute apex monomer score
                apex_monomer = np.min(np.array(bait_monomer_sec_id - xcorr_apex, prey_monomer_sec_id - xcorr_apex))

                return({'var_xcorr_shape': xcorr_shape, 'var_xcorr_shift': xcorr_shift, 'var_abundance_ratio': abundance_ratio, 'var_total_abundance_ratio': total_abundance_ratio, 'var_mic': mic, 'var_tic': tic, 'var_sec_overlap': relative_overlap, 'var_sec_intersection': longest_intersection, 'var_delta_monomer': delta_monomer, 'var_apex_monomer': apex_monomer})

# Scoring
class scoring:
    def __init__(self, outfile, chunck_size, threads, minimum_peptides, maximum_peptides, peakpicking):
        self.outfile = outfile
        self.chunck_size = chunck_size
        self.threads = threads
        self.minimum_peptides = minimum_peptides
        self.maximum_peptides = maximum_peptides
        self.peakpicking = peakpicking

        self.sec_boundaries = self.read_sec_boundaries()

        click.echo("Info: Read peptide chromatograms.")
        chromatograms = self.read_chromatograms()
        click.echo("Info: Filter peptide chromatograms.")
        self.chromatograms = self.filter_peptides(chromatograms)
        self.store_filtered()
        click.echo("Info: Read queries and SEC boundaries.")
        self.queries = self.read_queries()

        click.echo("Info: Score PPI.")
        self.compare()

    def read_chromatograms(self):
        # Read data
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT SEC.condition_id, SEC.replicate_id, SEC.sec_id, QUANTIFICATION.protein_id, QUANTIFICATION.peptide_id, peptide_intensity, MONOMER.sec_id AS monomer_sec_id FROM QUANTIFICATION INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id INNER JOIN SEC ON QUANTIFICATION.RUN_ID = SEC.RUN_ID INNER JOIN MONOMER ON QUANTIFICATION.protein_id = MONOMER.protein_id and SEC.condition_id = MONOMER.condition_id AND SEC.replicate_id = MONOMER.replicate_id WHERE peptide_count >= %s AND peptide_rank <= %s;' % (self.minimum_peptides, self.maximum_peptides), con)

        con.close()

        return df

    def filter_peptides(self, df):
        def peptide_detrend_zero(x):
            peptide_mean = np.mean(np.append(x['peptide_intensity'], np.zeros(len(self.sec_boundaries['sec_id'].unique())-x.shape[0])))
            return x[x['peptide_intensity'] > peptide_mean][['sec_id','peptide_intensity','monomer_sec_id']]

        def peptide_detrend_drop(x):
            peptide_mean = np.mean(x['peptide_intensity'])
            return x[x['peptide_intensity'] > peptide_mean][['sec_id','peptide_intensity','monomer_sec_id']]

        def protein_pick(x):
            xpep = x.groupby(['peptide_id','sec_id'])['peptide_intensity'].mean().reset_index()
            xprot = xpep.groupby(['sec_id'])['peptide_intensity'].mean().reset_index()

            xall = pd.merge(self.sec_boundaries, xprot[['sec_id','peptide_intensity']], on='sec_id', how='left').sort_values(['sec_id'])
            xall['peptide_intensity'] = np.nan_to_num(xall['peptide_intensity'].values) # Replace missing values with zeros

            peaks, _ = find_peaks(xall['peptide_intensity'], width=[3,])
            boundaries = peak_widths(xall['peptide_intensity'], peaks, rel_height=0.9)

            left_boundaries = np.floor(boundaries[2])
            right_boundaries = np.ceil(boundaries[3])

            sec_list = None
            for peak in list(zip(left_boundaries, right_boundaries)):
                if sec_list is None:
                    sec_list = np.arange(peak[0],peak[1]+1)
                else:
                    sec_list = np.append(sec_list, np.arange(peak[0],peak[1]+1))

            if len(x['replicate_id'].unique()) == 1:
                return x[x['sec_id'].isin(np.unique(sec_list))][['peptide_id','sec_id','peptide_intensity','monomer_sec_id']]
            else:
                return x[x['sec_id'].isin(np.unique(sec_list))][['replicate_id','peptide_id','sec_id','peptide_intensity','monomer_sec_id']]

        # Report statistics before filtering
        click.echo("Info: %s unique peptides before filtering." % len(df['peptide_id'].unique()))
        click.echo("Info: %s peptide chromatograms before filtering." % df[['condition_id','replicate_id','protein_id','peptide_id']].drop_duplicates().shape[0])
        click.echo("Info: %s data points before filtering." % df.shape[0])

        # Filter monomers
        df = df[df['sec_id'] <= df['monomer_sec_id']]

        if self.peakpicking == "detrend_zero":
            # Remove constant trends from peptides, average over all fractions
            df = df.groupby(['condition_id','replicate_id','protein_id','peptide_id']).apply(peptide_detrend_zero).reset_index(level=['condition_id','replicate_id','protein_id','peptide_id'])
        if self.peakpicking == "detrend_drop":
            # Remove constant trends from peptides, average over fractions with detections
            df = df.groupby(['condition_id','replicate_id','protein_id','peptide_id']).apply(peptide_detrend_drop).reset_index(level=['condition_id','replicate_id','protein_id','peptide_id'])
        elif self.peakpicking == "localmax_conditions":
            # Protein-level peakpicking
            df = df.groupby(['condition_id','protein_id']).apply(protein_pick).reset_index(level=['condition_id','protein_id'])
        elif self.peakpicking == "localmax_replicates":
            # Protein-level peakpicking
            df = df.groupby(['condition_id','replicate_id','protein_id']).apply(protein_pick).reset_index(level=['condition_id','replicate_id','protein_id'])

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

    def split_chunks(self, dfm):
        def index_marks(nrows, chunck_size):
            return range(1 * chunck_size, (nrows // chunck_size + 1) * chunck_size, chunck_size)

        indices = index_marks(dfm.shape[0], self.chunck_size)
        return np.split(dfm, indices)

    def compare(self):
        # Obtain experimental design
        exp_design = self.chromatograms[['condition_id','replicate_id']].drop_duplicates()

        # Iterate over experimental design
        for exp_ix, run in exp_design.iterrows():
            chromatograms = self.chromatograms[(self.chromatograms['condition_id']==run['condition_id']) & (self.chromatograms['replicate_id']==run['replicate_id'])]
            qm = chromatograms.pivot_table(index=['protein_id','peptide_id','monomer_sec_id'], columns='sec_id', values='peptide_intensity')

            # Ensure that all queries are covered by chromatograms
            proteins = chromatograms['protein_id'].unique()
            queries = self.queries[self.queries['bait_id'].isin(proteins) & self.queries['prey_id'].isin(proteins)]

            # Split data into chunks for parallel processing
            queries_chunks = self.split_chunks(queries)
            click.echo("Info: Total number of queries for condition %s and replicate %s: %s. Split into %s chuncks." % (run['condition_id'], run['replicate_id'], queries.shape[0], len(queries_chunks)))
            
            # Initialize multiprocessing
            pool = multiprocessing.Pool(processes=self.threads)

            with tqdm(total=len(queries_chunks)) as pbar:
                for i, result in tqdm(enumerate(pool.imap_unordered(partial(score_chunk, qm=qm, run=run), queries_chunks))):

                    con = sqlite3.connect(self.outfile)
                    pd.DataFrame(result).to_sql('FEATURE', con, index=False, if_exists='append')
                    con.close()

                    pbar.update()
