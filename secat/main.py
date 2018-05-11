import click
from tqdm import tqdm

from multiprocessing import Pool, freeze_support, RLock, cpu_count
import sqlite3
import os
from shutil import copyfile

import pandas as pd
import numpy as np

from preprocess import uniprot, net, sec, quantification, meta, query
from detect import prepare, process
from score import prepare_filter, filter_mw, filter_training, pyprophet, infer

from pyprophet.data_handling import transform_pi0_lambda

@click.group(chain=True)
@click.version_option()
def cli():
    """
    SECAT: Size-Exclusion Chromatography Algorithmic Toolkit.

    Visit https://github.com/grosenberger/secat for usage instructions and help.
    """

# SECAT import data
@cli.command()
@click.argument('infiles', nargs=-1, type=click.Path(exists=True))
@click.option('--out', 'outfile', required=True, type=click.Path(exists=False), help='Output SECAT file.')
# Reference files
@click.option('--sec', 'secfile', required=True, type=click.Path(exists=True), help='The input SEC calibration file.')
@click.option('--net', 'netfile', required=True, type=click.Path(exists=True), help='Reference binary protein-protein interaction file in STRING-DB or HUPO-PSI MITAB (2.5-2.7) format.')
@click.option('--uniprot', 'uniprotfile', required=True, type=click.Path(exists=True), help='Reference molecular weights file in UniProt XML format.')
@click.option('--columns', default=["run_id","sec_id","sec_mw","condition_id","replicate_id","protein_id","peptide_id","peptide_intensity"], show_default=True, type=(str,str,str,str,str,str,str,str), help='Column names for SEC & peptide quantification files')
# Parameters for decoys
@click.option('--decoy_intensity_bins', 'decoy_intensity_bins', default=1, show_default=True, type=int, help='Number of decoy bins for intensity.')
@click.option('--decoy_left_sec_bins', 'decoy_left_sec_bins', default=1, show_default=True, type=int, help='Number of decoy bins for left SEC fraction.')
@click.option('--decoy_right_sec_bins', 'decoy_right_sec_bins', default=1, show_default=True, type=int, help='Number of decoy bins for right SEC fraction.')
def preprocess(infiles, outfile, secfile, netfile, uniprotfile, columns, decoy_intensity_bins, decoy_left_sec_bins, decoy_right_sec_bins):
    """
    Import and preprocess SEC data.
    """

    # Prepare output file
    try:
        os.remove(outfile)
    except OSError:
        pass

    con = sqlite3.connect(outfile)

    # Generate UniProt table
    click.echo("Info: Parsing UniProt XML file %s." % uniprotfile)
    uniprot_data = uniprot(uniprotfile)
    uniprot_data.to_df().to_sql('PROTEIN', con, index=False)

    # Generate Network table
    click.echo("Info: Parsing network file %s." % netfile)
    net_data = net(netfile, uniprot_data)
    net_data.to_df().to_sql('NETWORK', con, index=False)

    # Generate SEC definition table
    click.echo("Info: Parsing SEC definition file %s." % secfile)
    sec_data = sec(secfile, columns)
    sec_data.to_df().to_sql('SEC', con, index=False)

    # Generate Peptide quantification table
    run_ids = sec_data.to_df()['run_id'].unique() # Extract valid run_ids from SEC definition table

    for infile in infiles:
        click.echo("Info: Parsing peptide quantification file %s." % infile)
        quantification_data = quantification(infile, columns, run_ids)
        quantification_data.to_df().to_sql('QUANTIFICATION' ,con, index=False, if_exists='append')

    # Generate peptide and protein meta data over all conditions and replicates
    click.echo("Info: Generating peptide and protein meta data.")
    meta_data = meta(quantification_data, sec_data, decoy_intensity_bins, decoy_left_sec_bins, decoy_right_sec_bins)
    meta_data.peptide_meta.to_sql('PEPTIDE_META', con, index=False)
    meta_data.protein_meta.to_sql('PROTEIN_META', con, index=False)

    # Generate interaction query data
    click.echo("Info: Generating interaction query data.")
    query_data = query(net_data, meta_data.protein_meta)
    query_data.to_df().to_sql('QUERY', con, index=False)

    # Remove any entries that are not necessary (proteins not covered by LC-MS/MS data)
    con.execute('DELETE FROM PROTEIN WHERE protein_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION);')
    con.execute('DELETE FROM NETWORK WHERE bait_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION) OR prey_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION);')
    con.execute('DELETE FROM SEC WHERE run_id NOT IN (SELECT DISTINCT(run_id) as run_id FROM QUANTIFICATION);')
    con.execute('DELETE FROM QUERY WHERE bait_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION) OR prey_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION);')
    con.execute('VACUUM;')

    # Close connection to file
    con.close()

    click.echo("Info: Data successfully preprocessed and stored in %s." % outfile)

# SECAT detect features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--out', 'outfile', required=False, type=click.Path(exists=False), help='Output SECAT file.')
# Parameters for peptides
@click.option('--min_peptides', 'min_peptides', default=3, show_default=True, type=int, help='Minimum number of required peptides per protein.')
@click.option('--max_peptides', 'max_peptides', default=6, show_default=True, type=int, help='Maximum number of (most intense) peptides per protein.')
@click.option('--det_peptides', 'det_peptides', default=6, show_default=True, type=int, help='Number of (most intense) peptides per query for detection.')
# Parameters for peak picking
@click.option('--peak_method', 'peak_method', default='gauss', show_default=True, type=click.Choice(['gauss', 'sgolay']), help='Use Gaussian or Savitzky-Golay smoothing.')
@click.option('--peak_width', 'peak_width', default=2, show_default=True, type=int, help='Force a certain minimal peak width (sec units; -1 to disable) on the data (e.g. extend the peak at least by this amount on both sides).')
@click.option('--signal_to_noise', 'signal_to_noise', default=0.75, show_default=True, type=int, help='Signal-to-noise threshold at which a peak will not be extended any more. Note that setting this too high (e.g. 1.0) can lead to peaks whose flanks are not fully captured.')
@click.option('--gauss_width', 'gauss_width', default=6, show_default=True, type=int, help='Specify expected gaussian width in SEC units at FWHM.')
@click.option('--sgolay_frame_length', 'sgolay_frame_length', default=15, show_default=True, type=int, help='Specify Savitzky-Golay frame length.')
@click.option('--sgolay_polynomial_order', 'sgolay_polynomial_order', default=3, show_default=True, type=int, help='Specify Savitzky-Golay polynomial order.')
@click.option('--sn_win_len', 'sn_win_len', default=30, show_default=True, type=int, help='Signal to noise window length.')
@click.option('--sn_bin_count', 'sn_bin_count', default=15, show_default=True, type=int, help='Signal to noise bin count.')
@click.option('--max_xcorr_coelution', 'max_xcorr_coelution', default=2.0, show_default=True, type=float, help='Do not consider features with an SEC coelution above the threshold.')
@click.option('--threads', 'threads', default=1, show_default=True, type=int, help='Number of threads used for parallel processing of SEC runs. -1 means all available CPUs.')
def detect(infile, outfile, min_peptides, max_peptides, det_peptides, peak_method, peak_width, signal_to_noise, gauss_width, sgolay_frame_length, sgolay_polynomial_order, sn_win_len, sn_bin_count, max_xcorr_coelution, threads):
    """
    Detect protein and interaction features in SEC data.
    """

    click.echo("Info: The signal processing module will display warning messages if your data is sparse. In most scenarios, these warnings can be ignored.")

    # Define outfile
    if outfile is None:
        outfile = infile
    else:
        copyfile(infile, outfile)
        outfile = outfile

    con = sqlite3.connect(outfile)
    con.execute('DROP TABLE IF EXISTS FEATURE;')
    con.close()

    # Prepare SEC experiments, e.g. individual conditions + replicates
    exps = prepare(outfile, min_peptides, max_peptides, det_peptides, peak_method, peak_width, signal_to_noise, gauss_width, sgolay_frame_length, sgolay_polynomial_order, sn_win_len, sn_bin_count, max_xcorr_coelution)

    # Execute workflow in parallel
    if threads == -1:
        n_cpus = cpu_count()
    else:
        n_cpus = threads

    freeze_support()
    p = Pool(processes=n_cpus, initializer=tqdm.set_lock, initargs=(RLock(),))
    p.map(process, exps)

    # Move temporary files to final table
    con = sqlite3.connect(outfile)
    for exp in exps:
        con.execute('ATTACH DATABASE "%s" AS tmp;' % exp['tmpoutfile'])
        con.execute('CREATE TABLE IF NOT EXISTS FEATURE AS SELECT * FROM tmp.FEATURE WHERE 0;')
        con.execute('INSERT INTO FEATURE SELECT * FROM tmp.FEATURE;')
        con.execute('DETACH DATABASE "tmp";')
        os.remove(exp['tmpoutfile'])
    con.close()
  
# SECAT score features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--out', 'outfile', required=False, type=click.Path(exists=False), help='Output SECAT file.')
@click.option('--complex_threshold_factor', 'complex_threshold_factor', default=2.0, show_default=True, type=float, help='Factor threshold to consider a feature a complex rather than a monomer.')
# Semi-supervised learning
@click.option('--xeval_fraction', default=0.5, show_default=True, type=float, help='Data fraction used for cross-validation of semi-supervised learning step.')
@click.option('--xeval_num_iter', default=10, show_default=True, type=int, help='Number of iterations for cross-validation of semi-supervised learning step.')
@click.option('--ss_initial_fdr', default=0.15, show_default=True, type=float, help='Initial FDR cutoff for best scoring targets.')
@click.option('--ss_iteration_fdr', default=0.05, show_default=True, type=float, help='Iteration FDR cutoff for best scoring targets.')
@click.option('--ss_num_iter', default=10, show_default=True, type=int, help='Number of iterations for semi-supervised learning step.')
# Statistics
@click.option('--parametric/--no-parametric', default=False, show_default=True, help='Do parametric estimation of p-values.')
@click.option('--pfdr/--no-pfdr', default=False, show_default=True, help='Compute positive false discovery rate (pFDR) instead of FDR.')
@click.option('--pi0_lambda', default=[0.1,0.5,0.05], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
@click.option('--pi0_method', default='bootstrap', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
@click.option('--pi0_smooth_df', default=3, show_default=True, type=int, help='Number of degrees-of-freedom to use when estimating pi_0 with a smoother.')
@click.option('--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0', default=False, show_default=True, help='If True and pi0_method = "smoother", pi0 will be estimated by applying a smoother to a scatterplot of log(pi0) estimates against the tuning parameter lambda.')
@click.option('--lfdr_truncate/--no-lfdr_truncate', show_default=True, default=True, help='If True, local FDR values >1 are set to 1.')
@click.option('--lfdr_monotone/--no-lfdr_monotone', show_default=True, default=True, help='If True, local FDR values are non-decreasing with increasing p-values.')
@click.option('--lfdr_transformation', default='probit', show_default=True, type=click.Choice(['probit', 'logit']), help='Either a "probit" or "logit" transformation is applied to the p-values so that a local FDR estimate can be formed that does not involve edge effects of the [0,1] interval in which the p-values lie.')
@click.option('--lfdr_adj', default=1.5, show_default=True, type=float, help='Numeric value that is applied as a multiple of the smoothing bandwidth used in the density estimation.')
@click.option('--lfdr_eps', default=np.power(10.0,-8), show_default=True, type=float, help='Numeric value that is threshold for the tails of the empirical p-value distribution.')
@click.option('--threads', 'threads', default=1, show_default=True, type=int, help='Number of threads used for parallel processing of SEC runs. -1 means all available CPUs.')
@click.option('--test/--no-test', default=False, show_default=True, help='Run in test mode with fixed seed.')
def score(infile, outfile, complex_threshold_factor, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test):
    """
    Score protein and interaction features in SEC data.
    """

    # Execute workflow in parallel
    if threads == -1:
        n_cpus = cpu_count()
    else:
        n_cpus = threads

    # p = Pool(processes=n_cpus)

    # Define outfile
    if outfile is None:
        outfile = infile
    else:
        copyfile(infile, outfile)
        outfile = outfile

    # Prepare SEC experiments, e.g. individual conditions + replicates
    exps = prepare_filter(outfile, complex_threshold_factor)

    # # Assess molecular weight
    # click.echo("Info: Filtering based on molecular weight.")
    # mw_data = p.map(filter_mw, exps)

    # con = sqlite3.connect(outfile)
    # pd.concat(mw_data).to_sql('FEATURE_MW', con, index=False, if_exists='replace')
    # con.close()

    # # Filter training data
    # click.echo("Info: Filtering based on elution profile scores.")
    # training_data = p.map(filter_training, exps)

    # con = sqlite3.connect(outfile)
    # pd.concat(training_data).to_sql('FEATURE_TRAINING', con, index=False, if_exists='replace')
    # con.close()

    # # Close parallel execution
    # p.close()

    # Run PyProphet training
    click.echo("Info: Running PyProphet.")
    scored_data = pyprophet(outfile, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, n_cpus, test)

    con = sqlite3.connect(outfile)
    scored_data.df.to_sql('FEATURE_SCORED', con, index=False, if_exists='replace')
    con.close()

    # Infer proteins
    click.echo("Info: Infering complexes and monomers.")
    infer_data = infer(outfile, 'FEATURE_SCORED')

    con = sqlite3.connect(outfile)
    infer_data.interactions.to_sql('COMPLEX', con, index=False, if_exists='replace')
    infer_data.proteins.to_sql('MONOMER', con, index=False, if_exists='replace')
    con.close()
