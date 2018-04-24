import click
from tqdm import tqdm

from multiprocessing import Pool, freeze_support, RLock, cpu_count
import sqlite3
import os
from shutil import copyfile

import pandas as pd

from preprocess import uniprot, net, sec, quantification, meta, queries
from detect import prepare, process
from score import mw, filter_training

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
    queries_data = queries(net_data, meta_data.protein_meta)
    queries_data.to_df().to_sql('QUERIES', con, index=False)

    # Remove any entries that are not necessary (proteins not covered by LC-MS/MS data)
    con.execute('DELETE FROM PROTEIN WHERE protein_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION);')
    con.execute('DELETE FROM NETWORK WHERE bait_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION) OR prey_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION);')
    con.execute('DELETE FROM QUERIES WHERE bait_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION) OR prey_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION);')
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
@click.option('--max_peptides', 'max_peptides', default=3, show_default=True, type=int, help='Maximum number of (most intense) peptides per protein.')
@click.option('--det_peptides', 'det_peptides', default=3, show_default=True, type=int, help='Number of (most intense) peptides per assay for detection.')
# Parameters for peak picking
@click.option('--peak_method', 'peak_method', default='gauss', show_default=True, type=click.Choice(['gauss', 'sgolay']), help='Use Gaussian or Savitzky-Golay smoothing.')
@click.option('--peak_width', 'peak_width', default=2, show_default=True, type=int, help='Force a certain minimal peak width (sec units; -1 to disable) on the data (e.g. extend the peak at least by this amount on both sides).')
@click.option('--signal_to_noise', 'signal_to_noise', default=0.75, show_default=True, type=int, help='Signal-to-noise threshold at which a peak will not be extended any more. Note that setting this too high (e.g. 1.0) can lead to peaks whose flanks are not fully captured.')
@click.option('--gauss_width', 'gauss_width', default=6, show_default=True, type=int, help='Specify expected gaussian width in SEC units at FWHM.')
@click.option('--sgolay_frame_length', 'sgolay_frame_length', default=15, show_default=True, type=int, help='Specify Savitzky-Golay frame length.')
@click.option('--sgolay_polynomial_order', 'sgolay_polynomial_order', default=3, show_default=True, type=int, help='Specify Savitzky-Golay polynomial order.')
@click.option('--sn_win_len', 'sn_win_len', default=30, show_default=True, type=int, help='Signal to noise window length.')
@click.option('--sn_bin_count', 'sn_bin_count', default=15, show_default=True, type=int, help='Signal to noise bin count.')
@click.option('--threads', 'threads', default=1, show_default=True, type=int, help='Number of threads used for parallel processing of SEC runs. -1 means all available CPUs.')
def detect(infile, outfile, min_peptides, max_peptides, det_peptides, peak_method, peak_width, signal_to_noise, gauss_width, sgolay_frame_length, sgolay_polynomial_order, sn_win_len, sn_bin_count, threads):
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

    # Prepare SEC experiments, e.g. individual conditions + replicates
    exps = prepare(outfile, min_peptides, max_peptides, det_peptides, peak_method, peak_width, signal_to_noise, gauss_width, sgolay_frame_length, sgolay_polynomial_order, sn_win_len, sn_bin_count)

    # Execute workflow in parallel
    if threads == -1:
        n_cpus = cpu_count()
    else:
        n_cpus = threads

    freeze_support()
    p = Pool(processes=n_cpus, initializer=tqdm.set_lock, initargs=(RLock(),))
    dfs = p.map(process, exps)

    df = pd.concat(dfs)

    con = sqlite3.connect(outfile)
    df.to_sql('FEATURES', con, index=False, if_exists='replace')
    con.close()

# SECAT score features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--out', 'outfile', required=False, type=click.Path(exists=False), help='Output SECAT file.')
@click.option('--complex_threshold_factor', 'complex_threshold_factor', default=2.0, show_default=True, type=float, help='Factor threshold to consider a feature a complex rather than a monomer.')
@click.option('--threads', 'threads', default=1, show_default=True, type=int, help='Number of threads used for parallel processing of SEC runs. -1 means all available CPUs.')
def score(infile, outfile, complex_threshold_factor, threads):
    """
    Score protein and interaction features in SEC data.
    """

    # Define outfile
    if outfile is None:
        outfile = infile
    else:
        copyfile(infile, outfile)
        outfile = outfile

    # mw_data = mw(outfile, complex_threshold_factor)

    # con = sqlite3.connect(outfile)
    # mw_data.to_sql('FEATURES_MW', con, index=False, if_exists='replace')
    # con.close()

    filter_training(outfile)

