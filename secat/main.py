import click

import sqlite3
import os
from shutil import copyfile

import pandas as pd
import numpy as np

from preprocess import uniprot, net, sec, quantification, meta, query
from score import monomer, scoring
from learn import pyprophet
from quantify import quantitative_matrix, quantitative_test
from plot import plot_features

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
@click.option('--negnet', 'negnetfile', required=False, type=click.Path(exists=True), help='Reference binary negative protein-protein interaction file in STRING-DB or HUPO-PSI MITAB (2.5-2.7) format.')
@click.option('--uniprot', 'uniprotfile', required=True, type=click.Path(exists=True), help='Reference molecular weights file in UniProt XML format.')
@click.option('--columns', default=["run_id","sec_id","sec_mw","condition_id","replicate_id","run_id","protein_id","peptide_id","peptide_intensity"], show_default=True, type=(str,str,str,str,str,str,str,str,str), help='Column names for SEC & peptide quantification files')
# Parameters for decoys
@click.option('--decoy_intensity_bins', 'decoy_intensity_bins', default=1, show_default=True, type=int, help='Number of decoy bins for intensity.')
@click.option('--decoy_left_sec_bins', 'decoy_left_sec_bins', default=1, show_default=True, type=int, help='Number of decoy bins for left SEC fraction.')
@click.option('--decoy_right_sec_bins', 'decoy_right_sec_bins', default=1, show_default=True, type=int, help='Number of decoy bins for right SEC fraction.')
@click.option('--min_interaction_confidence', 'min_interaction_confidence', default=0.0, show_default=True, type=float, help='Minimum interaction confidence for prior information from network.')
def preprocess(infiles, outfile, secfile, netfile, negnetfile, uniprotfile, columns, decoy_intensity_bins, decoy_left_sec_bins, decoy_right_sec_bins, min_interaction_confidence):
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

    # Generate Negative Network table
    if negnetfile != None:
        click.echo("Info: Parsing negative network file %s." % negnetfile)
        negnet_data = net(negnetfile, uniprot_data)
        negnet_data.to_df().to_sql('NEGATIVE_NETWORK', con, index=False)
    else:
        negnet_data = None

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
    query_data = query(net_data, negnet_data, meta_data.protein_meta, min_interaction_confidence)
    query_data.to_df().to_sql('QUERY', con, index=False)

    # Remove any entries that are not necessary (proteins not covered by LC-MS/MS data)
    con.execute('DELETE FROM PROTEIN WHERE protein_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION);')
    con.execute('DELETE FROM NETWORK WHERE bait_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION) OR prey_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION);')
    con.execute('DELETE FROM SEC WHERE run_id NOT IN (SELECT DISTINCT(run_id) as run_id FROM QUANTIFICATION);')
    con.execute('DELETE FROM QUERY WHERE bait_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION) OR prey_id NOT IN (SELECT DISTINCT(protein_id) as protein_id FROM QUANTIFICATION);')

    # Add indices
    con.execute('CREATE INDEX IF NOT EXISTS idx_protein_protein_id ON PROTEIN (protein_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_network_bait_id ON NETWORK (bait_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_network_prey_id ON NETWORK (prey_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_network_bait_id_prey_id ON NETWORK (bait_id, prey_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_quantification_run_id ON QUANTIFICATION (run_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_quantification_protein_id ON QUANTIFICATION (protein_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_quantification_peptide_id ON QUANTIFICATION (peptide_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_peptide_meta_peptide_id ON PEPTIDE_META (peptide_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_protein_meta_protein_id ON PROTEIN_META (protein_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_query_bait_id ON QUERY (bait_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_query_prey_id ON QUERY (prey_id);')
    con.execute('CREATE INDEX IF NOT EXISTS idx_query_bait_id_prey_id ON QUERY (bait_id, prey_id);')
    con.execute('VACUUM;')

    # Close connection to file
    con.close()

    click.echo("Info: Data successfully preprocessed and stored in %s." % outfile)

# SECAT score features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--out', 'outfile', required=False, type=click.Path(exists=False), help='Output SECAT file.')
@click.option('--monomer_threshold_factor', 'monomer_threshold_factor', default=4.0, show_default=True, type=float, help='Factor threshold to consider a feature a complex rather than a monomer.')
@click.option('--monomer_elution_width', 'monomer_elution_width', default=10.0, show_default=True, type=float, help='Expected monomer peak width in SEC units. Used to adjust left monomer threshold padding.')
@click.option('--minimum_peptides', 'minimum_peptides', default=3, show_default=True, type=int, help='Minimum number of peptides required to score an interaction.')
@click.option('--maximum_peptides', 'maximum_peptides', default=20, show_default=True, type=int, help='Maximum number of peptides used to score an interaction.')
@click.option('--minimum_overlap', 'minimum_overlap', default=5, show_default=True, type=int, help='Minimum number of fractions required to score an interaction.')
@click.option('--minimum_peptide_snr', 'minimum_peptide_snr', default=1.0, show_default=True, type=float, help='Minimum signal-to-noise ratio per peptide.')
@click.option('--chunck_size', 'chunck_size', default=50000, show_default=True, type=int, help='Chunck size for processing.')
def score(infile, outfile, monomer_threshold_factor, monomer_elution_width, minimum_peptides, maximum_peptides, minimum_overlap, minimum_peptide_snr, chunck_size):
    """
    Score interaction features in SEC data.
    """

    # Define outfile
    if outfile is None:
        outfile = infile
    else:
        copyfile(infile, outfile)
        outfile = outfile

    # Find monomer thresholds
    click.echo("Info: Detect monomers.")
    monomer_data = monomer(outfile, monomer_threshold_factor, monomer_elution_width)

    con = sqlite3.connect(outfile)
    monomer_data.df.to_sql('MONOMER', con, index=False, if_exists='replace')
    con.close()

    # Signal processing
    click.echo("Info: Signal processing.")
    feature_data = scoring(outfile, chunck_size, minimum_peptides, maximum_peptides, minimum_overlap, minimum_peptide_snr)

    con = sqlite3.connect(outfile)
    feature_data.df.to_sql('FEATURE', con, index=False, if_exists='replace')
    con.close()

# SECAT learn features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--out', 'outfile', required=False, type=click.Path(exists=False), help='Output SECAT file.')
# Prefiltering
@click.option('--minimum_snr', 'minimum_snr', default=1.0, show_default=True, type=int, help='Minimum signal-to-noise ratio per peptide.')
@click.option('--minimum_mass_ratio', 'minimum_mass_ratio', default=0.2, show_default=True, type=float, help='Minimum number of fractions required to score an interaction.')
@click.option('--maximum_sec_shift', 'maximum_sec_shift', default=3.0, show_default=True, type=float, help='Maximum lag in SEC units between interactions and subunits.')
@click.option('--minimum_learning_confidence', 'minimum_learning_confidence', default=0.9, show_default=True, type=float, help='Minimum prior interaction confidence for semi-supervised learning.')
# Semi-supervised learning
@click.option('--xeval_fraction', default=0.5, show_default=True, type=float, help='Data fraction used for cross-validation of semi-supervised learning step.')
@click.option('--xeval_num_iter', default=10, show_default=True, type=int, help='Number of iterations for cross-validation of semi-supervised learning step.')
@click.option('--ss_initial_fdr', default=0.15, show_default=True, type=float, help='Initial FDR cutoff for best scoring targets.')
@click.option('--ss_iteration_fdr', default=0.15, show_default=True, type=float, help='Iteration FDR cutoff for best scoring targets.')
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
def learn(infile, outfile, minimum_snr, minimum_mass_ratio, maximum_sec_shift, minimum_learning_confidence, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test):
    """
    Learn true/false interaction features in SEC data.
    """

    # Define outfile
    if outfile is None:
        outfile = infile
    else:
        copyfile(infile, outfile)
        outfile = outfile

    # Run PyProphet training
    click.echo("Info: Running PyProphet.")

    scored_data = pyprophet(outfile, minimum_snr, minimum_mass_ratio, maximum_sec_shift, minimum_learning_confidence, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test)

    con = sqlite3.connect(outfile)
    scored_data.df.to_sql('FEATURE_SCORED', con, index=False, if_exists='replace')
    con.close()


# SECAT quantify features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--out', 'outfile', required=False, type=click.Path(exists=False), help='Output SECAT file.')
def quantify(infile, outfile):
    """
    Quantify protein and interaction features in SEC data.
    """

    # Define outfile
    if outfile is None:
        outfile = infile
    else:
        copyfile(infile, outfile)
        outfile = outfile


    click.echo("Info: Prepare quantitative matrix")
    qm = quantitative_matrix(outfile)

    con = sqlite3.connect(outfile)
    qm.complex.to_sql('COMPLEX_QM', con, index=False, if_exists='replace')
    con.close()

    click.echo("Info: Assess differential features")
    qt = quantitative_test(outfile)

    con = sqlite3.connect(outfile)
    qt.edge_directional.to_sql('EDGE_DIRECTIONAL', con, index=False, if_exists='replace')
    qt.edge.to_sql('EDGE', con, index=False, if_exists='replace')
    qt.edge_level.to_sql('EDGE_LEVEL', con, index=False, if_exists='replace')
    qt.node.to_sql('NODE', con, index=False, if_exists='replace')
    qt.node_level.to_sql('NODE_LEVEL', con, index=False, if_exists='replace')
    con.close()

# SECAT export features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
def export(infile):
    """
    Export SECAT results.
    """

    outfile_nodes = infile.split(".secat")[0] + "_nodes.csv"
    outfile_nodes_level = infile.split(".secat")[0] + "_nodes_level.csv"
    outfile_edges_directional = infile.split(".secat")[0] + "_edges_directional.csv"
    outfile_edges = infile.split(".secat")[0] + "_edges.csv"
    outfile_edges_level = infile.split(".secat")[0] + "_edges_level.csv"

    con = sqlite3.connect(infile)
    node_data = pd.read_sql('SELECT * FROM node;' , con)
    node_level_data = pd.read_sql('SELECT * FROM node_level;' , con)
    edge_directional_data = pd.read_sql('SELECT * FROM edge_directional;' , con)
    edge_data = pd.read_sql('SELECT * FROM edge;' , con)
    edge_level_data = pd.read_sql('SELECT * FROM edge_level;' , con)
    con.close()

    node_data.to_csv(outfile_nodes, index=False)
    node_level_data.to_csv(outfile_nodes_level, index=False)
    edge_directional_data.to_csv(outfile_edges_directional, index=False)
    edge_data.to_csv(outfile_edges, index=False)
    edge_level_data.to_csv(outfile_edges_level, index=False)

# SECAT plot chromatograms
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--interaction_id', required=False, type=str, help='Plot features for specified interaction_id.')
@click.option('--interaction_qvalue', default=None, show_default=True, type=float, help='Maximum q-value to plot interactions.')
@click.option('--bait_id', required=False, type=str, help='Plot features for specified bait_id.')
@click.option('--bait_qvalue', default=None, show_default=True, type=float, help='Maximum q-value to plot baits.')
@click.option('--peptide_rank', default=6, show_default=True, type=int, help='Number of most intense peptides to plot.')
def plot(infile, interaction_id, interaction_qvalue, bait_id, bait_qvalue, peptide_rank):
    """
    Plot SECAT results
    """

    pf = plot_features(infile, interaction_id, interaction_qvalue, bait_id, bait_qvalue, peptide_rank)
