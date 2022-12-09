import click

from sqlite3 import connect
from os import remove, path
from sys import exit
from shutil import copyfile

from pandas import concat, read_sql
from numpy import power

from .preprocess import uniprot, net, sec, quantification, normalization, meta, query
from .score import monomer, scoring
from .learn import pyprophet, combine
from .quantify import quantitative_matrix, enrichment_test
from .plot import plot_features, check_sqlite_table

from pyprophet.data_handling import transform_threads, transform_pi0_lambda

# import cProfile

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
@click.option('--net', 'netfile', required=False, type=click.Path(exists=True), help='Reference binary protein-protein interaction file in STRING-DB or HUPO-PSI MITAB (2.5-2.7) format.')
@click.option('--posnet', 'posnetfile', required=False, type=click.Path(exists=True), help='Reference binary positive protein-protein interaction file in STRING-DB or HUPO-PSI MITAB (2.5-2.7) format.')
@click.option('--negnet', 'negnetfile', required=False, type=click.Path(exists=True), help='Reference binary negative protein-protein interaction file in STRING-DB or HUPO-PSI MITAB (2.5-2.7) format.')
@click.option('--uniprot', 'uniprotfile', required=True, type=click.Path(exists=True), help='Reference molecular weights file in UniProt XML format.')
@click.option('--columns', default=["run_id","sec_id","sec_mw","condition_id","replicate_id","run_id","protein_id","peptide_id","peptide_intensity"], show_default=True, type=(str,str,str,str,str,str,str,str,str), help='Column names for SEC & peptide quantification files')


# Parameters for normalization
@click.option('--normalize/--no-normalize', default=True, show_default=True, help='Normalize quantification data by sliding window cycling LOWESS normaklization.')
@click.option('--normalize_window','normalize_window', default=5, show_default=True, type=int, help='Number of SEC fractions per sliding window.')
@click.option('--normalize_padded/--no-normalize_padded', default=True, show_default=True, help='Use padding for first and last SEC fractions.')
# Parameters for decoys
@click.option('--decoy_intensity_bins', 'decoy_intensity_bins', default=1, show_default=True, type=int, help='Number of decoy bins for intensity.')
@click.option('--decoy_left_sec_bins', 'decoy_left_sec_bins', default=1, show_default=True, type=int, help='Number of decoy bins for left SEC fraction.')
@click.option('--decoy_right_sec_bins', 'decoy_right_sec_bins', default=1, show_default=True, type=int, help='Number of decoy bins for right SEC fraction.')
@click.option('--decoy_oversample','decoy_oversample', default=2, show_default=True, type=int, help='Number of iterations to sample decoys.')
@click.option('--decoy_subsample/--no-decoy_subsample', default=False, show_default=True, help='Whether decoys should be subsampled to be approximately of similar number as targets.')
@click.option('--decoy_exclude/--no-decoy_exclude', default=True, show_default=True, help='Whether decoy interactions also covered by targets should be excluded.')
@click.option('--min_interaction_confidence', 'min_interaction_confidence', default=0.0, show_default=True, type=float, help='Minimum interaction confidence for prior information from network.')
@click.option('--interaction_confidence_bins', 'interaction_confidence_bins', default=100, show_default=True, type=int, help='Number of interaction confidence bins for grouped error rate estimation.')
@click.option('--interaction_confidence_quantile/--no-interaction_confidence_quantile', default=True, show_default=True, help='Whether interaction confidence bins should be grouped by quantiles.')
@click.option('--use_cached_uniprot', 'cache', default=True, required=False, show_default=True, type=bool, help='Whether to use the Uniprot table parsed from a previous run')
def preprocess(infiles, outfile, secfile, netfile, posnetfile, negnetfile, uniprotfile, columns, normalize, normalize_window, normalize_padded, decoy_intensity_bins, decoy_left_sec_bins, decoy_right_sec_bins, decoy_oversample, decoy_subsample, decoy_exclude, min_interaction_confidence, interaction_confidence_bins, interaction_confidence_quantile, cache):
    """
    Import and preprocess SEC data.
    """

    # Prepare output file
    try:
        remove(outfile)
    except OSError:
        pass

    con = connect(outfile)

    # Generate SEC definition table
    click.echo("Info: Parsing SEC definition file %s." % secfile)
    sec_data = sec(secfile, columns)
    sec_data.to_df().to_sql('SEC', con, index=False)

    # Generate Peptide quantification table
    run_ids = sec_data.to_df()['run_id'].unique() # Extract valid run_ids from SEC definition table

    quantification_list = []
    for infile in infiles:
        click.echo("Info: Parsing peptide quantification file %s." % infile)
        quantification_list.append(quantification(infile, columns, run_ids).to_df())
    quantification_data = concat(quantification_list)

    # Normalize quantitative data
    if normalize:
        click.echo("Info: Normalizing quantitative data.")
        quantification_data = normalization(quantification_data, sec_data.to_df(), normalize_window, normalize_padded, outfile).to_df()

    # Store quantification data
    quantification_data.to_sql('QUANTIFICATION' ,con, index=False, if_exists='append')

    # Generate peptide and protein meta data over all conditions and replicates
    click.echo("Info: Generating peptide and protein meta data.")
    meta_data = meta(quantification_data, sec_data.to_df(), decoy_intensity_bins, decoy_left_sec_bins, decoy_right_sec_bins)
    meta_data.peptide_meta.to_sql('PEPTIDE_META', con, index=False)
    meta_data.protein_meta.to_sql('PROTEIN_META', con, index=False)

    # Generate UniProt table
    click.echo("Info: Parsing UniProt XML file %s." % uniprotfile)
    uniprot_data = uniprot(uniprotfile, cache)
    uniprot_data.to_df().to_sql('PROTEIN', con, index=False)

    # Generate Network table
    if netfile != None:
        click.echo("Info: Parsing network file %s." % netfile)
    else:
        click.echo("Info: No reference network file was provided.")
        decoy_exclude = False
    net_data = net(netfile, uniprot_data, meta_data)
    net_data.to_df().to_sql('NETWORK', con, index=False)

    # Generate Positive Network table
    if posnetfile != None:
        click.echo("Info: Parsing positive network file %s." % posnetfile)
        posnet_data = net(posnetfile, uniprot_data, meta_data)
        posnet_data.to_df().to_sql('POSITIVE_NETWORK', con, index=False)
    else:
        posnet_data = None

    # Generate Negative Network table
    if negnetfile != None:
        click.echo("Info: Parsing negative network file %s." % negnetfile)
        negnet_data = net(negnetfile, uniprot_data, meta_data)
        negnet_data.to_df().to_sql('NEGATIVE_NETWORK', con, index=False)
    else:
        negnet_data = None

    # Generate interaction query data
    click.echo("Info: Generating interaction query data.")
    query_data = query(net_data, posnet_data, negnet_data, meta_data.protein_meta, min_interaction_confidence, interaction_confidence_bins, interaction_confidence_quantile, decoy_oversample, decoy_subsample, decoy_exclude)
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
    # con.execute('VACUUM;')

    con.commit()

    # Close connection to file
    con.close()

    click.echo("Info: Data successfully preprocessed and stored in %s." % outfile)

# SECAT score features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--out', 'outfile', required=False, type=click.Path(exists=False), help='Output SECAT file.')
@click.option('--monomer_threshold_factor', 'monomer_threshold_factor', default=2.0, show_default=True, type=float, help='Factor threshold to consider a feature a complex rather than a monomer.')
@click.option('--minimum_peptides', 'minimum_peptides', default=1, show_default=True, type=int, help='Minimum number of peptides required to score an interaction.')
@click.option('--maximum_peptides', 'maximum_peptides', default=3, show_default=True, type=int, help='Maximum number of peptides used to score an interaction.')
@click.option('--peakpicking', default='none', show_default=True, type=click.Choice(['none', 'detrend_zero', 'detrend_drop', 'localmax_conditions', 'localmax_replicates']), help='Either "none", "detrend_zero", "detrend_drop", "localmax_conditions" or "localmax_replicates"; the method for peakpicking of the peptide chromatograms. detrend_drop averages over all fractions with peptides; detrend_zero averages over all fractions (less agressive). localmax_conditions averages peak-picking over replicates of the same conditions; localmax_replicates conducts peak-picking for all samples separately.')
@click.option('--chunck_size', 'chunck_size', default=50000, show_default=True, type=int, help='Chunck size for processing.')
@click.option('--threads', default=1, show_default=True, type=int, help='Number of threads used for parallel processing. -1 means all available CPUs.', callback=transform_threads)
def score(infile, outfile, monomer_threshold_factor, minimum_peptides, maximum_peptides, peakpicking, chunck_size, threads):
    """
    Score interaction features in SEC data.
    """

    # Define outfile
    if outfile is None:
        outfile = infile
    else:
        # TODO: Consider replacing this with subprocess.call(["cp", "infile", "outfile"]) for speed improvement
        copyfile(infile, outfile)
        outfile = outfile

    # Find monomer thresholds
    click.echo("Info: Detect monomers.")
    monomer_data = monomer(outfile, monomer_threshold_factor)

    con = connect(outfile)
    monomer_data.df.to_sql('MONOMER', con, index=False, if_exists='replace')
    con.close()

    # Signal processing
    click.echo("Info: Signal processing.")

    # Drop features if they already exist
    con = connect(outfile)
    c = con.cursor()
    c.execute('DROP TABLE IF EXISTS FEATURE;')
    con.close()

    scoring(outfile, chunck_size, threads, minimum_peptides, maximum_peptides, peakpicking)
    # cProfile.runctx('scoring(outfile, chunck_size, minimum_peptides, maximum_peptides, peakpicking)', globals(), locals = {'outfile': outfile, 'chunck_size': chunck_size, 'minimum_peptides': minimum_peptides, 'maximum_peptides': maximum_peptides, 'peakpicking': peakpicking}, filename="score_performance.cprof")

# SECAT learn features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--out', 'outfile', required=False, type=click.Path(exists=False), help='Output SECAT file.')
@click.option('--apply_model', 'apply_model', required=False, type=click.Path(exists=False), help='Apply pretrained SECAT model')
# Prefiltering
@click.option('--minimum_abundance_ratio', 'minimum_abundance_ratio', default=0.1, show_default=True, type=float, help='Minimum abundance ratio required to score an interaction.')
@click.option('--maximum_sec_shift', 'maximum_sec_shift', default=10, show_default=True, type=float, help='Maximum lag in SEC units between interactions and subunits.')
@click.option('--cb_decoys/--no-cb_decoys', default=False, show_default=True, help='Use only decoys from same confidence bin instead of full set for learning.')
# Semi-supervised learning
@click.option('--xeval_fraction', default=0.8, show_default=True, type=float, help='Data fraction used for cross-validation of semi-supervised learning step.')
@click.option('--xeval_num_iter', default=3, show_default=True, type=int, help='Number of iterations for cross-validation of semi-supervised learning step.')
@click.option('--ss_initial_fdr', default=0.1, show_default=True, type=float, help='Initial FDR cutoff for best scoring targets.')
@click.option('--ss_iteration_fdr', default=0.05, show_default=True, type=float, help='Iteration FDR cutoff for best scoring targets.')
@click.option('--ss_num_iter', default=10, show_default=True, type=int, help='Number of iterations for semi-supervised learning step.')
@click.option('--xgb_autotune/--no-xgb_autotune', default=False, show_default=True, help='Autotune hyperparameters after semi-supervised learning.')
# Statistics
@click.option('--parametric/--no-parametric', default=False, show_default=True, help='Do parametric estimation of p-values.')
@click.option('--pfdr/--no-pfdr', default=False, show_default=True, help='Compute positive false discovery rate (pFDR) instead of FDR.')
@click.option('--pi0_lambda', default=[0.01,0.5,0.01], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
@click.option('--pi0_method', default='bootstrap', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
@click.option('--pi0_smooth_df', default=3, show_default=True, type=int, help='Number of degrees-of-freedom to use when estimating pi_0 with a smoother.')
@click.option('--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0', default=False, show_default=True, help='If True and pi0_method = "smoother", pi0 will be estimated by applying a smoother to a scatterplot of log(pi0) estimates against the tuning parameter lambda.')
@click.option('--lfdr_truncate/--no-lfdr_truncate', show_default=True, default=True, help='If True, local FDR values >1 are set to 1.')
@click.option('--lfdr_monotone/--no-lfdr_monotone', show_default=True, default=True, help='If True, local FDR values are non-decreasing with increasing p-values.')
@click.option('--lfdr_transformation', default='probit', show_default=True, type=click.Choice(['probit', 'logit']), help='Either a "probit" or "logit" transformation is applied to the p-values so that a local FDR estimate can be formed that does not involve edge effects of the [0,1] interval in which the p-values lie.')
@click.option('--lfdr_adj', default=1.5, show_default=True, type=float, help='Numeric value that is applied as a multiple of the smoothing bandwidth used in the density estimation.')
@click.option('--lfdr_eps', default=power(10.0,-8), show_default=True, type=float, help='Numeric value that is threshold for the tails of the empirical p-value distribution.')
#other
@click.option('--plot_reports/--no-plot_reports', default=False, show_default=True, help='Plot reports for all confidence bins.')
@click.option('--threads', default=1, show_default=True, type=int, help='Number of threads used for parallel processing. -1 means all available CPUs.', callback=transform_threads)
@click.option('--test/--no-test', default=False, show_default=True, help='Run in test mode with fixed seed to ensure reproducibility.')
@click.option('--export_tables/--no-export_tables', default=False, show_default=True, help='Saves two csv tables. One for the interations in used for modeling interactions, and another of the target interactions. Including all scores.')
def learn(infile, outfile, apply_model, minimum_abundance_ratio, maximum_sec_shift, cb_decoys, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, xgb_autotune, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, plot_reports, threads, test, export_tables):
    """
    Learn true/false interaction features in SEC data.
    """

    # Define outfile
    if outfile is None:
        outfile = infile
    else:
        # TODO: Consider replacing this with subprocess.call(["cp", "infile", "outfile"]) for speed improvement
        copyfile(infile, outfile)
        outfile = outfile

    # Run PyProphet training
    click.echo("Info: Running PyProphet.")

    # Drop feature scores if they already exist
    con = connect(outfile)
    c = con.cursor()
    c.execute('DROP TABLE IF EXISTS FEATURE_SCORED;')
    con.close()

    pyprophet(outfile, apply_model, minimum_abundance_ratio, maximum_sec_shift, cb_decoys, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, xgb_autotune, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, plot_reports, threads, test, export_tables)

    # Combine all replicates
    click.echo("Info: Combine evidence across replicate runs.")

    combined_data = combine(outfile, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, pfdr)

    con = connect(outfile)
    if export_tables != False:
        network_interaction_name = path.splitext(infile)[0] + "_net_int_scored.csv"
        combined_data.df.to_csv(network_interaction_name, index=False)
    combined_data.df.to_sql('FEATURE_SCORED_COMBINED', con, index=False, if_exists='replace')
    con.close()

# SECAT quantify features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--out', 'outfile', required=False, type=click.Path(exists=False), help='Output SECAT file.')
@click.option('--control_condition', default="center", type=str, help='Specify control condition identifier. Setting this parameter to "center" will compare all conditions against all and use the mean as reference for quantification.')
@click.option('--paired/--no-paired', default=False, show_default=True, help='Whether replicates should be paired, e.g. replicates 1 of conditions A & B were measured with heavy and light SILAC labels as part of the same runs.')
@click.option('--maximum_interaction_qvalue', default=0.05, show_default=True, type=float, help='Maximum q-value to consider interactions for quantification.')
@click.option('--min_abs_log2fx', default=1.0, show_default=True, type=float, help='Minimum absolute log2 fold-change for integrated nodes.')
@click.option('--minimum_peptides', 'minimum_peptides', default=1, show_default=True, type=int, help='Minimum number of peptides required to quantify an interaction.')
@click.option('--maximum_peptides', 'maximum_peptides', default=3, show_default=True, type=int, help='Maximum number of peptides used to quantify an interaction.')
@click.option('--missing_peptides', 'missing_peptides', default="zero", type=str, help='Whether missing peptide abundances should be set to 0 ("zero") or dropped ("drop") for fold change computation.')
@click.option('--peptide_log2fx/--no-peptide_log2fx', default=True, show_default=True, help='Whether peptide-level log2fx should be computed instead of protein-level. Protein-level is more robust if measured peptides are variable between conditions or replicates.')
@click.option('--threads', default=1, show_default=True, type=int, help='Number of threads used for parallel processing. -1 means all available CPUs.', callback=transform_threads)
def quantify(infile, outfile, control_condition, paired, maximum_interaction_qvalue, min_abs_log2fx, minimum_peptides, maximum_peptides, missing_peptides, peptide_log2fx, threads):
    """
    Quantify protein and interaction features in SEC data.
    """

    # Define outfile
    if outfile is None:
        outfile = infile
    else:
        # TODO: Consider replacing this with subprocess.call(["cp", "infile", "outfile"]) for speed improvement
        copyfile(infile, outfile)
        outfile = outfile

    click.echo("Info: Prepare quantitative matrices.")
    qm = quantitative_matrix(outfile, maximum_interaction_qvalue, minimum_peptides, maximum_peptides)

    con = connect(outfile)
    qm.monomer_peptide.to_sql('MONOMER_QM', con, index=False, if_exists='replace')
    qm.complex_peptide.to_sql('COMPLEX_QM', con, index=False, if_exists='replace')
    con.close()

    click.echo("Info: Assess differential features.")
    et = enrichment_test(outfile, control_condition, paired, min_abs_log2fx, missing_peptides, peptide_log2fx, threads)


    con = connect(outfile)
    et.edge.to_sql('EDGE', con, index=False, if_exists='replace')
    et.edge_level.to_sql('EDGE_LEVEL', con, index=False, if_exists='replace')
    et.node.to_sql('NODE', con, index=False, if_exists='replace')
    et.node_level.to_sql('NODE_LEVEL', con, index=False, if_exists='replace')
    et.protein_level.to_sql('PROTEIN_LEVEL', con, index=False, if_exists='replace')

    con.close()

# SECAT export features
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--maximum_interaction_qvalue', default=0.05, show_default=True, type=float, help='Maximum q-value to consider interactions for quantification.')
def export(infile, maximum_interaction_qvalue):
    """
    Export SECAT results.
    """

    outfile_interactions = path.splitext(infile)[0] + "_interactions.csv"
    outfile_network = path.splitext(infile)[0] + "_network.csv"
    outfile_nodes = path.splitext(infile)[0] + "_differential_nodes.csv"
    outfile_nodes_level = path.splitext(infile)[0] + "_differential_nodes_level.csv"
    outfile_edges = path.splitext(infile)[0] + "_differential_edges.csv"
    outfile_edges_level = path.splitext(infile)[0] + "_differential_edges_level.csv"
    outfile_proteins_level = path.splitext(infile)[0] + "_differential_proteins_level.csv"

    con = connect(infile)

    if check_sqlite_table(con, 'FEATURE_SCORED_COMBINED'):
        interaction_data = read_sql('SELECT DISTINCT bait_id, prey_id FROM FEATURE_SCORED_COMBINED WHERE decoy == 0 and qvalue <= %s;' % maximum_interaction_qvalue , con)
        interaction_data.to_csv(outfile_interactions, index=False)
    if check_sqlite_table(con, 'FEATURE_SCORED_COMBINED') and check_sqlite_table(con, 'MONOMER_QM'):
        network_data = read_sql('SELECT DISTINCT bait_id, prey_id FROM FEATURE_SCORED_COMBINED WHERE decoy == 0 and qvalue <= %s UNION SELECT DISTINCT bait_id, prey_id FROM MONOMER_QM;' % maximum_interaction_qvalue , con)
        network_data.to_csv(outfile_network, index=False)
    if check_sqlite_table(con, 'NODE'):
        node_data = read_sql('SELECT * FROM NODE LEFT OUTER JOIN PROTEIN ON bait_id = protein_id;' , con)
        node_data.sort_values(by=['pvalue']).to_csv(outfile_nodes, index=False)
    if check_sqlite_table(con, 'NODE_LEVEL'):
        node_level_data = read_sql('SELECT * FROM NODE_LEVEL LEFT OUTER JOIN PROTEIN ON bait_id = protein_id;' , con)
        node_level_data.sort_values(by=['pvalue']).to_csv(outfile_nodes_level, index=False)
    if check_sqlite_table(con, 'EDGE'):
        edge_data = read_sql('SELECT * FROM EDGE;' , con)
        edge_data.sort_values(by=['pvalue']).to_csv(outfile_edges, index=False)
    if check_sqlite_table(con, 'EDGE_LEVEL'):
        edge_level_data = read_sql('SELECT * FROM EDGE_LEVEL;' , con)
        edge_level_data.sort_values(by=['pvalue']).to_csv(outfile_edges_level, index=False)
    if check_sqlite_table(con, 'PROTEIN_LEVEL'):
        protein_level_data = read_sql('SELECT * FROM PROTEIN_LEVEL LEFT OUTER JOIN PROTEIN ON bait_id = protein_id;' , con)
        protein_level_data.sort_values(by=['pvalue']).to_csv(outfile_proteins_level, index=False)

    con.close()


# SECAT plot chromatograms
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--level', default='bait', show_default=True, type=click.Choice(['bait', 'interaction']), help='Plot either all interactions of bait proteins or individual interactions')
@click.option('--id', required=False, type=str, help='Plot specific UniProt bait_id (Q10000) or interaction_id (Q10000_P10000)')
@click.option('--max_qvalue', default=0.01, show_default=True, type=float, help='Maximum q-value to plot baits or interactions.')
@click.option('--min_abs_log2fx', default=1.0, show_default=True, type=float, help='Minimum absolute log2 fold-change for integrated nodes.')
@click.option('--mode', default='quantitative', show_default=True, type=click.Choice(['quantitative', 'detection']), help='Select mode to order interaction plots by.')
@click.option('--combined/--no-combined', default=False, show_default=True, help='Select interactions and baits according to combined q-values.')
@click.option('--peptide_rank', default=6, show_default=True, type=int, help='Number of most intense peptides to plot.')
def plot(infile, level, id, max_qvalue, min_abs_log2fx, mode, combined, peptide_rank):
    """
    Plot SECAT results
    """

    pf = plot_features(infile, level, id, max_qvalue, min_abs_log2fx, mode, combined, peptide_rank)

# SECAT print statistics
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Input SECAT file.')
@click.option('--min_abs_log2fx', default=1.0, show_default=True, type=float, help='Minimum absolute log2 fold-change for integrated nodes.')
def statistics(infile, min_abs_log2fx):
    """
    Print SECAT statistics
    """

    con = connect(infile)

    if check_sqlite_table(con, 'QUANTIFICATION') and check_sqlite_table(con, 'SEC') and check_sqlite_table(con, 'PROTEIN'):
        click.echo("Protein information")
        click.echo(10*"-")
        pepprot = read_sql('SELECT * FROM QUANTIFICATION INNER JOIN SEC ON QUANTIFICATION.run_id = SEC.run_id WHERE protein_id IN (SELECT DISTINCT protein_id FROM PROTEIN);' , con)

        click.echo("Total proteins: %s" % len(pepprot['protein_id'].drop_duplicates()))
        click.echo("Total proteins per run:")
        click.echo(pepprot.groupby(['condition_id', 'replicate_id'])['protein_id'].nunique())
        click.echo(10*"-")

        click.echo("Total peptides: %s" % len(pepprot[['peptide_id']].drop_duplicates()))
        click.echo("Total peptides per run:")
        click.echo(pepprot.groupby(['condition_id', 'replicate_id'])['peptide_id'].nunique())
        click.echo(10*"=")

    if check_sqlite_table(con, 'FEATURE_SCORED_COMBINED'):
        click.echo("PPI Information")
        click.echo(10*"-")
        intact_combined = read_sql('SELECT *, bait_id || "_" || prey_id AS interaction_id FROM FEATURE_SCORED_COMBINED WHERE decoy==0;' , con)
        intact = read_sql('SELECT *, bait_id || "_" || prey_id AS interaction_id FROM FEATURE_SCORED WHERE decoy==0;' , con)

        click.echo("Total interactions (q-value<0.01): %s" % intact_combined[intact_combined['qvalue']<0.01]['interaction_id'].nunique())
        click.echo("Total interactions per run (q-value<0.01):")
        click.echo(intact[intact['qvalue']<0.01].groupby(['condition_id', 'replicate_id'])['interaction_id'].nunique())
        click.echo(10*"-")

        click.echo("Total interactions (q-value<0.05): %s" % intact_combined[intact_combined['qvalue']<0.05]['interaction_id'].nunique())
        click.echo("Total interactions per run (q-value<0.05):")
        click.echo(intact[intact['qvalue']<0.05].groupby(['condition_id', 'replicate_id'])['interaction_id'].nunique())
        click.echo(10*"-")

        click.echo("Total interactions (q-value<0.1): %s" % intact_combined[intact_combined['qvalue']<0.1]['interaction_id'].nunique())
        click.echo("Total interactions per run (q-value<0.1):")
        click.echo(intact[intact['qvalue']<0.1].groupby(['condition_id', 'replicate_id'])['interaction_id'].nunique())
        click.echo(10*"=")

    if check_sqlite_table(con, 'PROTEIN_LEVEL'):
        click.echo("Quantitative Information (min_abs_log2fx > %s)" % min_abs_log2fx)
        click.echo(10*"-")

        df_node_level = read_sql('SELECT * FROM NODE_LEVEL;', con)

        df_node_level_filtered = df_node_level[df_node_level['abs_log2fx'] > min_abs_log2fx]
        df_node = df_node_level_filtered.sort_values('pvalue_adjusted').groupby(['condition_1','condition_2','bait_id']).head(1).reset_index()

        click.echo("Info: Total dysregulated proteins detected:")
        click.echo("%s (at FDR < 0.01)" % (df_node[df_node['pvalue_adjusted'] < 0.01][['bait_id']].drop_duplicates().shape[0]))
        click.echo("%s (at FDR < 0.05)" % (df_node[df_node['pvalue_adjusted'] < 0.05][['bait_id']].drop_duplicates().shape[0]))
        click.echo("%s (at FDR < 0.1)" % (df_node[df_node['pvalue_adjusted'] < 0.1][['bait_id']].drop_duplicates().shape[0]))
        click.echo("%s (unfiltered)" % (df_node[['bait_id']].drop_duplicates().shape[0]))

        for level in df_node_level_filtered['level'].unique():
            click.echo("Info: Dysregulated (%s-mode) proteins detected:" % (level))
            click.echo("%s (at FDR < 0.01)" % (df_node_level_filtered[(df_node_level_filtered['level'] == level) & (df_node_level_filtered['pvalue_adjusted'] < 0.01)][['bait_id']].drop_duplicates().shape[0]))
            click.echo("%s (at FDR < 0.05)" % (df_node_level_filtered[(df_node_level_filtered['level'] == level) & (df_node_level_filtered['pvalue_adjusted'] < 0.05)][['bait_id']].drop_duplicates().shape[0]))
            click.echo("%s (at FDR < 0.1)" % (df_node_level_filtered[(df_node_level_filtered['level'] == level) & (df_node_level_filtered['pvalue_adjusted'] < 0.1)][['bait_id']].drop_duplicates().shape[0]))
            click.echo("%s (unfiltered)" % (df_node_level_filtered[(df_node_level_filtered['level'] == level)][['bait_id']].drop_duplicates().shape[0]))
            
        click.echo(10*"=")
