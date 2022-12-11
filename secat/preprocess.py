import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import click
from sklearn import preprocessing
import statsmodels.api as sm
import sys
import os
import numpy as np

from lxml import etree
import itertools

from pandas.api.types import is_numeric_dtype

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class uniprot:
    def __init__(self, uniprotfile, cache):
        self.namespaces = {'uniprot': "http://uniprot.org/uniprot"}
        self.cache = cache
        self.df = self.read(uniprotfile)
    
    def read(self, uniprotfile):
        if uniprotfile.endswith("xml.gz"):
            cache_filename = uniprotfile.strip("xml.gz") + "parquet"
        elif uniprotfile.endswith("xml"):
            cache_filename = uniprotfile.strip("xml") + "parquet"
        
        if self.cache and os.path.exists(cache_filename):
            print("Found cached table")
            return pd.read_parquet(cache_filename)
        
        def _extract(lst):
            if len(lst) >= 1:
                return lst[0]
            else:
                return None

        df = pd.DataFrame(columns=["protein_id", "protein_name", "gene", "ensembl_id", "protein_mw"])

        root = etree.parse(uniprotfile)

        mw = root.xpath('//uniprot:entry/uniprot:sequence/@mass', namespaces = self.namespaces)
        if root.xpath('//uniprot:entry/uniprot:organism/uniprot:dbReference[@type="NCBI Taxonomy"]/@id', namespaces = self.namespaces)[0] == '559292':
            ensembl = root.xpath('//uniprot:entry/uniprot:gene/uniprot:name[@type = "ordered locus"]/text()', namespaces = self.namespaces)
            ensembl_path = './uniprot:gene/uniprot:name[@type = "ordered locus"]/text()'
        else:
            ensembl = root.xpath('//uniprot:entry/uniprot:dbReference[@type="Ensembl"]/uniprot:property[@type="protein sequence ID"]/@value', namespaces = self.namespaces)
            ensembl_path = './uniprot:dbReference[@type="Ensembl"]/uniprot:property[@type="protein sequence ID"]/@value'

        # TODO: Need to make this parsing more memory efficient
        entries = root.xpath('//uniprot:entry', namespaces = self.namespaces)
        rows = []
        for entry in tqdm(entries, total=len(entries)):
            accession = entry.xpath('./uniprot:accession/text()', namespaces = self.namespaces)
            name = entry.xpath('./uniprot:name/text()', namespaces = self.namespaces)
            gene = entry.xpath('./uniprot:gene/uniprot:name/text()', namespaces = self.namespaces)
            mw = entry.xpath('./uniprot:sequence/@mass', namespaces = self.namespaces)
            ensembl = entry.xpath(ensembl_path, namespaces = self.namespaces)

            row = pd.Series({
                'protein_id': _extract(accession), 
                'protein_name': _extract(name), 
                'gene': _extract(gene), 
                'ensembl_id': ensembl, 
                'protein_mw': float(_extract(mw))
            })
            rows.append(row)
        
        # Append each Series object as a new row to df
        df = pd.concat([df, *[row.to_frame().T for row in rows]], ignore_index=True)
        return df

    def to_df(self):
        return self.df[['protein_id','protein_name', 'gene', 'protein_mw']]

    def expand(self):
        ensembl = self.df.apply(
            lambda x: pd.Series(x['ensembl_id'], dtype='object'),
            axis=1
        ).stack().reset_index(level=1, drop=True)
        ensembl.name = 'ensembl_id'

        return self.df.drop('ensembl_id', axis=1).join(ensembl).reset_index(drop=True)[["protein_id", "protein_name", "gene", "ensembl_id", "protein_mw"]]

class mitab:
    def __init__(self, mitabfile):
        self.df = self.read(mitabfile)

    def read(self, mitabfile):
        def _extract_uniprotkb(string):
            return [i.split("uniprotkb:")[1].split("-")[0] for i in [u for u in string.split("|") if "uniprotkb" in u]]

        def _extract_score(string):
            if 'intact-miscore:' in string:
                return float([u for u in string.split("|") if "intact-miscore" in u][0].split("intact-miscore:")[1])
            elif 'score:' in string:
                return float([u for u in string.split("|") if "score" in u][0].split("score:")[1])
            else:
                return 0

        df = pd.read_csv(mitabfile, sep="\t", header = None, usecols=[0,1,2,3,14], engine='c')
        df.columns = ["bait_id","prey_id","bait_id_alt","prey_id_alt","interaction_confidence"]

        df['bait_id_alt'] = df['bait_id_alt'].replace(np.nan,'',regex=True)
        df['prey_id_alt'] = df['prey_id_alt'].replace(np.nan,'',regex=True)

        df['bait_id'] = df['bait_id'] + "|" + df['bait_id_alt']
        df['prey_id'] = df['prey_id'] + "|" + df['prey_id_alt']
        df = df.drop(columns=['bait_id_alt','prey_id_alt'])

        # Reduce DB to UniProtKB entries with scores
        df = df[df['bait_id'].str.contains('uniprotkb:') & df['prey_id'].str.contains('uniprotkb:') & (df['interaction_confidence'].str.contains('score:') | df['interaction_confidence'].str.contains('shortestPath:'))]

        if df.shape[0] == 0:
            sys.exit("Error: the MITAB file doesn't contain any valid entries.")
        else:
            click.echo("Info: MITAB file contains %s entries." % df.shape[0])

        # Extract UniProtKB ids
        df.bait_id = df.bait_id.apply(_extract_uniprotkb)
        df.prey_id = df.prey_id.apply(_extract_uniprotkb)

        # Explode lists
        df = df.explode('bait_id').reset_index(drop=True)
        df = df.explode('prey_id').reset_index(drop=True)

        click.echo("Info: MITAB file contains %s entries considering all alternative identifiers." % df.shape[0])

        # Extract score
        df.interaction_confidence = df.interaction_confidence.apply(_extract_score)

        df = df.groupby(["bait_id","prey_id"])["interaction_confidence"].max().reset_index()
        click.echo("Info: MITAB file contains %s unique entries." % df.shape[0])

        # Normalize score
        if df.interaction_confidence.max() > 1 or df.interaction_confidence.min() < 0:
            scaler = preprocessing.MinMaxScaler()
            scores = np.array(np.transpose([df.interaction_confidence.values]))
            scaler.fit(scores)
            df.interaction_confidence = scaler.transform(scores)

        return df

class stringdb:
    def __init__(self, stringdbfile, uniprot):
        self.df = self.read(stringdbfile, uniprot)

    def read(self, stringdbfile, uniprot):
        df = pd.read_csv(stringdbfile, sep=" ", engine='c')

        df[['protein1o','protein1s']] = df.protein1.str.split('.', expand=True)
        df[['protein2o','protein2s']] = df.protein2.str.split('.', expand=True)

        df = df[['protein1s','protein2s','combined_score']]
        df['combined_score'] = df['combined_score'] / 1000.0

        #Map protein 1
        df = pd.merge(df, uniprot.expand(), left_on='protein1s', right_on='ensembl_id')[['protein_id','protein2s','combined_score']]
        df.columns = ["bait_id","protein2s","combined_score"]
        #Map protein 2
        df = pd.merge(df, uniprot.expand(), left_on='protein2s', right_on='ensembl_id')[['bait_id','protein_id','combined_score']]
        df.columns = ["bait_id","prey_id","interaction_confidence"]


        return df

class bioplex:
    def __init__(self, bioplexfile):
        self.df = self.read(bioplexfile)

    def read(self, bioplexfile):
        df = pd.read_csv(bioplexfile, sep="\t")

        df = df[['UniprotA','UniprotB','p(Interaction)']]
        df.columns = ["bait_id","prey_id","interaction_confidence"]

        return df

class preppi:
    def __init__(self, preppifile):
        self.df = self.read(preppifile)

    def read(self, preppifile):
        df = pd.read_csv(preppifile, sep="\t")

        # Criterion for direct physical interaction
        # df = df[df['str_max_score'] * df['red_score'] > 100]

        # Estimate probability
        df['interaction_confidence'] = df['final_score'] / (df['final_score'] + 600)

        df = df[['prot1','prot2','interaction_confidence']]
        df.columns = ["bait_id","prey_id","interaction_confidence"]

        return df

class binary:
    def __init__(self, binaryfile):
        self.df = self.read(binaryfile)

    def read(self, binaryfile):
        df = pd.read_csv(binaryfile, sep=" ")

        df.columns = ["bait_id","prey_id"]
        df['interaction_confidence'] = 1

        return df

class net:
    def __init__(self, netfile, uniprot, meta):
        self.formats = ['mitab','stringdb','bioplex','preppi', 'binary','none']
        self.format = self.identify(netfile)

        if self.format == 'mitab':
            network = mitab(netfile).df
        elif self.format == 'stringdb':
            network = stringdb(netfile, uniprot).df
        elif self.format == 'bioplex':
            network = bioplex(netfile).df
        elif self.format == 'preppi':
            network = preppi(netfile).df
        elif self.format == 'binary':
            network = binary(netfile).df
        elif self.format == 'none':
            protein_ids = sorted(list(meta.protein_meta['protein_id'].unique()))
            full_queries = list(itertools.combinations(protein_ids, 2))
            click.echo("Info: Assessing all %s potential interactions of the %s proteins." % (len(full_queries), len(protein_ids)))
            network = pd.DataFrame(full_queries, columns=['bait_id', 'prey_id'])
            network['interaction_confidence'] = 1

        # Ensure that interactions are unique
        df = self.unique_interactions(network)

        # Remove interactions between same baits and preys
        df = df[df['bait_id'] != df['prey_id']]

        self.df = df

    def identify(self, netfile):
        if netfile == None:
            return self.formats[5]

        header = pd.read_csv(netfile, sep=None, nrows=1, engine='python')

        columns = list(header.columns.values)

        # STRING-DB
        if columns == ['protein1', 'protein2', 'combined_score']:
            return self.formats[1]
        # BioPlex
        elif columns == ['GeneA','GeneB','UniprotA','UniprotB','SymbolA','SymbolB','p(Wrong)','p(No Interaction)','p(Interaction)']:
            return self.formats[2]
        # PrePPI
        elif columns == ['prot1','prot2','str_score','protpep_score','str_max_score','red_score','ort_score','phy_score','coexp_score','go_score','total_score','dbs','pubs','exp_score','final_score']:
            return self.formats[3]
        # MITAB 2.5, 2.6, 2.7
        elif len(header.columns) in [11, 15, 35, 36, 42]:
            return self.formats[0]
        # Binary
        elif len(header.columns) == 2:
            return self.formats[4]
        else:
            sys.exit("Error: Reference network file format is not supported.")

    def get_interaction_id(self, x):
        return '__'.join(sorted([x[0], x[1]]))
    
    def unique_interactions(self, network):
        # Subset network to bait_id and prey_id columns
        bait_prey = network[["bait_id", "prey_id"]]
        
        # Turn subset into list of lists where each sublist is of length 2 (one bait_id one prey_id) 
        bait_prey = bait_prey.values.tolist()
        
        # Use multiprocessing.Pool to distribute get_interaction_id() across list of lists
        with multiprocessing.Pool(os.cpu_count()) as p:
            network['interaction_id'] = p.map(self.get_interaction_id, bait_prey)
        
        network = network.groupby('interaction_id')['interaction_confidence'].max().reset_index()
        network[['bait_id','prey_id']] = network.interaction_id.str.split('__', expand=True)
        return network[['bait_id','prey_id','interaction_confidence']]

    def to_df(self):
        return self.df

class sec:
    def __init__(self, secfile, columns):
        self.run_id_col = columns[0]
        self.sec_id_col = columns[1]
        self.sec_mw_col = columns[2]
        self.condition_id_col = columns[3]
        self.replicate_id_col = columns[4]
        self.formats = ['secdef']
        self.format = self.identify(secfile)
        self.df = self.read(secfile)

    def identify(self, secfile):
        header = pd.read_csv(secfile, sep=None, nrows=1, engine='python')

        columns = list(header.columns.values)

        # Check if valid
        if self.run_id_col in columns and self.sec_mw_col in columns and self.condition_id_col in columns and self.replicate_id_col in columns:
            return 'secdef'
        else:
            sys.exit("Error: SEC definition file format is not supported. Try changing the 'columns' parameter.")

    def read(self, secfile):
        df = pd.read_csv(secfile, sep=None, engine='python')

        # Organize and rename columns
        df = df[[self.run_id_col, self.sec_id_col, self.sec_mw_col, self.condition_id_col, self.replicate_id_col]]
        df.columns = ['run_id', 'sec_id', 'sec_mw', 'condition_id', 'replicate_id']

        df = df.sort_values(by=['condition_id','replicate_id','sec_id'])

        if not is_numeric_dtype(df['sec_id']):
            sys.exit("Error: SEC definition file does not contain numerical '%s' column." % self.sec_id_col)

        if not is_numeric_dtype(df['sec_mw']):
            sys.exit("Error: SEC definition file does not contain numerical '%s' column." % self.sec_mw_col)

        # run_id, condition_id and replicate_id are categorial values
        df['run_id'] = df['run_id'].apply(str)
        df['condition_id'] = df['condition_id'].apply(str)
        df['replicate_id'] = df['replicate_id'].apply(str)

        return df

    def to_df(self):
        return self.df

class quantification:
    def __init__(self, infile, columns, run_ids):
        self.run_id_col = columns[5]
        self.protein_id_col = columns[6]
        self.peptide_id_col = columns[7]
        self.intensity_id_col = columns[8]
        self.formats = ['matrix','long']
        self.format, self.header = self.identify(infile)
        self.run_ids = run_ids

        if self.format == 'matrix':
            self.df = self.read_matrix(infile)
        elif self.format == 'long':
            self.df = self.read_long(infile)

    def identify(self, infile):
        header = pd.read_csv(infile, sep=None, nrows=1, engine='python')

        columns = list(header.columns.values)

        # Matrix
        if self.run_id_col not in columns and self.protein_id_col in columns and self.peptide_id_col in columns:
            return self.formats[0], columns
        # Long list
        elif self.run_id_col in columns and self.protein_id_col in columns and self.peptide_id_col and self.intensity_id_col in columns:
            return self.formats[1], columns
        else:
            sys.exit("Error: Peptide quantification file format is not supported. Try changing the 'columns' parameter.")

    def read_matrix(self, infile):
        # Identify run_ids in header
        run_id_columns = set(self.run_ids).intersection(self.header)

        # Read data
        mx = pd.read_csv(infile, sep=None, engine='python')

        df = pd.melt(mx, id_vars=[self.protein_id_col, self.peptide_id_col], value_vars=run_id_columns, var_name=self.run_id_col, value_name=self.intensity_id_col)

        # Organize and rename columns
        df = df[[self.run_id_col, self.protein_id_col, self.peptide_id_col, self.intensity_id_col]]
        df.columns = ['run_id', 'protein_id', 'peptide_id', 'peptide_intensity']

        # run_id, condition_id and replicate_id are categorial values, peptide_intensity must be float
        if sum([1 for id in df['run_id'] if not isinstance(id, str)]) > 0:
            df['run_id'] = df['run_id'].apply(lambda x: os.path.basename(str(x)))
        df['protein_id'] = df['protein_id'].apply(str)
        df['peptide_id'] = df['peptide_id'].apply(str)
        df['peptide_intensity'] = df['peptide_intensity'].apply(float)

        df = df.sort_values(by=['protein_id','peptide_id','run_id'])

        # Proteotypic peptides only
        if "/" in df['protein_id'][0]:
            df = df.loc[df['protein_id'].str.startswith("1/")]
            df.protein_id = df.protein_id.str[2:]
        else:
            df = df.loc[df['protein_id'].str.find(';')==-1]

        # Simple validation
        if len(run_id_columns) < 10:
            sys.exit("Error: Peptide quantification file could not be linked to SEC definition file. Try changing the 'columns' parameter.")

        # Remove zero values to save space
        df = df[df['peptide_intensity'] > 0]

        return df

    def read_long(self, infile):
        def parse_protein_id(x):
            if 'sp|' in x:
                return x.split('|')[1]
            else:
                return x

        # Read data
        df = pd.read_csv(infile, sep="\t", engine='c')

        # Exclude decoys if present
        if 'decoy' in df.columns:
            df = df.loc[df['decoy'] == 0]

        # Organize and rename columns
        df = df[[self.run_id_col, self.protein_id_col, self.peptide_id_col, self.intensity_id_col]]
        df.columns = ['run_id', 'protein_id', 'peptide_id', 'peptide_intensity']

        # run_id, condition_id and replicate_id are categorial values, peptide_intensity must be float
        if sum([1 for id in df['run_id'] if not isinstance(id, str)]) > 0:
            df['run_id'] = df['run_id'].apply(lambda x: os.path.basename(str(x)))
        df['protein_id'] = df['protein_id'].apply(str)
        df['peptide_id'] = df['peptide_id'].apply(str)
        df['peptide_intensity'] = df['peptide_intensity'].apply(float)

        df = df.sort_values(by=['protein_id','peptide_id','run_id'])

        # Proteotypic peptides only
        if "/" in df['protein_id'][0]:
            df = df.loc[df['protein_id'].str.startswith("1/")]
            df.protein_id = df.protein_id.str[2:]
        else:
            df = df.loc[df['protein_id'].str.find(';')==-1]

        # Parse protein identifiers if necessary
        df['protein_id'] = df['protein_id'].apply(parse_protein_id)

        # Simple validation
        run_id_columns = set(self.run_ids).intersection(set(df['run_id'].unique()))

        if len(run_id_columns) < 10:
            sys.exit("Error: Peptide quantification file could not be linked to SEC definition file. Try changing the 'columns' parameter.")

        # Remove zero values to save space
        df = df[df['peptide_intensity'] > 0]

        return df

    def to_df(self):
        return self.df


class normalization:
    def __init__(self, quantification_data, sec_data, window_size, padded, outfile):
        self.quantification_data = quantification_data
        self.sec_data = sec_data
        self.window_size = window_size
        self.df = self.slide_normalize(padded)

        # plot input data
        self.plot(self.quantification_data, self.sec_data, os.path.splitext(os.path.basename(outfile))[0]+"_raw.pdf")
        # plot normalized data
        self.plot(self.df, self.sec_data, os.path.splitext(os.path.basename(outfile))[0]+"_norm.pdf")
        # plot number of pep identifications
        self.plot_count(self.df, self.sec_data, os.path.splitext(os.path.basename(outfile))[0]+"_count.pdf")

    def slide_normalize(self, padded):
        def window(seq, n=2):
            "Returns a sliding window (of width n) over data from the iterable"
            "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
            it = iter(seq)
            result = tuple(itertools.islice(it, n))
            if len(result) == n:
                yield result
            for elem in it:
                result = result[1:] + (elem,)
                yield result

        quantification_data = self.quantification_data.copy()
        quantification_data['peptide_intensity'] = np.log2(quantification_data['peptide_intensity'])
        quantification_list = []

        if padded:
            # Add padding to lower and upper boundaries to ensure that each fractions is covered by equal number of windows
            min_sec_id = min(self.sec_data['sec_id']) - self.window_size + 1
            max_sec_id = max(self.sec_data['sec_id']) + self.window_size
        else:
            min_sec_id = min(self.sec_data['sec_id'])
            max_sec_id = max(self.sec_data['sec_id'])+1
        
        windows = [w for w in window(range(min_sec_id, max_sec_id),  n=self.window_size)]
        mx_to_normalize = []
        for w in windows:
            runs = self.sec_data.loc[self.sec_data['sec_id'].isin(w)]['run_id']
            mx = quantification_data.loc[quantification_data['run_id'].isin(runs)]
            mx_to_normalize.append(mx)
        
        # With multiprocessing
        with multiprocessing.Pool(os.cpu_count()) as p:
            quantification_list = p.map(self.normalize, mx_to_normalize)
        
        quantification_norm = pd.concat(quantification_list)
        quantification_norm = quantification_norm.groupby(['run_id','protein_id','peptide_id'])['peptide_intensity'].mean().reset_index()

        quantification_norm['peptide_intensity'] = np.exp2(quantification_norm['peptide_intensity'])
        
        return(quantification_norm)
 
    def normalize(self, quantification_data):
        def normalizeCyclicLoess(x, span=0.7, iterations = 3):
            n = len(x.columns)
            for k in range(0,iterations):
                a = x.mean(axis=1, skipna=True)
                for i in range(0,n):
                    m = x.iloc[:,i] - a

                    # Fit lowess model
                    lwd = 0.01 * np.diff([np.nanmin(m),np.nanmax(m)])
                    lw = sm.nonparametric.lowess(endog=m.values, exog=a.values, frac=span, it=3, delta=lwd, return_sorted=False)
                    x.iloc[:,i] = x.iloc[:,i].values - lw
            return x

        mx = pd.pivot_table(quantification_data, values='peptide_intensity', index=['protein_id','peptide_id'], columns='run_id').reset_index()
        mx_idx = mx[['protein_id','peptide_id']]
        mx = mx.drop(['protein_id','peptide_id'], axis=1)

        mx_norm = normalizeCyclicLoess(mx)
        mx_new = pd.merge(mx_idx, mx_norm, left_index=True, right_index=True)
        quantification_data_new = mx_new.melt(id_vars=['protein_id','peptide_id'], var_name='run_id', value_name='peptide_intensity')
        return quantification_data_new.dropna()[quantification_data.columns]

    def plot(self, quantification_data, sec_data, filename):
        if plt is None:
            raise ImportError("Error: The matplotlib package is required to create a report.")

        quantification_sum = quantification_data.groupby(['run_id'])['peptide_intensity'].sum().reset_index()

        dfsum = pd.merge(quantification_sum, sec_data, on='run_id')
        dfsum['sample_id'] = dfsum['condition_id'].astype(str) + '_' + dfsum['replicate_id'].astype(str)

        dfplot = pd.pivot_table(dfsum, values='peptide_intensity', index='sec_id', columns='sample_id').reset_index()

        with PdfPages(filename) as pdf:
            plt.figure(figsize=(10, 5))
            for sample in dfplot.drop('sec_id', axis=1):
                plt.plot(dfplot['sec_id'], dfplot[sample], label=sample)
            plt.legend()
            plt.xlabel("SEC fraction")
            plt.ylabel("total intensity")
            pdf.savefig()
            plt.clf()
            plt.close()

    def plot_count(self, quantification_data, sec_data, filename):
        if plt is None:
            raise ImportError("Error: The matplotlib package is required to create a report.")
        
        quantification_count = quantification_data.groupby(['run_id'])['peptide_intensity'].count().reset_index() #edit
        dfcount = pd.merge(quantification_count, sec_data, on='run_id') #edit
        dfcount['sample_id'] = dfcount['condition_id'].astype(str) + '_' + dfcount['replicate_id'].astype(str) #edit
        dfplot_count = pd.pivot_table(dfcount, values='peptide_intensity', index='sec_id', columns='sample_id').reset_index() #edit
        
        with PdfPages(filename) as pdf: #all new edit
            plt.figure(figsize=(10, 5))
            for sample in dfplot_count.drop('sec_id', axis=1):
                plt.plot(dfplot_count['sec_id'], dfplot_count[sample], label=sample)
            plt.legend()
            plt.xlabel("SEC fraction")
            plt.ylabel("number of peptides")
            pdf.savefig()
            plt.clf()
            plt.close()

    def to_df(self):
        return self.df

class meta:
    def __init__(self, quantification_data, sec_data, decoy_intensity_bins, decoy_left_sec_bins, decoy_right_sec_bins):
        self.decoy_intensity_bins = decoy_intensity_bins
        self.decoy_left_sec_bins = decoy_left_sec_bins
        self.decoy_right_sec_bins = decoy_right_sec_bins

        self.peptide_meta, self.protein_meta = self.generate(quantification_data, sec_data)
 
    def generate(self, quantification_data, sec_data):
        df = pd.merge(quantification_data, sec_data, on='run_id')

        # Peptide-level meta data
        top_pep_tg = df.groupby(['peptide_id'])
        top_pep = top_pep_tg['peptide_intensity'].sum().reset_index()
        top_pep.columns = ['peptide_id','sumIntensity']
        top_pep_merged = pd.merge(df[['peptide_id','protein_id']], top_pep, on='peptide_id', how='inner')
        top_pep_rank = top_pep_merged[['protein_id','peptide_id','sumIntensity']].drop_duplicates()
        top_pep_rank['peptide_rank'] = top_pep_rank.groupby(['protein_id'])['sumIntensity'].rank(ascending=False)

        # Store peptide-level meta data
        peptide_meta = top_pep_rank[['peptide_id','peptide_rank']]

        # Protein-level meta data
        num_pep = top_pep_rank.groupby(['protein_id'])['peptide_id'].count().reset_index()
        num_pep.columns = ['protein_id','peptide_count']

        # Generate intensity bins
        inpep_intensity = df.groupby(['protein_id'])
        inpep_intensity_ranks = inpep_intensity['peptide_intensity'].sum().reset_index()
        inpep_intensity_ranks.columns = ['protein_id','sum_intensity']
        inpep_intensity_ranks['intensity_rank'] = inpep_intensity_ranks['sum_intensity'].rank(ascending=False)
        inpep_intensity_ranks['intensity_bin'] = pd.cut(inpep_intensity_ranks['intensity_rank'], bins=self.decoy_intensity_bins, right=False, labels=False)

        # Generate left sec bins
        inpep_min_sec = df.groupby(['protein_id'])
        inpep_min_sec_ranks = inpep_min_sec['sec_id'].min().reset_index()
        inpep_min_sec_ranks.columns = ['protein_id','min_sec']
        inpep_min_sec_ranks['sec_min_rank'] = inpep_min_sec_ranks['min_sec'].rank(ascending=False)
        inpep_min_sec_ranks['sec_min_bin'] = pd.cut(inpep_min_sec_ranks['sec_min_rank'], bins=self.decoy_left_sec_bins, right=False, labels=False)

        # Generate right sec bins
        inpep_max_sec = df.groupby(['protein_id'])
        inpep_max_sec_ranks = inpep_max_sec['sec_id'].max().reset_index()
        inpep_max_sec_ranks.columns = ['protein_id','max_sec']
        inpep_max_sec_ranks['sec_max_rank'] = inpep_max_sec_ranks['max_sec'].rank(ascending=False)
        inpep_max_sec_ranks['sec_max_bin'] = pd.cut(inpep_max_sec_ranks['sec_max_rank'], bins=self.decoy_right_sec_bins, right=False, labels=False)

        # Store protein-level meta data
        protein_meta = pd.merge(pd.merge(pd.merge(num_pep[['protein_id','peptide_count']], inpep_intensity_ranks[['protein_id','intensity_bin']], on='protein_id', how='inner'), inpep_min_sec_ranks[['protein_id','sec_min_bin','min_sec']], on='protein_id', how='inner'), inpep_max_sec_ranks[['protein_id','sec_max_bin','max_sec']], on='protein_id', how='inner')

        return peptide_meta, protein_meta

class query:
    def __init__(self, net_data, posnet_data, negnet_data, protein_meta_data, min_interaction_confidence, interaction_confidence_bins, interaction_confidence_quantile, decoy_oversample, decoy_subsample, decoy_exclude):
        self.min_interaction_confidence = min_interaction_confidence
        self.interaction_confidence_bins = interaction_confidence_bins
        self.interaction_confidence_quantile = interaction_confidence_quantile
        self.decoy_oversample = decoy_oversample
        self.decoy_subsample = decoy_subsample
        self.decoy_exclude = decoy_exclude
        self.df = self.generate_query(net_data, posnet_data, negnet_data, protein_meta_data)
 
    def generate_query(self, net_data, posnet_data, negnet_data, protein_meta_data):
        def _random_nonidentical_array(data):
            np.random.shuffle(data['bait_id'].values)
            return data

        # Merge data
        queries = pd.merge(net_data.to_df(), protein_meta_data, left_on='prey_id', right_on='protein_id', how='inner')
        queries['decoy'] = False
        queries['learning'] = False

        # Generate confidence bin assignment
        if len(queries['interaction_confidence'].unique()) >= self.interaction_confidence_bins:
            if self.interaction_confidence_quantile:
                queries['confidence_bin'] = pd.qcut(queries['interaction_confidence'], q=self.interaction_confidence_bins, labels=False, duplicates='drop')
            else:
                queries['confidence_bin'] = pd.cut(queries['interaction_confidence'], bins=self.interaction_confidence_bins, labels=False)
        else:
            queries['confidence_bin'] = 1

        # Append positive network
        if posnet_data is not None:
            posqueries = pd.merge(posnet_data.to_df(), protein_meta_data, left_on='prey_id', right_on='protein_id', how='inner')
            posqueries['decoy'] = False
            posqueries['learning'] = True
            posqueries['confidence_bin'] = queries['confidence_bin'].max()+1
            queries = pd.concat([queries, posqueries], ignore_index=True)

        # Append decoys
        if negnet_data is None:
            decoy_queries = queries.copy()
            decoy_queries = decoy_queries.groupby(['learning','confidence_bin','intensity_bin','sec_min_bin','sec_max_bin'])

            for it in range(0, self.decoy_oversample):
                if it == 0:
                    decoy_queries_shuffled = decoy_queries.apply(_random_nonidentical_array).reset_index()
                else:
                    decoy_queries_shuffled = pd.concat([decoy_queries_shuffled, decoy_queries.apply(_random_nonidentical_array).reset_index()])
            decoy_queries = decoy_queries_shuffled.drop(['decoy'], axis=1).drop_duplicates()
        else:
            decoy_queries = pd.merge(negnet_data.to_df(), protein_meta_data, left_on='prey_id', right_on='protein_id', how='inner')
            decoy_queries['learning'] = False

        # Exclude conflicting data from decoys (present in both targets and decoys)
        if self.decoy_exclude:
            decoy_queries = pd.merge(decoy_queries, queries[['bait_id','prey_id','decoy']], on=['bait_id','prey_id'], how='left')
            decoy_queries = decoy_queries.fillna(True)
            decoy_queries = decoy_queries[decoy_queries['decoy'] == True]
        else:
            decoy_queries['decoy'] = True

        # Filter for minimum interaction confidence
        queries = queries[queries['interaction_confidence'] >= self.min_interaction_confidence]
        decoy_queries = decoy_queries[decoy_queries['bait_id'] != decoy_queries['prey_id']]
        if (decoy_queries.shape[0] > queries.shape[0]) and self.decoy_subsample:
            decoy_queries = decoy_queries.sample(queries.shape[0]) # Same number of decoys as targets

        # Add confidence bin from target network if negative network is provided
        if negnet_data is not None:
            decoy_queries['confidence_bin'] = queries.sample(decoy_queries.shape[0], replace=True)['confidence_bin'].values

        # Add learning flag to decoys
        if posnet_data is not None:
            decoy_queries.loc[decoy_queries['confidence_bin'] != decoy_queries['confidence_bin'].max(), 'learning'] = False
            decoy_queries.loc[decoy_queries['confidence_bin'] == decoy_queries['confidence_bin'].max(), 'learning'] = True

        click.echo("Target queries per confidence bin:\n%s" % (queries['confidence_bin'].value_counts()))
        click.echo("Decoy queries per confidence bin:\n%s" % (decoy_queries['confidence_bin'].value_counts()))

        return pd.concat([queries, decoy_queries], sort=True)[['bait_id','prey_id','decoy','confidence_bin','learning']].drop_duplicates()

    def to_df(self):
        return self.df
