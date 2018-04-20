import numpy as np
import pandas as pd
import click
from sklearn import preprocessing
import sys

from lxml import etree

from pandas.api.types import is_numeric_dtype

class uniprot:
    def __init__(self, uniprotfile):
        self.namespaces = {'uniprot': "http://uniprot.org/uniprot"}
        self.df = self.read(uniprotfile)

    def read(self, uniprotfile):
        def _extract(lst):
            if len(lst) >= 1:
                return lst[0]
            else:
                return None

        df = pd.DataFrame(columns=["protein_id", "protein_name", "ensembl_id", "protein_mw"])

        root = etree.parse(uniprotfile)

        accessions = root.xpath('//uniprot:entry/uniprot:accession/text()', namespaces = self.namespaces)
        names = root.xpath('//uniprot:entry/uniprot:name/text()', namespaces = self.namespaces)
        mw = root.xpath('//uniprot:entry/uniprot:sequence/@mass', namespaces = self.namespaces)
        ensembl = root.xpath('//uniprot:entry/uniprot:dbReference[@type="Ensembl"]/uniprot:property[@type="protein sequence ID"]/@value', namespaces = self.namespaces)

        for entry in root.xpath('//uniprot:entry', namespaces = self.namespaces):
            accession = entry.xpath('./uniprot:accession/text()', namespaces = self.namespaces)
            name = entry.xpath('./uniprot:name/text()', namespaces = self.namespaces)
            mw = entry.xpath('./uniprot:sequence/@mass', namespaces = self.namespaces)
            ensembl = entry.xpath('./uniprot:dbReference[@type="Ensembl"]/uniprot:property[@type="protein sequence ID"]/@value', namespaces = self.namespaces)

            df = df.append({'protein_id': _extract(accession), 'protein_name': _extract(name), 'ensembl_id': ensembl, 'protein_mw': _extract(mw)}, ignore_index=True)

        return df

    def to_df(self):
        return self.df[['protein_id','protein_name','protein_mw']]

    def expand(self):
        ensembl = self.df.apply(lambda x: pd.Series(x['ensembl_id']),axis=1).stack().reset_index(level=1, drop=True)
        ensembl.name = 'ensembl_id'

        return self.df.drop('ensembl_id', axis=1).join(ensembl).reset_index(drop=True)[["protein_id", "protein_name", "ensembl_id", "protein_mw"]]

class mitab:
    def __init__(self, mitabfile):
        self.df = self.read(mitabfile)

    def read(self, mitabfile):
        def _extract_uniprotkb(string):
            return [u for u in string.split("|") if "uniprotkb" in u][0].split("uniprotkb:")[1]

        def _extract_score(string):
            return float([u for u in string.split("|") if "score" in u][0].split("score:")[1])

        df = pd.read_table(mitabfile, header = None, usecols=[0,1,14])
        df.columns = ["bait_id","prey_id","interaction_confidence"]

        # Reduce DB to UniProtKB entries with scores
        df = df[df['bait_id'].str.contains('uniprotkb:') & df['prey_id'].str.contains('uniprotkb:') & df['interaction_confidence'].str.contains('score:')]

        if df.shape[0] == 0:
            sys.exit("Error: the MITAB file doesn't contain any valid entries.")
        else:
            click.echo("Info: MITAB file contains %s entries." % df.shape[0])

        # Extract UniProtKB ids
        df.bait_id = df.bait_id.apply(_extract_uniprotkb)
        df.prey_id = df.prey_id.apply(_extract_uniprotkb)

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
        df = pd.read_table(stringdbfile, sep=" ")

        _, df['protein1s'] = df['protein1'].str.split('.', 1).str
        _, df['protein2s'] = df['protein2'].str.split('.', 1).str

        df = df[['protein1s','protein2s','combined_score']]
        df['combined_score'] = df['combined_score'] / 1000.0

        # Map protein1
        df = pd.merge(df, uniprot.expand(), left_on='protein1s', right_on='ensembl_id')[['protein_id','protein2s','combined_score']]
        df.columns = ["bait_id","protein2s","combined_score"]
        # Map protein2
        df = pd.merge(df, uniprot.expand(), left_on='protein2s', right_on='ensembl_id')[['bait_id','protein_id','combined_score']]
        df.columns = ["bait_id","prey_id","interaction_confidence"]

        return df

class net:
    def __init__(self, netfile, uniprot):
        self.formats = ['stringdb','mitab']
        self.format = self.identify(netfile)

        if self.format == 'stringdb':
            network = stringdb(netfile, uniprot).df
        elif self.format == 'mitab':
            network = mitab(netfile).df

        self.df = self.expand(network, uniprot)

    def identify(self, netfile):
        header = pd.read_table(netfile, sep=None, nrows=1, engine='python')

        columns = list(header.columns.values)

        # STRING-DB
        if columns == ['protein1', 'protein2', 'combined_score']:
            return self.formats[0]
        # MITAB 2.5, 2.6, 2.7
        elif len(header.columns) in [15, 36, 42]:
            return self.formats[1]
        else:
            sys.exit("Error: Reference network file format is not supported.")

    def expand(self, network, uniprot):
        network_rev = network.copy()

        network_rev = network_rev[['prey_id','bait_id','interaction_confidence']]
        network_rev.columns = ["bait_id","prey_id","interaction_confidence"]

        proteins = uniprot.to_df()[['protein_id','protein_id']]
        proteins.columns = ["bait_id","prey_id"]
        proteins['interaction_confidence'] = 1.0

        return pd.concat([proteins, network, network_rev])

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
        header = pd.read_table(secfile, sep=None, nrows=1, engine='python')

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
        self.run_id_col = columns[0]
        self.protein_id_col = columns[5]
        self.peptide_id_col = columns[6]
        self.intensity_id_col = columns[7]
        self.formats = ['matrix','long']
        self.format, self.header = self.identify(infile)
        self.run_ids = run_ids

        if self.format == 'matrix':
            self.df = self.read_matrix(infile)
        elif self.format == 'long':
            self.df = self.read_long(infile)

    def identify(self, infile):
        header = pd.read_table(infile, sep=None, nrows=1, engine='python')

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
        df['run_id'] = df['run_id'].apply(str)
        df['protein_id'] = df['protein_id'].apply(str)
        df['peptide_id'] = df['peptide_id'].apply(str)
        df['peptide_intensity'] = df['peptide_intensity'].apply(float)

        df = df.sort_values(by=['protein_id','peptide_id','run_id'])

        # Simple validation
        if len(run_id_columns) < 10:
            sys.exit("Error: Peptide quantification file could not be linked to SEC definition file. Try changing the 'columns' parameter.")

        # Remove zero values to save space
        df = df[df['peptide_intensity'] > 0]

        return df

    def read_long(self, infile):
        # Read data
        df = pd.read_csv(infile, sep=None, engine='python')

        # Organize and rename columns
        df = df[[self.run_id_col, self.protein_id_col, self.peptide_id_col, self.intensity_id_col]]
        df.columns = ['run_id', 'protein_id', 'peptide_id', 'peptide_intensity']

        # run_id, condition_id and replicate_id are categorial values, peptide_intensity must be float
        df['run_id'] = df['run_id'].apply(str)
        df['protein_id'] = df['protein_id'].apply(str)
        df['peptide_id'] = df['peptide_id'].apply(str)
        df['peptide_intensity'] = df['peptide_intensity'].apply(float)

        df = df.sort_values(by=['protein_id','peptide_id','run_id'])

        # Simple validation
        run_id_columns = set(self.run_ids).intersection(set(df['run_id'].unique()))

        if len(run_id_columns) < 10:
            sys.exit("Error: Peptide quantification file could not be linked to SEC definition file. Try changing the 'columns' parameter.")

        # Remove zero values to save space
        df = df[df['peptide_intensity'] > 0]

        return df

    def to_df(self):
        return self.df

class meta:
    def __init__(self, quantification_data, sec_data, decoy_intensity_bins, decoy_left_sec_bins, decoy_right_sec_bins):
        self.decoy_intensity_bins = decoy_intensity_bins
        self.decoy_left_sec_bins = decoy_left_sec_bins
        self.decoy_right_sec_bins = decoy_right_sec_bins

        self.peptide_meta, self.protein_meta = self.generate(quantification_data, sec_data)

    def generate(self, quantification_data, sec_data):
        df = pd.merge(quantification_data.to_df(), sec_data.to_df(), on='run_id')

        # Peptide-level meta data
        top_pep_tg = df.groupby(['peptide_id'])
        top_pep_s = top_pep_tg['peptide_intensity'].sum()
        top_pep = pd.DataFrame(top_pep_s)
        top_pep['peptide_id']=top_pep.index
        top_pep.columns = ['sumIntensity','peptide_id']
        top_pep_merged = pd.merge(df[['peptide_id','protein_id']], top_pep, on='peptide_id', how='inner')
        top_pep_rank = top_pep_merged[['protein_id','peptide_id','sumIntensity']].drop_duplicates()
        top_pep_rank['peptide_rank'] = top_pep_rank.groupby(['protein_id'])['sumIntensity'].rank(ascending=False)

        # Store peptide-level meta data
        peptide_meta = top_pep_rank[['peptide_id','peptide_rank']]

        # Protein-level meta data
        num_pep = pd.DataFrame(top_pep_rank.groupby(['protein_id'])['peptide_id'].count())
        num_pep['protein_id']=num_pep.index
        num_pep.columns = ['peptide_count','protein_id']

        # Generate intensity bins
        inpep_intensity = df.groupby(['protein_id'])
        inpep_intensity_sum = inpep_intensity['peptide_intensity'].sum()
        inpep_intensity_ranks = pd.DataFrame(inpep_intensity_sum)
        inpep_intensity_ranks['protein_id']=inpep_intensity_ranks.index
        inpep_intensity_ranks.columns = ['sum_intensity','protein_id']
        inpep_intensity_ranks['intensity_rank'] = inpep_intensity_ranks['sum_intensity'].rank(ascending=False)
        inpep_intensity_ranks['intensity_bin'] = pd.cut(inpep_intensity_ranks['intensity_rank'], bins=self.decoy_intensity_bins, right=False, labels=False)

        # Generate left sec bins
        inpep_min_sec = df.groupby(['protein_id'])
        inpep_min_sec_cnt = inpep_min_sec['sec_id'].min()
        inpep_min_sec_ranks = pd.DataFrame(inpep_min_sec_cnt)
        inpep_min_sec_ranks['protein_id']=inpep_min_sec_ranks.index
        inpep_min_sec_ranks.columns = ['min_sec','protein_id']
        inpep_min_sec_ranks['sec_min_rank'] = inpep_min_sec_ranks['min_sec'].rank(ascending=False)
        inpep_min_sec_ranks['sec_min_bin'] = pd.cut(inpep_min_sec_ranks['sec_min_rank'], bins=self.decoy_left_sec_bins, right=False, labels=False)

        # Generate right sec bins
        inpep_max_sec = df.groupby(['protein_id'])
        inpep_max_sec_cnt = inpep_max_sec['sec_id'].max()
        inpep_max_sec_ranks = pd.DataFrame(inpep_max_sec_cnt)
        inpep_max_sec_ranks['protein_id']=inpep_max_sec_ranks.index
        inpep_max_sec_ranks.columns = ['max_sec','protein_id']
        inpep_max_sec_ranks['sec_max_rank'] = inpep_max_sec_ranks['max_sec'].rank(ascending=False)
        inpep_max_sec_ranks['sec_max_bin'] = pd.cut(inpep_max_sec_ranks['sec_max_rank'], bins=self.decoy_right_sec_bins, right=False, labels=False)

        # Store protein-level meta data
        protein_meta = pd.merge(pd.merge(pd.merge(num_pep[['protein_id','peptide_count']], inpep_intensity_ranks[['protein_id','intensity_bin']], on='protein_id', how='inner'), inpep_min_sec_ranks[['protein_id','sec_min_bin','min_sec']], on='protein_id', how='inner'), inpep_max_sec_ranks[['protein_id','sec_max_bin','max_sec']], on='protein_id', how='inner')

        return peptide_meta, protein_meta

class queries:
    def __init__(self, net_data, protein_meta_data):
        self.df = self.generate_queries(net_data, protein_meta_data)

    def generate_queries(self, net_data, protein_meta_data):
        def _random_nonidentical_array(data):
            np.random.shuffle(data['bait_id'].values)
            return data

        # Merge data
        queries = pd.merge(net_data.to_df(), protein_meta_data, left_on='prey_id', right_on='protein_id', how='inner')
        queries['decoy'] = False

        # Append decoys
        decoy_queries = queries.copy()
        decoy_queries = decoy_queries.groupby(['intensity_bin','sec_min_bin','sec_max_bin'])

        for it in range(0, 1):
            if it == 0:
                decoy_queries_shuffled = decoy_queries.apply(_random_nonidentical_array).reset_index()
            else:
                decoy_queries_shuffled = pd.concat([decoy_queries_shuffled, decoy_queries.apply(random_nonidentical_array).reset_index()])
        decoy_queries = decoy_queries_shuffled

        # Exclude conflicting data from decoys (present in both targets and decoys)
        decoy_queries = pd.merge(decoy_queries.drop(['decoy'], axis=1), queries[['bait_id','prey_id','decoy']], on=['bait_id','prey_id'], how='left')
        decoy_queries = decoy_queries.fillna(True)
        decoy_queries = decoy_queries[decoy_queries['decoy'] == True]

        return pd.concat([queries, decoy_queries]).drop_duplicates()

    def to_df(self):
        return self.df
