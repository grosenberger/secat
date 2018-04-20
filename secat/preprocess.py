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
            self.df = stringdb(netfile, uniprot).df
        elif self.format == 'mitab':
            self.df = mitab(netfile).df

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

