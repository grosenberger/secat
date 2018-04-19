import numpy as np
import pandas as pd
import click
from sklearn import preprocessing
import sys

from lxml import etree

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


    def table(self):
        return self.df


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
        with open(netfile) as f:
            header = f.readline()
        f.close()

        # STRING-DB
        if header.split("\n")[0].split(" ") == ['protein1', 'protein2', 'combined_score']:
            return self.formats[0]
        # MITAB 2.5, 2.6, 2.7
        elif len(header.split("\n")[0].split("\t")) in [15, 36, 42]:
            return self.formats[1]
        else:
            sys.exit("Error: Reference network file format is not supported.")

