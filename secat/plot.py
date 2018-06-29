import pandas as pd
import numpy as np
import click
import sqlite3
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class plot_features:
    def __init__(self, infile, interaction_id, interaction_qvalue, bait_id, bait_qvalue, peptide_rank):
        self.infile = infile
        self.interaction_id = interaction_id
        self.interaction_qvalue = interaction_qvalue
        self.bait_id = bait_id
        self.bait_qvalue = bait_qvalue
        self.peptide_rank = peptide_rank

        self.feature_data = self.read_features()
        self.peptide_data = self.read_peptides()

        if self.interaction_id is not None:
            self.plot_interaction(interaction_id)

        if self.bait_id is not None:
            self.plot_bait(bait_id)

        if self.interaction_qvalue is not None:
            interaction_ids = self.read_interactions()
            for interaction_id in interaction_ids:
                self.plot_interaction(interaction_id)

        if self.bait_qvalue is not None:
            bait_ids = self.read_baits()
            for bait_id in bait_ids:
                self.plot_bait(bait_id)

    def plot_interaction(self, interaction_id):
        feature_data = self.feature_data
        peptide_data = self.peptide_data

        feature_data = feature_data[feature_data['interaction_id'] == interaction_id]
        proteins = pd.DataFrame({"protein_id": pd.concat([feature_data['bait_id'], feature_data['prey_id']])}).drop_duplicates()
        peptide_data = pd.merge(peptide_data, proteins, how='inner', on='protein_id')
        out = os.path.splitext(os.path.basename(self.infile))[0]+"_"+interaction_id+".pdf"

        with PdfPages(out) as pdf:
            f = self.generate_plot(peptide_data, feature_data)
            pdf.savefig()

    def plot_bait(self, bait_id):
        feature_data = self.feature_data
        peptide_data = self.peptide_data

        feature_data = feature_data[feature_data['bait_id'] == bait_id]
        out = os.path.splitext(os.path.basename(self.infile))[0]+"_"+bait_id+".pdf"

        with PdfPages(out) as pdf:
            for prey_id in feature_data['prey_id'].drop_duplicates().values:
                f = self.generate_plot(peptide_data[(peptide_data['protein_id'] == bait_id) | (peptide_data['protein_id'] == prey_id)], feature_data[feature_data['prey_id'] == prey_id])
                pdf.savefig()

    def read_features(self):
        con = sqlite3.connect(self.infile)
        df = pd.read_sql('''
SELECT DISTINCT feature_meta.condition_id || '_' || feature_meta.replicate_id AS tag,
                feature_super.feature_id,
                feature_super.bait_id,
                feature_super.prey_id,
                FEATURE_ALIGNED.interaction_id,
                feature_meta.rt,
                feature_meta.leftWidth,
                feature_meta.rightWidth,
                feature_scored.pep
FROM FEATURE_SUPER
INNER JOIN FEATURE_META ON FEATURE_SUPER.FEATURE_ID = FEATURE_META.FEATURE_ID
INNER JOIN FEATURE_ALIGNED ON FEATURE_SUPER.feature_id = FEATURE_ALIGNED.feature_id
AND FEATURE_SUPER.prey_id = FEATURE_ALIGNED.prey_id
INNER JOIN EDGE_LEVEL ON FEATURE_ALIGNED.interaction_id = EDGE_LEVEL.interaction_id
LEFT JOIN feature_scored ON feature_super.feature_id = feature_scored.feature_id
AND feature_super.prey_id = feature_scored.prey_id;
''', con)
        con.close()

        return df

    def read_peptides(self):
        con = sqlite3.connect(self.infile)
        df = pd.read_sql('''
SELECT condition_id,
       replicate_id,
       protein_id,
       QUANTIFICATION.peptide_id,
       peptide_intensity,
       sec_id,
       condition_id || '_' || replicate_id AS tag
FROM QUANTIFICATION
INNER JOIN SEC ON QUANTIFICATION.run_id = SEC.run_id
INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id
WHERE peptide_rank <= %s;
''' % (self.peptide_rank), con)
        con.close()

        return df

    def read_interactions(self):
        con = sqlite3.connect(self.infile)
        df = pd.read_sql('''
SELECT DISTINCT interaction_id FROM EDGE WHERE qvalue <= %s;
''' % (self.interaction_qvalue), con)
        con.close()

        return df['interaction_id'].values

    def read_baits(self):
        con = sqlite3.connect(self.infile)
        df = pd.read_sql('''
SELECT DISTINCT bait_id FROM EDGE NODE qvalue <= %s;
''' % (self.bait_qvalue), con)
        con.close()

        return df['bait_id'].values

    def generate_plot(self, peptide_data, feature_data):
        interaction_id = feature_data['interaction_id'].drop_duplicates().values[0]
        bp_pairs = feature_data[['bait_id','prey_id']].drop_duplicates().sort_values(by=['bait_id']).reset_index()

        tags = peptide_data.sort_values(by=['tag'])['tag'].drop_duplicates().values.tolist()

        f, axarr = plt.subplots(len(tags), 2, sharex=True, sharey=True, figsize=(15,15))
        f.suptitle(interaction_id)

        xmin = 0 # peptide_data['sec_id'].min()
        xmax = peptide_data['sec_id'].max()
        ymin = 0 #peptide_data['peptide_intensity'].min()
        ymax = peptide_data['peptide_intensity'].max() * 1.2

        for bp_index, bp_pair in bp_pairs.iterrows():
            for tag in tags:
                axarr[tags.index(tag), bp_index].set_xlim(xmin, xmax)
                axarr[tags.index(tag), bp_index].set_ylim(ymin, ymax)
                proteins = peptide_data['protein_id'].drop_duplicates().values
                for protein in proteins:
                        if protein == bp_pair['bait_id']:
                            protein_color = 'red'
                        else:
                            protein_color = 'black'
                        peptides = peptide_data[peptide_data['protein_id'] == protein]['peptide_id'].drop_duplicates().values
                        for peptide in peptides:
                            points = peptide_data[(peptide_data['peptide_id'] == peptide) & (peptide_data['tag'] == tag)].sort_values(by=['sec_id'])

                            axarr[tags.index(tag), bp_index].plot(points['sec_id'], points['peptide_intensity'], color=protein_color)
                            axarr[tags.index(tag), bp_index].legend([Line2D([0], [0], color='red'), Line2D([0], [0], color='black')], [bp_pair['bait_id'], bp_pair['prey_id']])
                            axarr[tags.index(tag), bp_index].set_title(tag, loc = 'center', pad = -15)
                features = feature_data[(feature_data['bait_id'] == bp_pair['bait_id']) & (feature_data['prey_id'] == bp_pair['prey_id']) & (feature_data['tag'] == tag)]
                for feature_index, feature in features.iterrows():
                    if np.isfinite(feature['pep']):
                        boxalpha = 0.1 + 0.2 * (1-feature['pep'])
                        axarr[tags.index(tag), bp_index].text(feature['RT'], ymax*0.9, np.around(feature['pep'], 2), fontsize=6, horizontalalignment='center', bbox=dict(facecolor='white', alpha=1.0))
                    else:
                        boxalpha = 0.1
                    axarr[tags.index(tag), bp_index].axvspan(feature['leftWidth'], feature['rightWidth'], color='red', alpha=boxalpha)
                    axarr[tags.index(tag), bp_index].axvline(x=feature['RT'], color='orange', alpha=0.5)

        return f

