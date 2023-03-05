from pdb import set_trace
import pandas as pd
import numpy as np
import click
import sqlite3
import os
from os import path
import sys
from sqlite3 import connect
from tqdm import tqdm
from .plot import check_sqlite_table

class export_tables:
    def __init__(self, infile, level, id, max_qvalue, min_abs_log2fx, mode, combined, peptide_rank, extra=True):
        self.infile = infile
        self.level = level
        self.id = id
        self.max_qvalue = max_qvalue
        self.min_abs_log2fx = min_abs_log2fx
        self.mode = mode
        self.combined = combined
        self.peptide_rank = peptide_rank
        
        con = connect(infile)
        click.echo("Exporting basic data...")
        self.export_basic_data(con)
        
        if extra:
            click.echo("Exporting extra data...")
            # Set global SEC boundaries
            self.sec_min, self.sec_max = self.read_sec_boundaries(con)
            
            # Read peptide and feature data
            self.feature_data = self.read_features(con)
            self.peptide_data = self.read_peptides(con)
            self.protein_data = self.read_proteins(con)

            # Read meta data if available
            self.interactions_dmeta = self.read_interactions_dmeta(con)
            self.interactions_qmeta = self.read_interactions_qmeta(con)
            
            self.monomer_qmeta = self.read_monomer_qmeta(con)
            
            if self.level == 'interaction' and self.id is not None:
                self.export_interaction(self.id)
            elif self.level == 'bait' and self.id is not None:
                self.export_bait(self.id, 0)
            elif self.level == 'interaction' and self.id is None and self.max_qvalue is not None:
                interaction_ids, result_ids, decoys = self.read_interactions(con)
                click.echo(f"No interaction specified so exporting all {len(interaction_ids)} interactions")
                progress_bar = tqdm(zip(interaction_ids, result_ids, decoys), desc="Interactions", total=len(interaction_ids))
                
                for interaction_id, result_id, decoy in progress_bar:
                    self.export_interaction(interaction_id, result_id, decoy)
            elif self.level == 'bait' and self.id is None and self.max_qvalue is not None:
                bait_ids, result_ids = self.read_baits(con)
                click.echo(f"No bait specified so exporting all {len(bait_ids)} baits")
                progress_bar = tqdm(zip(bait_ids, result_ids), total=len(bait_ids), desc="Baits")
                
                for bait_id, result_id in progress_bar:
                    self.export_bait(bait_id, result_id)
                    
        con.close()
    
    def export_basic_data(self, con):
        outfile_interactions = path.splitext(self.infile)[0] + "_interactions.csv"
        outfile_network = path.splitext(self.infile)[0] + "_network.csv"
        outfile_nodes = path.splitext(self.infile)[0] + "_differential_nodes.csv"
        outfile_nodes_level = path.splitext(self.infile)[0] + "_differential_nodes_level.csv"
        outfile_edges = path.splitext(self.infile)[0] + "_differential_edges.csv"
        outfile_edges_level = path.splitext(self.infile)[0] + "_differential_edges_level.csv"
        outfile_proteins_level = path.splitext(self.infile)[0] + "_differential_proteins_level.csv"
        
        if check_sqlite_table(con, 'FEATURE_SCORED_COMBINED'):
            interaction_data = pd.read_sql('SELECT DISTINCT bait_id, prey_id FROM FEATURE_SCORED_COMBINED WHERE decoy == 0 and qvalue <= %s;' % self.max_qvalue , con)
            interaction_data.to_csv(outfile_interactions, index=False)
        if check_sqlite_table(con, 'FEATURE_SCORED_COMBINED') and check_sqlite_table(con, 'MONOMER_QM'):
            network_data = pd.read_sql('SELECT DISTINCT bait_id, prey_id FROM FEATURE_SCORED_COMBINED WHERE decoy == 0 and qvalue <= %s UNION SELECT DISTINCT bait_id, prey_id FROM MONOMER_QM;' % self.max_qvalue , con)
            network_data.to_csv(outfile_network, index=False)
        if check_sqlite_table(con, 'NODE'):
            node_data = pd.read_sql('SELECT * FROM NODE LEFT OUTER JOIN PROTEIN ON bait_id = protein_id;' , con)
            node_data.sort_values(by=['pvalue']).to_csv(outfile_nodes, index=False)
        if check_sqlite_table(con, 'NODE_LEVEL'):
            node_level_data = pd.read_sql('SELECT * FROM NODE_LEVEL LEFT OUTER JOIN PROTEIN ON bait_id = protein_id;' , con)
            node_level_data.sort_values(by=['pvalue']).to_csv(outfile_nodes_level, index=False)
        if check_sqlite_table(con, 'EDGE'):
            edge_data = pd.read_sql('SELECT * FROM EDGE;' , con)
            edge_data.sort_values(by=['pvalue']).to_csv(outfile_edges, index=False)
        if check_sqlite_table(con, 'EDGE_LEVEL'):
            edge_level_data = pd.read_sql('SELECT * FROM EDGE_LEVEL;' , con)
            edge_level_data.sort_values(by=['pvalue']).to_csv(outfile_edges_level, index=False)
        if check_sqlite_table(con, 'PROTEIN_LEVEL'):
            protein_level_data = pd.read_sql('SELECT * FROM PROTEIN_LEVEL LEFT OUTER JOIN PROTEIN ON bait_id = protein_id;' , con)
            protein_level_data.sort_values(by=['pvalue']).to_csv(outfile_proteins_level, index=False)
            
    
    def export_interaction(self, interaction_id, result_id=000000, decoy=False):
        feature_data = self.feature_data
        protein_data = self.protein_data
        protein_data['picked'] = True
        peptide_data = self.peptide_data

        bait_id, prey_id = interaction_id.split("_")

        feature_data = feature_data[feature_data['interaction_id'] == interaction_id]
        proteins = pd.DataFrame({"protein_id": [bait_id, prey_id]}).drop_duplicates()
        peptide_data = pd.merge(peptide_data, proteins, how='inner', on='protein_id')

        peptide_data = pd.merge(peptide_data, protein_data, on=['condition_id', 'replicate_id', 'protein_id', 'sec_id'], how='left')
        peptide_data.loc[peptide_data['picked'].isnull(), 'picked'] = False

        if decoy:
            peptide_out = os.path.splitext(os.path.basename(self.infile))[0]+"_DECOY_"+interaction_id+"_peptide"+".csv"
            feature_out = os.path.splitext(os.path.basename(self.infile))[0]+"_DECOY_"+interaction_id+"_feature"+".csv"
        else:
            peptide_out = os.path.splitext(os.path.basename(self.infile))[0]+"_"+interaction_id+"_peptide"+".csv"
            feature_out = os.path.splitext(os.path.basename(self.infile))[0]+"_"+interaction_id+"_feature"+".csv"
            
        peptide_data.to_csv(peptide_out, index=False)
        feature_data.to_csv(feature_out, index=False)
    
    def export_bait(self, bait_id, result_id):
        feature_data = self.feature_data
        protein_data = self.protein_data
        protein_data['picked'] = True
        peptide_data = self.peptide_data

        peptide_data = pd.merge(peptide_data, protein_data, on=['condition_id', 'replicate_id', 'protein_id', 'sec_id'], how='left')
        peptide_data.loc[peptide_data['picked'].isnull(), 'picked'] = False

        interaction_data = self.interactions_dmeta[['bait_id','prey_id','interaction_id']].drop_duplicates()
        interaction_data = interaction_data[(interaction_data['bait_id'] == bait_id) | (interaction_data['prey_id'] == bait_id)]

        # Add monomer
        interaction_data = pd.concat([pd.DataFrame({'bait_id': [bait_id], 'prey_id': [bait_id], 'interaction_id': [bait_id + "_" + bait_id]}), interaction_data], sort=False)
        peptide_dfs = []
        feature_dfs = []
        for idx, interaction in interaction_data.iterrows():
            feature_data_int = feature_data[feature_data['interaction_id'] == interaction['interaction_id']].copy()
            
            if feature_data_int.shape[0] > 0:
                proteins_int = pd.DataFrame({"protein_id": pd.concat([feature_data_int['bait_id'], feature_data_int['prey_id']])}).drop_duplicates()
                peptide_data_int = pd.merge(peptide_data, proteins_int, how='inner', on='protein_id') 
                peptide_data_int['picked'] = peptide_data_int['picked'].astype('bool')
                
                prey_id = feature_data_int['prey_id'].values[0]
                interaction_id = bait_id + "_" + prey_id
                peptide_data_int['interaction_id'] = interaction_id
                feature_data_int['interaction_id'] = interaction_id
                
                peptide_dfs.append(peptide_data_int)
                feature_dfs.append(feature_data_int)
                
            elif interaction['interaction_id'] == (bait_id + "_" + bait_id):
                peptide_data_int = peptide_data[peptide_data['protein_id'] == bait_id].copy()
                peptide_data_int['picked'] = peptide_data_int['picked'].astype('bool')
                
                interaction_id = bait_id + "_" + bait_id
                peptide_data_int['interaction_id'] = interaction_id
                feature_data_int['interaction_id'] = interaction_id
                
                peptide_dfs.append(peptide_data_int)
                feature_dfs.append(feature_data_int)
        
        peptide_df = pd.concat(peptide_dfs)
        feature_df = pd.concat(feature_dfs)
        peptide_out = os.path.splitext(os.path.basename(self.infile))[0]+"_"+bait_id+"_peptide"+".csv"
        feature_out = os.path.splitext(os.path.basename(self.infile))[0]+"_"+bait_id+"_feature"+".csv"                
        
        peptide_df.to_csv(peptide_out, index=False)
        feature_df.to_csv(feature_out, index=False)
    
    def read_sec_boundaries(self, con):
        df = pd.read_sql('SELECT min(sec_id) AS min_sec_id, max(sec_id) AS max_sec_id FROM SEC;', con)
        return df['min_sec_id'].values[0], df['max_sec_id'].values[0]

    def read_features(self, con):
        df = pd.read_sql(
            """
            SELECT *, 
            condition_id || "_" || replicate_id AS tag, 
            bait_id || "_" || prey_id AS interaction_id 
            FROM FEATURE_SCORED;
            """, 
            con
        )
        return df

    def read_proteins(self, con):
        df = pd.read_sql('SELECT * FROM PROTEIN_PEAKS;', con)
        return df

    def read_peptides(self, con):
        df = pd.read_sql(
            """
            SELECT 
                SEC.condition_id || "_" || SEC.replicate_id AS tag, 
                SEC.condition_id, SEC.replicate_id,
                SEC.sec_id, 
                QUANTIFICATION.protein_id, 
                QUANTIFICATION.peptide_id, 
                peptide_intensity, 
                MONOMER.sec_id AS monomer_sec_id 
            FROM QUANTIFICATION 
            INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id 
            INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id 
            INNER JOIN SEC ON QUANTIFICATION.RUN_ID = SEC.RUN_ID 
            INNER JOIN MONOMER ON QUANTIFICATION.protein_id = MONOMER.protein_id AND SEC.condition_id = MONOMER.condition_id AND SEC.replicate_id = MONOMER.replicate_id 
            WHERE peptide_rank <= %s;
            """ % (self.peptide_rank), 
            con
        )
        return df

    def read_interactions(self, con):
        if self.combined:
            table = 'EDGE'
        else:
            table = 'EDGE_LEVEL'

        if check_sqlite_table(con, 'EDGE') and self.mode == 'quantitative':
            df = pd.read_sql('SELECT DISTINCT bait_id || "_" || prey_id AS interaction_id, 0 as decoy FROM %s WHERE pvalue_adjusted < %s AND abs_log2fx > %s ORDER BY pvalue ASC;' % (table, self.max_qvalue, self.min_abs_log2fx), con)
        elif self.mode == 'detection':
            df = pd.read_sql('SELECT DISTINCT bait_id || "_" || prey_id AS interaction_id, decoy FROM FEATURE_SCORED_COMBINED WHERE qvalue < %s GROUP BY bait_id, prey_id ORDER BY qvalue ASC;' % (self.max_qvalue), con)
        else:
            sys.exit("Error: Mode for interaction plotting not supported.")

        return df['interaction_id'].values, df.index+1, df['decoy'].values

    def read_interactions_dmeta(self, con):
        if check_sqlite_table(con, 'COMPLEX_QM') and (self.mode == 'quantitative'):
            df = pd.read_sql('SELECT FEATURE_SCORED_COMBINED.bait_id AS bait_id, FEATURE_SCORED_COMBINED.prey_id AS prey_id, FEATURE_SCORED_COMBINED.bait_id || "_" || FEATURE_SCORED_COMBINED.prey_id AS interaction_id, BAIT_META.protein_name AS bait_name, PREY_META.protein_name AS prey_name, min(FEATURE_SCORED_COMBINED.pvalue) AS pvalue, min(FEATURE_SCORED_COMBINED.qvalue) AS qvalue FROM FEATURE_SCORED_COMBINED INNER JOIN (SELECT * FROM PROTEIN) AS BAIT_META ON FEATURE_SCORED_COMBINED.bait_id = BAIT_META.protein_id INNER JOIN (SELECT * FROM PROTEIN) AS PREY_META ON FEATURE_SCORED_COMBINED.prey_id = PREY_META.protein_id INNER JOIN (SELECT DISTINCT bait_id, prey_id FROM COMPLEX_QM) AS COMPLEX_QM ON FEATURE_SCORED_COMBINED.bait_id = COMPLEX_QM.bait_id AND FEATURE_SCORED_COMBINED.prey_id = COMPLEX_QM.prey_id GROUP BY FEATURE_SCORED_COMBINED.bait_id, FEATURE_SCORED_COMBINED.prey_id;', con)
        elif self.mode == 'detection':
            df = pd.read_sql('SELECT FEATURE_SCORED_COMBINED.bait_id AS bait_id, FEATURE_SCORED_COMBINED.prey_id AS prey_id, FEATURE_SCORED_COMBINED.bait_id || "_" || FEATURE_SCORED_COMBINED.prey_id AS interaction_id, BAIT_META.protein_name AS bait_name, PREY_META.protein_name AS prey_name, min(FEATURE_SCORED_COMBINED.pvalue) AS pvalue, min(FEATURE_SCORED_COMBINED.qvalue) AS qvalue FROM FEATURE_SCORED_COMBINED INNER JOIN (SELECT * FROM PROTEIN) AS BAIT_META ON FEATURE_SCORED_COMBINED.bait_id = BAIT_META.protein_id INNER JOIN (SELECT * FROM PROTEIN) AS PREY_META ON FEATURE_SCORED_COMBINED.prey_id = PREY_META.protein_id GROUP BY FEATURE_SCORED_COMBINED.bait_id, FEATURE_SCORED_COMBINED.prey_id;', con)
        else:
            df = None

        return df

    def read_interactions_qmeta(self, con):
        df = None

        if check_sqlite_table(con, 'EDGE'):
            df = pd.read_sql('SELECT condition_1, condition_2, bait_id, prey_id, pvalue, pvalue_adjusted, level, bait_id || "_" || prey_id AS interaction_id FROM EDGE_LEVEL;', con)
        else:
            df = None

        return df

    def read_monomer_qmeta(self, con):
        df = None

        if check_sqlite_table(con, 'NODE'):
            df = pd.read_sql('SELECT condition_1, condition_2, bait_id, pvalue, pvalue_adjusted, level FROM NODE_LEVEL;', con)
        else:
            df = None

        return df

    def read_baits(self, con):
        if not check_sqlite_table(con, 'NODE'):
            sys.exit("Error: Your experimental design is not supported. At least two conditions are necessary for differential analysis. Switch 'level' to 'interaction' for visualization.")

        if self.combined:
            table = 'NODE'
        else:
            table = 'NODE_LEVEL'

        if self.mode == 'quantitative':
            df = pd.read_sql('SELECT DISTINCT bait_id, min(pvalue) as pvalue FROM %s WHERE pvalue_adjusted < %s AND abs_log2fx > %s GROUP BY bait_id;' % (table, self.max_qvalue, self.min_abs_log2fx), con)

        df = df.sort_values(by=['pvalue']).reset_index()

        return df['bait_id'].values, df.index+1
