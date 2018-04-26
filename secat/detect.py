import pandas as pd
import numpy as np
import sys
import difflib
import csv
import re
import click
import sqlite3

import multiprocessing as mp
import mmh3
from tqdm import tqdm

import pyopenms as po

def prepare(outfile, min_peptides, max_peptides, det_peptides, peak_method, peak_width, signal_to_noise, gauss_width, sgolay_frame_length, sgolay_polynomial_order, sn_win_len, sn_bin_count):

    con = sqlite3.connect(outfile)
    df = pd.read_sql('SELECT DISTINCT condition_id, replicate_id FROM SEC;', con)
    con.close()

    experiments = []
    for idx, exp in enumerate(df.to_dict('records')):
        experiments.append({'idx': idx, 'outfile': outfile, 'condition_id': exp['condition_id'], 'replicate_id': exp['replicate_id'], 'min_peptides': min_peptides, 'max_peptides': max_peptides, 'det_peptides': det_peptides, 'peak_method': peak_method, 'peak_width': peak_width, 'signal_to_noise': signal_to_noise, 'gauss_width': gauss_width, 'sgolay_frame_length': sgolay_frame_length, 'sgolay_polynomial_order': sgolay_polynomial_order, 'sn_win_len': sn_win_len, 'sn_bin_count': sn_bin_count})

    return experiments

def process(exp):
    signalprocess(exp)

class signalprocess:
    def __init__(self, exp):
        self.idx = exp['idx']
        self.outfile = exp['outfile']
        self.condition_id = exp['condition_id']
        self.replicate_id = exp['replicate_id']
        self.min_peptides = exp['min_peptides']
        self.max_peptides = exp['max_peptides']
        self.det_peptides = exp['det_peptides']
        self.peak_method = exp['peak_method']
        self.peak_width = exp['peak_width']
        self.signal_to_noise = exp['signal_to_noise']
        self.gauss_width = exp['gauss_width']
        self.sgolay_frame_length = exp['sgolay_frame_length']
        self.sgolay_polynomial_order = exp['sgolay_polynomial_order']
        self.sn_win_len = exp['sn_win_len']
        self.sn_bin_count = exp['sn_bin_count']

        self.chromatograms, self.protein_index, self.peptide_index = self.generate_chromatograms()

        self.df = self.read()

    def generate_chromatograms(self):
        # Read data
        con = sqlite3.connect(self.outfile)
        df = pd.read_sql('SELECT sec_id, QUANTIFICATION.protein_id, QUANTIFICATION.peptide_id, peptide_rank, peptide_intensity, intensity_bin, sec_min_bin, sec_max_bin FROM QUANTIFICATION INNER JOIN PROTEIN_META ON QUANTIFICATION.protein_id = PROTEIN_META.protein_id INNER JOIN PEPTIDE_META ON QUANTIFICATION.peptide_id = PEPTIDE_META.peptide_id INNER JOIN SEC ON QUANTIFICATION.RUN_ID = SEC.RUN_ID WHERE peptide_count >= %s AND peptide_rank <= %s AND condition_id == "%s" AND replicate_id == "%s";' % (self.min_peptides, self.max_peptides, self.condition_id, self.replicate_id), con)
        con.close()

        protein_index = df['protein_id'].drop_duplicates().tolist()
        peptide_index = df['peptide_id'].drop_duplicates().tolist()

        # Generate Chromatograms
        sec_min = df['sec_id'].min()
        sec_max = df['sec_id'].max()

        datg = df.groupby('peptide_id')

        chromatograms = {}

        for name, group in datg:
            c = po.MSChromatogram()
            c.setName(str(name))
            c.setNativeID(str(name))
            c.setMetaValue("precursor_mz", protein_index.index(group['protein_id'].tolist()[0]))
            c.setMetaValue("product_mz", peptide_index.index(group['peptide_id'].tolist()[0]))
            group_idx = group.set_index('sec_id')[['peptide_intensity']].to_dict()
            for i in range(sec_min, sec_max+1):
                p = po.ChromatogramPeak()
                p.setRT(i)
                if set([i,i-1,i+1]).issubset(set(group_idx['peptide_intensity'].keys())) or set([i,i-1,i-2]).issubset(set(group_idx['peptide_intensity'].keys())) or set([i,i+1,i+2]).issubset(set(group_idx['peptide_intensity'].keys())):
                    p.setIntensity(group_idx['peptide_intensity'][i])
                else:
                    p.setIntensity(0)
                c.push_back(p)

            chromatograms[name] = c

        return chromatograms, protein_index, peptide_index

    def convert_feature(self, feature, pepprot):
        bait = re.split("bait_",feature.getMetaValue("PeptideRef"))[1]

        subfeatures = []

        if feature.metaValueExists("id_target_num_transitions"):
            for i in range(int(feature.getMetaValue("id_target_num_transitions"))):
                for prey in pepprot[feature.getMetaValue("id_target_transition_names").split(";")[i]]:

                    if (float(feature.getIntensity()) / float(feature.getMetaValue("nr_peaks"))) > 0:
                        var_stoichiometry_score = float(feature.getMetaValue("id_target_area_intensity").split(";")[i]) / (float(feature.getIntensity()) / float(feature.getMetaValue("nr_peaks")))
                    else:
                        var_stoichiometry_score = 0

                    subfeature = {
                                    "condition_id": self.condition_id,
                                    "replicate_id": self.replicate_id,
                                    "bait_feature_id": mmh3.hash(self.condition_id + "_" + self.replicate_id + "_" + bait + "_" + prey + "_" + str(feature.getUniqueId())) & 0xffffffff,
                                    "feature_id": mmh3.hash(self.condition_id + "_" + self.replicate_id + "_" + bait + "_" + prey + "_" + str(feature.getUniqueId()) + "_" + feature.getMetaValue("id_target_transition_names").split(";")[i]) & 0xffffffff,
                                    "bait_id": bait,
                                    "prey_id": prey,
                                    "decoy": False, # decoy
                                    "RT": feature.getRT(),
                                    "leftWidth": feature.getMetaValue("leftWidth"),
                                    "rightWidth": feature.getMetaValue("rightWidth"),
                                    "bait_sn_ratio": feature.getMetaValue("sn_ratio"),
                                    "bait_xcorr_coelution": feature.getMetaValue("var_xcorr_coelution"),
                                    "bait_xcorr_shape": feature.getMetaValue("var_xcorr_shape"),
                                    "bait_intensity_fraction": feature.getMetaValue("var_intensity_score"),
                                    "bait_log_sn": feature.getMetaValue("var_log_sn_score"),
                                    "bait_elution_model_fit": feature.getMetaValue("var_elution_model_fit_score"),
                                    "bait_intensity": feature.getIntensity(),
                                    "prey_peptide_id": feature.getMetaValue("id_target_transition_names").split(";")[i],
                                    "prey_peptide_intensity": float(feature.getMetaValue("id_target_area_intensity").split(";")[i]),
                                    "prey_peptide_total_intensity": float(feature.getMetaValue("id_target_total_area_intensity").split(";")[i]),
                                    "prey_peptide_total_mi": float(feature.getMetaValue("id_target_total_mi").split(";")[i]),
                                    "main_var_xcorr_shape_score": float(feature.getMetaValue("id_target_ind_xcorr_shape").split(";")[i]),
                                    "var_xcorr_coelution_score": float(feature.getMetaValue("id_target_ind_xcorr_coelution").split(";")[i]),
                                    "var_log_sn_score": float(feature.getMetaValue("id_target_ind_log_sn_score").split(";")[i]),
                                    "var_mi_score": float(feature.getMetaValue("id_target_ind_mi_score").split(";")[i]),
                                    "var_mi_ratio_score": float(feature.getMetaValue("id_target_ind_mi_ratio_score").split(";")[i]),
                                    "var_intensity_score": float(feature.getMetaValue("id_target_intensity_score").split(";")[i]),
                                    "var_intensity_ratio_score": float(feature.getMetaValue("id_target_intensity_ratio_score").split(";")[i]),
                                    "var_stoichiometry_score": var_stoichiometry_score
                                    }
                    subfeatures.append(subfeature)
        if feature.metaValueExists("id_decoy_num_transitions"):
            for i in range(int(feature.getMetaValue("id_decoy_num_transitions"))):
                for prey in pepprot[feature.getMetaValue("id_decoy_transition_names").split(";")[i]]:

                    if (float(feature.getIntensity()) / float(feature.getMetaValue("nr_peaks"))) > 0:
                        var_stoichiometry_score = float(feature.getMetaValue("id_decoy_area_intensity").split(";")[i]) / (float(feature.getIntensity()) / float(feature.getMetaValue("nr_peaks")))
                    else:
                        var_stoichiometry_score = 0

                    subfeature = {
                                    "condition_id": self.condition_id,
                                    "replicate_id": self.replicate_id,
                                    "bait_feature_id": mmh3.hash("DECOY_" + self.condition_id + "_" + self.replicate_id + "_" + bait + "_" + prey + "_" + str(feature.getUniqueId())) & 0xffffffff,
                                    "feature_id": mmh3.hash("DECOY_" + self.condition_id + "_" + self.replicate_id + "_" + bait + "_" + prey + "_" + str(feature.getUniqueId()) + "_" + feature.getMetaValue("id_decoy_transition_names").split(";")[i]) & 0xffffffff,
                                    "bait_id": bait,
                                    "prey_id": prey,
                                    "decoy": True, # decoy
                                    "RT": feature.getRT(),
                                    "leftWidth": feature.getMetaValue("leftWidth"),
                                    "rightWidth": feature.getMetaValue("rightWidth"),
                                    "bait_sn_ratio": feature.getMetaValue("sn_ratio"),
                                    "bait_xcorr_coelution": feature.getMetaValue("var_xcorr_coelution"),
                                    "bait_xcorr_shape": feature.getMetaValue("var_xcorr_shape"),
                                    "bait_intensity_fraction": feature.getMetaValue("var_intensity_score"),
                                    "bait_log_sn": feature.getMetaValue("var_log_sn_score"),
                                    "bait_elution_model_fit": feature.getMetaValue("var_elution_model_fit_score"),
                                    "bait_intensity": feature.getIntensity(),
                                    "prey_peptide_id": feature.getMetaValue("id_decoy_transition_names").split(";")[i],
                                    "prey_peptide_intensity": float(feature.getMetaValue("id_decoy_area_intensity").split(";")[i]),
                                    "prey_peptide_total_intensity": float(feature.getMetaValue("id_decoy_total_area_intensity").split(";")[i]),
                                    "prey_peptide_total_mi": float(feature.getMetaValue("id_decoy_total_mi").split(";")[i]),
                                    "main_var_xcorr_shape_score": float(feature.getMetaValue("id_decoy_ind_xcorr_shape").split(";")[i]),
                                    "var_xcorr_coelution_score": float(feature.getMetaValue("id_decoy_ind_xcorr_coelution").split(";")[i]),
                                    "var_log_sn_score": float(feature.getMetaValue("id_decoy_ind_log_sn_score").split(";")[i]),
                                    "var_mi_score": float(feature.getMetaValue("id_decoy_ind_mi_score").split(";")[i]),
                                    "var_mi_ratio_score": float(feature.getMetaValue("id_decoy_ind_mi_ratio_score").split(";")[i]),
                                    "var_intensity_score": float(feature.getMetaValue("id_decoy_intensity_score").split(";")[i]),
                                    "var_intensity_ratio_score": float(feature.getMetaValue("id_decoy_intensity_ratio_score").split(";")[i]),
                                    "var_stoichiometry_score": var_stoichiometry_score
                                    }
                    subfeatures.append(subfeature)

        return pd.DataFrame(subfeatures)

    def score(self,queries):
        baits = set()
        pepprot = {}
        tg = po.LightMRMTransitionGroupCP()
        tg.setTransitionGroupID("bait_"+queries['bait_id'].unique()[0])
        # Append transitions and chromatograms
        for _, tr_it in queries.iterrows():
            tr = po.LightTransition()
            tr.transition_name = str(tr_it['peptide_id'])
            tr.peptide_ref = "bait_"+str(tr_it['bait_id'])
            baits.add("bait_"+str(tr_it['bait_id']))
            tr.precursor_mz = self.protein_index.index(tr_it['prey_id'])
            tr.product_mz = self.peptide_index.index(tr_it['peptide_id'])
            tr.library_intensity = 100
            if tr_it['decoy']:
                tr.decoy = True
                tr.detecting_transition = False
            else:
                tr.decoy = False
                if (tr_it['prey_id'] == tr_it['bait_id'] and tr_it['peptide_rank'] <= int(self.det_peptides)):
                    tr.detecting_transition = True
                else:
                    tr.detecting_transition = False
            tr.identifying_transition = True

            if tr_it['peptide_id'] not in pepprot.keys():
                pepprot[tr_it['peptide_id']] = [tr_it['prey_id']]
            else:
                pepprot[tr_it['peptide_id']].append(tr_it['prey_id'])

            tg.addTransition(tr, tr.transition_name)
            tg.addChromatogram(self.chromatograms[tr.transition_name], tr.transition_name)

        # Generate peptide - protein mapping
        targeted = po.LightTargetedExperiment()
        peptides = []
        proteins = []
        for bait in baits:
            pr_pep = po.LightCompound()
            pr_pep.id = bait
            pr_pep.peptide_group_label = "light"
            pr_pep.charge = 1
            pr_pep.rt = 0
            pr_pep.sequence = "PEPTIDEA"
            pr_pep.protein_refs = [bait]
            peptides.append(pr_pep)
            pr_prot = po.LightProtein()
            pr_prot.id = bait
            proteins.append(pr_prot)

        targeted.compounds = peptides
        targeted.proteins = proteins

        # Create empty files as input and finally as output
        swath_maps_dummy = []

        trafo = po.TransformationDescription()
        features = po.FeatureMap()

        # Set up OpenSwath analyzer (featurefinder) and run
        featurefinder = po.MRMFeatureFinderScoring()
        featurefinder_params = po.MRMFeatureFinderScoring().getDefaults()

        tgpicker = po.MRMTransitionGroupPicker()
        tgpicker_params = po.MRMTransitionGroupPicker().getDefaults()

        peakpicker = po.PeakPickerMRM()
        peakpicker_params = po.PeakPickerMRM().getDefaults()

        peakpicker_params.setValue("sgolay_frame_length", int(self.sgolay_frame_length), 'The number of subsequent data points used for smoothing.\nThis number has to be uneven. If it is not, 1 will be added.')
        peakpicker_params.setValue("sgolay_polynomial_order", int(self.sgolay_polynomial_order), 'Order of the polynomial that is fitted.')
        peakpicker_params.setValue("gauss_width", float(self.gauss_width), 'Gaussian width in seconds, estimated peak size.')
        if self.peak_method == 'gauss':
            peakpicker_params.setValue("use_gauss", "true", 'Use Gaussian filter for smoothing (alternative is Savitzky-Golay filter)')
        elif self.peak_method == 'sgolay':
            peakpicker_params.setValue("use_gauss", "false", 'Use Gaussian filter for smoothing (alternative is Savitzky-Golay filter)')

        peakpicker_params.setValue("peak_width", float(self.peak_width), 'Force a certain minimal peak_width on the data (e.g. extend the peak at least by this amount on both sides) in seconds. -1 turns this feature off.')
        peakpicker_params.setValue("signal_to_noise", float(self.signal_to_noise), 'Signal-to-noise threshold at which a peak will not be extended any more. Note that setting this too high (e.g. 1.0) can lead to peaks whose flanks are not fully captured.')
        peakpicker_params.setValue("sn_win_len", float(self.sn_win_len), 'Signal to noise window length.')
        peakpicker_params.setValue("sn_bin_count", int(self.sn_bin_count), 'Signal to noise bin count.')

        tgpicker_params.insert("PeakPickerMRM:",peakpicker_params)
        tgpicker.setParameters(tgpicker_params)

        featurefinder_params.insert("TransitionGroupPicker:",tgpicker_params)

        featurefinder_params.setValue("Scores:use_uis_scores",'true', '')
        featurefinder_params.setValue("Scores:use_rt_score",'false', '')
        featurefinder_params.setValue("Scores:use_library_score",'false', '')

        featurefinder_params.setValue("uis_threshold_sn", -1, '')
        featurefinder_params.setValue("uis_threshold_peak_area", 0, '')

        featurefinder.setParameters(featurefinder_params);
        featurefinder.prepareProteinPeptideMaps_(targeted)

        tgpicker.pickTransitionGroup(tg);
        results = []
        if len(tg.getFeatures()) > 0 and len(tg.getTransitions()) > 0 and len(tg.getChromatograms()) > 0:
            featurefinder.scorePeakgroups(tg, trafo, swath_maps_dummy, features, False)

            for feature in features:
                results.append(self.convert_feature(feature, pepprot))

            con = sqlite3.connect(self.outfile)
            pd.concat(results).to_sql('FEATURE', con, index=False, if_exists='append')
            con.close()

    def read(self):
        con = sqlite3.connect(self.outfile)
        netdata = pd.read_sql('SELECT bait_id, QUERY.prey_id AS prey_id, decoy, PEPTIDES.peptide_id AS peptide_id, peptide_rank FROM QUERY INNER JOIN (SELECT DISTINCT protein_id AS prey_id, peptide_id FROM QUANTIFICATION INNER JOIN SEC ON QUANTIFICATION.run_id = SEC.run_id WHERE condition_id == "%s" and replicate_id == "%s") AS PEPTIDES ON QUERY.prey_id == PEPTIDES.prey_id INNER JOIN PEPTIDE_META ON PEPTIDES.peptide_id = PEPTIDE_META.peptide_id INNER JOIN PROTEIN_META ON QUERY.prey_id == PROTEIN_META.protein_id WHERE peptide_count >= %s AND peptide_rank <= %s;' % (self.condition_id, self.replicate_id, self.min_peptides, self.max_peptides), con)
        con.close()

        # Generate MRMTransitionGroup and compute scores
        netgroups = netdata.drop_duplicates().groupby('bait_id')

        for _, gr_it in tqdm(netdata.drop_duplicates().groupby('bait_id'), desc=self.condition_id + "_" + self.replicate_id, position=self.idx):
            res = self.score(gr_it)
