import logging
import numpy as np
import pandas as pd
from collections import defaultdict


class NodeDetectionEvaluator:

    def __init__(self, nodes_truth, nodes_predict, tolerance, counts=False, selection=None, unmatched=False, ignore=False):
        self.nodes_truth = nodes_truth
        self.nodes_predict = nodes_predict
        self.tolerance = tolerance
        self.counts = counts
        self.unmatched = unmatched
        self.logger = logging.getLogger()
        self.matched = 0
        self.mislabeled = 0
        self.misclassified = 0
        self.false = 0
        self.missed = 0
        self.selection = selection
        self.unmatched_counts = {}
        self.matched_errors = defaultdict(list)
        self.ignore_missing = ignore

    def _reset_counters(self):
        self.matched = 0
        self.mislabeled = 0
        self.misclassified = 0
        self.false = 0
        self.missed = 0
        self.unmatched_counts = {}
        
        
    def evaluate(self, object_id):
        object_truth = self.nodes_truth.df[(self.nodes_truth.df['ObjectID'] == object_id) & \
                          (self.nodes_truth.df['Direction'] != 'ES')].copy()
        object_predict = self.nodes_predict.df[(self.nodes_predict.df['ObjectID'] == object_id) & \
                                    (self.nodes_predict.df['Direction'] != 'ES')].copy()
        if self.selection:
            object_truth = object_truth[object_truth.Direction == self.selection].copy()
            object_predict = object_predict[object_predict.Direction == self.selection].copy()
        object_predict['matched'] = False
        object_predict['classification'] = None
        object_predict['distance'] = None
        object_truth['classification'] = None
        object_truth['distance'] = None
        count_matched = 0
        count_mislabeled = 0
        count_misclassified = 0
        count_missed = 0
        count_false = 0

        for gt_idx, gt_row in object_truth.iterrows():
            self.logger.debug(f"Evaluating node: object={object_id}, index={gt_row['TimeIndex']}, direction={gt_row['Direction']}, type={gt_row['Node']}")
            matching_participant_events = object_predict[
                (object_predict['matched'] == False) &
                (object_predict['TimeIndex'] >= gt_row['TimeIndex'] - self.tolerance) &
                (object_predict['TimeIndex'] <= gt_row['TimeIndex'] + self.tolerance) &
                (object_predict['Direction'] == gt_row['Direction'])
            ]

            if len(matching_participant_events) > 0:
                self.logger.debug(f'Found {len(matching_participant_events)} matching predict nodes')
                p_idx = matching_participant_events.index[0]
                p_row = matching_participant_events.iloc[0]
                distance = p_row['TimeIndex'] - gt_row['TimeIndex']
                self.logger.debug(f'Using first matching event with time index error of {distance}')
                if p_row['Node'] == gt_row['Node'] and p_row['Type'] == gt_row['Type']:
                    self.logger.debug('Matching event is correct.')
                    count_matched += 1
                    object_truth.loc[gt_idx, 'classification'] = 'TP'
                    object_truth.loc[gt_idx, 'distance'] = distance
                    object_predict.loc[p_idx, 'classification'] = 'TP'
                    object_predict.loc[p_idx, 'distance'] = distance
                    self.matched_errors[gt_row['Node']].append(distance)
                else:
                    if p_row['Node'] != gt_row['Node']:
                        self.logger.debug(f"Matching event is incorrect node type: truth {gt_row['Node']} vs predict {p_row['Node']}")
                        count_mislabeled += 1
                    else:
                        self.logger.debug(f"Matching event is incorrect propulsion type: truth {gt_row['Type']} vs predict {p_row['Type']}")
                        count_misclassified += 1
                    object_truth.loc[gt_idx, 'classification'] = 'FP'
                    object_truth.loc[gt_idx, 'distance'] = distance
                    object_predict.loc[p_idx, 'classification'] = 'FP'
                    object_predict.loc[p_idx, 'distance'] = distance
                object_predict.loc[matching_participant_events.index[0], 'matched'] = True
            else:
                self.logger.debug(f'No matching event found.')
                if self.unmatched:
                    self.logger.info(f'Object {object_id} {gt_row["Direction"]} truth node type {gt_row["Node"]} at index {gt_row["TimeIndex"]} is unmatched.')
                self.unmatched_counts[gt_row["Node"]] = self.unmatched_counts.get(gt_row["Node"], 0) + 1
                count_missed += 1
                object_truth.loc[gt_idx, 'classification'] = 'FN'
                
        additional_fp = object_predict[~object_predict['matched']].copy()
        self.logger.debug(f'There were {len(additional_fp)} unmatched predicted events.')
        count_false = len(additional_fp)
        object_predict.loc[additional_fp.index, 'classification'] = 'FP'

        if self.counts:
            self.logger.info(f'Object {object_id} Counts: matched={count_matched}, mislabeled={count_mislabeled}, misclassified={count_misclassified}, missed={count_missed}, false={count_false}')

        self.matched += count_matched
        self.mislabeled += count_mislabeled
        self.misclassified += count_misclassified
        self.false += count_false
        self.missed += count_missed

        tp = count_matched
        fp = count_false + count_mislabeled + count_misclassified
        fn = count_missed
 
        distances = object_predict[object_predict['classification'] == 'TP']['distance'].tolist()

        entry = {
            'objectid': object_id,
            'match': count_matched,
            'mislabel': count_mislabeled,
            'misclass': count_misclassified,
            'miss': count_missed,
            'false': count_false,
        }
 
        return tp, fp, fn, object_truth, object_predict, distances, entry
    
    def score(self, output_file=None, group=None):

        # reset counters
        self._reset_counters()

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_distances = []
        entries = []

        object_ids = self.nodes_truth.df['ObjectID'].unique().tolist()
        object_ids.sort()
        predict_ids = set(self.nodes_predict.df['ObjectID'].unique().tolist())
        for object_id in object_ids:
            if group and object_id not in group:
                continue
            if object_id not in predict_ids and self.ignore_missing:
                continue
            tp, fp, fn, _, object_predict, distances, entry = self.evaluate(object_id)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_distances.extend(distances)
            entries.append(entry)

        detections = self.matched + self.mislabeled + self.misclassified
        events = detections + self.missed
        pd = detections / events
        pfa = self.false / events
        
        self.logger.info(f'Total Counts: matched={self.matched}, mislabeled={self.mislabeled}, misclassified={self.misclassified}')
        self.logger.info(f'Total Counts: missed={self.missed}, false={self.false}')
        self.logger.info(f'Total Counts: TP={total_tp}, FP={total_fp}, FN={total_fn}')
        for key, value in self.matched_errors.items():
            distances = np.array(value)
            self.logger.info(f'Matched Distances: Type:{key} Mean:{distances.mean():.2f} STD:{distances.std():.2f}  Num:{len(distances)}')
        for key, value in self.unmatched_counts.items():
            self.logger.info(f'Unmatched Counts: {key}: {value}')
        
        precision = total_tp / (total_tp + total_fp) \
            if (total_tp + total_fp) != 0 else 0
        recall = total_tp / (total_tp + total_fn) \
            if (total_tp + total_fn) != 0 else 0
        f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
            if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0
        rmse = np.sqrt((sum(d ** 2 for d in total_distances) / len(total_distances))) if total_distances else 0
        
        if output_file:
            df = pd.DataFrame(entries)
            df.to_csv(output_file)

        return precision, recall, f2, rmse, pd, pfa


    def plot(self, object_id):
        from .plot import plot_timeline
        tp, fp, fn, object_truth, object_predict, distances, _ = self.evaluate(object_id)
        rmse = np.sqrt((sum(d ** 2 for d in distances) / len(distances))) if distances else 0
        title_info = f"Object {object_id}: TPs={tp}, FPs={fp}, FNs={fn}, RMSE={rmse:.2f}"
        plot_timeline(object_truth, object_predict, title_info, self.tolerance)
