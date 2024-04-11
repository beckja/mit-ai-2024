import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .data import SatelliteTimeSeriesData, Node, NodeDetectionSet, TruthNodeDetectionSet
from .eval import NodeDetectionEvaluator


class Processor:

    def __init__(self, cfg):
        self.logger = logging.getLogger()
        self.trace_object = cfg.getint('trace_object', fallback=None)
        self.object_range = cfg.get('object_range', fallback=None)
        if self.object_range:
            tmp = self.object_range.split('-')
            self.object_range = (int(tmp[0]), int(tmp[1]))
        self.ew_detector = None
        self.ns_detector = None
        self.ew_failed_list = []
        self.ns_failed_list = []

    def run(self, datapath):

        # Searching for training data within the dataset folder
        satdatafiles = Path(datapath).glob('*.csv')
        if not satdatafiles:
            raise ValueError(f'No csv files found in {datapath}')
        
        nodes_all = []
        for satdatafile in satdatafiles:

            # retrieve and prepare object data
            satdata = SatelliteTimeSeriesData(satdatafile)
            if self.trace_object and satdata.object_id != self.trace_object:
                continue
            if self.object_range and (satdata.object_id < self.object_range[0] or satdata.object_id > self.object_range[1]):
                continue

            self.logger.info(f'Processing data file {satdatafile}...')
            nodes_object = self.predict_object_nodes(satdata)
            if len(nodes_object) == 0:
                continue

            nodes_all.append(nodes_object)
        
        if self.ew_detector:
            for key, value in self.ew_detector.alg_counts.items():
                self.logger.info(f'EW Detector Algorithm Counts: {key:30s} = {value}')
            wide_deadband_objects = list(self.ew_detector.wide_deadband_objects)
            wide_deadband_objects.sort()
            self.logger.info(f'Wide Deadband Objects: {wide_deadband_objects}')
        
        if self.ns_detector:
            for key, value in self.ns_detector.alg_counts.items():
                self.logger.info(f'NS Detector Algorithm Counts: {key:30s} = {value}')

        self.logger.info(f'Failed EW Processing Objects: {self.ew_failed_list}')
        self.logger.info(f'Failed NS Processing Objects: {self.ns_failed_list}')

        return NodeDetectionSet(pd.concat(nodes_all))


class LearningClassifierModel:

    def __init__(self):
        #self.model_EW = None
        #self.model_NS = None
        #self.le_EW = None
        #self.le_NS = None
        #TODO: move these back into train()?
        self.le_EW = LabelEncoder()
        self.le_NS = LabelEncoder()
        self.model_EW = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_NS = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def load(self, model_path):
        # Load the trained models (don't use the utils module, use pickle)
        self.model_EW = pickle.load(open(model_path / 'model_EW.pkl', 'rb'))
        self.model_NS = pickle.load(open(model_path / 'model_NS.pkl', 'rb'))
        self.le_EW = pickle.load(open(model_path / 'le_EW.pkl', 'rb'))
        self.le_NS = pickle.load(open(model_path / 'le_NS.pkl', 'rb'))

    def dump(self, model_path):
        # Save the trained random forest models (and label encoders) to disk
        # Create the folder trained_model if it doesn't exist
        Path(model_path).mkdir(exist_ok=True)
        pickle.dump(self.model_EW, open(model_path / 'model_EW.pkl', 'wb'))
        pickle.dump(self.model_NS, open(model_path / 'model_NS.pkl', 'wb'))
        pickle.dump(self.le_EW, open(model_path / 'le_EW.pkl', 'wb'))
        pickle.dump(self.le_NS, open(model_path / 'le_NS.pkl', 'wb'))


class LearningProcessor(Processor):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_path = Path(cfg['model_path'])
        self.logger.info(f'Model Path: {self.model_path.absolute()}')
        self.model = LearningClassifierModel()

    def run(self, datapath):

        self.model.load(self.model_path)
        nodes_object = super().run(datapath)

        return nodes_object


class DevkitLearningProcessor(LearningProcessor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.valid_ratio = cfg.getfloat('valid_ratio')
        self.logger.info(f'Valid Ratio: {self.valid_ratio}')
        self.lag_steps = cfg.getint('lag_steps')
        self.logger.info(f'Lag Steps: {self.lag_steps}')

    def predict_object_nodes(self, satdata):
        satdata, updated_feature_cols = self.tabularize_data(satdata)

        # Make predictions on the object test data for EW
        satdata['Predicted_EW'] = self.model.le_EW.inverse_transform(
            self.model.model_EW.predict(satdata[updated_feature_cols])
        )

        # Make predictions on the object test data for NS
        satdata['Predicted_NS'] = self.model.le_NS.inverse_transform(
            self.model.model_NS.predict(satdata[updated_feature_cols])
        )

        nodes_object = self.convert_classifier_output(satdata)
        return nodes_object

    def train(self, datapath):

        self.logger.info(f'Training {type(self).__name__}...')

        # Define the directory paths
        train_data_dir = datapath / 'train'

        # Load the ground truth data
        nodes_truth = TruthNodeDetectionSet(datapath / 'train_labels.csv')

        satdatafiles = Path(train_data_dir).glob('*.csv')
        if not satdatafiles:
            raise ValueError(f'No csv files found in {datapath}')

        # Apply the function to the ground truth data
        self.logger.info('Compiling data tables...')
        frames = []
        for satdatafile in satdatafiles:
            self.logger.info(f'Processing data file {satdatafile}...')
            satdata = SatelliteTimeSeriesData(satdatafile)
            satdata, updated_feature_cols = self.tabularize_data(satdata, nodes_truth)
            frames.append(satdata)
        data = pd.concat(frames)

        # For each ObjectID, show the first rows of the columns TimeIndex, ObjectID, EW, and NS
        #data[['ObjectID', 'TimeIndex' , 'EW', 'NS']].groupby('ObjectID').head(2).head(10)

        # Create a validation set without mixing the ObjectIDs
        self.logger.info(f'Splitting into train and validation sets with valid ratio {self.valid_ratio}...')
        object_ids = data['ObjectID'].unique()
        train_ids, valid_ids = train_test_split(object_ids, 
                                                test_size=self.valid_ratio, 
                                                random_state=42)

        self.logger.info('Copying data into new tables...')
        data_train = data[data['ObjectID'].isin(train_ids)].copy()
        data_valid = data[data['ObjectID'].isin(valid_ids)].copy()

        ground_truth_train = nodes_truth.df[nodes_truth.df['ObjectID'].isin(train_ids)].copy()
        ground_truth_valid = nodes_truth.df[nodes_truth.df['ObjectID'].isin(valid_ids)].copy()

        # Count the number of objects in the training and validation sets
        self.logger.info(
            f"Number of objects in the training set: {len(data_train['ObjectID'].unique())}"
        )
        self.logger.info(
            f"Number of objects in the validation set: {len(data_valid['ObjectID'].unique())}"
        )

        # Next we will make sure that there every label, both in the direction EW and NS,
        # is present both in the training and validation partitions

        # Get the unique values of EW and NS in train and test data
        data_train_EW = set(data_train['EW'].unique())
        data_train_NS = set(data_train['NS'].unique())
        data_valid_EW = set(data_train['EW'].unique())
        data_valid_NS = set(data_valid['NS'].unique())

        # Get the values of EW and NS that are in test data but not in train data
        missing_EW = data_valid_EW.difference(data_train_EW)
        missing_NS = data_valid_NS.difference(data_train_NS)

        # Check if all the values in EW are also present in NS
        if not set(data_train['EW'].unique()).issubset(set(data_train['NS'].unique())):
            # Get the values of EW that are not present in NS
            missing_EW_NS = set(data_train['EW'].unique()).difference(
                set(data_train['NS'].unique())
            )
        else:
            missing_EW_NS = None

        # Print the missing values of EW and NS
        self.logger.info(
            f"Missing values of EW in test data: {missing_EW}"
        )
        self.logger.info(
            f"Missing values of NS in test data: {missing_NS}"
        )
        self.logger.info(
            f"Values of EW not present in NS: {missing_EW_NS}"
        )

        # Convert categorical data to numerical data

        # Encode the 'EW' and 'NS' columns
        data_train['EW_encoded'] = self.model.le_EW.fit_transform(data_train['EW'])
        data_train['NS_encoded'] = self.model.le_NS.fit_transform(data_train['NS'])

        # Fit the model to the training data for EW
        self.model.model_EW.fit(data_train[updated_feature_cols], data_train['EW_encoded'])

        # Fit the model to the training data for NS
        self.model.model_NS.fit(data_train[updated_feature_cols], data_train['NS_encoded'])

        # Make predictions on the training data for EW
        data_train['Predicted_EW'] = self.model.le_EW.inverse_transform(
            self.model.model_EW.predict(data_train[updated_feature_cols])
        )

        # Make predictions on the validation data for NS
        data_train['Predicted_NS'] = self.model.le_NS.inverse_transform(
            self.model.model_NS.predict(data_train[updated_feature_cols])
        )

        # Print the first few rows of the test data with predictions for both EW and NS
        data_train[['TimeIndex', 'ObjectID', 'EW', 
                    'Predicted_EW', 'NS', 'Predicted_NS']].groupby('ObjectID').head(3)

        if self.valid_ratio > 0:
            # Make predictions on the validation data for EW
            data_valid['Predicted_EW'] = self.model.le_EW.inverse_transform(
                self.model.model_EW.predict(data_valid[updated_feature_cols])
            )

            # Make predictions on the validation data for NS
            data_valid['Predicted_NS'] = self.model.le_NS.inverse_transform(
                self.model.model_NS.predict(data_valid[updated_feature_cols])
            )

        # The `NodeDetectionEvaluator` class in the evaluation module allows not only to
        # compute the general score for a given dataset, but get evaluations per object, and
        # even plots that show how the predictions look like in a timeline

        nodes_train = self.convert_classifier_output(data_train)
        evaluator = NodeDetectionEvaluator(
            NodeDetectionSet(ground_truth_train),
            NodeDetectionSet(nodes_train),
            tolerance=6
        )
        precision, recall, f2, rmse = evaluator.score()
        self.logger.info(f'Precision for the train set: {precision:.2f}')
        self.logger.info(f'Recall for the train set: {recall:.2f}')
        self.logger.info(f'F2 for the train set: {f2:.2f}')
        self.logger.info(f'RMSE for the train set: {rmse:.2f}')

        # Plot the evaluation timeline for a random ObjectID from the training set
        evaluator.plot(np.random.choice(data_train['ObjectID'].unique()))

        # Loop over the Object IDs in the training set and call the evaluation
        # function for each object and aggregate the results
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for oid in data_train['ObjectID'].unique():
            tp, fp, fn, gt_object, p_object, distances, _ = evaluator.evaluate(oid)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        self.logger.info(f'Total true positives: {total_tp}')
        self.logger.info(f'Total false positives: {total_fp}')
        self.logger.info(f'Total false negatives: {total_fn}')

        if self.valid_ratio > 0:
            nodes_valid = self.convert_classifier_output(data_valid)
            evaluator = NodeDetectionEvaluator(
                NodeDetectionSet(ground_truth_valid),
                NodeDetectionSet(nodes_valid),
                tolerance=6
            )
        precision, recall, f2, rmse = evaluator.score()
        self.logger.info(f'Precision for the validation set: {precision:.2f}')
        self.logger.info(f'Recall for the validation set: {recall:.2f}')
        self.logger.info(f'F2 for the validation set: {f2:.2f}')
        self.logger.info(f'RMSE for the validation set: {rmse:.2f}')

        # Plot the evaluation timeline for a random ObjectID from the training set
        evaluator.plot(np.random.choice(data_valid['ObjectID'].unique()))

        # Save the trained random forest models (and label encoders) to disk
        # Create the folder trained_model if it doesn't exist
        self.model.dump(self.model_path)

    def tabularize_data(self, satdata, ground_truth=None, fill_na=True):
        '''Prepare the data in a tabular format'''
    
        lagged_features = []
        new_feature_cols = list(satdata.feature_cols)  # Create a copy of feature_cols
        # Create lagged features for each column in feature_cols
        for col in satdata.feature_cols:
            for i in range(1, self.lag_steps+1):
                lag_col_name = f'{col}_lag_{i}'
                satdata.df[lag_col_name] = satdata.df.groupby('ObjectID')[col].shift(i)
                new_feature_cols.append(lag_col_name)  # Add the lagged feature to new_feature_cols
        
        # Add the lagged features to the DataFrame all at once
        satdata.df = pd.concat([satdata.df] + lagged_features, axis=1)

        if ground_truth is None:
            merged_df = satdata.df
        else:
            ground_truth_EW, ground_truth_NS = ground_truth.get_object(satdata.df['ObjectID'][0])

            # Merge the input data with the ground truth
            merged_df = pd.merge(satdata.df, 
                                ground_truth_EW.sort_values('TimeIndex'), 
                                on=['TimeIndex', 'ObjectID'],
                                how='left')
            merged_df = pd.merge_ordered(merged_df, 
                                        ground_truth_NS.sort_values('TimeIndex'), 
                                        on=['TimeIndex', 'ObjectID'],
                                        how='left')

            # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
            merged_df['EW'].ffill(inplace=True)
            merged_df['NS'].ffill(inplace=True)

        if fill_na:
            merged_df.bfill(inplace=True)
            
        return merged_df, new_feature_cols

    def convert_classifier_output(self, classifier_output):

        # Split the 'Predicted_EW' and 'Predicted_NS' columns into 
        # 'Node' and 'Type' columns
        ew_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_EW']].copy()
        ew_df[['Node', 'Type']] = ew_df['Predicted_EW'].str.split('-', expand=True)
        ew_df['Direction'] = 'EW'
        ew_df.drop(columns=['Predicted_EW'], inplace=True)

        ns_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_NS']].copy()
        ns_df[['Node', 'Type']] = ns_df['Predicted_NS'].str.split('-', expand=True)
        ns_df['Direction'] = 'NS'
        ns_df.drop(columns=['Predicted_NS'], inplace=True)

        # Concatenate the processed EW and NS dataframes
        final_df = pd.concat([ew_df, ns_df], ignore_index=True)

        # Sort dataframe based on 'ObjectID', 'Direction' and 'TimeIndex'
        final_df.sort_values(['ObjectID', 'Direction', 'TimeIndex'], inplace=True)

        # Apply the function to each group of rows with the same 'ObjectID' and 'Direction'
        groups = final_df.groupby(['ObjectID', 'Direction'])
        keep = groups[['Node', 'Type']].apply(lambda group: group.shift() != group).any(axis=1)

        # Filter the DataFrame to keep only the rows we're interested in
        keep.index = final_df.index
        final_df = final_df[keep]

        # Reset the index and reorder the columns
        final_df = final_df.reset_index(drop=True)
        final_df = final_df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']]
        final_df = final_df.sort_values(['ObjectID', 'TimeIndex', 'Direction'])

        return final_df


class Update3LearningProcessor(DevkitLearningProcessor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ew_detector = NodeDetector()
    
    def tabularize_data(self, satdata, ground_truth=None, fill_na=True):
        self.ew_detector.fix_longitude_crossovers(satdata.df)
        return super().tabularize_data(satdata, ground_truth, fill_na)


class NodeDetector:

    def __init__(self, cfg):
        self.logger = logging.getLogger()
        self.trace_object = cfg.getint('trace_object', fallback=None)
        self.alg_counts = {}

    @staticmethod
    def fix_longitude_crossovers(df):

        longitudes = df['Longitude (deg)'].copy()

        diff = np.abs(longitudes - longitudes.shift(1))
        rollover = False
        rollovers = diff[np.abs(diff) > 300]

        while len(rollovers) > 0:
            rollover = True
            if longitudes.iloc[rollovers.index[0]] > longitudes.iloc[rollovers.index[0]-1]:
                longitudes[rollovers.index[0]:] -= 360
            else:
                longitudes[rollovers.index[0]:] += 360
            diff = np.abs(longitudes - longitudes.shift(1))
            rollovers = diff[np.abs(diff) > 300]

        # re-center on zero
        if rollover and (longitudes < -180).all():
            longitudes += 360
        elif rollover and (longitudes > 180).all():
            longitudes -= 360
        
        df['Longitude (deg)'] = longitudes

    def tabularize_data(self, satdata):
        
        steps_per_day = satdata.steps_per_day
        df = satdata.df.copy()

        # check for longitude 180 crossover
        self.fix_longitude_crossovers(df)
        longitudes = df['Longitude (deg)']

        # set up longitude and inclination deltas
        df['DeltaLon24h'] = longitudes - longitudes.shift(steps_per_day)
        df['DeltaLon24h'] = df['DeltaLon24h'].bfill()
        inclinations = df['Inclination (deg)']
        df['DeltaIncl2h'] = inclinations - inclinations.shift(1)
        df['DeltaIncl2h'] = df['DeltaIncl2h'].bfill()
        
        return df


class HeuristicEastWestNodeDetector(NodeDetector):

    def __init__(self, cfg):
        self.lon_std_thr = cfg.getfloat('lon_std_thr')
        self.drift_thr = cfg.getfloat('drift_thr')
        self.adjust_factor = cfg.getfloat('adjust_factor')
    
    def tabularize_data(self, satdata):
        
        steps_per_day = satdata.steps_per_day
        df = satdata.df.copy()

        # check for longitude 180 crossover  (this is the update1 fix not in the heuristic classifier)
        # lon_max_24 = df['Longitude (deg)'].rolling(steps_per_day).max()
        # lon_min_24 = df['Longitude (deg)'].rolling(steps_per_day).min()
        # delta = lon_max_24 - lon_min_24
        # if delta.max() > 300:
        #     longitudes = df['Longitude (deg)'].copy()
        #     longitudes[longitudes < 0] += 360
        #     df['Longitude (deg)'] = longitudes
        
        # Get std for longitude over the preceding 24-hour window
        df['Lon STD 24h'] = df['Longitude (deg)'].rolling(steps_per_day).std().shift(1)
        df['Lon STD 24h'].bfill(inplace=True)
        df['Lon STD Max'] = df['Lon STD 24h'].rolling(steps_per_day).max().shift(1)
        df['Lon STD Min'] = df['Lon STD 24h'].rolling(steps_per_day).min().shift(1)
        
        return df

    def run(self, satdata):

        satcat = satdata.object_id
        self.logger.debug(f'Running EW detection on object {satcat}')
        steps_per_day = satdata.steps_per_day
        df = self.tabularize_data(satdata)
        lon_std = df['Lon STD 24h']

        EW_detections = NodeDetectionSet()
        ssEW = Node(
            index=0,
            ntype="SS",
            signal="EW"
        )
        EW_detections.add(satcat, ssEW)

        drifting = False
        for i in range(steps_per_day + 1, len(df) - steps_per_day):              # if at least 1 day has elapsed since t0
            max_lon_std_24h = df['Lon STD Max'][i]
            min_lon_std_24h = df['Lon STD Min'][i]
            avg_lon_std_24h = np.abs(max_lon_std_24h - min_lon_std_24h) / 2
            adjust_thr = self.adjust_factor * avg_lon_std_24h
        
            # ID detection
            if (lon_std[i] > self.lon_std_thr) and not drifting:                    # if sd is elevated & last sd was at baseline
                self.logger.debug(f'Detected drift at time index {i}')
                before = np.mean(df["Longitude (deg)"][i-steps_per_day:i])    # mean of previous day's longitudes
                after = np.mean(df["Longitude (deg)"][i:i+steps_per_day])     # mean of next day's longitudes
                # if not temporary noise, then real ID
                if np.abs(before - after) > self.drift_thr:                           # if means are different
                    drifting = True                                          # the sd is not yet back at baseline
                    if i < steps_per_day + 2:
                        EW_detections.set_type(0, 'NK')
                        self.logger.debug(f'Setting initial node mode to NK at time index{i}')
                    else:
                        self.logger.debug(f'Setting ID node at time index {i}')
                        EW_detections.add(satcat, Node(index=i, ntype='ID', signal='EW'))
            # IK detection
            elif (lon_std[i] <= self.lon_std_thr) & drifting:           # elif sd is not elevated and drift has already been initialized
                self.logger.debug(f'Detected end of drift at time index {i}')
                drift_ended = True                                                    # toggle end-of-drift boolean 
                for j in range(steps_per_day):                                        # for the next day, check...
                    if np.abs(df["Longitude (deg)"][i] - df["Longitude (deg)"][i+j]) > self.drift_thr:       # if the longitude changes from the current value
                        drift_ended = False                                           # the drift has not ended
                        self.logger.debug(f'Rejected end of drift at time index {i}')
                if drift_ended:                                                       # if the drift has ended
                    drifting = False                                           # the sd is back to baseline
                    self.logger.debug(f'Setting IK node at time index {i}')
                    EW_detections.add(satcat, Node(index=i, ntype='IK', signal='EW'))
        
            # Last step
            elif (i == (len(lon_std) - steps_per_day - 1)) & drifting:
                self.logger.debug(f'Setting IK node at last time index {i} to mark end of drift period')
                EW_detections.add(satcat, Node(index=i, ntype='IK', signal='EW'))
        
            # AD detection
            elif ((lon_std[i] - max_lon_std_24h > adjust_thr) or \
                (min_lon_std_24h - lon_std[i] > adjust_thr)) & \
                drifting:          # elif sd is elevated and drift has already been initialized
                self.logger.debug(f'Detected change in drift at time index {i}')
                if i >= steps_per_day + 3:
                    self.logger.debug(f'Setting AD node at time index {i}')
                    EW_detections.add(satcat, Node(index=i, ntype='AD', signal='EW'))
                else:
                    self.logger.debug(f'Rejected drift adjust -- too close to start')

        es = Node(
            index=len(df),
            ntype="ES",
            signal="ES",
            mtype="ES"
        )
        EW_detections.add(satdata.object_id, es)
        
        EW_filtered_detections = self.filter_EW_nodes(satdata, EW_detections)

        return EW_filtered_detections
    
    def filter_EW_nodes(self, satdata, EW_detections):

        EW_filtered_detections = NodeDetectionSet()
        EW_filtered_detections.add(EW_detections.df.iloc[0])  # ss node

        steps_per_day = satdata.steps_per_day

        df = self.tabularize_data(satdata)

        IK_detections = EW_detections.df[EW_detections.df.Node == 'IK']
        ID_detections = EW_detections.df[EW_detections.df.Node == 'ID']
        AD_detections = EW_detections.df[EW_detections.df.Node == 'AD']

        if len(IK_detections) == 1:
            if len(ID_detections) == 1:
                # keep the current ID
                EW_filtered_detections.add(ID_detections.iloc[0])
                index_range = [ID_detections.iloc[0].TimeIndex, IK_detections.iloc[0].TimeIndex]
            else:
                index_range = [0, IK_detections.iloc[0].TimeIndex]
            add_next_ad = True
            if len(AD_detections) == 1:
                EW_filtered_detections.add(AD_detections.iloc[0])
            elif len(AD_detections) == 0:
                pass
            else:
                for j in range(len(AD_detections)):
                    ad = AD_detections.iloc[j]
                    ad_next = AD_detections.iloc[j+1] if j < (len(AD_detections)-1) else None
                    if ad.TimeIndex > index_range[0] and ad.TimeIndex < index_range[1]:
                        if add_next_ad & (ad_next is not None):
                            if ad_next.TimeIndex - ad.TimeIndex > steps_per_day:   # more than 24 hours
                                EW_filtered_detections.add(ad)
                            else:
                                EW_filtered_detections.add(ad)
                                add_next_ad = False
                        elif add_next_ad & (ad_next is None):
                            EW_filtered_detections.add(ad)
                        elif (not add_next_ad) & (ad_next is not None):
                            if ad_next.TimeIndex - ad.TimeIndex > steps_per_day:    # more than 24 hours
                                add_next_ad = True
            if IK_detections.iloc[0].TimeIndex != (len(df)-steps_per_day-1):  # not in final 24 hours
                EW_filtered_detections.add(IK_detections.iloc[0])
        
        toggle = True
        for i in range(len(IK_detections)-1):                                 # for each longitudinal shift detection
            if toggle:                                                            # if the last ID was not discarded
                if ID_detections.iloc[i+1].TimeIndex - IK_detections.iloc[i].TimeIndex > 1.5 * steps_per_day:# if the time between the current IK & next ID is longer than 36 hours
                    # keep the current ID
                    # keep the current IK
                    index_range = [ID_detections.iloc[i].TimeIndex, IK_detections.iloc[i].TimeIndex]
                    EW_filtered_detections.add(ID_detections.iloc[i])
                    add_next_ad = True
                    for j in range(len(AD_detections)):
                        ad = AD_detections.iloc[j]
                        ad_next = AD_detections.iloc[j+1] if j < (len(AD_detections)-1) else None
                        if ad.TimeIndex > index_range[0] and ad.TimeIndex < index_range[1]:
                            if add_next_ad & (ad_next is not None):
                                if ad_next.TimeIndex - ad.TimeIndex > steps_per_day:   # more than 24 hours apart
                                    EW_filtered_detections.add(ad)
                                else:
                                    EW_filtered_detections.add(ad)
                                    add_next_ad = False
                            elif add_next_ad & (ad_next is None):
                                EW_filtered_detections.add(ad)
                            elif (not add_next_ad) & (ad_next is not None):
                                if ad_next.TimeIndex - ad.TimeIndex > steps_per_day:   # more than 24 hours apart
                                    add_next_ad = True
                    if IK_detections.iloc[0].TimeIndex != (
                        len(df)-steps_per_day-1):           # not in final 24 hours
                        EW_filtered_detections.add(IK_detections.iloc[0])
                    if i == len(IK_detections)-2:                             # if the next drift is the last drift
                        # keep the next ID
                        EW_filtered_detections.add(ID_detections.iloc[i+1])
                        add_next_ad = True
                        for j in range(len(AD_detections)):
                            ad = AD_detections.iloc[j]
                            ad_next = AD_detections.iloc[j+1] if j < (len(AD_detections)-1) else None
                            if ad.TimeIndex > index_range[0] and ad.TimeIndex < index_range[1]:
                                if add_next_ad & (ad_next is not None):
                                    if ad_next.TimeIndex - ad.TimeIndex > steps_per_day:   # more than 24 hours apart
                                        EW_filtered_detections.add(ad)
                                    else:
                                        EW_filtered_detections.add(ad)
                                        add_next_ad = False
                                elif add_next_ad & (ad_next is None):
                                    EW_filtered_detections.add(ad)
                                elif (not add_next_ad) & (ad_next is not None):
                                    if ad_next.TimeIndex - ad.TimeIndex > steps_per_day:    # more than 24 hours apart
                                        add_next_ad = True
                        if IK_detections.iloc[i].TimeIndex != (
                            len(df)-steps_per_day-1):         # not in final 24 hours
                            # keep the next IK
                            EW_filtered_detections.add(IK_detections.iloc[i])
                else:                                                             # if the next ID and the current IK are 48 hours apart or less
                    index_range = [ID_detections.iloc[i].TimeIndex, IK_detections.iloc[i+1].TimeIndex]
                    EW_filtered_detections.add(ID_detections.iloc[i])
                    add_next_ad = True
                    for j in range(len(AD_detections)):
                        ad = AD_detections.iloc[j]
                        ad_next = AD_detections.iloc[j+1] if j < (len(AD_detections)-1) else None
                        if ad.TimeIndex > index_range[0] and ad.TimeIndex < index_range[1]:
                            if add_next_ad & (ad_next is not None):
                                if ad_next.TimeIndex - ad.TimeIndex > steps_per_day:       # more than 24 hours apart
                                    EW_filtered_detections.add(ad)
                                else:
                                    EW_filtered_detections.add(ad)
                                    add_next_ad = False
                            elif add_next_ad & (ad_next is None):
                                EW_filtered_detections.add(ad)
                            elif (not add_next_ad) & (ad_next is not None):
                                if ad_next.TimeIndex - ad.TimeIndex > steps_per_day:     # more than 24 hours apart
                                    add_next_ad = True
                    if IK_detections.iloc[i+1] != (
                        len(df)-steps_per_day-1):
                        EW_filtered_detections.add(IK_detections.iloc[i+1])
                    toggle = False                                                # skip the redundant drift
            else:
                toggle = True
        EW_filtered_detections.add(EW_detections.df.iloc[-1])
    
        return EW_filtered_detections


class Update1EastWestNodeDetector(HeuristicEastWestNodeDetector):
    
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def tabularize_data(self, satdata):
        
        steps_per_day = satdata.steps_per_day
        df = satdata.df.copy()

        # check for longitude 180 crossover  (this is the update1 fix not in the heuristic classifier)
        lon_max_24 = df['Longitude (deg)'].rolling(steps_per_day).max()
        lon_min_24 = df['Longitude (deg)'].rolling(steps_per_day).min()
        delta = lon_max_24 - lon_min_24
        if delta.max() > 300:
            longitudes = df['Longitude (deg)'].copy()
            longitudes[longitudes < 0] += 360
            df['Longitude (deg)'] = longitudes
        
        # Get std for longitude over the preceding 24-hour window
        df['Lon STD 24h'] = df['Longitude (deg)'].rolling(steps_per_day).std().shift(1)
        df['Lon STD 24h'].bfill(inplace=True)
        df['Lon STD Max'] = df['Lon STD 24h'].rolling(steps_per_day).max().shift(1)
        df['Lon STD Min'] = df['Lon STD 24h'].rolling(steps_per_day).min().shift(1)
        
        return df
    

class Update2EastWestNodeDetector(HeuristicEastWestNodeDetector):

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def tabularize_data(self, satdata):

        df = satdata.df.copy()
        steps_per_day = satdata.steps_per_day

        # check for longitude crossovers
        self.fix_longitude_crossovers(df)
        
        # Get std for longitude over the preceding 24-hour window
        df['Lon STD 24h'] = df['Longitude (deg)'].rolling(steps_per_day).std().shift(1)
        df['Lon STD 24h'].bfill(inplace=True)
        df['Lon STD Max'] = df['Lon STD 24h'].rolling(steps_per_day).max().shift(1)
        df['Lon STD Min'] = df['Lon STD 24h'].rolling(steps_per_day).min().shift(1)
        
        return df


class HeuristicNorthSouthNodeDetector(NodeDetector):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.XIPS_inc_per_day = 0.0005  #TODO: move to config file
        self.didt_threshold = 5.5e-7

    def run(self, satdata, EW_detections):

        NS_detections = NodeDetectionSet()

        EW_indices = list(EW_detections.df.TimeIndex.values)

        steps_per_day = satdata.steps_per_day
        step_size_hours = int(24 / steps_per_day)
        df = satdata.df
        inclinations = df["Inclination (deg)"]

        ssNS = Node(
            index=0,
            ntype="SS",
            signal="NS"
        )
        NS_detections.add(satdata.object_id, ssNS)
        NS_detections.set_type(0, 'NK')

        # loop through EW node transitions
        for node_index in range(len(EW_indices)-1):

            # if first node is 'NK' (i.e. we're drifting), skip
            #NOTE: this appears to assume NS station-keeping never occurs when drifting EW (probably true)
            if EW_detections.df.iloc[node_index].Type == 'NK':
                continue

            data_indices = EW_indices[node_index:node_index+2]
            sk_mode = EW_detections.df.iloc[node_index].Type
            end_of_sk = EW_detections.df.iloc[node_index+1].Node == 'ID'
            self.logger.debug(f'indices={data_indices} sk_mode={sk_mode} end_of_sk={end_of_sk}')

            first = True if data_indices[0] == 0 else False
            dexs = []
            mode_incs = inclinations[data_indices[0]:data_indices[1]].to_numpy()
            mode_steps = data_indices[1] - data_indices[0]
            rate = (steps_per_day / mode_steps) * (np.max(mode_incs) - np.min(mode_incs))  # deg / day

            # if change in inclination is slower than min XIPS rate and period starts in the first 24 hours but ends outside it
            if (rate < self.XIPS_inc_per_day) and (data_indices[0] < steps_per_day) and (data_indices[1] > steps_per_day):

                # if the next EW node is ID set next NS to ID
                if end_of_sk:
                    n = Node(
                        index=data_indices[1],
                        ntype="ID",
                        signal="NS",
                        mtype="NK"
                    )
                    NS_detections.add(satdata.object_id, n)
                    NS_detections.set_type(-1, 'NK')

                # set start of study type to match current EW node type    
                NS_detections.set_type(0, sk_mode)

            # else if change in inclination is slower than min XIPS but we're past first 24 hours
            elif (rate < self.XIPS_inc_per_day):

                if len(dexs) > 0:
                    n = Node(
                        index=dexs[0],
                        ntype="IK",
                        signal="NS",
                        mtype=sk_mode
                    )
                    NS_detections.add(satdata.object_id, n)

                    # if next EW node is ID
                    if end_of_sk:

                        # set next NS node to ID
                        n = Node(
                            index=data_indices[1],
                            ntype="ID",
                            signal="NS",
                            mtype="NK"
                        )
                        NS_detections.add(satdata.object_id, n)
                        NS_detections.set_type(-1, 'NK')

            # else change in inclination is faster
            else:

                # compute di/dt via simple difference
                didt = [0.0]
                for ii in range(len(mode_incs)-1):
                    didt.append((mode_incs[ii+1]-mode_incs[ii])/(step_size_hours*60*60))

                # loop over di/dt values                
                prev = 1.0
                for ii in range(len(didt)-1):

                    if np.abs(didt[ii]) > self.didt_threshold:

                        # threshold exceeded
                        dexs.append(ii+data_indices[0])

                        # if ratio of (pre-mean - post-mean) / std / prev < 1
                        ratio = np.abs(np.mean(mode_incs[0:ii]) - np.mean(mode_incs[ii:len(mode_incs)])) / np.std(mode_incs[0:ii])
                        if ratio/prev < 1.0:

                            if first and len(dexs)==2:

                                NS_detections.set_type(0, EW_detections.df.iloc[0].Type)
                                first = False

                        elif len(dexs)==2:

                            first = False

                        prev = ratio

                # found some detections
                if len(dexs) > 0:

                    n = Node(
                        index=dexs[0],
                        ntype="IK",
                        signal="NS",
                        mtype=sk_mode
                    )
                    NS_detections.add(satdata.object_id, n)
                    NS_detections.set_type(-1, sk_mode)
                    NS_detections.set_type(0, 'NK')

                    if end_of_sk:

                        n = Node(
                            index=data_indices[1],
                            ntype="ID",
                            signal="NS",
                            mtype="NK"
                        )
                        NS_detections.add(satdata.object_id, n)
                        NS_detections.set_type(-1, 'NK')

                elif node_index == 0:

                    NS_detections.set_type(0, EW_detections.df.iloc[0].Type)

                else:

                    NS_detections.set_type(0, 'NK')

        return NS_detections


class HeuristicEastWestNodeClassifier:

    def __init__(self, cfg):
        self.logger = logging.getLogger()
        self.ck_threshold = 5.1

    def run(self, EW_detections, longitudes):
        '''Set all EW node station-keeping modes.

        SS node is assumed to be preset to 'NK' if drifting at start of period.
        '''
        for index, row in EW_detections.df.iterrows():
            if row.Node == 'ES':
                EW_detections.set_type(index, 'ES')
            elif row.Node == 'ID' or row.Node == 'AD':
                EW_detections.set_type(index, 'NK')
            else:
                mode_lons = longitudes[row.TimeIndex:EW_detections.df.iloc[index+1].TimeIndex]
                EW_db = np.max(mode_lons) - np.min(mode_lons)
                EW_sd = np.std(mode_lons)
                EW = (EW_db - EW_sd) / EW_sd
                if row.Node == 'IK' or (row.Node == 'SS' and row.Type != 'NK'):
                    EW_detections.set_type(index, 'CK' if EW < self.ck_threshold else 'EK')
    

class DevkitHeuristicProcessor(Processor):
    '''Classifier based on the following paper: Solera, Haley & Roberts, Thomas & 
    Linares, Richard. (2023). Geosynchronous Satellite Pattern-of-Life Node Detection
    and Classification. Draft submitted to 9th AIAA Space Traffic 
    '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ew_detector = HeuristicEastWestNodeDetector(cfg)
        self.ns_detector = HeuristicNorthSouthNodeDetector(cfg)
        self.ew_classifier = HeuristicEastWestNodeClassifier(cfg)
    
    def predict_object_nodes(self, satdata):
        
        # Run LS detection
        EW_detections = self.ew_detector.run(satdata)
        self.ew_classifier.run(EW_detections, satdata.df['Longitude (deg)'])
        self.logger.info(f'EW Detections:\n{EW_detections.df}')

        NS_detections = self.ns_detector.run(satdata, EW_detections)
        self.logger.info(f'NS Detections:\n{NS_detections.df}')

        predictions_df = pd.concat([EW_detections.df, NS_detections.df], ignore_index=True)

        return predictions_df


class AstrokinetixEastWestNodeDetector(NodeDetector):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.delta_lon_24h_drift_threshold = cfg.getfloat('delta_lon_24h_drift_threshold')
        self.delta_lon_24h_sk_threshold = cfg.getfloat('delta_lon_24h_sk_threshold')
        self.min_drift_rate_change_threshold = cfg.getfloat('min_drift_rate_change_threshold')
        self.eccentricity_settling_threshold = cfg.getfloat('eccentricity_settling_threshold')
        self.high_eccentricity_threshold = cfg.getfloat('high_eccentricity_threshold')
        self.slow_drift_threshold = cfg.getfloat('slow_drift_threshold')
        self.min_slow_drift_rate = cfg.getfloat('min_slow_drift_rate')
        self.sk_slot_accuracy = cfg.getfloat('sk_slot_accuracy')
        self.unstable_drift_threshold = cfg.getfloat('unstable_drift_threshold')
        self.sk_std_tolerance_multiplier = cfg.getfloat('sk_std_tolerance_multiplier')
        self.max_sk_drift_rate_std = cfg.getfloat('max_sk_drift_rate_std')
        self.false_slow_drift_period = cfg.getfloat('false_slow_drift_period')
        self.max_slow_drift_lag = cfg.getint('max_slow_drift_lag')
        self.sk_wide_deadband_threshold = cfg.getfloat('sk_wide_deadband_threshold')
        self.drift_anomaly_threshold = cfg.getfloat('drift_anomaly_threshold')
        self.wide_deadband_objects = set()
    
    def run(self, satdata):

        object_id = satdata.object_id

        # turn on trace debugging
        if self.trace_object and self.trace_object == object_id:
            self.logger.info(f'Tracing EW detection of object {object_id}')
            level = self.logger.getEffectiveLevel()
            self.logger.setLevel(logging.DEBUG)
        
        # set up additional tabular observation data
        steps_per_day = satdata.steps_per_day
        df = self.tabularize_data(satdata)
        longitudes = df['Longitude (deg)']
        longitude_mean = longitudes.rolling(steps_per_day).mean()
        longitude_mean.bfill(inplace=True)
        delta_lon_24h = df['DeltaLon24h']  # changes are immediate
        drift_rate_mean = delta_lon_24h.rolling(steps_per_day).mean().shift(1)  # changes lag 24 hours
        drift_rate_mean.bfill(inplace=True)
        drift_rate_std = delta_lon_24h.rolling(steps_per_day).std()
        drift_rate_std.bfill(inplace=True)
        # use max std over prior 24 hours for drift detection
        drift_rate_std_max = drift_rate_std.rolling(steps_per_day).max().shift(1)
        drift_rate_std_max.bfill(inplace=True)
        drift_rate_delta_from_mean = np.abs(delta_lon_24h - drift_rate_mean)
        eccentricity = df['Eccentricity']
        eccentricity_mean = eccentricity.rolling(steps_per_day).mean().shift(1)
        eccentricity_delta_from_mean = np.abs(eccentricity - eccentricity_mean)
        delta_ecc_24h = eccentricity - eccentricity.shift(steps_per_day)
        delta_ecc_24h.bfill(inplace=True)
        delta_lon_2h = longitudes - longitudes.shift(1)
        
        # initialize EW detections
        EW_detections = NodeDetectionSet()
        ssEW = Node(
            index=0,
            ntype="SS",
            signal="EW"
        )
        EW_detections.add(object_id, ssEW)

        # initialize status flags, markers, and counts
        drifting = False
        last_AD = 0
        last_ID = 0
        last_IK = 0
        last_SK = 0
        last_unstable_drift = 0
        last_false_slow_drift = 0
        last_drift_anomaly = 0
        last_drift_anomaly_offset = 0
        count_from_last_drift_reversal = 0
        wide_deadband = False
        high_eccentricity = False
        slow_drift = False
        slow_drift_rate = 0
        slow_drift_rate_unstable = True

        # initialize SK slot
        slot_factor = 1 / self.sk_slot_accuracy
        sk_slot = np.round(slot_factor * longitudes.iloc[0:steps_per_day].mean()) / slot_factor  # baseline if initially stationkeeping

        for i in range(len(df)-3):
            if not drifting:
                last_SK_restart = max(last_IK, last_false_slow_drift)
                if i - last_SK_restart < 30 * steps_per_day:
                    # adjust slot estimate based on long-term mean
                    sk_longitude_mean = longitudes.iloc[last_SK_restart:i+1].mean()
                    sk_longitude_std = longitudes.iloc[last_SK_restart:i+1].std()
                    sk_slot = np.round(slot_factor * sk_longitude_mean) / slot_factor
                    sk_slot_error = np.abs(sk_longitude_mean - sk_slot)
                    # check for wide deadband case and adjust drift tolerance
                    sk_drift_tolerance = sk_slot_error + self.sk_std_tolerance_multiplier * sk_longitude_std
                    if i > last_SK_restart + steps_per_day:
                        istart = last_SK_restart + steps_per_day
                        sk_mean_variation = longitude_mean.iloc[istart:i+1].max() - longitude_mean.iloc[istart:i+1].min()
                        self.logger.debug(f'Index {i} longitude mean variation: {sk_mean_variation:.2f}')
                        if sk_mean_variation > self.sk_wide_deadband_threshold:
                            wide_deadband = True
                            self.wide_deadband_objects.add(object_id)
                            sk_drift_tolerance = sk_slot_error + 2 * self.sk_std_tolerance_multiplier * sk_longitude_std
                    # adjust detection thresholds near start of SK period
                    #FIXME: why do this after false slow drift?
                    #FIXME: why not do this after SK maneuver?
                    if i - last_SK_restart < 5 * steps_per_day:
                        if last_SK_restart == 0:
                            istart = 0
                        else:
                            istart = last_SK_restart + steps_per_day - 1  # allow for some settling after SK maneuver, IK node or false slow-drift
                        sk_drift_rate_std_max = min(delta_lon_24h.iloc[istart:i].expanding().std().max(), self.max_sk_drift_rate_std)
                        sk_drift_rate_tolerance = max(3 * sk_drift_rate_std_max, self.delta_lon_24h_drift_threshold)
                    # update mean longitude bounds
                    if i > last_IK + steps_per_day:
                        istart = last_IK + steps_per_day
                    else:
                        istart = last_IK
                    longitude_mean_upper_bound = longitude_mean.iloc[istart:i+1].max()
                    longitude_mean_lower_bound = longitude_mean.iloc[istart:i+1].min()
                    longitude_excursion = False
                    count_from_last_longitude_excursion = 0
                    self.logger.debug(f'Mean longitude bounds: upper={longitude_mean_upper_bound:.4f} lower={longitude_mean_lower_bound:.4f}')
                else:
                    # track mean longitude excursions
                    if longitude_excursion:
                        if longitude_mean.iloc[i] > longitude_mean_upper_bound:
                            self.logger.debug(f'Index {i} upper bound excursion continues {longitude_mean_upper_bound:.4f}')
                        elif longitude_mean.iloc[i] < longitude_mean_lower_bound:
                            self.logger.debug(f'Index {i} lower bound excursion continues {longitude_mean_lower_bound:.4f}')
                        else:
                            self.logger.debug(f'Index {i} ending longitude excursion')
                            longitude_excursion = False # we've stopped the current excursion
                        count_from_last_longitude_excursion +=1
                    else:
                        if longitude_mean.iloc[i] > longitude_mean_upper_bound:
                            self.logger.debug(f'Index {i} upper bound exceeded {longitude_mean_upper_bound:.4f}')
                            longitude_excursion = True
                        elif longitude_mean.iloc[i] < longitude_mean_lower_bound:
                            self.logger.debug(f'Index {i} lower bound exceeded {longitude_mean_lower_bound:.4f}')
                            longitude_excursion = True
                        else:
                            self.logger.debug(f'Index {i} longitude mean within bounds')
                        count_from_last_longitude_excursion = 0
                if i - last_IK < 5 * steps_per_day:
                    # initialize counters
                    count_from_last_sk_maneuver = 0
                    count_from_last_drift_reversal = 0
                else:
                    # update counters
                    count_from_last_sk_maneuver += 1
                    if drift_rate_mean.iloc[i] * drift_rate_mean.iloc[i-1] < 0:
                        count_from_last_drift_reversal = 0
                    else:
                        count_from_last_drift_reversal += 1
                self.logger.debug(f'Index {i} Lon:{longitudes.iloc[i]:.4f} MeanLon:{longitude_mean.iloc[i]:.4f} Delta24:{delta_lon_24h.iloc[i]:.4f}')
                self.logger.debug(f'Index {i} counts since: drftrvrsl:{count_from_last_drift_reversal}'
                                  f'  skmnvr:{count_from_last_sk_maneuver}'
                                  f'  excursion:{count_from_last_longitude_excursion}')
                # has eccentricity increased significantly?
                if eccentricity.iloc[i] > self.high_eccentricity_threshold:
                    alg_path = 'SK_END_DELTA_ECCENTRICITY'
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    drifting = True
                    slow_drift = False
                    last_unstable_drift = i
                    if i < steps_per_day:
                        EW_detections.set_type(0, 'NK')
                        self.logger.debug(f'Setting SS node to NK due to delta eccentricity')
                    else:
                        EW_detections.add(
                            object_id,
                            Node(
                                index=i,
                                signal='EW',
                                ntype='ID',
                            )
                        )
                        last_ID = i
                        self.logger.debug(f'Detected ID node at {i} due to delta eccentricity')
                    if eccentricity.iloc[i] > self.high_eccentricity_threshold:
                        high_eccentricity = True
                    self.logger.debug(f'High-Eccentricity: {high_eccentricity}')
                    self.logger.debug(f'Slow Drift: {slow_drift}')
                    drift_rate = delta_lon_24h.iloc[min(i+steps_per_day, len(df)-1)]
                    self.logger.debug(f'Drift Rate: {drift_rate:.02f} deg/day')
                    self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                    continue
                # check for a station-keeping maneuver before checking for drift initiation
                if i - last_IK > steps_per_day and drift_rate_std.iloc[i] > 3 * sk_drift_rate_std_max:
                    if np.abs(drift_rate_mean.iloc[i+3]) < np.abs(drift_rate_mean.iloc[i]):
                        self.logger.debug(f'drmean: {drift_rate_mean.iloc[i]:.3f}  drmean+3: {drift_rate_mean.iloc[i+3]:.3f}')
                        alg_path = 'SK_MANEUVER_DETECT'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        last_SK = i
                        count_from_last_sk_maneuver = 0
                        continue
                # allow 24-hours to settle after SK maneuver
                if last_SK != 0 and i < last_SK + steps_per_day:
                    alg_path = 'SK_MANEUVER_CONTINUE'
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    continue
                # check for longitude drift anomaly (sudden shift back toward long-term mean)
                if np.abs(delta_lon_2h.iloc[i]) > max(self.drift_anomaly_threshold, 5 * delta_lon_2h.iloc[i-3:i].abs().max()):
                    if np.abs(longitudes.iloc[i] - sk_longitude_mean) < np.abs(longitudes.iloc[i-1] - sk_longitude_mean):
                        alg_path = 'LONGITUDE_DRIFT_ANOMALY'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        last_drift_anomaly = i
                        last_drift_anomaly_offset = np.abs(delta_lon_2h.iloc[i])
                        continue
                # was the recent drift anomaly a single-step event or is it something else?
                if i < last_drift_anomaly + steps_per_day and np.abs(delta_lon_2h.iloc[i]) < 0.2 * last_drift_anomaly_offset:
                        alg_path = 'CONTINUE_LONGITUDE_DRIFT_ANOMALY'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        continue
                else:
                    last_drift_anomaly = 0
                # has the longitude changed from 24 hours prior?
                if np.abs(delta_lon_24h.iloc[i:i+4]).min() > sk_drift_rate_tolerance:  # changing for 6+ hours
                    alg_path = 'SK_END_DELTA_LON'
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    drifting = True
                    slow_drift = False
                    last_unstable_drift = i
                    if i < steps_per_day:
                        EW_detections.set_type(0, 'NK')
                        self.logger.debug(f'Setting SS node to NK due to delta lon')
                    else:
                        EW_detections.add(
                            object_id,
                            Node(
                                index=i,
                                signal='EW',
                                ntype='ID',
                            )
                        )
                        last_ID = i
                        self.logger.debug(f'Detected ID node at {i} due to delta lon')
                    if eccentricity.iloc[i] > self.high_eccentricity_threshold:
                        high_eccentricity = True
                    self.logger.debug(f'High-Eccentricity: {high_eccentricity}')
                    self.logger.debug(f'Slow Drift: {slow_drift}')
                    drift_rate = delta_lon_24h.iloc[min(i+steps_per_day, len(df)-1)]
                    self.logger.debug(f'Drift Rate: {drift_rate:.02f} deg/day')
                    self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                # look for slow drift out of orbital slot based on longitude variation
                elif (i > last_SK_restart + 20 * steps_per_day and 
                        np.abs(longitude_mean.iloc[i] - sk_slot) > sk_drift_tolerance):
                    drifting = True
                    slow_drift = True
                    slow_drift_rate = np.abs(drift_rate_mean.iloc[i])
                    slow_drift_rate_unstable = False
                    lag_estimate = int((longitude_mean.iloc[i] - sk_slot) /
                            (delta_lon_24h.iloc[i] / steps_per_day))
                    lag_estimate = min(self.max_slow_drift_lag * steps_per_day, lag_estimate)  # don't go back too far
                    i_slow_drift = max(0, last_ID, i - lag_estimate)
                    if i_slow_drift == 0 and len(EW_detections) == 1:
                        # reset start node to drifting
                        alg_path = 'RESET_START_NODE_FOR_SLOW_DRIFT'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        EW_detections.set_type(0, 'NK')
                        self.logger.debug(f'Detected slow drift at {i} due to longitude mean and setting Node index to {i_slow_drift}')
                        self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                        continue
                    elif i_slow_drift < last_IK:
                        # remove last IK node
                        alg_path = 'REMOVE_LAST_IK_NODE_FOR_SLOW_DRIFT'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        EW_detections.df = EW_detections.df.iloc[:-1]
                        self.logger.debug(f'Removing IK node at {last_IK} due to slow-drift detected at {i} with estimated start at {i_slow_drift}')
                        self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                        continue
                    alg_path = 'SK_END_SLOW_DRIFT'
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    EW_detections.add(
                        object_id,
                        Node(
                            index=i_slow_drift,
                            signal='EW',
                            ntype='ID',
                        )
                    )
                    last_ID = i  # use i rather than i_node to catch short-term excursions beyond tolerance
                    self.logger.debug(f'Detected slow drift at {i} due to longitude mean and setting Node index to {i_slow_drift}')
                    if eccentricity.iloc[i] > self.high_eccentricity_threshold:
                        high_eccentricity = True
                    self.logger.debug(f'High-Eccentricity: {high_eccentricity}')
                    self.logger.debug(f'Slow Drift: {slow_drift}')
                    self.logger.debug(f'Drift Rate: {slow_drift_rate:.02f} deg/day')
                    self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                # look for slow drift out of orbital slot based on time from drift rate reversal
                elif (not wide_deadband and i > last_SK_restart + 30 * steps_per_day and 
                            count_from_last_drift_reversal > 20 * steps_per_day):
                    drifting = True
                    slow_drift = True
                    slow_drift_rate = np.abs(drift_rate_mean.iloc[i])
                    slow_drift_rate_unstable = False
                    i_slow_drift = i - count_from_last_drift_reversal
                    alg_path = 'SK_END_SLOW_DRIFT_ALT'
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    EW_detections.add(
                        object_id,
                        Node(
                            index=i_slow_drift,
                            signal='EW',
                            ntype='ID',
                        )
                    )
                    last_ID = i  # use i rather than i_node to catch short-term excursions beyond tolerance
                    self.logger.debug(f'Detected slow drift at {i} due to longitude mean and setting Node index to {i_slow_drift}')
                    if eccentricity.iloc[i] > self.high_eccentricity_threshold:
                        high_eccentricity = True
                    self.logger.debug(f'High-Eccentricity: {high_eccentricity}')
                    self.logger.debug(f'Slow Drift: {slow_drift}')
                    self.logger.debug(f'Drift Rate: {slow_drift_rate:.02f} deg/day')
                    self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                else:
                    alg_path = 'SK_CONTINUE'
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    continue
            else: # drifting
                drift_rate_change_threshold = max(3 * drift_rate_std_max.iloc[i], self.min_drift_rate_change_threshold)
                delta_lon_24h_sk_threshold = self.delta_lon_24h_sk_threshold
                if slow_drift:
                    self.logger.debug(f'dlon24:{delta_lon_24h.iloc[i]:.5f} drmean:{drift_rate_mean.iloc[i]:.5f} drslow:{slow_drift_rate:.5f}')
                    # use a tighter tolerance for station-keeping detection
                    delta_lon_24h_sk_threshold /= 2
                    # are we in the false drift period of a slow-drift at the start of drift segment?
                    if last_ID > last_AD and i < last_ID + self.false_slow_drift_period * steps_per_day:
                        # is mean longitude moving back toward sk_longitude_mean?
                        if (longitude_mean.iloc[i] - longitude_mean.iloc[i-1]) * (longitude_mean.iloc[i] - sk_longitude_mean) < 0:
                            if i_slow_drift == 0:
                                alg_path = 'FALSE_SLOW_DRIFT_AT_ZERO'
                                # reset start node to NaN
                                EW_detections.set_type(0, np.nan)
                            else:
                                alg_path = 'FALSE_SLOW_DRIFT'
                                # remove last ID node
                                EW_detections.df = EW_detections.df[:-1]
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            drifting = False
                            slow_drift = False
                            self.logger.debug(f'False slow drift detected at {i} and ID node at {i_slow_drift} removed')
                            last_ID = 0
                            last_false_slow_drift = i
                            self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                            continue
                        # is current drift rate change enough to trigger an AD, but also moving closer to zero
                        elif np.abs(delta_lon_24h.iloc[i] - drift_rate_mean.iloc[i]) > drift_rate_change_threshold \
                                and (np.abs(delta_lon_24h.iloc[i]) < np.abs(drift_rate_mean.iloc[i]) \
                                or delta_lon_24h.iloc[i] * drift_rate_mean.iloc[i] < 0):
                            if i_slow_drift == 0:
                                alg_path = 'FALSE_SLOW_DRIFT_AT_ZERO_ALT'
                                # reset start node to NaN
                                EW_detections.set_type(0, np.nan)
                            else:
                                alg_path = 'FALSE_SLOW_DRIFT_ALT'
                                # remove last ID node
                                EW_detections.df = EW_detections.df[:-1]
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            drifting = False
                            slow_drift = False
                            self.logger.debug(f'False slow drift detected at {i} and ID node at {i_slow_drift} removed')
                            last_ID = 0
                            last_false_slow_drift = i
                            self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                            continue
                    # is a late slow drift still settling?
                    if slow_drift_rate_unstable:
                        if max(np.abs(drift_rate_mean.iloc[i]), self.min_slow_drift_rate) < slow_drift_rate:
                            alg_path = 'UNSTABLE_SLOW_DRIFT'
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            slow_drift_rate = max(np.abs(drift_rate_mean.iloc[i]), self.min_slow_drift_rate)
                            self.logger.debug(f'Adjusting slow drift rate to {slow_drift_rate:.03f}')
                            last_unstable_drift = i
                            continue
                        else:
                            slow_drift_rate_unstable = False
                # has the eccentricity dropped to near zero?
                if np.abs(delta_ecc_24h.iloc[i]) > .0001 and eccentricity.iloc[i] < .0005:
                    if i < last_ID + 2:  # immediate reversal
                        alg_path = 'REMOVE_FALSE_ID_BASED_ON_ECC'
                        EW_detections.df = EW_detections.df[:-1]
                        self.logger.debug(f'Removed false ID node at time index {last_ID} due to eccentricity change {delta_ecc_24h.iloc[i]:.5f}')
                        last_ID = 0
                    else:
                        alg_path = 'SK_INIT_ECC'
                        last_IK = i
                        EW_detections.add(
                            object_id,
                            Node(
                                index=last_IK,
                                signal='EW',
                                ntype='IK'
                            )
                        )
                        sk_slot = np.round(slot_factor * longitudes.iloc[i]) / slot_factor # set baseline longitude
                        self.logger.debug(f'Detected IK node at {i} based on eccentricity {eccentricity.iloc[i]:.4f}')
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    drifting = False
                    slow_drift = False
                    high_eccentricity = False
                    self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                    continue
                # has the longitude stopped changing from 24 hours prior?
                if np.abs(delta_lon_24h.iloc[i-3:i+1]).max() < delta_lon_24h_sk_threshold:  # stationary for 8+ hours
                    # are we still in a slow drift from an ID?
                    if slow_drift and last_ID > last_AD:
                        # is mean drift rate still above initial rate at detection?
                        if np.abs(drift_rate_mean.iloc[i]) >= slow_drift_rate:
                            alg_path = 'SLOW_DRIFT_CONTINUE'
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            continue
                        # is mean longitude still moving away from last SK mean longitude?
                        if ((longitude_mean.iloc[i] - longitude_mean.iloc[i-1]) * 
                                (longitude_mean.iloc[i] - longitude_mean.iloc[last_ID]) > 0):
                            alg_path = 'SLOW_DRIFT_CONTINUE_ALT'
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            continue
                    # are we too close to most recent drift onset?
                    if i <= last_ID + steps_per_day:
                        alg_path = 'REMOVE_ID_AND_RESUME_SK'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        # remove false ID node
                        EW_detections.df = EW_detections.df.iloc[:-1]
                        drifting = False
                        slow_drift = False
                        self.logger.debug(f'Removed false ID node at {last_ID}')
                        self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                        continue  # resume previous SK period
                    # are we too close to most recent drift adjust?
                    elif i <= last_AD + 1.5 * steps_per_day:
                        if slow_drift:
                            # remove false AD node
                            EW_detections.df = EW_detections.df.iloc[:-1]
                            drifting = False
                            slow_drift = False
                            self.logger.debug(f'Removed false AD node')
                            self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                        else:
                            alg_path = 'IK_REJECT_TOO_CLOSE_AD'
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            continue
                    alg_path = 'SK_INIT'
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    drifting = False
                    slow_drift = False
                    last_IK = i - steps_per_day
                    EW_detections.add(
                        object_id,
                        Node(
                            index=last_IK,
                            signal='EW',
                            ntype='IK'
                        )
                    )
                    sk_slot = np.round(slot_factor * longitudes.iloc[last_IK:i].mean()) / slot_factor # set baseline longitude
                    high_eccentricity = False
                    self.logger.debug(f'Detected IK node at {i} and setting at {last_IK}')
                    self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                else:  # check for AD
                    # are we too close to the start of the study?
                    if i <= 2 * steps_per_day:
                        alg_path = 'TOO_CLOSE_TO_START_FOR_AD'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        continue
                    # are we too close to most recent drift onset or change?
                    if last_unstable_drift == i - 1 and i < max(last_ID, last_AD) + 3 * steps_per_day:
                        self.logger.debug(f'dlon24:{delta_lon_24h.iloc[i]:.4f} drmean:{drift_rate_mean.iloc[i]:.4f} drstd:{drift_rate_std.iloc[i]:.4f}')
                        # has drift rate dropped near zero in past 6 hours?
                        if delta_lon_24h.iloc[i-2:i+1].abs().max() < delta_lon_24h_sk_threshold:
                            alg_path = 'BYPASS_SETTLING'
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            continue
                        # has AD drift rate settled below slow drift threshold?
                        # if not slow_drift and last_AD > last_ID and np.abs(drift_rate_mean.iloc[i]) < self.slow_drift_threshold:
                        #     alg_path = 'TRANSITION_TO_SLOW_DRIFT'
                        #     self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        #     self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        #     slow_drift = True
                        #     slow_drift_rate_unstable = True
                        #     slow_drift_rate = np.abs(drift_rate_mean.iloc[i]) # use mean for settling
                        #     last_unstable_drift = i
                        #     i_slow_drift = last_AD
                        #     continue
                        alg_path = 'DRIFT_STILL_SETTLING'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        # is drift rate standard deviation elevated? (still adjusting)
                        if drift_rate_std.iloc[i] > self.unstable_drift_threshold:
                            self.logger.debug(f'Drift rate unstable due to elevated drift rate STD: {drift_rate_std.iloc[i]:.4f}')
                            last_unstable_drift = i
                        # or is difference between current drift and 24-hour mean still outside threshold?
                        elif (delta_lon_24h.iloc[i:i+2] - drift_rate_mean.iloc[i]).abs().max() > self.min_drift_rate_change_threshold:
                            self.logger.debug(f'Drift rate unstable due to difference between current and mean drift rates: {drift_rate_delta_from_mean.iloc[i]:.4f}')
                            last_unstable_drift = i
                        # or is eccentricity mean still settling?
                        elif eccentricity_delta_from_mean.iloc[i] > self.eccentricity_settling_threshold:
                            self.logger.debug(f'Drift rate unstable due to difference between current and mean eccentricity: {eccentricity_delta_from_mean.iloc[i]:.5f}')
                            last_unstable_drift = i
                        continue
                    # is it too soon after drift has stabilized?
                    if i < last_unstable_drift + 0.5 * steps_per_day:
                        alg_path = 'DRIFT_TOO_CLOSE_LAST'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        continue
                    # have we shifted from circular to elliptical drift?
                    if eccentricity.iloc[i] > max(2 * eccentricity_mean.iloc[i], self.eccentricity_settling_threshold):
                        alg_path = 'DRIFT_ECCENTRICITY_INCREASE'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        # fudge factor to match truth
                        lag = 0
                        if slow_drift:
                            lag = 7
                        EW_detections.add(
                            object_id,
                            Node(
                                index=i-lag,
                                signal='EW',
                                ntype='AD'
                            )
                        )
                        last_AD = i - lag
                        high_eccentricity = True
                        slow_drift = False
                        last_unstable_drift = i
                        self.logger.debug(f'Detected AD node at {i} with increased eccentricity {eccentricity.iloc[i]:.05f} lag:{lag}')
                        self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                    # have we shifted from elliptical to circular drift?
                    elif high_eccentricity and eccentricity.iloc[i] < 0.5 * eccentricity_mean.iloc[i]:
                        alg_path = 'DRIFT_ECCENTRICITY_DECREASE'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        EW_detections.add(
                            object_id,
                            Node(
                                index=i,
                                signal='EW',
                                ntype='AD'
                            )
                        )
                        last_AD = i
                        high_eccentricity = False
                        i_slow_drift = i
                        slow_drift = True
                        slow_drift_rate = np.abs(delta_lon_24h.iloc[i])
                        slow_drift_rate_unstable = True
                        last_unstable_drift = i
                        self.logger.debug(f'Detected AD node at {i} with decreased eccentricity {eccentricity.iloc[i]:.05f}')
                        self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                    # has the drift rate changed from the prior 24-hour mean for at least 2 steps?
                    elif (delta_lon_24h.iloc[i:i+2] - drift_rate_mean.iloc[i]).abs().min() > drift_rate_change_threshold:
                        self.logger.debug(f'drmean:{drift_rate_mean.iloc[i]:.3f} dlon24:{delta_lon_24h.iloc[i]:.3f} dlon24+1:{delta_lon_24h.iloc[i+1]:.3f} thr:{drift_rate_change_threshold:.3f}')
                        alg_path = 'DRIFT_RATE_CHANGE'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        EW_detections.add(
                            object_id,
                            Node(
                                index=i,
                                signal='EW',
                                ntype='AD'
                            )
                        )
                        last_AD = i
                        last_unstable_drift = i
                        self.logger.debug(f'Detected AD node at {i} with drift rate change')
                        if eccentricity.iloc[i] > self.high_eccentricity_threshold:
                            high_eccentricity = True
                        else:
                            high_eccentricity = False
                        if np.abs(delta_lon_24h.iloc[i]) > self.slow_drift_threshold:
                            slow_drift = False
                        else:
                            slow_drift = True
                            i_slow_drift = i
                            slow_drift_rate = max(np.abs(delta_lon_24h.iloc[i]), self.min_slow_drift_rate)  # don't set too low to initiate IK
                            slow_drift_rate_unstable = True
                        self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                    # has eccentricity changed?
                    elif np.abs(delta_ecc_24h.iloc[i]) > .0001:
                        alg_path = 'DELTA_ECC_THRESHOLD'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        EW_detections.add(
                            object_id,
                            Node(
                                index=i,
                                signal='EW',
                                ntype='AD'
                            )
                        )
                        last_AD = i
                        last_unstable_drift = i
                        self.logger.debug(f'Detected AD node at {i} with eccentricity change {delta_ecc_24h.iloc[i]}')
                        if eccentricity.iloc[i] > self.high_eccentricity_threshold:
                            high_eccentricity = True
                        else:
                            high_eccentricity = False
                        if np.abs(delta_lon_24h.iloc[i]) > self.slow_drift_threshold:
                            slow_drift = False
                        else:
                            slow_drift = True
                            i_slow_drift = i
                            slow_drift_rate = max(np.abs(delta_lon_24h.iloc[i]), self.min_slow_drift_rate)  # don't set too low to initiate IK
                            slow_drift_rate_unstable = True
                        self.logger.debug(f'EW Detections:\n{EW_detections.df}')
                    else:
                        alg_path = 'DRIFT_UNCHANGED'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        continue
                    self.logger.debug(f'High-Eccentricity: {high_eccentricity}')
                    self.logger.debug(f'Slow Drift: {slow_drift}')
                    drift_rate = delta_lon_24h.iloc[min(i+steps_per_day, len(df)-1)]
                    self.logger.debug(f'Drift Rate: {drift_rate:.02f} deg/day')
        
        EW_detections.add(
            object_id,
            Node(
                index=len(df),
                signal='ES',
                ntype='ES'
            )
        )
        self.logger.debug(f'EW Detections:\n{EW_detections.df}')

        # reset logger level after trace
        if self.trace_object and self.trace_object == object_id:
            self.logger.setLevel(level)

        return EW_detections


class AstrokinetixNorthSouthNodeDetector(NodeDetector):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sk_delta_incl_threshold = cfg.getfloat('sk_delta_incl_threshold')
        self.sk_delta_incl_hf_threshold = cfg.getfloat('sk_delta_incl_hf_threshold')
        self.max_sk_maneuver_period = cfg.getfloat('max_sk_maneuver_period')
        self.inclination_drift_threshold = cfg.getfloat('inclination_drift_threshold')
        self.sk_maneuver_max_inclination = cfg.getfloat('sk_maneuver_max_inclination')

    def run(self, satdata, EW_detections):

        object_id = satdata.object_id

        # turn on trace debugging
        if self.trace_object and self.trace_object == object_id:
            self.logger.info(f'Tracing NS detection of object {object_id}')
            level = self.logger.getEffectiveLevel()
            self.logger.setLevel(logging.DEBUG)

        # set up additional tabular observation data
        steps_per_day = satdata.steps_per_day
        df = self.tabularize_data(satdata)
        inclinations = df["Inclination (deg)"]
        delta_incl_2h = df['DeltaIncl2h']
        delta_incl_5day_std = delta_incl_2h.rolling(5*steps_per_day).std()
        delta_incl_5day_std.bfill(inplace=True)

        # initialize status flags
        ew_drifting = EW_detections.df.iloc[0].Type == 'NK'
        if ew_drifting:
            ew_start = len(df)
        else:
            ew_start = 0
        drifting = ew_drifting
        last_IK = -1
        last_IM = -1
        last_ID = -1
        last_maneuver = -1  # needs to be less than first index
        maneuver_count = 0
        maneuver_list = []
        last_exceedance = -1
        peak_inclinations = []
        peak_inclination_exceeded = False
        last_period_exceedance = -1
        max_period_exceeded = False
        hi_frequency = False
        sk_delta_incl_threshold = self.sk_delta_incl_threshold

        # initialize NS detections
        NS_detections = NodeDetectionSet()
        ssNS = Node(
            index=0,
            ntype="SS",
            signal="NS"
        )
        NS_detections.add(satdata.object_id, ssNS)
        if drifting:
            NS_detections.set_type(0, 'NK')

        EW_indices = list(EW_detections.df.TimeIndex.values)

        # loop through observations
        for i in range(len(df)):

            # track EW drift status
            if i in EW_indices:
                ew_index = EW_indices.index(i)
                if EW_detections.df.iloc[ew_index].Node == 'ID':
                    ew_drifting = True
                elif EW_detections.df.iloc[ew_index].Node == 'IK':
                    ew_drifting = False
                    ew_start = i
                    # reset maneuver tracking
                    maneuver_count = 0
                    maneuver_list = []
                    peak_inclinations = []
                    self.logger.debug(f'EW IK at index {i} -- maneuvers and peak inclination reset')

            if not drifting:

                # update counters
                count_from_last_maneuver = i - last_maneuver

                self.logger.debug(f'Index {i} incl={inclinations.iloc[i]:.4f} dinc2={delta_incl_2h.iloc[i]:.4f} #mnvrs={maneuver_count} lastmnvr={last_maneuver}')

                # are we still prior to declaring station-keeping?
                if last_IM > last_IK:

                    # is EW station-keeping active?
                    if not ew_drifting:

                        # has a maneuver just occurred?
                        if count_from_last_maneuver == 1 and ew_start < i:

                            # initiate station-keeping
                            if last_IM == i - 1:
                                alg_path = 'RESET_IM_TO_IK'
                                NS_detections.set_node(-1, 'IK')
                                last_IK = last_IM
                            else:
                                alg_path = 'ADD_IK_NODE'
                                n = Node(
                                    index=i - 1,
                                    signal='NS',
                                    ntype='IK',
                                )
                                NS_detections.add(satdata.object_id, n)
                                last_IK = i - 1
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            peak_inclinations = [maneuver_list[-1][1]]
                            self.logger.debug(f'Max inclination set to {np.array(peak_inclinations).mean():.4f}')
                            peak_inclination_exceeded = False
                            last_exceedance = i - 1
                            continue

                            # # are there enough maneuvers to check in-range status?
                            # if len(maneuver_list) > 2 and maneuver_list[-3][0] >= ew_start:

                            #     # is candidate inclination below next two pre-maneuver inclinations?
                            #     icandidate = maneuver_list[-3][0]
                            #     test_inclination = maneuver_list[-3][2]
                            #     limit = min(maneuver_list[-2][1], maneuver_list[-1][1], self.sk_maneuver_max_inclination)
                            #     if test_inclination < limit:
                            #         if icandidate == last_IM:
                            #             alg_path = 'RESET_IM_TO_IK_INRANGE'
                            #             NS_detections.set_node(-1, 'IK')
                            #             last_IK = last_IM
                            #         else:
                            #             alg_path = 'ADD_IK_NODE_INRANGE'
                            #             n = Node(
                            #                 index=icandidate,
                            #                 signal='NS',
                            #                 ntype='IK',
                            #             )
                            #             NS_detections.add(satdata.object_id, n)
                            #             last_IK = icandidate
                            #         self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            #         self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            #         peak_inclinations = [maneuver_list[-2][1], maneuver_list[-1][1]]  # 1st 2 peaks after IK
                            #         self.logger.debug(f'Max inclination set to {np.array(peak_inclinations).mean():.4f}')
                            #         peak_inclination_exceeded = False
                            #         last_exceedance = icandidate - 1
                            #         continue
                else:
                    # check if inclination range has been exceeded
                    if len(peak_inclinations) > 0 and inclinations.iloc[i] > np.array(peak_inclinations).mean():
                        # track start of most recent exceedances
                        if not peak_inclination_exceeded:
                            peak_inclination_exceeded = True
                            last_exceedance = i
                            self.logger.debug(f'Max inclination {np.array(peak_inclinations).mean():.4f} exceeded at index {i}')
                    
                    # check if max maneuver period has been exceeded
                    if not max_period_exceeded and len(maneuver_list) > 2:
                        periods = []
                        for imaneuver in range(1, len(maneuver_list)):
                            this_period = maneuver_list[imaneuver][0] - maneuver_list[imaneuver-1][0]
                            periods.append(this_period)
                        #max_period = np.array(periods).max()
                        max_period = int(np.array(periods).mean())
                        #if not hi_frequency:
                        #    max_period = int(np.median(np.array(periods)))
                        #max_period = periods[-1]
                        #max_period = np.array(periods).min()
                        if i > last_maneuver + max_period:
                            max_period_exceeded = True
                            last_period_exceedance = i
                            self.logger.debug(f'Max period {max_period} exceeded at index {i}')

                if i > last_IK + 3 * steps_per_day:

                    # check for SK maneuver (delta incl must be negative)
                    if delta_incl_2h.iloc[i] < -sk_delta_incl_threshold:
                        alg_path = 'SK_MANEUVER'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        last_maneuver = i
                        maneuver_count += 1
                        maneuver_list.append((i, inclinations.iloc[i-1], inclinations.iloc[i]))
                        peak_inclination_exceeded = False
                        peak_inclinations.append(inclinations.iloc[i-1])
                        max_period_exceeded = False
                        self.logger.debug(f'Max inclination set to {np.array(peak_inclinations).mean():.4f}')
                        if not hi_frequency and maneuver_count > 3:
                            indices = [index for (index, hi, lo) in maneuver_list]
                            if (indices[-1] - indices[0]) / (len(indices) - 1) < 3 * steps_per_day:
                                hi_frequency = True
                                sk_delta_incl_threshold = self.sk_delta_incl_hf_threshold
                        continue

                # check if SK maneuvers not observed
                if i > last_maneuver + self.max_sk_maneuver_period * steps_per_day:

                    if last_maneuver == last_IK or maneuver_count == 1:

                        if last_IK == -1:
                            alg_path = 'RESET_SS_NODE_TO_NK'
                            # reset SS node to NK
                            NS_detections.set_type(0, 'NK')
                        else:
                            alg_path = 'REMOVE_LAST_IK_NODE'
                            # remove false IK node (single SK maneuver)
                            NS_detections.df = NS_detections.df.iloc[:-1]
                            last_IK = -1
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        drifting = True
                        continue

                    # are we station-keeping?
                    elif last_IK >= last_IM:
                        alg_path = 'INITIATE_DRIFT'
                        self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                        self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                        # set ID node to last exceedance (or last maneuver if no exceedances)
                        if last_period_exceedance > 0:
                            idrift = last_period_exceedance
                        else:
                            idrift = max(last_exceedance, last_maneuver)
                        n = Node(
                            index=idrift,
                            signal='NS',
                            ntype='ID'
                        )
                        NS_detections.add(satdata.object_id, n)
                        drifting = True
                        last_ID = idrift
                        continue
                
                if i in EW_indices:

                    ew_index = EW_indices.index(i)

                    # is the EW node an ID node?
                    if EW_detections.df.iloc[ew_index].Node == 'ID':

                        # has there not been an SK maneuver yet?
                        if last_maneuver == -1:
                            alg_path = 'RESET_SS_NODE_TO_NK_ON_EW_ID'
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            # reset SS node to NK
                            NS_detections.set_type(0, 'NK')
                            drifting = True
                            continue

                        # are we currently station-keeping?
                        if last_IK >= last_IM:

                            # have we already exceeded the max period?
                            if max_period_exceeded:
                                alg_path = 'EW_ID_AFTER_MAX_PERIOD'
                                n = Node(
                                    index=last_period_exceedance,
                                    signal='NS',
                                    ntype='ID'
                                )
                                last_ID = last_period_exceedance
                                
                            else:
                                alg_path = 'MATCH_EW_ID_NODE'
                                n = Node(
                                    index=i,
                                    signal='NS',
                                    ntype='ID'
                                )
                                last_ID = i

                            NS_detections.add(satdata.object_id, n)
                            drifting = True
                            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                            continue
                
                alg_path = 'SK_CONTINUE'
                self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                continue

            else:  # drifting

                # did a maneuver occur?
                if delta_incl_2h.iloc[i] < -sk_delta_incl_threshold:

                    alg_path = 'INITIAL_MANEUVER_DETECT'
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    n = Node(
                        index=i,
                        signal='NS',
                        ntype='IM',
                    )
                    NS_detections.add(satdata.object_id, n)
                    drifting = False
                    last_IM = i
                    last_maneuver = i
                    maneuver_count = 1
                    maneuver_list = [(i, inclinations.iloc[i-1], inclinations.iloc[i])]
                    continue

                else:
                    alg_path = 'DRIFT_CONTINUE'
                    self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
                    self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
                    continue
        
        # catch late drift onset
        if not drifting and last_IK >= last_IM and max_period_exceeded and count_from_last_maneuver > steps_per_day:
            alg_path = 'INITIATE_DRIFT_POST_RUN'
            self.alg_counts[alg_path] = self.alg_counts.get(alg_path, 0) + 1
            self.logger.debug(f'Object {object_id} time index {i} alg path {alg_path}')
            if last_period_exceedance > 0:
                idrift = last_period_exceedance
            else:
                idrift = max(last_exceedance, last_maneuver)
            n = Node(
                index=idrift,
                signal='NS',
                ntype='ID'
            )
            NS_detections.add(satdata.object_id, n)
        
        NS_detections.add(
            object_id,
            Node(
                index=len(df),
                signal='ES',
                ntype='ES'
            )
        )

        return NS_detections


class PropulsionClassifier:

    def __init__(self, cfg):
        self.logger = logging.getLogger()

    @staticmethod
    def get_sk_segments(nodes_df, direction='EW'):

        sk_segments = []
        non_sk_segments = []
        df = nodes_df[np.logical_or(nodes_df.Direction == direction, nodes_df.Direction == 'ES')]
        sk_nodes = df[np.logical_or(df.Node == 'ID', df.Node == 'IK')]

        if len(sk_nodes) == 0:

            if df.iloc[0].Type == 'NK':
                non_sk_segments.append([0, nodes_df.iloc[-1].TimeIndex])
            else:
                sk_segments.append([0, nodes_df.iloc[-1].TimeIndex])

        else:

            if df.iloc[0].Node == sk_nodes.iloc[0].Node:
                raise Exception('Node order is invalid')
            segments = [[0, sk_nodes.iloc[0].TimeIndex]]
            for i in range(len(sk_nodes) - 1):
                if sk_nodes.iloc[i].Node == sk_nodes.iloc[i+1].Node:
                    raise Exception('Node order is invalid')
                segments.append([sk_nodes.iloc[i].TimeIndex, sk_nodes.iloc[i+1].TimeIndex])
            segments.append([sk_nodes.iloc[-1].TimeIndex, nodes_df.iloc[-1].TimeIndex])
            if df.iloc[0].Type == 'NK':
                drifting = True
            else:
                drifting = False
            for segment in segments:
                if drifting:
                    non_sk_segments.append(segment)
                else:
                    sk_segments.append(segment)
                drifting = not drifting

        return sk_segments, non_sk_segments


class AstrokinetixEastWestClassifier(PropulsionClassifier):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ck_threshold = 5.1

    def run(self, EW_detections, longitudes):
        '''Set all EW node station-keeping modes.

        SS node is assumed to be preset to 'NK' if drifting at start of period.
        '''
        for index, row in EW_detections.df.iterrows():
            if row.Node == 'ES':
                EW_detections.set_type(index, 'ES')
            elif row.Node == 'ID' or row.Node == 'AD':
                EW_detections.set_type(index, 'NK')
        
        sk_segments, _ = self.get_sk_segments(EW_detections.df, direction='EW')
        self.logger.info(f'EW SK segments:{sk_segments}')
        for segment in sk_segments:
            mode_lons = longitudes.iloc[segment[0]:segment[1]]
            EW_db = mode_lons.max() - mode_lons.min()
            EW_sd = mode_lons.std()
            EW = (EW_db - EW_sd) / EW_sd
            index = EW_detections.df[EW_detections.df.TimeIndex == segment[0]].index.values[0]
            EW_detections.set_type(index, 'CK' if EW < self.ck_threshold else 'EK')


class AstrokinetixNorthSouthClassifier(PropulsionClassifier):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ek_max_rate = 0.026
        self.ek_min_std = 0.0004
        self.sk_delta_incl_threshold = cfg.getfloat('sk_delta_incl_threshold')

    def run(self, NS_detections, inclinations):
        '''Process and classify North-South station-keeping segments.
        Assumes that SS node is already set to NK type when drifting at start.
        '''

        delta_incl_2h = inclinations - inclinations.shift(1)

        # set drifting nodes and ES nodes
        for index, row in NS_detections.df.iterrows():
            if row.Node == 'ES':
                NS_detections.set_type(index, 'ES')
            elif row.Node == 'ID' or row.Node == 'AD' or row.Node == 'IM':
                NS_detections.set_type(index, 'NK')

        # classify SK nodes        
        sk_segments, _ = self.get_sk_segments(NS_detections.df, direction='NS')
        self.logger.info(f'NS SK segments:{sk_segments}')
        for segment in sk_segments:

            segment_deltas = delta_incl_2h.iloc[segment[0]:segment[1]]
            median_rate = -segment_deltas[segment_deltas < -self.sk_delta_incl_threshold].median()
            non_thrust_std = segment_deltas[segment_deltas > -self.sk_delta_incl_threshold].std()
            index = NS_detections.df[NS_detections.df.TimeIndex == segment[0]].index.values[0]
            ns_sk_indices = segment_deltas[segment_deltas < -self.sk_delta_incl_threshold].index.values.tolist()
            if ns_sk_indices:
                # this isn't like EW -- we don't expect a range for each maneuver, just a single step
                if len(ns_sk_indices) > 1:
                    sk_avg_period = (ns_sk_indices[-1] - ns_sk_indices[0]) / (len(ns_sk_indices) - 1)
                else:
                    sk_avg_period = np.nan
            else:
                sk_avg_period = np.nan

            #if max_rate < self.ek_max_rate:
            #if non_thrust_std > self.ek_min_std:
            if sk_avg_period < 35 and median_rate < .0025:
                NS_detections.set_type(index, 'EK')
            else:
                NS_detections.set_type(index, 'CK')
        
        return


class AstrokinetixProcessor(Processor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ew_detector = AstrokinetixEastWestNodeDetector(cfg)
        self.ew_classifier = AstrokinetixEastWestClassifier(cfg)
        self.ns_detector = AstrokinetixNorthSouthNodeDetector(cfg)
        self.ns_classifier = AstrokinetixNorthSouthClassifier(cfg)
    
    def predict_object_nodes(self, satdata):
        
        try:
            EW_detections = self.ew_detector.run(satdata)
        except Exception as e:
            print(getattr(e, 'message', repr(e)))
            self.ew_failed_list.append(satdata.object_id)
            self.logger.warn(f'EW detection failed for object {satdata.object_id}')
            return pd.DataFrame([])
        
        try:
            self.ew_classifier.run(EW_detections, satdata.df['Longitude (deg)'])
            self.logger.info(f'EW Detections:\n{EW_detections.df}')
        except Exception as e:
            print(getattr(e, 'message', repr(e)))
            self.logger.info(f'EW Detections:\n{EW_detections.df}')
            self.ew_failed_list.append(satdata.object_id)
            self.logger.warn(f'EW classification failed for object {satdata.object_id}')
            return pd.DataFrame([])

        try:
            NS_detections = self.ns_detector.run(satdata, EW_detections)
        except Exception as e:
            print(getattr(e, 'message', repr(e)))
            self.ns_failed_list.append(satdata.object_id)
            self.logger.warn(f'NS detection failed for object {satdata.object_id}')
            return pd.DataFrame([])

        try:
            self.ns_classifier.run(NS_detections, satdata.df['Inclination (deg)'])
            self.logger.info(f'NS Detections:\n{NS_detections.df}')
        except Exception as e:
            print(getattr(e, 'message', repr(e)))
            self.logger.info(f'NS Detections:\n{NS_detections.df}')
            self.ns_failed_list.append(satdata.object_id)
            self.logger.warn(f'NS classification failed for object {satdata.object_id}')
            return pd.DataFrame([])

        # remove IM nodes
        NS_detections.df = NS_detections.df[NS_detections.df.Node != 'IM']
        NS_detections.df.reset_index(drop=True, inplace=True)

        predictions_df = pd.concat([EW_detections.df, NS_detections.df.iloc[:-1]], ignore_index=True) # drop duplicate ES node

        return predictions_df
