import logging
import pandas as pd
import numpy as np
from pathlib import Path

from .data import SatelliteTimeSeriesData, Node, NodeDetectionSet


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
