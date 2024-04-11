import logging
import pandas as pd
from pathlib import Path

from datetime import datetime, timedelta
import numpy as np


class SatelliteTimeSeriesData:
    '''Class to hold satellite time series data

    Data is read from .csv files with the following data columns:
        Eccentricity
        Semimajor Axis (m)
        Inclination (deg)
        RAAN (deg)
        Argument of Periapsis (deg)
        Mean Anomaly (deg)  -- removed for phase 1
        True Anomaly (deg)
        Latitude (deg)
        Longitude (deg)
        Altitude (m)
        X (m)
        Y (m)
        Z (m)
        Vx (m/s)
        Vy (m/s)
        Vz (m/s)

    The cartesian coordinates are expressed in the J2000 frame.
    The data is stored in a Pandas dataframe.

    Additionally, derived data is appended to the dataframe:
        Object ID   [from the file name]
        Time Index  [data row count]
    
    Time series data are provided in 2-hour steps.  Initial time is unknown.
    '''

    def __init__(self, datafile):

        self.logger = logging.getLogger()
        
        if not isinstance(datafile, Path):
            datafile = Path(datafile)
        if not datafile.exists():
            raise FileNotFoundError(f'File {datafile} does not exist.')
        
        self.object_id = int(datafile.stem)
        self.steps_per_day = 12  # 12 2-hour time steps
        self.feature_cols = [
           "Eccentricity",
           "Semimajor Axis (m)",
           "Inclination (deg)",
           "RAAN (deg)",
           "Argument of Periapsis (deg)",
           "True Anomaly (deg)",
           "Latitude (deg)",
           "Longitude (deg)",
           "Altitude (m)",
           "X (m)",
           "Y (m)",
           "Z (m)",
           "Vx (m/s)",
           "Vy (m/s)",
           "Vz (m/s)"
        ]

        self.df = pd.read_csv(datafile)
        # verify data file contains expected feature columns
        set1 = set(self.feature_cols)
        set2 = set(self.df.columns.to_list())
        if not set1.issubset(set2):
            raise ValueError(f'Satellite data file {datafile} missing feature columns {set1 - set2}')

        self.df['ObjectID'] = self.object_id
        self.df['TimeIndex'] = range(len(self.df))


class Node:

    def __init__(self,
                 t: timedelta = None,
                 index : int = None,
                 next_index : int = None,
                 ntype: str = None,
                 signal: str = None,
                 lon: float = None,
                 mtype: str = None):
        '''
        t -> timestamp\n
        index -> index associated with timestep array (optional)\n
        ntype -> node type; options: "ID", "IK", "AD", "IM" (optional)\n
        lon -> satellite longitude (deg) at t0 (optional)
        '''
        self.t = t
        self.index = index
        self.type = ntype
        self.signal = signal
        self.lon = lon
        if mtype:
            self.mtype = mtype
        else:
            self.mtype = np.nan


class NodeDetectionSet:
    '''Class contains the set of detected or truth nodes.

    Node files are .csv with following columns:
        ObjectID
        TimeIndex
        Direction
        Node
        Type
    
    Direction Labels: EW, NS
    Node Labels:      SS, ES, ID, AD, IK
    Type Labels:      NK, CK, EK, HK
    '''

    def __init__(self, arg1=None):
        if isinstance(arg1, pd.DataFrame):
            self.df = arg1
            self.df.sort_values(['ObjectID', 'TimeIndex', 'Direction'], inplace=True)
        elif isinstance(arg1, Path):
            self.load(arg1)
        elif arg1 is not None:
            raise ValueError(f'Unknown argument to {type(self).__name__}: {arg1}')
        else:
            self.df = pd.DataFrame(columns=['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type'])
        self.df['Direction'] = pd.Categorical(self.df['Direction'], categories=['EW', 'NS'] + ['ES'])
        self.df['Node'] = pd.Categorical(self.df['Node'], categories=['ID', 'AD', 'IK', 'IM'] + ['SS', 'ES'])
        self.df['Type'] = pd.Categorical(self.df['Type'], categories=['NK', 'CK', 'EK', 'HK'] + ['ES'])
    
    def load(self, datafile):
        self.df = pd.read_csv(datafile)

    def save(self, datafile):
        self.df.to_csv(datafile, index=False)

    def add(self, arg1, arg2=None):
        if not arg2:
            detection = arg1.to_dict()
        else:
            detection = {"ObjectID":arg1, "TimeIndex":arg2.index, 
                    "Direction":arg2.signal, "Node":arg2.type, "Type":'NK' if arg2.type == 'ID' else np.nan}
        self.df = pd.DataFrame(self.df.to_dict('records') + [detection])
        self.df['Direction'] = pd.Categorical(self.df['Direction'], categories=['EW', 'NS'] + ['ES'])
        self.df['Node'] = pd.Categorical(self.df['Node'], categories=['ID', 'AD', 'IK', 'IM'] + ['SS', 'ES'])
        self.df['Type'] = pd.Categorical(self.df['Type'], categories=['NK', 'CK', 'EK', 'HK'] + ['ES'])
    
    def set_node(self, index, node):
        # use .iloc[].name to convert index into name for .loc[] assignment
        self.df.loc[self.df.iloc[index].name, 'Node'] = node

    def set_type(self, index, type):
        # use .iloc[].name to convert index into name for .loc[] assignment
        self.df.loc[self.df.iloc[index].name, 'Type'] = type
    
    def __len__(self):
        return len(self.df)
    
    def get_object(self, object_id):
        return self.df[self.df.ObjectID == object_id]


class TruthNodeDetectionSet(NodeDetectionSet):

    def __init__(self, datafile):
        super().__init__(datafile)
    
    def get_object_old(self, object_id):
        '''Retrieve truth data for a specific object ID.
        
        Returns separate EW and NS dataframes with following columns:
            ObjectID
            TimeIndex
            EW / NS
        
        EW/NS entries are formated as <node>-<type>.
        '''

        # extract entries for current object
        object_truth = self.df[self.df['ObjectID'] == object_id]

        # Separate the 'EW' and 'NS' types in the ground truth
        # Create copy before modifying
        object_truth_EW = object_truth[object_truth['Direction'] == 'EW'].copy()
        object_truth_NS = object_truth[object_truth['Direction'] == 'NS'].copy()
        
        # Create 'EW' and 'NS' labels and fill 'unknown' values
        object_truth_EW['EW'] = object_truth_EW['Node'].astype(str) + '-' + object_truth_EW['Type'].astype(str)
        object_truth_NS['NS'] = object_truth_NS['Node'].astype(str) + '-' + object_truth_NS['Type'].astype(str)
        object_truth_EW.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
        object_truth_NS.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)

        return object_truth_EW, object_truth_NS
