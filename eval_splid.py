#!/usr/bin/env python

import sys
import logging
import pandas as pd
from pathlib import Path
import argparse

from splid import configure_console_logger
from splid.data import NodeDetectionSet
from splid.eval import NodeDetectionEvaluator




def merge_label_files(label_folder):
    """
    Merges all label files in a given folder into a single pandas DataFrame. 
    The filenames must be in the format <ObjectID>.csv, and the object id will
    be extracted from the filename and added as a column to the DataFrame.

    Args:
        label_folder (str): The path to the folder containing the label files.

    Returns:
        pandas.DataFrame: A DataFrame containing the merged label data.
    """
    label_data = []
    label_folder = Path(label_folder).expanduser()
    for file_path in label_folder.ls():
        df = pd.read_csv(file_path)
        oid_s = os.path.basename(file_path).split('.')[0]  # Extract ObjectID from filename
        df['ObjectID'] = int(oid_s)
        label_data.append(df)

    label_data = pd.concat(label_data)
    label_data = label_data[['ObjectID'] + list(label_data.columns[:-1])]
    return label_data


def run_evaluator(truth_path, predict_path, plot_object=None, selection=None, output_file=None, group=None, ignore=False):

    predict_path = Path(predict_path).expanduser()
    if predict_path.is_dir():
        predict_df = merge_label_files(predict_path)
        nodes_predict = NodeDetectionSet(predict_df)
    else:
        nodes_predict = NodeDetectionSet(predict_path)
    
    truth_path = Path(truth_path).expanduser()
    if truth_path.is_dir():
        truth_df = merge_label_files(truth_path)
        nodes_truth = NodeDetectionSet(truth_df)
    else:
        nodes_truth = NodeDetectionSet(truth_path)

    # Create a NodeDetectionEvaluator instance
    evaluator = NodeDetectionEvaluator(nodes_truth, nodes_predict, tolerance=args.tol, counts=args.counts, selection=selection, unmatched=args.unmatched, ignore=ignore)
    precision, recall, f2, rmse, pd, pfa = evaluator.score(output_file=output_file, group=group)
    logger.info('Competition Composite Score')
    logger.info(f'Precision: {precision:.2f}')
    logger.info(f'Recall: {recall:.2f}')
    logger.info(f'F2: {f2:.2f}')
    logger.info(f'RMSE: {rmse:.2f}')
    logger.info(f'Pd: {100*pd:.1f}%')
    logger.info(f'Pfa: {100*pfa:.1f}%')

    # Plot the evaluation for the selected object (if any)
    if plot_object:
        evaluator.plot(object_id=plot_object)
    return precision, recall, f2, rmse

if __name__ == "__main__":
    if 'ipykernel' in sys.modules:
        run_evaluator(plot_object=True)
    else:
        # Parse the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Provide more detailed output during processing',
        )
        parser.add_argument(
            '--predict_path',
            type=str,
            required=True,
            help='Path to the prediction file or folder.'
        )
        parser.add_argument(
            '--truth_path',
            type=str,
            required=True,
            help='Path to the truth file or folder.'
        )
        parser.add_argument(
            '--tol',
            type=int,
            default=6,
            help='Specify match tolerance in time steps',
        )
        parser.add_argument(
            '-o', '--output_file',
            type=str,
            default=None,
            help='File path and name of output file for detailed counts',
        )
        parser.add_argument(
            '-g', '--group',
            type=str,
            required=False,
            help='Filename of listing of selected objects to assess performance of',
        )
        parser.add_argument(
            '--plot_object',
            type=int,
            required=False,
            help='Object ID to plot.'
        )
        parser.add_argument(
            '-u', '--unmatched',
            action='store_true',
            help='Turn on output of missed truth nodes',
        )
        parser.add_argument(
            '-c', '--counts',
            action='store_true',
            help='Display performance counts for each object',
        )
        parser.add_argument(
            '-i', '--ignore_missing',
            action='store_true',
            help='Exclude missing objects from computed scores or tallies',
        )
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--ew',
            action='store_true',
            help='Evaluate only EW node performance'
        )
        group.add_argument(
            '--ns',
            action='store_true',
            help='Evaluate only NS node performance'
        )
        args = parser.parse_args()

        configure_console_logger()
        logger = logging.getLogger()
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        selection = None
        if args.ew:
            selection = 'EW'
        elif args.ns:
            selection = 'NS'
        
        group = None
        if args.group:
            group_file = Path(args.group)
            if not group_file.exists():
                raise FileNotFoundError(f'Group file {group_file} not found.')
            with group_file.open('rt') as f:
                group = []
                text = f.read()
                lines = text.split('\n')
                tokens = []
                for line in lines:
                    tokens += line.split(',')
                    for token in tokens:
                        numbers = token.strip().split('-')
                        if len(numbers) == 2:
                            group.extend(range(int(numbers[0]), int(numbers[1])+1))
                        elif len(numbers) == 1:
                            group.append(int(numbers[0]))
                        else:
                            raise ValueError('Entry {token} in {group_file} is invalid.')
            group = set(group)
            logger.info(f'Using group {group_file} with entries:\n{text}')
                        
        run_evaluator(args.truth_path, args.predict_path, args.plot_object, 
                      selection=selection, output_file=args.output_file,
                      group=group, ignore=args.ignore_missing)