#!/usr/bin/env python

import sys
import argparse
import logging
import time
from pathlib import Path
from configparser import ConfigParser

from splid import configure_console_logger
from splid import process


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Run script for MIT ARCLab Prize for Innovation in Space 2024'
    )
    parser.add_argument(
        '--cfg',
        metavar='cfgfile',
        default='splid.ini',
        help='SPLID configuration file',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Provide more detailed output during processing',
    )
    parser.add_argument(
        '-t', '--train',
        metavar='training_path',
        default=None,
        help='Run training mode with data at given path',    )
    args = parser.parse_args()

    configure_console_logger()
    logger = logging.getLogger()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    cfgfile = Path(args.cfg)
    if not cfgfile.exists():
        logger.error(f'Configuration file {cfgfile} not found.')
        sys.exit(1)
    logger.info(f'Running splid with config file {cfgfile}...')

    config = ConfigParser()
    cfgfile = Path(args.cfg)
    config.read(cfgfile.as_posix())

    # check for verbose option in config file
    verbose = config['general'].getboolean('verbose')
    if verbose:
        logger.setLevel(logging.DEBUG)

    start_time = time.time()

    #TODO: add config key checks before retrieval 
    try:
        section = config['general']['section']
        cfg = config[section]
    except:
        logger.error(f"No section {config['general']['section']} found in config file {cfgfile}.")
        sys.exit(2)
    try:
        Processor = getattr(process, cfg['processor'])
        logger.info(f"Using processor {cfg['processor']}")
    except:
        logger.error(f"Processor class {cfg['processor']} is not valid.")
        sys.exit(3)

    processor = Processor(cfg)
    if args.train is not None:
        processor.train(Path(args.train))
    else:
        nodes = processor.run(datapath=Path(config['general']['input_data_path']))

        # Save the prediction into a csv file 
        output_file_path = Path(config['general']['output_file_path'])
        trace_object = cfg.getint('trace_object', fallback=None)
        if trace_object:
            output_file_path = output_file_path.parent.joinpath(f"trace_{trace_object}.csv")
        nodes.save(output_file_path)
        logger.info(f"Saved predictions to: {output_file_path}")
    
    stop_time = time.time()
    run_time = stop_time - start_time
    logger.info(f'Total run time: {run_time:.1f} seconds')

    # work-around for submittal issues with evalAI
    if config['general'].getboolean('pause'):
        time.sleep(360)
