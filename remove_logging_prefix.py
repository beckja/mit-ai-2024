#! /usr/bin/env python

import sys
from pathlib import Path


filename = sys.argv[1]
print(f'Filename {filename} is being processed')

with open(filename, 'rt') as fin:
    with open('stripped' + filename, 'wt') as fout:
        for line in fin.readlines():
            tokens = line.split(']')
            if len(tokens) == 1:
                fout.write(tokens[0])# + '\n')
            elif len(tokens) == 2:
                fout.write(tokens[1])# + '\n')
            else:
                fout.write(tokens[1])
                for i in range(2,len(tokens)):
                    fout.write(']' + tokens[i])

