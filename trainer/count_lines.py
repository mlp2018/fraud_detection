#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
# from builtins import (bytes, chr, dict, filter, hex, input,
#                       int, map, next, oct, open, pow, range, round,
#                       str, super, zip)
from sys import argv
from tensorflow.python.lib.io import file_io

def main():
    infile, outfile = argv[1:]
    with file_io.FileIO(outfile, mode='wb') as fout:
        print('Hello world!')
        with file_io.FileIO(infile, mode='rb') as fin:
            x = 0
            for line in fin.readlines():
                # print(line)
                # print(line, file=fout)
                x += 1
            print(x, file=fout)
        with file_io.FileIO(infile, mode='r') as fin:
            print(sum(map(lambda x: 1, fin)), file=fout)

if __name__ == '__main__':
    main()
