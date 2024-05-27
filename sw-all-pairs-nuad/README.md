# Overview

This small project implements a command line interface to score the
local alignment of all pairs of strings provided using a vectorized
version of the Smith-Waterman algorithm (with Got oh affine gap
penalty extension).  In particular, the reverse complement of each
sequence is aligned with the original of every other.  Thus, if there
are N sequences, then N(N-1)/2 scores are returned.

# Building from source

## Requirements to build

- a C99 compliant compiler and standard library (e.g., a modern version of gcc or clang)

## Building

Unpack the tarball:
`tar xfz sw-score-all-pairs.tar.gz`

Change into source directory:
`cd sw-score-all-pairs`

Run Make to build:
`make`

This will build the binary `sw-score-all-pairs`.  It can
be moved to any directory or used in place.

If you prefer to use a C99 compliant compiler other than gcc, adjust
the CC variable in the Makefile or on the command line before invoking
`make`.

# Using the program

The program has been designed with some simplifying assumptions that
can be modified in the future, if required.  Firstly, all strings to
be aligned are assumed to be of the same length.  Secondly, it is
assumed that the alignment of all pairs be scored.  More specifically,
align the reverse complement of each sequence with the original of
every other.

## command line syntax

usage: sw-score-all-pairs <match> <mismatch> <gap_open> <gap_extend> <num> <length> <sequence>

	<match>      score for matching characters
	<mismatch>   score for mismatching characters
	<gap_open>   score to open a gap
	<gap_extend> score to extend a gap
	<num>        total number of equi-length sequences to compare
	<length>     length of every sequence
	<sequence>   concatenation of every sequence - must have length <num>*<length>

## notes on parameters

<match> should be positive, <mismatch> should be negative, <gap_open>
should be negative, and <gap_extend> should be at most 0.

<sequence> must contain characters from the set 'acgtACGT'.
<sequence> has a particular format to for easy piping with other
programs.  In particular, if <num> is 4 and <length> is 10, then the
concatenation of those sequences, provided as <sequence>, must have
length 40:

                     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
                     \        /\        /\        /\        /
                        seq0      seq1      seq2      seq3
                    id:  0         1         2         3
    
We refer to the substrings of this concatenation by their ids, indexed
from 0.  In the example above, ids 0...3.

## output

If there are N sequences, the output will consist of N(N-1)/2 triples,
each given on a newline.  Each triple <i j score> will report the
alignment score of the reverse complement of sequence with id i to
sequence with id j.  Using our above example, output may look like the
following:

                    0 1 30
                    0 2 0
                    0 3 37
                    1 2 20
                    1 3 33
                    2 3 0

## examples

### example 1
Here is the command and output for checking the sequence set
{aaaaaaaaaa, tttttttttt, cccccccccc, gggggggggg} using reasonable
scoring parameters.

                    ./sw-score-all-pairs 10 -15 -33 0 4 8 aaaaaaaattttttttccccccccgggggggg
                    
                    0 1 80
                    0 2 0
                    0 3 0
                    1 2 0
                    1 3 0
                    2 3 80

Since the reverse complement of seq0 (aaaaaaaaaa) is equal to seq1
(tttttttttt), they get an optimal alignment score of 80: length 8 * 10
(match score); similarly for seq2 (cccccccccc) and seq3 (gggggggggg).
However, all other sequences are orthogonal and thus have an optimal
alignment score of 0.

### example 2
Included in this tarball is the file `random_seq.txt` which contains a
random DNA sequence of length 42000.  Assuming it is the concatenation
of 1000 sequences, each of length 42, the following command will score
all pairs (as described above) using reasonable scoring parameters:

                   ./sw-score-all-pairs 10 -15 -33 -3 1000 42 `cat random_seq.txt` > scores.txt


# Authors

Coded by Chris Thachuk (2015) with the main vectorized implementation
being only a slight modification of code found in the SHRiMP aligner
