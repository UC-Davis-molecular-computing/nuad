# dsd

dsd stands for "DNA sequence designer". It is a Python library that enables one to specify constraints on a DNA nanostructure made from synthetic DNA. 

dsd allows one to go from a design with abstract "domains", such as `a`, `a*`, `b`, `b*`, to concrete DNA sequences, for example, 
`a` = 5'-CCCAG-3', `a*` = 5'-CTGGG-3', `b` = 5'-AAAAAAC-3', `b*` = 5'-GTTTTTT-3', obeying the constraints.

There are some pre-built constraints, for example limiting the number of G's in a domain or checking the partition function energy of a multi-domain strand (i.e., the strand's quantitative "secondary structure") according to the 3rd-party tools NUPACK and ViennaRNA. The user can also specify custom constraints.

# TODO write instructions
