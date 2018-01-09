This directory contains a list of networks which can be select in the
scaling benchmark.

For example, if you want to use `network_1` instead of `network_0` then
you just need to edit the line corresponding to the import of the
`network` module in the Python file `main.py` from:

    from networks import network_0 as network

to

    from networks import network_1 as network


## Descriptions

- `network`: all the blocks needed to do spike sorting (i.e. with preprocessing, clustering,
template matching).
- `network_0`: a reader block and a writer block.
- `network_1`: a reader, filter and writer blocks.
- `network_2`: all the block needed to do peak detection.