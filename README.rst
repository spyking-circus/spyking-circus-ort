SpyKING CIRCUS ORT
==================

.. image:: http://spyking-circus.readthedocs.io/en/latest/_images/circus.png
   :alt: SpyKING CIRCUS logo


*An online solution for spike sorting of large-scale extracellular recordings*

SpyKING CIRCUS ORT is a python code to allow online spike sorting on multi channel recordings. 
A preprint with the details of the offline algorithm can be found on BioRxiv at http://biorxiv.org/content/early/2016/08/04/067843. 
It has been tested on datasets coming from *in vitro* retina 
with 252 electrodes MEA, from *in vivo* hippocampus with tetrodes, *in vivo* and *in vitro* cortex 
data with 30 and up to 4225 channels, with good results. Synthetic tests on these data show 
that cells firing at more than 0.5Hz can be detected, and their spikes recovered with error 
rates at around 1%, even resolving overlapping spikes and synchronous firing. It seems to 
be compatible with optogenetic stimulation, based on experimental data obtained in the retina.

This **online** implementation of SpyKING CIRCUS is currently still under **active** development. Please do not hesitate to report issues with the issue tracker

:copyright: Copyright 2006-2018 by the SpyKING CIRCUS team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
