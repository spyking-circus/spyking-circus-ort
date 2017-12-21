- [ ] Create a script `pregenerate.py` to pregenerate the synthetic data
  - [ ] First, find/generate the probe to use during the pregeneration
    - [x] Modify the `Probe` class in `circusort.io` to handle path
    management outside `__init__`
    - [x] Implement method to generate a probe
    - [x] Implement method to save a probe
    - [x] Implement method to load a probe
    - [ ] Be less restrictive concerning the probe filenames
  - [x] Find/generate the templates to use during the pregeneration
    - [x] Implement method to generate some templates
    - [x] Implement method to save some templates
    - [x] Implement method to load some templates
  - [x] Find/generate the trains to use during the pregeneration
    - [x] Implement method to generate some trains
    - [x] Implement method to save some trains
    - [x] Implement method to load some trains
  - [ ] Load cells from the `generation` directory
  - [ ] Add position parameters (time dependent) to the generated cells
  - [x] Generate the signal (i.e. raw data)
    - [x] ~~Add gaussian noise to the signal~~
    - [x] ~~Add template waveforms to the signal~~
    - [x] Use the `synthetic_generator` and `writer` block to generate
    the signal
    - [x] Let background thread generate trains chunk by chunk based on
    the global trains
    - [x] Move code from `io.pregenerate` to `net.pregenerator`
    - [x] Use 16 bit signed (or unsigned) integer to generate data
  - [x] Pregenerate signal precisely for a given duration
- [ ] Correct the main script in `main.py`
  - [x] Add a reader block (i.e. read pregenerated data)
  - [ ] Correct the part of the code to analyse the results
  - [x] Change output format from RAW to HDF5
  See [[h5py] Multiprocess concurrent write and read](http://docs.h5py.org/en/latest/swmr.html?highlight=append#multiprocess-concurrent-write-and-read)
- [ ] Move the utils from `utils.py` to `circusort`'s core
- [ ] Explain how to use a non-empty initial template dictionary (in the
README)
- [x] Correct README (bugs Pierre)
 