- [ ] Set up an architecture similar to the one used in
`scaling/electrodes` but with a fixed number of electrodes (i.e. 256)
and a varying number of cells.
  - [x] Add `README` file.
  - [x] Add `main.py` file.
  - [x] Add `networks` directory.
  - [x] Add `outputs` directory.
  - [ ] Add an example (with its main output figure) in the README.
- [ ] Understand why 4 `RunTimeError` appear at some point (probably
during the introspection). The errors were raised by some `Tkinter`
callbacks because the maximum recursion depth was exceeded.
- [ ] Modify the `updater` block to send templates in a control manner
to the `fitter` block.
