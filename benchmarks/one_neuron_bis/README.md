# Description

This benchmark was created to check that the detected peak trains correspond to
the generated peak train.

# Usage

 1. `cd` into this directory
 2. launch `ipython`
 3. enter the following command: `%run main.py`
 4. and then:
  - `ans.plot_signal_and_peaks(t_min=5.0, t_max=10.0, thold=7.0)`
  - `ans.compare_peak_trains()`
  - `ans.compare_peaks_number()`
