# gatemon-analysis
Library functions to fit and plot measurement data from microwave spectroscopy

The modules are
- `microwave_analysis.resonances` with fit functions for various complex
  valued resonance shapes and factory functions to make `lmfit` models for
  the fit functions.
- `microwave_analysis.lmfitmany` with functions to run the same `lmfit` model
  along the last axis of a large dataset. Can also run in multiple
  subprocesses to parallelize on multiple CPU cores.
- `microwave_analysis.twoport` with functions for calculating properties of
  two-port networks, i.e. converting scattering matrix to ABCD matrix.
- `microwave_analysis.sonnetparser` with function to read output files from
  the Sonnet EM software.

To make use of this package, either
- Put it in the same folder as the script where you try to import it.
- Put the folder containing this package into the PYTHONPATH environment variable, eg.
  https://stackoverflow.com/q/3402168 or https://stackoverflow.com/a/32609129
