# Data directory

This directory contains a small real mouse droplet example and can also be used
for local input files.

Included file:

- `ace_mouse_tiny.h5ad`: a 960-cell subset of the public mouse droplet data used
  by the manuscript workflow. It contains raw counts in `layers["raw_counts"]`
  and metadata in `obs`.

Large single-cell datasets used in manuscript-scale analyses are not committed
to git. The included tiny dataset is sufficient for the reviewer quickstart and
notebook. For larger analyses, pass your own processed `.h5ad` file to the
example scripts with `--adata`.

See [../docs/data_format.md](../docs/data_format.md) for the required `AnnData`
structure.
