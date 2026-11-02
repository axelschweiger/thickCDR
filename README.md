# python prgrams to work with sea ice thickness CDR
# matching cdr records with model data in netCDF format

# match_thickCDR_model.py is the main program library with the driver function doing the work. 

The CompareThickCDR_and_model.ipynb notebook shows how the data are used. 

There are several external programs are used. Those are located in: 
git@github.com:axelschweiger/im.git
https://github.com/axelschweiger/im

None of this is written for public use... so milage may vary on how well this works.

## Extracting model values at CDR points

Model runs are listed in `RUNS` in `match_thickCDR_model.py`. To add a run, add an entry there, then:

    python match_thickCDR_model.py <model_id>            # one pass, all sources
    python match_thickCDR_model.py <model_id> --twostep  # old per-source matchDB + merged files

The one-pass mode writes `extractDir/<grid>/<model_id>/<model_id>.extract.<var0>.<var1>.nc`
(about 1-2 min per run, ~1.3 GB memory; the two-step mode took ~10 min and 9 GB for 1c).
`readMerged()` uses that file if it exists, otherwise the per-source `*.merged.*` files.
Both modes give the same points and model times. About 0.04% of points are almost exactly
equidistant from two grid cells and the two modes can pick different cells for those.

Monthly files (`heff`, `snow`) are time-stamped on the 1st of the month. They are moved to
mid-month before matching. Observations are matched to the nearest model time within 15 days.

`toThick()` returns a copy and subtracts the 0.29 m submarine draft bias only once. It is safe
to call again on the result.
