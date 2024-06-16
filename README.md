# OOI Profiler Nitrate Data Retriever

This package simplifies the retrieval of *in situ* nitrate data from [Ocean Observatories Initiative Endurance Array](https://oceanobservatories.org/array/coastal-endurance/) profiling moorings. The retrieved data is then binned and quality controlled before being saved to the local machine.

## Installation

Assuming you have [Python](https://docs.python.org/3/) and [conda](https://conda.io/projects/conda/en/latest/index.html) installed:

1. Clone the repository and switch to the directory:
    ```
    git clone https://github.com/andrew-s28/ooi-profiler-nitrate-retriever.git
    cd ooi-profiler-nitrate-retriever/
    ```

2. Create and activate a [conda](https://conda.io/projects/conda/en/latest/index.html) environment:
    ```
    conda env create -f environment.yml
    conda activate ooi
    ```

3. You can now retrieve data:
    ```
    python ./scripts/retrieve_profiler_data.py CE01ISSP -s 2017-08-15 -e 2020-12-16 -p ./data
    ```

## Usage

The only stations currently implemented are the [Oregon Inshore Surface Piercing Profiler (CE01ISSP)](https://oceanobservatories.org/site/ce01issp/) and the [Oregon Shelf Surface Piercing Profiler Mooring (CE02SHSP)](https://oceanobservatories.org/site/ce02shsp/). This script accesses the [NUTNRJ dataset](https://oceanobservatories.org/instrument-series/nutnrj/), which includes *in situ* nitrate concentration data as well as typical oceanographic data (e.g., salinity, temperature, and density). Both a binned dataset and the raw data without any modification are saved to the local machine.

> :warning: **The binning and QC process is expensive**: The QC process implemented here requires running a least squares regression on every observation, of which there can be hundreds of thousands. The binning process requires using the groupby operation, [which is expensive in xarray](https://docs.xarray.dev/en/v2023.06.0/user-guide/dask.html#optimization-tips). Several speedups have been implemented (namely, using [joblib](https://joblib.readthedocs.io/en/stable/) to parallelize the regression and [flox](https://flox.readthedocs.io/en/latest/) to speed up the grouping), but the code can still take over an hour to finish if downloading all available data onto a typical consumer machine.

Only a site name is required. The only currently implemented sites are [CE01ISSP](https://oceanobservatories.org/site/ce01issp/) and [CE02SHSP](https://oceanobservatories.org/site/ce02shsp/). Separate sites by a space to get data from them both
```
python ./scripts/retrieve_profiler_data.py CE01ISSP CE02SHSP
```

Find help using the `--help` flag:
```
python ./scripts/retrieve_profiler_data.py --help
```
```
usage: retrieve_profiler_data.py [-h] [-p [path]] [-f [file]] [-s [start]] [-e [end]] [-n [njobs]] site [site ...]

Download OOI Endurance Array profiler nitrate data, QC fits, bin, and save.

positional arguments:
  site                  Site to download from, either CE01ISSP or CE02SHSP

options:
  -h, --help            show this help message and exit
  -p [path], --path [path]
                        Path to folder for saving output files
  -s [start], --start [start]
                        Start date in YYYY-MM-DD format (default: 2010-01-01)
  -e [end], --end [end]
                        End date in YYYY-MM-DD format (default: 2024-06-16)
  -n [njobs], --njobs [njobs]
                        number of jobs to run in parallel - see joblib.Parallel docs for more details
```

## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
