# OOI Profiler Nitrate Data Retriever

This package simplifies the retrieval of *in situ* nitrate data from [Ocean Observatories Initiative Endurance Array](https://oceanobservatories.org/array/coastal-endurance/) profiling moorings. The retrieved data is then binned and quality controlled before being saved locally.

## Installation

It is recommended to use [uv](https://docs.astral.sh/uv/) to run this script:

1. Clone the repository and switch to the directory:

  ```bash
  $ git clone https://github.com/andrew-s28/ooi-profiler-nitrate-retriever.git
  $ cd ooi-profiler-nitrate-retriever/
  ```

2. Run the script directly with [uv run](https://docs.astral.sh/uv/reference/cli/#uv-run):

  ```bash
  $ uv run ./scripts/retrieve_profiler_data.py CE01ISSP -s 2017-08-15 -e 2020-12-16 -p ./data
  ```

  uv handles all of the dependencies auto-magically!

3. If you instead prefer to use a virtual environment, you can also use [uv sync](https://docs.astral.sh/uv/reference/cli/#uv-sync) for that:

  ```bash
  $ uv sync
  $ .venv/bin/activate
  $ python ./scripts/retrieve_profiler_data.py CE01ISSP -s 2017-08-15 -e 2020-12-16 -p ./data
  ```

You can also use pip or conda and the included `requirements.txt` file.

## Usage

The only stations currently implemented are the [Oregon Inshore Surface Piercing Profiler (CE01ISSP)](https://oceanobservatories.org/site/ce01issp/) and the [Oregon Shelf Surface Piercing Profiler Mooring (CE02SHSP)](https://oceanobservatories.org/site/ce02shsp/). This script accesses the [NUTNRJ dataset](https://oceanobservatories.org/instrument-series/nutnrj/), which includes *in situ* nitrate concentration data as well as typical oceanographic data (e.g., salinity, temperature, and density). Both a binned dataset and the raw data without any modification are saved to the local machine.

> :warning: **The binning process is resource intensive**: The binning process requires using the groupby operation, [which is expensive in xarray](https://docs.xarray.dev/en/v2023.06.0/user-guide/dask.html#optimization-tips). Optimization using [flox](https://flox.readthedocs.io/en/latest/) to speed up the groupby has been implemented, but the code can still take ten to twenty minutes to finish if downloading all available data onto a typical consumer machine.

Only a site name is required. The only currently implemented sites are [CE01ISSP](https://oceanobservatories.org/site/ce01issp/) and [CE02SHSP](https://oceanobservatories.org/site/ce02shsp/). Separate sites by a space to get data from more than one:

```bash
$ uv run ./scripts/retrieve_profiler_data.py CE01ISSP CE02SHSP
```

Find help using the `--help` flag:

```bash
$ uv run ./scripts/retrieve_profiler_data.py --help
```
```
usage: retrieve_profiler_data.py [-h] [-p [path]] [-s [start]] [-e [end]] site [site ...]

Download OOI Endurance Array profiler nitrate data, QC fits, bin, and save.

positional arguments:
  site                 Site to download from, either CE01ISSP or CE02SHSP

options:
  -h, --help           show this help message and exit
  -p, --path [path]    Path to folder for saving output files
  -s, --start [start]  Start date in YYYY-MM-DD format (default: 2010-01-01)
  -e, --end [end]      End date in YYYY-MM-DD format (default: 2025-03-24)
```

## Example

You can view an interactive Python notebook in the [examples directory](examples/ooi_profiler_nitrate_analysis.ipynb) that goes through each step used in the data retriever script. After cloning the repository and setting up your environment, simply open and run the notebook `ooi_profiler_nitrate_analysis.ipynb` to view the steps taken in the access, quality control, binning, and validation of OOI profiler nitrate datasets.

## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
