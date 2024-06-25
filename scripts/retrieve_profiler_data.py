"""
Download OOI Endurance Array profiler nitrate data, QC fits, bin, and save.

"""
# Python Standard Library
import argparse
from datetime import datetime
import io
import os
import re
import warnings

# External Modules
from bs4 import BeautifulSoup
import gsw
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm as tq
import xarray as xr
from flox.xarray import xarray_reduce


def _filter_dates(nc_files: list, start_date: str, end_date: str) -> list:
    date_regex = r'[0-9]+T[0-9]+\.[0-9]+-[0-9]+T[0-9]+\.[0-9]+'
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    # get start and end dates of each deployment from the file names
    nc_files_start_dates = [datetime.strptime(re.search(date_regex, f).group(0).split('-')[0], '%Y%m%dT%H%M%S.%f') for f in nc_files]
    nc_files_end_dates = [datetime.strptime(re.search(date_regex, f).group(0).split('-')[1], '%Y%m%dT%H%M%S.%f') for f in nc_files]
    # filter by start date, including any data files that end after the start date
    nc_files = [f for i, f in enumerate(nc_files) if nc_files_end_dates[i] >= start_date]
    # filter by end date, including any data files that start before the end date
    nc_files = [f for i, f in enumerate(nc_files) if nc_files_start_dates[i] <= end_date]
    return nc_files


def _get_attrs(ds) -> tuple[dict, dict]:
    global_attrs = ds.attrs
    var_attrs = {}
    for var in list(ds.variables):
        var_attrs.update({var: ds[var].attrs})
    return global_attrs, var_attrs


def _assign_attrs(ds: xr.Dataset, global_attrs: dict, var_attrs: dict) -> xr.Dataset:
    ds.attrs = global_attrs
    for var in list(ds.variables):
        ds[var].attrs = var_attrs[var]
    return ds


def list_files(url: str, start_date: str, end_date: str, tag=r'.*\.nc$') -> list[str]:
    """
    Function to create a list of the netCDF data files in the THREDDS catalog
    created by a request to the M2M system. Obtained from 2022 OOIFB workshop

    Args:
        url (str): URL to a THREDDS catalog specific to a data request
        tag (regexp, optional): Regex pattern used to distinguish files of interest. Defaults to r'.*\\.nc$'.

    Returns:
        array: list of files in the catalog with the URL path set relative to the catalog
    """
    with requests.session() as s:
        page = s.get(url).text

    soup = BeautifulSoup(page, 'html.parser')
    pattern = re.compile(tag)
    nc_files = [node.get('href') for node in soup.find_all('a', string=pattern)]
    nc_files = [re.sub('catalog.html\\?dataset=', '', file) for file in nc_files]
    nc_files = _filter_dates(nc_files, start_date, end_date)
    return nc_files


def _drop_unused_vars(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.drop_vars([
        'nutnr_nitrogen_in_nitrate_qc_executed',
        'nutnr_spectrum_average',
        'nutnr_fit_base_2',
        'nutnr_fit_base_1',
        'year',
        'salinity_corrected_nitrate_qartod_results',
        'salinity_corrected_nitrate_qc_results',
        'sea_water_pressure_qc_executed',
        'nutnr_current_main',
        'sea_water_practical_salinity_qc_executed',
        'nitrate_concentration_qc_results',
        'humidity',
        'voltage_main',
        'sea_water_pressure_qc_results',
        'spectral_channels',
        'temp_spectrometer',
        'temp_lamp',
        'day_of_year',
        'nutnr_nitrogen_in_nitrate',
        'nitrate_concentration',
        'nutnr_absorbance_at_254_nm',
        'nutnr_absorbance_at_350_nm',
        'temp_interior',
        'nutnr_bromide_trace',
        'sea_water_temperature_qc_results',
        'nutnr_integration_time_factor',
        'lamp_time',
        'nitrate_concentration_qc_executed',
        'time_of_sample',
        'nutnr_nitrogen_in_nitrate_qc_results',
        'nitrate_concentration_qartod_results',
        'sea_water_practical_salinity_qc_results',
        'sea_water_temperature_qc_executed',
        'nutnr_voltage_int',
        'salinity_corrected_nitrate_qc_executed',
        'suspect_timestamp',
        'aux_fitting_1',
        'nutnr_fit_rmse',
        'dark_val',
        'aux_fitting_2',
        'voltage_lamp',
    ]).drop(
        'wavelength'
    )
    return ds


def _comput_tsrho(ds: xr.Dataset) -> xr.Dataset:
    # renames and computes variables
    ds = ds.swap_dims({'obs': 'time'})
    # sea_water_temperature and sea_water_practical_salinity have junk values
    ds = ds.drop_vars(['sea_water_temperature', 'sea_water_practical_salinity', 'sea_water_pressure'])
    ds = ds.rename({
        'ctdpf_j_cspp_instrument_recovered-sea_water_temperature': 'temperature',
        'ctdpf_j_cspp_instrument_recovered-sea_water_practical_salinity': 'practical_salinity',
        'nutnr_dark_value_used_for_fit': 'dark_val',
        'int_ctd_pressure': 'pressure',
        })
    ds['absolute_salinity'] = gsw.conversions.SA_from_SP(
        SP=ds['practical_salinity'],
        p=ds['pressure'],
        lon=ds['lon'],
        lat=ds['lat']
    )
    ds['absolute_salinity'].attrs = {
        'long_name': 'Seawater Absolute Salinity',
        'standard_name': 'sea_water_absolute_salinity',
        'units': 'g/kg'
    }
    ds['conservative_temperature'] = gsw.conversions.CT_from_t(
        SA=ds['absolute_salinity'],
        t=ds['temperature'],
        p=ds['pressure']
    )
    ds['conservative_temperature'].attrs = {
        'long_name': 'Seawater Conservative Temperature',
        'standard_name': 'sea_water_conservative_temperature',
        'units': 'degrees_Celsius',
        'units_metadata': 'temperature: on_scale'
    }
    ds['sigma_t'] = gsw.density.rho(
        SA=ds['absolute_salinity'],
        CT=ds['conservative_temperature'],
        p=ds['pressure']
    ) - 1000
    ds['sigma_t'].attrs = {
        'long_name': 'Seawater Density Anomaly',
        'standard_name': 'sea_water_sigma_t',
        'units': 'kg/m^3',
    }
    ds['sigma_theta'] = gsw.density.sigma0(
        SA=ds['absolute_salinity'],
        CT=ds['conservative_temperature']
    )
    ds['sigma_theta'].attrs = {
        'long_name': 'Seawater Potential Density Anomaly',
        'standard_name': 'sea_water_sigma_theta',
        'units': 'kg/m^3',
        'reference_pressure': 0
    }
    return ds


def _nutnr_qc(
    spectral_channels: np.ndarray,
    dark_val: np.ndarray,
    salinity: np.ndarray,
    time: np.ndarray,
    rmse_lim: float,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    # does the actual QC as vectorized numpy operations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in log10")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10")

        wavelengths = np.arange(190, 394.01, .8)[30:]
        spectral_channels = (spectral_channels.T - dark_val).T[:, 30:]
        mask = np.full(time.shape, True, dtype=bool)
        a_all = np.empty(len(time))
        a_all[:] = np.nan
        step_1, step_2, step_3, step_4 = 0, 0, 0, 0

        # mask observations with low salinity values
        mask[(salinity <= 28) & (mask)] = False
        step_1 = np.count_nonzero(~mask)

        # mask spectra with non-finite or negative values
        mask[(np.any(~np.isfinite(spectral_channels), axis=1)) & (mask)] = False
        mask[(np.any(spectral_channels <= 0, axis=1)) & (mask)] = False
        step_2 = np.count_nonzero(~mask) - step_1

        # fit spectra to linear model between wavelengths ~360 nm to 380 nm and remove bad fits 
        wl_fit = wavelengths[217-30:241-30]
        spec_fit = np.log10(spectral_channels[:, 217-30:241-30])
        rmse_all = np.empty(len(spec_fit))
        rmse_all[:] = np.nan
        for i, sp in enumerate(spec_fit):
            if np.any(~np.isfinite(sp)):
                if mask[i]:
                    print("Why is this happening? This shouldn't be happening.") # any non-finite values should have been removed in step 2
                    mask[i] = False
                    step_2 += 1
            else:
                a, ssres, _, _ = np.linalg.lstsq(np.vstack([np.ones(wl_fit.shape), wl_fit]).T, sp, rcond=None)
                residuals = sp - wl_fit*a[1] - a[0]
                rmse = np.sqrt(np.sum(residuals**2)/(residuals.size-2))
                rmse_all[i] = rmse
                if rmse > rmse_lim:
                    if mask[i]:
                        mask[i] = False
                        step_3 += 1
                else:
                    a_all[i] = a[1]

        # remove spectra with slope far from the mean slope
        a_all[~mask] = np.nan
        mask[(np.abs(a_all - np.nanmean(a_all)) >= 3*np.nanstd(a_all)) & (mask)] = False
        step_4 = np.count_nonzero(~mask) - step_1 - step_2 - step_3
    return mask, a_all, step_1, step_2, step_3, step_4


def _nutnr_qc_wrapper(ds: xr.Dataset, rmse_lim: float = 0.02) -> xr.Dataset:
    """
    Remove bad fits in OOI nutnr datasets

    Args:
        ds (Dataset): OOI nutnr dataset
        rmse_lim (int, optional): Maximum RMSE for fit to be kept. Defaults to 1000.
    """
    # provide numpy arrays to apply QC
    mask, a_all, step_1, step_2, step_3, step_4 = _nutnr_qc(
        ds.spectral_channels.values,
        ds.dark_val.values,
        ds.practical_salinity.values,
        ds.time.values,
        rmse_lim=rmse_lim
    )
    # mask the data
    ds = ds.where(xr.DataArray(mask, coords={'time': ds.time.values}), drop=True)
    return ds, len(ds.time.values), step_1, step_2, step_3, step_4


def split_profiles(ds: xr.Dataset) -> list:
    """
    Split the data set into individual profiles, where each profile is a
    collection of data from a single deployment and profile sequence. The
    resulting data sets are returned in a list.

    :param ds: data set containing the profile data
    :return: a list of data sets, one for each profile
    """
    # split the data into profiles, assuming at least 120 seconds between profiles
    dt = ds.where(ds['time'].diff('time') > np.timedelta64(120, 's'), drop=True).get_index('time')

    # process each profile, adding the results to a list of profiles
    profiles = []
    jback = np.timedelta64(30, 's')  # 30 second jump back to avoid collecting data from the following profile
    for i, d in enumerate(dt):
        # print(d)
        # pull out the profile
        if i == 0:
            profile = ds.sel(time=slice(ds['time'].values[0], d - jback))
        else:
            profile = ds.sel(time=slice(dt[i - 1], d - jback))

        # add the profile to the list
        profiles.append(profile)

    # grab the last profile and append it to the list
    profile = ds.sel(time=slice(d, ds['time'].values[-1]))
    profiles.append(profile)
    return profiles


def profiler_binning(d, z, z_lab='depth', t_lab='time', offset=0.5) -> xr.DataArray | xr.Dataset:
    """
    Bins a profiler time series into daily bins and depth bins.
    Removes any non-numeric data types, including any time types,
    outside of the coordinates.

    input:
    d = xr.Dataset with coordinates depth and time
    z = depth bins array
    z_lab, t_lab = labels for depth, time in d

    returns:
    Binned xr.Dataset
    Args:
        d (xr.dataset): OOI profiler dataset
        z (array): edges of depth/pressure bins
        z_lab (str, optional): name of depth/pressure in dataset. Defaults to 'depth'.
        t_lab (str, optional): name of time in dataset. Defaults to 'time'.
        offset (float, optional): Distance from location to CTD (positive when CTD is higher).
            Defaults to 0.5.

    Returns:
        xr.dataset: binned dataset
    """
    types = [d[i].dtype for i in d]
    vars = list(d.keys())
    exclude = []
    for i, t in enumerate(types):
        if not (np.issubdtype(t, np.number)):
            exclude.append(vars[i])
    d = d.drop_vars(exclude)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        out = xarray_reduce(
            d,
            d[t_lab],
            d[z_lab],
            func='nanmean',
            expected_groups=(None, z),
            isbin=[False, True],
            method='map-reduce',
            skipna=True,
        )

    depth = np.array([x.mid + offset for x in out.depth_bins.values])
    out[z_lab] = ([z_lab + '_bins'], depth)
    out = out.swap_dims({z_lab + '_bins': z_lab})
    out = out.drop_vars([z_lab + '_bins'])

    return out


def _parse_args() -> tuple[str, str, str, str]:
    parser = argparse.ArgumentParser(
        description="Download OOI Endurance Array profiler nitrate data, QC fits, bin, and save."
    )
    parser.add_argument('site', metavar='site', type=str, nargs='+',
                        help="Site to download from, either CE01ISSP or CE02SHSP")
    parser.add_argument('-p', '--path', metavar='path', type=str, nargs='?',
                        help="Path to folder for saving output files",
                        default='./output/')
    parser.add_argument('-s', '--start', metavar='start', type=str, nargs='?', default='2010-01-01',
                        help="Start date in YYYY-MM-DD format (default: 2010-01-01)")
    parser.add_argument('-e', '--end', metavar='end', type=str, nargs='?', default=datetime.now().strftime('%Y-%m-%d'),
                        help=f"End date in YYYY-MM-DD format (default: {datetime.now().strftime('%Y-%m-%d')})")
    args = parser.parse_args()
    args = vars(args)
    sites = args['site']
    out_path = args['path']
    start_date = args['start']
    end_date = args['end']
    sites = [site.upper() for site in sites]
    return sites, out_path, start_date, end_date


def _make_dirs(out_path: str, site: str) -> None:
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, site)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(os.path.join(out_path, 'raw')):
        os.mkdir(os.path.join(out_path, 'raw'))
    # if not os.path.exists(os.path.join(out_path, 'qced')):
    #     os.mkdir(os.path.join(out_path, 'qced'))
    # if not os.path.exists(os.path.join(out_path, 'binned')):
    #     os.mkdir(os.path.join(out_path, 'binned'))
    return out_path


def _get_datasets(site: str, start_date: str, end_date: str) -> list:
    if site == 'CE01ISSP':
        # setup defaults to use in subsequent data queries
        refdes = site + "-SP001-06-NUTNRJ000"
        method = "recovered_cspp"
        stream = "nutnr_j_cspp_instrument_recovered"
    elif site == 'CE02SHSP':
        refdes = site + "-SP001-05-NUTNRJ000"
        method = "recovered_cspp"
        stream = "nutnr_j_cspp_instrument_recovered"
    else:
        raise ValueError("Invalid site selection, choose one or more of CE01ISSP (Oregon inshore) or CE02SHSP (Oregon midshelf).")

    # construct the OOI Gold Copy THREDDS catalog URL for this data set
    base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/"
    url = base_url + ('-').join([refdes, method, stream]) + '/catalog.html'
    tag = r'NUTNRJ000.*.nc$'  # setup regex for files we want
    nc_files = list_files(url, start_date, end_date, tag)
    base_url = 'https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/'
    nc_url = [base_url + i + '#mode=bytes' for i in nc_files]  # create urls for download
    # nc_url = nc_url[:5]
    # load datasets
    ds = []
    for i, f in (enumerate(tq(nc_url, desc='Downloading datasets'))):
        r = requests.get(f, timeout=(3.05, 120))
        if r.ok:
            ds.append(xr.open_dataset(io.BytesIO(r.content)))
            ds[i].load()
    return ds


if __name__ == '__main__':
    # parse command line arguments
    sites, out_path_top, start_date, end_date = _parse_args()

    for site in sites:
        # setup output folders
        out_path = _make_dirs(out_path_top, site)

        # get datasets
        ds = _get_datasets(site, start_date, end_date)
        ds_orig = ds.copy()

        for i, d in enumerate(ds):
            # compute derived variables
            ds[i] = _comput_tsrho(d)
            # save metadata to include in final dataset
            global_attrs, var_attrs = _get_attrs(ds[i])

        # QC nitrate data and remove short datasets
        deployments, total_data, good_data, step_1, step_2, step_3, step_4 = [], [], [], [], [], [], []
        for i, d in enumerate(tq(ds, desc='QCing datasets')):
            deployments.append(np.unique(d.deployment)[0])
            total_data.append(d.time.size)
            out = _nutnr_qc_wrapper(d)
            ds[i] = out[0]
            good_data.append(out[1])
            step_1.append(out[2])
            step_2.append(out[3])
            step_3.append(out[4])
            step_4.append(out[5])

        qc_results = pd.DataFrame(
            {
                'deployment': deployments,
                'total_data': total_data,
                'good_data': good_data,
                'bad_data': np.subtract(np.array(total_data), np.array(good_data)),
                'bad_salinity': step_1,
                'bad_values': step_2,
                'bad_fit': step_3,
                'bad_slope': step_4,
            },
            columns=['deployment', 'total_data', 'good_data', 'bad_data', 'bad_salinity', 'bad_values', 'bad_fit', 'bad_slope'],
        )

        qc_results.to_csv(os.path.join(out_path, f'{site}_qc_results.csv'))

        # remove datasets with no data after QC
        for i, d in enumerate(ds):
            if d.time.size == 0:
                ds.pop(i)
                print(f"Deployment {ds_orig[i].deployment.values[0]} removed due to lack of data after QC.")
        ds_qced = ds.copy()

        # find minimum and maximum depth bins over all nitrate data
        sur = np.min(xr.concat(ds, dim='time')['depth'])
        bot = np.max(xr.concat(ds, dim='time')['depth'])

        # setup depth/pressure bins
        step = 1
        sur = np.floor(sur)
        bot = np.ceil(bot)
        # pressure_grid is centers of bins
        pressure_grid = np.arange(sur+step/2, bot+step, step)
        # pressure_bins is edges of bins
        pressure_bins = np.nan*np.empty(len(pressure_grid)+1)
        pressure_bins[0] = pressure_grid[0] - step/2
        pressure_bins[-1] = pressure_grid[-1] + step/2
        for i in range(len(pressure_bins)-2):
            pressure_bins[i+1] = np.average([pressure_grid[i], pressure_grid[i+1]])

        # bin datasets into new list
        deployments = []
        times = []
        for d in tq(ds, desc='Splitting and binning profiles'):
            deployments.append(split_profiles(d))
            # print(deployments[-1])
            profiles = []
            profile_times = []
            for profile in deployments[-1]:
                profile_binned = profiler_binning(profile, pressure_bins, offset=0.5)
                profiles.append(profile_binned.mean(dim='time'))
                profile_times.append(profile_binned.time.mean().values)
            deployments[-1] = profiles
            times.append(xr.DataArray(profile_times, dims='time'))
        deployments = [xr.concat(profiles, profile_times) for profiles, profile_times in zip(deployments, times)]

        # remove datasets with less than 10 profiles data
        for i, d in enumerate(deployments):
            if d.time.size < 10:
                deployments.pop(i)
                print(f"Deployment {d.deployment.mean().values} removed due to having less than 10 profiles after QC and binning.")

        # concatenate binned datasets
        ds_bin = xr.concat(deployments, dim='time')
        ds_bin = ds_bin.drop_duplicates('time', keep='first')
        ds_bin = ds_bin.where(~np.isinf(ds_bin.salinity_corrected_nitrate))
        ds_bin = _assign_attrs(ds_bin, global_attrs, var_attrs)
        ds_bin = _drop_unused_vars(ds_bin)

        if site == 'CE01ISSP':
            ds_bin = ds_bin.where(ds_bin.deployment != 20, drop=True)

        if site == 'CE01ISSP':
            # custom baseline subtractions for deployments at inshore, based on nitrate-density relationship and some overlapping bottle samples
            for i, dep in enumerate(deployments):
                d = np.unique(dep.deployment)[0]
                if d == 1:
                    baseline = 10
                if d == 2:
                    baseline = -4
                elif d == 5:
                    baseline = 2
                elif d == 6:
                    baseline = 12
                elif d == 7:
                    baseline = 1
                elif d == 8:
                    baseline = 3
                elif d == 10:
                    baseline = -2
                elif d == 14:
                    baseline = 0
                elif d == 13:
                    baseline = -2
                elif d == 15:
                    baseline = 2
                elif d == 16:
                    baseline = 2
                elif d == 17:
                    baseline = 3
                elif d == 19:
                    baseline = 5
                elif d == 21:
                    baseline = 1
                else:
                    baseline = 0
                deployments[i]['salinity_corrected_nitrate'] = dep['salinity_corrected_nitrate'] - baseline
        elif site == 'CE02SHSP':
            for i, d in enumerate(deployments):
                baseline = d.where(d.depth < 2).salinity_corrected_nitrate.median().values
                deployments[i]['salinity_corrected_nitrate'] = d['salinity_corrected_nitrate'] - baseline

        ds_bin_baseline_subtracted = xr.concat(deployments, dim='time')
        ds_bin_baseline_subtracted = _assign_attrs(ds_bin_baseline_subtracted, global_attrs, var_attrs)
        ds_bin_baseline_subtracted = _drop_unused_vars(ds_bin_baseline_subtracted)

        if site == 'CE01ISSP':
            ds_bin_baseline_subtracted = ds_bin_baseline_subtracted.where(ds_bin_baseline_subtracted.deployment != 20, drop=True)

        print("Saving datasets...", end='', flush=True)

        # save start and end dates for combined datasets
        if datetime.strptime(start_date, '%Y-%m-%d') < pd.to_datetime(pd.Timestamp(ds_bin.time.min().values)):
            save_start_date = pd.to_datetime(pd.Timestamp(ds_bin.time.min().values))
            save_start_date = save_start_date.strftime('%Y-%m-%d')
        else:
            save_start_date = start_date
        if datetime.strptime(end_date, '%Y-%m-%d') > pd.to_datetime(pd.Timestamp(ds_bin.time.max().values)):
            save_end_date = pd.to_datetime(pd.Timestamp(ds_bin.time.max().values))
            save_end_date = save_end_date.strftime('%Y-%m-%d')
        else:
            save_end_date = end_date

        # save binned data
        out_file = site + '_nitrate_binned_' + save_start_date + '_' + save_end_date + '.nc'
        ds_bin.to_netcdf(os.path.join(out_path, out_file))
        ds_bin.close()

        # save binned and baseline subtracted data
        out_file = site + '_nitrate_binned_baseline_subtracted_' + save_start_date + '_' + save_end_date + '.nc'
        ds_bin_baseline_subtracted.to_netcdf(os.path.join(out_path, out_file))
        ds_bin_baseline_subtracted.close()

        # save qced, unbinned datasets
        # for i, d in enumerate(ds_qced):
        #     deployment = np.unique(d.deployment)[0]
        #     deployment = f'{deployment:02.0f}'
        #     d.to_netcdf(os.path.join(out_path, f'qced/{site}_dep{deployment}_nitrate_qced_{save_start_date}_{save_end_date}.nc'), mode='w')

        # save qced, binned datasets
        # for d in deployments:
        #     if d.time.size == 0:
        #         deployment = "unknown"
        #     else:
        #         deployment = np.unique(d.deployment)[0]
        #         deployment = f'{deployment:02.0f}'
        #     save_start_date = pd.to_datetime(pd.Timestamp(d.time.min().values))
        #     save_start_date = save_start_date.strftime('%Y-%m-%d')
        #     save_end_date = pd.to_datetime(pd.Timestamp(d.time.max().values))
        #     save_end_date = save_end_date.strftime('%Y-%m-%d')
        #     d.to_netcdf(os.path.join(out_path, f'binned/{site}_dep{deployment}_nitrate_binned_baseline_subtracted_{save_start_date}_{save_end_date}.nc'), mode='w')

        # save raw datasets
        for d in ds_orig:
            save_start_date = pd.to_datetime(pd.Timestamp(d.time.min().values))
            save_start_date = save_start_date.strftime('%Y-%m-%d')
            save_end_date = pd.to_datetime(pd.Timestamp(d.time.max().values))
            save_end_date = save_end_date.strftime('%Y-%m-%d')
            d.to_netcdf(os.path.join(out_path, f'raw/{site}_dep{d.deployment[0]:02.0f}_{save_start_date}_{save_end_date}.nc'), mode='w')
        print(f"done! {site} data saved to {out_path}.")
