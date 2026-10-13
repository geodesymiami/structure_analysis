#!/usr/bin/env python3
"""Structure_Analysis: fit height-dependent PSI building motion from CSVs.

Examples
--------
structure_analysis.py --asc ascending.csv --desc descending.csv \
    --building-points building.csv --out results
structure_analysis.py --desc descending.csv --bbox 25.94:25.95,-80.13:-80.11 \
    --building-name 'Building' --out results

The default is the three-parameter linear joint model. The exponential model
is available for comparison. Horizontal movement is only estimated jointly.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath
from matplotlib.widgets import PolygonSelector
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def create_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--asc', help='Ascending displacement CSV or output directory')
    parser.add_argument('--desc', help='Descending displacement CSV or output directory')
    parser.add_argument('--building-points', help='CSV with track and point_id selections')
    parser.add_argument('--building-points-dir', help='Analyze every building selection CSV in a folder and its subfolders')
    parser.add_argument('--building-name', default='Building')
    parser.add_argument('--bbox', help='LATMIN:LATMAX,LONMIN:LONMAX; selection mode')
    parser.add_argument('--bbox-asc', help='Ascending bbox; overrides --bbox')
    parser.add_argument('--bbox-desc', help='Descending bbox; overrides --bbox')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--model', choices=['linear', 'exponential', 'uniform'], default='linear')
    parser.add_argument('--height-mode', choices=['dem_error_minus_dem', 'dem_error', 'dem'],
                        default='dem_error_minus_dem')
    parser.add_argument('--min-coherence', type=float, default=0.05)
    parser.add_argument('--asc-geometry', help='Override geometryRadar.h5 for ASC')
    parser.add_argument('--desc-geometry', help='Override geometryRadar.h5 for DESC')
    parser.add_argument('--asc-stack', help='Override slcStack.h5 for ASC')
    parser.add_argument('--desc-stack', help='Override slcStack.h5 for DESC')
    parser.add_argument('--basemap', action='store_true',
                        help='Show OSM basemap during selection (requires contextily)')
    parser.add_argument('--dpi', type=int, default=250)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--strict-qc', action='store_true')
    parser.add_argument('--min-ps', type=int, default=25)
    return parser


def source_path(source, track):
    path = Path(os.path.expandvars(source)).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = sorted(path.glob('*thermalcorrected*.csv'))
    if not candidates:
        candidates = sorted(path.glob('*.csv'))
    if len(candidates) != 1:
        raise ValueError('%s: expected one CSV in %s, found %s' %
                         (track, path, len(candidates)))
    return candidates[0]


def find_auxiliary(csv_file, explicit, name):
    if explicit:
        candidate = Path(os.path.expandvars(explicit)).expanduser().resolve()
    else:
        candidate = csv_file.parent.parent / 'inputs' / name
    if not candidate.is_file():
        raise FileNotFoundError('%s missing: %s; supply an explicit override' % (name, candidate))
    return candidate


def get_bbox(text):
    if not text:
        raise ValueError('Interactive selection requires --bbox or --bbox-asc/--bbox-desc')
    try:
        a, b = text.split(',')
        south, north = sorted(float(x) for x in a.split(':'))
        west, east = sorted(float(x) for x in b.split(':'))
    except Exception as exc:
        raise ValueError('bbox format: LATMIN:LATMAX,LONMIN:LONMAX') from exc
    return west, east, south, north


def key_ids(series):
    return series.astype('string').str.strip().str.replace(r'^([+-]?\d+)\.0+$', r'\1', regex=True)


def selection_for_track(frame, selected, track):
    ids = selected.loc[selected['track'] == track, '_id'].tolist()
    if not ids:
        return None
    frame = frame.copy()
    frame['_id'] = key_ids(frame['point_id'])
    if frame['_id'].duplicated().any():
        raise ValueError('%s: duplicated source point_id' % track)
    source = frame.set_index('_id', drop=False)
    missing = [i for i in ids if i not in source.index]
    if missing:
        raise ValueError('%s: %d selected IDs not in CSV (e.g. %s)' %
                         (track, len(missing), missing[:3]))
    return source.loc[ids].drop(columns='_id').reset_index(drop=True)


def draw_selection(frame, track, bbox, building, basemap):
    west, east, south, north = get_bbox(bbox)
    frame = frame.loc[frame['X'].between(west, east) &
                      frame['Y'].between(south, north)].copy()
    if frame.empty:
        raise ValueError('%s: no points in bbox' % track)
    coords = frame[['X', 'Y']].to_numpy(float)
    selected_vertices = []
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=frame['dem_error'].to_numpy(float),
                    s=7, cmap='viridis', zorder=3)
    fig.colorbar(sc, ax=ax, label='dem_error (m)')
    ax.set(xlim=(west, east), ylim=(south, north), xlabel='Longitude', ylabel='Latitude',
           title='%s — %s | draw polygon; close window to accept' % (building, track))
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean([south, north]))))
    if basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom='auto', attribution_size=6)
        except Exception as exc:
            print('[WARNING] basemap unavailable: %s' % exc)
    def onselect(vertices):
        selected_vertices[:] = vertices
    selector = PolygonSelector(ax, onselect, useblit=True)
    plt.show()
    selector.disconnect_events()
    plt.close(fig)
    if len(selected_vertices) < 3:
        raise ValueError('%s: no polygon completed' % track)
    inside = PolygonPath(selected_vertices).contains_points(coords)
    chosen = frame.loc[inside].copy()
    if chosen.empty:
        raise ValueError('%s: selected polygon contains no points' % track)
    return chosen, selected_vertices


def geometry_match(frame, geometry_file, track):
    """Retain original radar-geocoded locations for geometry lookup.

    The geographic lookup can be ambiguous for duplicate float32 coordinates.
    """
    with h5py.File(geometry_file, 'r') as src:
        lon = src['longitude'][:]
        lat = src['latitude'][:]
        inc = src['incidenceAngle'][:]
        azi = src['azimuthAngle'][:]
    x = frame['X'].to_numpy(float)
    y = frame['Y'].to_numpy(float)
    lat0 = np.nanmean(y)
    factor = np.cos(np.deg2rad(lat0))
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(inc) & np.isfinite(azi)
    valid &= (lon >= x.min() - 0.002) & (lon <= x.max() + 0.002)
    valid &= (lat >= y.min() - 0.002) & (lat <= y.max() + 0.002)
    rows, cols = np.where(valid)
    if not len(rows):
        raise ValueError('%s: no matching geometry pixels' % track)
    coords = np.column_stack((lon[rows, cols] * factor, lat[rows, cols]))
    distance, idx = cKDTree(coords).query(np.column_stack((x * factor, y)))
    if np.any(distance > 5e-5):
        raise ValueError('%s: some PS positions are too far from radar geometry' % track)
    row, col = rows[idx], cols[idx]
    i, a = np.deg2rad(inc[row, col]), np.deg2rad(azi[row, col])
    frame = frame.copy()
    frame['geometry_row'] = row
    frame['geometry_col'] = col
    frame['los_E'] = -np.sin(i) * np.sin(a)
    frame['los_N'] = np.sin(i) * np.cos(a)
    frame['los_U'] = np.cos(i)
    # Duplicate lon/lat coordinates cannot identify a unique radar pixel.
    pair = np.column_stack((lon[rows, cols], lat[rows, cols]))
    n_unique = len(np.unique(pair, axis=0))
    if n_unique < len(rows):
        print('[WARNING] %s geometry has %d coordinate duplicates in local window; '
              'nearest-coordinate identity may be ambiguous.' % (track, len(rows) - n_unique))
    return frame


def prepare(frame, stack_file, mode, track):
    columns = sorted(c for c in frame if len(c) == 9 and c[0] == 'D' and c[1:].isdigit())
    if len(columns) < 3:
        raise ValueError('%s: need at least three DYYYYMMDD columns' % track)
    with h5py.File(stack_file, 'r') as src:
        dates = set(x.decode() if isinstance(x, bytes) else str(x) for x in src['date'][:])
        center = float(src.attrs.get('CENTER_LINE_UTC', 0.0))
    if not all(c[1:] in dates for c in columns):
        raise ValueError('%s: CSV dates differ from slcStack.h5' % track)
    times = pd.to_datetime([c[1:] for c in columns], format='%Y%m%d') + pd.to_timedelta(center, unit='s')
    if mode == 'dem_error_minus_dem':
        z = frame['dem_error'].to_numpy(float) - frame['dem'].to_numpy(float)
    else:
        z = frame[mode].to_numpy(float)
    raw = frame[columns].to_numpy(float)
    finite = np.isfinite(z) & np.isfinite(frame[['los_E', 'los_U']].to_numpy(float)).all(axis=1)
    finite &= np.isfinite(raw).all(axis=1)  # no silent interpolation
    frame = frame.loc[finite].reset_index(drop=True)
    raw, z = raw[finite], z[finite]
    if len(frame) == 0:
        raise ValueError('%s: no complete PS time-series' % track)
    values = raw - raw[:, :1]
    return dict(name=track, frame=frame, dates=times, z=z, y=values,
                e=frame['los_E'].to_numpy(float), u=frame['los_U'].to_numpy(float))


def design(track, model, t0):
    z = track['z'][:, None]
    elapsed = (track['dates'] - track['dates'][0]).total_seconds().to_numpy() / (365.25 * 86400.)
    t = (track['dates'] - t0).total_seconds().to_numpy() / (365.25 * 86400.)
    a = (track['dates'][0] - t0).total_seconds() / (365.25 * 86400.)
    uniform = track['u'][:, None] * np.broadcast_to(elapsed, (len(z), len(elapsed)))
    if model == 'uniform':
        return [uniform], ['uniform'], elapsed, t, a
    if model == 'linear':
        vertical = uniform * z
    else:
        vertical = None
    terms = [uniform]
    labels = ['uniform']
    if vertical is not None:
        terms.append(vertical)
        labels.append('height')
    if track.get('joint', False):
        terms.append(1e-3 * track['e'][:, None] * z * elapsed[None, :])
        labels.append('ew')
    return terms, labels, elapsed, t, a


def fit_model(tracks, model):
    t0 = min(t['dates'][0] for t in tracks)
    joint = len(tracks) == 2
    for tr in tracks:
        tr['joint'] = joint
    names = ['vU']
    if model != 'uniform':
        names += ['s0' if model == 'exponential' else 'gU']
        if model == 'exponential':
            names += ['tau']
        if joint:
            names += ['thetaE']
    x0 = [-2.0]
    lo, hi = [-100.0], [100.0]
    if model != 'uniform':
        x0 += [-0.05]; lo += [-10.0]; hi += [10.0]
        if model == 'exponential':
            x0 += [3.0]; lo += [0.05]; hi += [50.0]
        if joint:
            x0 += [20.0]; lo += [-5000.0]; hi += [5000.0]
    def components(tr, params):
        terms, labels, elapsed, t, a = design(tr, model, t0)
        result = {'uniform': terms[0] * params[0]}
        if model != 'uniform':
            if model == 'linear':
                result['height'] = terms[1] * params[1]
            else:
                tau = params[2]
                time_fun = tau * (np.exp(-a / tau) - np.exp(-t / tau))
                result['height'] = tr['u'][:, None] * tr['z'][:, None] * params[1] * time_fun
            if joint:
                result['ew'] = terms[-1] * params[-1]
        else:
            result['height'] = np.zeros_like(result['uniform'])
        if 'ew' not in result:
            result['ew'] = np.zeros_like(result['uniform'])
        result['prediction'] = result['uniform'] + result['height'] + result['ew']
        result['residual'] = tr['y'] - result['prediction']
        return result
    def residual(params):
        return np.concatenate([components(tr, params)['residual'].ravel() for tr in tracks])
    solution = least_squares(residual, x0, bounds=(lo, hi), method='trf', x_scale='jac',
                             max_nfev=1000)
    params = dict(zip(names, map(float, solution.x)))
    return params, {tr['name']: components(tr, solution.x) for tr in tracks}, solution


def save_results(tracks, model, parameters, components, fit, out, dpi, make_plots):
    records = []
    for track in tracks:
        name = track['name']
        comp = components[name]
        frame = track['frame'].copy()
        frame['height_proxy_m'] = track['z']
        frame.to_csv(out / ('selected_points_%s.csv' % name), index=False)
        df = frame[['point_id', 'height_proxy_m', 'X', 'Y', 'los_E', 'los_U']].copy()
        df['observed_final_mm'] = track['y'][:, -1]
        for col in ('uniform', 'height', 'ew', 'prediction', 'residual'):
            df[col + '_final_mm'] = comp[col][:, -1]
        df.to_csv(out / ('point_fit_%s.csv' % name), index=False)
        series = pd.DataFrame({'date': track['dates'].strftime('%Y-%m-%d')})
        series['observed_mm'] = track['y'].mean(axis=0)
        for col in ('uniform', 'height', 'ew', 'prediction', 'residual'):
            series[col + '_mm'] = comp[col].mean(axis=0)
        series.to_csv(out / ('mean_timeseries_%s.csv' % name), index=False)
        rmse = float(np.sqrt(np.mean(comp['residual']**2)))
        records.append({'track': name, 'n_ps': len(frame), 'n_dates': len(track['dates']),
                        'rmse_mm': rmse, 'height_min_m': float(np.min(track['z'])),
                        'height_max_m': float(np.max(track['z']))})
        if make_plots:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
            axes[0].plot(track['dates'], track['y'].mean(axis=0), 'o', ms=3, label='Observed')
            axes[0].plot(track['dates'], comp['prediction'].mean(axis=0), '-', lw=1.5, label='Modeled')
            axes[0].set(xlabel='Date', ylabel='LOS displacement (mm)', title='%s mean displacement' % name)
            axes[0].legend()
            for label in ('uniform', 'height', 'ew', 'residual'):
                if label == 'ew' and len(tracks) == 1:
                    continue
                axes[1].plot(track['dates'], comp[label].mean(axis=0), label=label)
            axes[1].set(xlabel='Date', ylabel='LOS displacement (mm)', title='Component means')
            axes[1].legend()
            fig.savefig(out / ('timeseries_%s.png' % name), dpi=dpi)
            plt.close(fig)
            order = np.argsort(track['z'])
            maps = [('Observed', track['y']),
                    ('Uniform vertical', comp['uniform']),
                    ('Height-dependent', comp['height']),
                    ('E-W contribution', comp['ew']),
                    ('Unmodeled residual', comp['residual'])]
            scale = max(1., float(np.nanpercentile(np.abs(track['y']), 99)))
            fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True,
                                     constrained_layout=True)
            for ax, (label, image) in zip(axes, maps):
                im = ax.imshow(image[order], interpolation='nearest', aspect='auto',
                               origin='lower', cmap='RdBu_r', vmin=-scale, vmax=scale,
                               extent=(0, len(track['dates'])-1, 0, len(order)))
                ax.set_title(label)
                ax.set_xlabel('Acquisition index')
            axes[0].set_ylabel('PS ranked by height proxy')
            fig.colorbar(im, ax=axes, label='LOS displacement (mm)', shrink=0.72)
            fig.suptitle('%s — %s (%s)' % (name, model, track['name']))
            fig.savefig(out / ('components_%s.png' % name), dpi=dpi)
            plt.close(fig)
    pd.DataFrame(records).to_csv(out / 'track_summary.csv', index=False)
    metadata = {'model': model, 'joint': len(tracks) == 2, 'parameters': parameters,
                'success': bool(fit.success), 'message': fit.message,
                'rmse_mm': float(np.sqrt(np.mean(fit.fun**2))),
                'height_mode_warning': 'Height proxy is not independently validated as physical structural height',
                'single_track_warning': 'No independent horizontal motion estimated' if len(tracks) == 1 else None}
    (out / 'model_summary.json').write_text(json.dumps(metadata, indent=2))
    print('[FIT] model=%s joint=%s rmse=%.3f mm' %
          (model, len(tracks) == 2, metadata['rmse_mm']))
    for key, value in parameters.items():
        print('      %-10s %.6f' % (key, value))


def list_buildings(args):
    """Find reusable building selections in a directory and subdirectories."""
    root = Path(os.path.expandvars(args.building_points_dir)).expanduser()
    if not root.is_dir():
        raise FileNotFoundError('Building selections directory: %s' % root)
    items = []
    for path in sorted(root.rglob('*.csv')):
        if path.stem.endswith('_selection_polygons'):
            continue
        # A true selection CSV must contain point_id and track.
        columns = pd.read_csv(path, nrows=0).columns
        if {'point_id', 'track'}.issubset(columns):
            items.append((path.stem.replace('_', ' '), path))
    if not items:
        raise ValueError('No building selection CSVs with track/point_id found under %s' % root)
    return items


def run_one(args):
    """Perform one selected-building inversion."""
    if not (args.asc or args.desc):
        raise ValueError('Supply --asc, --desc, or both')
    if not args.building_points and not (args.bbox or args.bbox_asc or args.bbox_desc):
        raise ValueError('No --building-points: provide --bbox for interactive selection')
    out = Path(os.path.expandvars(args.out)).expanduser()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is nonempty; choose a new --out folder: %s' % out)
    out.mkdir(parents=True, exist_ok=True)
    selected = None
    if args.building_points:
        selected = pd.read_csv(os.path.expandvars(args.building_points), dtype={'point_id': 'string'})
        if not {'track', 'point_id'}.issubset(selected.columns):
            raise ValueError('--building-points requires track and point_id columns')
        selected['track'] = selected['track'].str.strip().str.upper()
        if not selected['track'].isin(['ASC', 'DESC']).all():
            raise ValueError('Selection tracks must be ASC or DESC')
        selected['_id'] = key_ids(selected['point_id'])
        if selected.duplicated(['track', '_id']).any():
            raise ValueError('Duplicate (track, point_id) in selection')
    tracks = []
    selections = []
    for track, source in [('ASC', args.asc), ('DESC', args.desc)]:
        if not source or (selected is not None and not (selected['track'] == track).any()):
            continue
        csvfile = source_path(source, track)
        print('[LOAD] %s: %s' % (track, csvfile))
        frame = pd.read_csv(csvfile, dtype={'point_id': 'string'})
        if selected is not None:
            frame = selection_for_track(frame, selected, track)
        else:
            bbox = (args.bbox_asc if track == 'ASC' else args.bbox_desc) or args.bbox
            frame, verts = draw_selection(frame, track, bbox, args.building_name, args.basemap)
            selections.extend({'track': track, 'point_id': pid, 'building': args.building_name}
                              for pid in frame['point_id'])
            (out / ('polygon_%s.json' % track)).write_text(json.dumps(verts))
        if 'coherence' in frame:
            frame = frame.loc[frame['coherence'] >= args.min_coherence].copy()
        if args.strict_qc and len(frame) < args.min_ps:
            raise ValueError('%s only %d PS after filtering' % (track, len(frame)))
        if frame.empty:
            raise ValueError('%s: no PS after filtering' % track)
        geometry = find_auxiliary(csvfile, args.asc_geometry if track == 'ASC' else args.desc_geometry,
                                  'geometryRadar.h5')
        stack = find_auxiliary(csvfile, args.asc_stack if track == 'ASC' else args.desc_stack,
                               'slcStack.h5')
        frame = geometry_match(frame, geometry, track)
        tr = prepare(frame, stack, args.height_mode, track)
        print('[DATA] %s: %d PS, %d dates' % (track, len(tr['z']), len(tr['dates'])))
        tracks.append(tr)
    if not tracks:
        raise ValueError('No selected tracks to process')
    if selections:
        pd.DataFrame(selections).to_csv(out / 'building_points.csv', index=False)
    parameters, components, fit = fit_model(tracks, args.model)
    save_results(tracks, args.model, parameters, components, fit, out, args.dpi,
                 not args.no_plots)
    print('[DONE] %s' % out)
    return 0


def main(argv=None):
    args = create_parser().parse_args(argv)
    modes = sum(bool(x) for x in (args.building_points, args.building_points_dir))
    if modes > 1:
        raise ValueError('Choose either --building-points or --building-points-dir')
    if args.building_points_dir:
        parent = Path(os.path.expandvars(args.out)).expanduser()
        entries = list_buildings(args)
        if len({str(path.resolve()) for _, path in entries}) != len(entries):
            raise ValueError('The same building selection is listed more than once')
        completed, failed = 0, []
        for name, selection in entries:
            destination = parent / selection.stem
            print('\n[BUILDING] %s | %s' % (name, selection), flush=True)
            child = argparse.Namespace(**vars(args))
            child.building_points_dir = None
            child.building_points = str(selection)
            child.building_name = name
            child.out = str(destination)
            try:
                run_one(child)
                completed += 1
            except (ValueError, FileNotFoundError, KeyError) as exc:
                print('[FAILED] %s: %s' % (name, exc), file=sys.stderr)
                failed.append(name)
        print('\n[DONE] completed=%d failed=%d output=%s' % (completed, len(failed), parent))
        return 1 if failed else 0
    return run_one(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        sys.exit('ERROR: %s' % exc)

