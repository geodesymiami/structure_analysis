#!/usr/bin/env python3
"""Plot structural LOS components from structure_analysis.py results.

Examples
--------
plot_structure_analysis.py --results results/Ritz_Carlton --out RitzCarlton.png
plot_structure_analysis.py --results results/Ritz results/Turnberry \
    --layout buildings-y --track both --out Comparison.png
plot_structure_analysis.py --results-root results --out all_buildings.png

Observed LOS components are, uniform vertical projection, height-dependent
vertical projection, E-W horizontal projection, and Distortion/Differential Settlement.
Original X/Y are radar-grid geocoded locations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, FuncFormatter

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None


BASEMAP_PROVIDERS = {
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "grayscale": False,
        "cache_suffix": "osm",
    },
    "osm_gray": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "grayscale": True,
        "cache_suffix": "osm_gray",
    },
    "cartodb_light": {
        "url": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "grayscale": False,
        "cache_suffix": "cartodb_light",
    },
    "cartodb_dark": {
        "url": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "grayscale": False,
        "cache_suffix": "cartodb_dark",
    },
    "esri_gray": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "grayscale": False,
        "cache_suffix": "esri_gray",
        "max_zoom": 16,
    },
    "esri_satellite": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "grayscale": False,
        "cache_suffix": "esri_satellite",
        "max_zoom": 19,
    },
}

DEFAULT_BUILDINGS = [
    {
        "name": "Turnberry Ocean Club",
        "result_dir": "Turnberry_Ocean",
        "image": "turnberry_ocean_club_residences.jpg",
    },
    {
        "name": "The Estates at Acqualina North",
        "result_dir": "Estates_at_Acqualina_North",
        "image": "the_estates_at_acqualina_north.jpg",
    },
    {
        "name": "The Estates at Acqualina South",
        "result_dir": "Estates_at_Acqualina_South",
        "image": "the_estates_at_acqualina_south.jpg",
    },
    {
        "name": "Ritz Carlton Residences",
        "result_dir": "Ritz_Carlton",
        "image": "ritz_carlton.jpg",
    },
]

COMPONENTS = [
    ("total", "TOTAL\nDISPLACEMENT", "observed_final_mm", "Observed total displacement [mm]"),
    ("uniform", "UNIFORM\nSETTLEMENT", "uniform_final_mm", "Uniform LOS contribution [mm]"),
    ("height", "HEIGHT-DEPENDENT\nCOMPONENT", "height_final_mm", "Height-dependent LOS contribution [mm]"),
    ("tilt", "E-W\nTILT", "ew_final_mm", "E-W LOS contribution [mm]"),
    ("residual", "DISTORTION/\nDIFFERENTIAL SETTLEMENT", "residual_final_mm", "Distortion/Differential Settlement LOS displacement [mm]"),
]


def wrap_name(name, width=20):
    return "\n".join(
        textwrap.wrap(
            str(name),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def make_norm(vmin, vmax):
    if vmin < 0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def lonlat_formatter(decimals=5):
    fmt = "{: .".replace(" ", "") + str(decimals) + "f}"

    def _format(v, pos=None):
        s = fmt.format(v)
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    return FuncFormatter(_format)


def choose_xy(df, coord_mode="auto"):
    if coord_mode not in {"auto", "geocorr", "original"}:
        raise ValueError(coord_mode)

    if coord_mode in {"auto", "geocorr"}:
        if "X_geocorr" in df.columns and "Y_geocorr" in df.columns:
            xg = pd.to_numeric(df["X_geocorr"], errors="coerce")
            yg = pd.to_numeric(df["Y_geocorr"], errors="coerce")
            good = np.isfinite(xg) & np.isfinite(yg)
            if coord_mode == "geocorr" and not good.any():
                raise ValueError("Requested geocorr coordinates, but none are finite.")
            if good.any():
                x = xg.to_numpy(float)
                y = yg.to_numpy(float)
                if coord_mode == "auto" and "X" in df.columns and "Y" in df.columns:
                    xo = pd.to_numeric(df["X"], errors="coerce").to_numpy(float)
                    yo = pd.to_numeric(df["Y"], errors="coerce").to_numpy(float)
                    x = np.where(np.isfinite(x), x, xo)
                    y = np.where(np.isfinite(y), y, yo)
                return x, y, "geocorrected"

    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError("No usable coordinates found (X/Y or X_geocorr/Y_geocorr).")

    x = pd.to_numeric(df["X"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["Y"], errors="coerce").to_numpy(float)
    return x, y, "original"


def padded_extent(
    x,
    y,
    lon_pad_fraction=0.01,
    lat_pad_fraction=0.08,
    min_pad_lon=0.00003,
    min_pad_lat=0.00008,
    target_geo_ratio=0.78,
):
    """
    Tight east-west crop with deliberate north-south expansion.

    target_geo_ratio is the desired physical N-S / E-W span ratio after
    correcting longitude by cos(latitude).  Values around 0.75-0.85 produce
    a compact, building-scale map that is still rectangular but not ribbon-like.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if not good.any():
        raise ValueError("No finite coordinates for map extent.")

    xmin = float(np.nanmin(x[good]))
    xmax = float(np.nanmax(x[good]))
    ymin = float(np.nanmin(y[good]))
    ymax = float(np.nanmax(y[good]))

    dx = max(xmax - xmin, min_pad_lon)
    dy = max(ymax - ymin, min_pad_lat)

    # Keep longitude tight around the points.
    px = max(dx * lon_pad_fraction, min_pad_lon)
    # Add normal latitude padding first.
    py = max(dy * lat_pad_fraction, min_pad_lat)

    west = xmin - px
    east = xmax + px
    south = ymin - py
    north = ymax + py

    # Then expand ONLY north/south until the map reaches the target physical ratio.
    mean_lat = 0.5 * (south + north)
    coslat = max(np.cos(np.deg2rad(mean_lat)), 1.0e-6)
    width_geo = (east - west) * coslat
    height_geo = north - south
    desired_height = target_geo_ratio * width_geo
    if desired_height > height_geo:
        extra = 0.5 * (desired_height - height_geo)
        south -= extra
        north += extra

    return west, east, south, north

def lonlat_to_tile(lon, lat, zoom):
    lat = max(min(float(lat), 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    n = 2.0 ** int(zoom)
    xtile = int((float(lon) + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_edges(x, y, z):
    n = 2.0 ** int(z)
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_w, lon_e, lat_s, lat_n


def fetch_tile(x, y, z, cache_dir, provider_url, grayscale=False):
    if Image is None:
        raise RuntimeError(
            "Basemap requested but Pillow is not installed. Install with: conda install -c conda-forge pillow"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    tile_path = cache_dir / f"z{z}_x{x}_y{y}.png"

    if tile_path.exists():
        img = Image.open(tile_path).convert("RGB")
        if grayscale:
            img = ImageOps.grayscale(img).convert("RGB")
        return img

    url = provider_url.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": "building-analysis-components/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        img.save(tile_path)
        if grayscale:
            img = ImageOps.grayscale(img).convert("RGB")
        return img
    except Exception as exc:
        print(f"[WARNING] could not fetch basemap tile {z}/{x}/{y}: {exc}")
        return None


def effective_basemap_zoom(provider_key, zoom):
    if provider_key not in BASEMAP_PROVIDERS:
        raise ValueError(
            f"Unknown basemap provider '{provider_key}'. Choose from: {', '.join(sorted(BASEMAP_PROVIDERS))}"
        )
    provider = BASEMAP_PROVIDERS[provider_key]
    z = int(zoom)
    max_zoom = provider.get("max_zoom")
    if max_zoom is not None and z > int(max_zoom):
        print(f"[BASEMAP] {provider_key}: requested zoom {z} > supported zoom {int(max_zoom)}; using {int(max_zoom)}.")
        z = int(max_zoom)
    return z


def add_web_basemap(ax, extent, provider_key="esri_satellite", zoom=19, alpha=0.38, cache_root=Path(".tile_cache")):
    if Image is None:
        raise RuntimeError(
            "Basemap requested but Pillow is not installed. Install with: conda install -c conda-forge pillow"
        )
    if provider_key not in BASEMAP_PROVIDERS:
        raise ValueError(
            f"Unknown basemap provider '{provider_key}'. Choose from: {', '.join(sorted(BASEMAP_PROVIDERS))}"
        )

    west, east, south, north = map(float, extent)
    if not all(np.isfinite([west, east, south, north])):
        return False

    z = effective_basemap_zoom(provider_key, zoom)
    provider = BASEMAP_PROVIDERS[provider_key]
    provider_url = provider["url"]
    grayscale = bool(provider.get("grayscale", False))
    cache_dir = Path(cache_root) / provider.get("cache_suffix", provider_key)

    x0, y0 = lonlat_to_tile(west, north, z)
    x1, y1 = lonlat_to_tile(east, south, z)
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])

    ok = False
    for xt in range(xmin, xmax + 1):
        for yt in range(ymin, ymax + 1):
            img = fetch_tile(xt, yt, z, cache_dir, provider_url, grayscale=grayscale)
            if img is None:
                continue
            lon_w, lon_e, lat_s, lat_n = tile_edges(xt, yt, z)
            ax.imshow(
                img,
                extent=[lon_w, lon_e, lat_s, lat_n],
                origin="upper",
                alpha=float(alpha),
                interpolation="bilinear",
                zorder=0,
            )
            ok = True

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    return ok


def set_geo_aspect(ax, extent):
    west, east, south, north = extent
    mean_lat = 0.5 * (south + north)
    aspect = 1.0 / max(np.cos(np.deg2rad(mean_lat)), 1.0e-6)
    ax.set_aspect(aspect, adjustable="box")


def load_building_config(results_root, image_dir, requested=None):
    """Find result folders automatically; building photos are optional."""
    root = Path(results_root).expanduser()
    if requested:
        folders = [Path(item).expanduser() for item in requested]
    elif (root / "model_summary.json").is_file():
        folders = [root]
    else:
        folders = sorted(p for p in root.iterdir()
                         if p.is_dir() and (p / "model_summary.json").is_file())
    if not folders:
        raise ValueError(f"No structure_analysis.py result folders found in {root}")
    image_root = Path(image_dir).expanduser() if image_dir else None
    photo_names = {
        "Turnberry_Ocean_Club": "turnberry_ocean_club_residences.jpg",
        "Estates_at_Acqualina_North": "the_estates_at_acqualina_north.jpg",
        "Estates_at_Acqualina_South": "the_estates_at_acqualina_south.jpg",
        "Ritz_Carlton_Residences": "ritz_carlton.jpg",
        "Trump_Royale": "trump_royale.jpg",
    }
    buildings = []
    for folder in folders:
        if not (folder / "model_summary.json").exists():
            raise FileNotFoundError(f"Missing model_summary.json: {folder}")
        name = folder.name.replace("_", " ")
        fname = photo_names.get(folder.name, folder.name.lower() + ".jpg")
        image_path = image_root / fname if image_root else None
        if image_root and image_path is not None and not image_path.exists():
            candidates = list(image_root.glob(folder.name.lower() + ".*"))
            if candidates:
                image_path = candidates[0]
        buildings.append({"name": name, "result_dir": folder.name,
                          "result_path": folder, "image_path": image_path})
    return buildings


def load_building_summary(result_path):
    with open(Path(result_path) / "model_summary.json") as f:
        return json.load(f)


def load_track_timing(result_path, track):
    path = Path(result_path) / f"mean_timeseries_{track}.csv"
    data = pd.read_csv(path)
    dates = pd.to_datetime(data["datetime_utc"] if "datetime_utc" in data
                           else data["date"], errors="coerce").dropna()
    if len(dates) < 2:
        raise ValueError(f"At least two valid acquisition dates required: {path}")
    return dates.iloc[0], dates.iloc[-1]


def load_building_track(result_path, track, coord_mode):
    path = Path(result_path) / f"point_fit_{track}.csv"
    if not path.exists():
        return None, "missing"
    df = pd.read_csv(path, low_memory=False)
    for modern, legacy in (("height_final_mm", "creep_final_mm"),
                           ("ew_final_mm", "tilt_east_final_mm")):
        if modern not in df and legacy in df:
            df[modern] = df[legacy]
    if "ew_final_mm" not in df:
        df["ew_final_mm"] = 0.
    required = ["observed_final_mm", "uniform_final_mm", "height_final_mm",
                "ew_final_mm", "residual_final_mm"]
    missing = [field for field in required if field not in df]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    for field in required:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    x, y, mode = choose_xy(df, coord_mode=coord_mode)
    df["_plot_x"] = x
    df["_plot_y"] = y
    df["_track"] = track
    return df, mode


def physical_track_annotation_values(summary, first_dt, final_dt):
    years = (pd.Timestamp(final_dt) - pd.Timestamp(first_dt)).total_seconds() / (365.25*86400.)
    params = summary.get("parameters", {})
    uniform = params.get("vU", np.nan)
    gradient = params.get("gU", params.get("s0", np.nan))
    tau = params.get("tau", np.nan)
    if summary.get("model") == "exponential" and np.isfinite(tau) and tau > 0:
        # This is only a descriptive effective gradient using each track's duration.
        gradient_mm_m = gradient*tau*(-np.expm1(-years/tau))
    else:
        gradient_mm_m = gradient*years if np.isfinite(gradient) else np.nan
    return {"duration_yr": years,
            "uniform_vertical_mm": uniform*years if np.isfinite(uniform) else np.nan,
            "creep_gradient_mm_m": gradient_mm_m,
            "tilt_east_urad": params.get("thetaE", np.nan)*years}


def combine_physical_annotations(asc_physical, desc_physical):
    result = {}
    for key in ["duration_yr", "uniform_vertical_mm", "creep_gradient_mm_m", "tilt_east_urad"]:
        values = np.asarray([asc_physical.get(key, np.nan), desc_physical.get(key, np.nan)])
        result[key] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
    return result


def draw_building_image(ax, image_path, name, name_fontsize=20):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.45)

    ax.text(
        0.5,
        1.05,
        wrap_name(name, width=18),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=name_fontsize,
        fontweight="bold",
        linespacing=1.05,
    )

    if image_path is None or not Path(image_path).exists():
        ax.text(0.5, 0.5, "NO IMAGE", transform=ax.transAxes, ha="center", va="center", fontsize=12, fontweight="bold", alpha=0.55)
        return

    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")


def draw_building_header_side(ax, image_path, name, name_fontsize=18):
    """For buildings-y layout: put building name beside the image."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    text_ax = ax.inset_axes([0.00, 0.05, 0.36, 0.90])
    text_ax.axis("off")
    text_ax.text(
        0.98,
        0.50,
        wrap_name(name, width=15),
        transform=text_ax.transAxes,
        ha="right",
        va="center",
        fontsize=name_fontsize,
        fontweight="bold",
        linespacing=1.0,
    )

    img_ax = ax.inset_axes([0.40, 0.02, 0.58, 0.96])
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for spine in img_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.45)

    if image_path is None or not Path(image_path).exists():
        img_ax.text(0.5, 0.5, "NO IMAGE", transform=img_ax.transAxes, ha="center", va="center", fontsize=12, fontweight="bold", alpha=0.55)
        return

    img = mpimg.imread(image_path)
    img_ax.imshow(img)
    img_ax.set_aspect("equal", adjustable="box")
    img_ax.set_anchor("C")


def format_map_ticks(ax, show_yticklabels=False, show_xticklabels=True, max_ticks=4, tick_label_fontsize=11.5, show_coordinates=True):
    if show_coordinates:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize, length=3.0, pad=1.2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, min_n_ticks=2))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, min_n_ticks=2))
        ff = lonlat_formatter(decimals=5)
        ax.xaxis.set_major_formatter(ff)
        ax.yaxis.set_major_formatter(ff)
        ax.get_xaxis().get_offset_text().set_visible(False)
        ax.get_yaxis().get_offset_text().set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if not show_yticklabels:
            ax.set_yticklabels([])
        if not show_xticklabels:
            ax.set_xticklabels([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_alpha(0.65)


def component_values(df, key):
    if key == "total":
        return df["observed_final_mm"].to_numpy(float)
    if key == "uniform":
        return df["uniform_final_mm"].to_numpy(float)
    if key == "height":
        return df["height_final_mm"].to_numpy(float)
    if key == "tilt":
        return df["ew_final_mm"].to_numpy(float)
    if key == "residual":
        return df["residual_final_mm"].to_numpy(float)
    raise KeyError(key)


def _max_abs_cm(values_mm):
    arr = np.asarray(values_mm, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanmax(np.abs(arr))) / 10.0


def annotate_component(ax, key, values_mm, physical):
    max_cm = _max_abs_cm(values_mm)

    if key == "uniform":
        val_cm = physical["uniform_vertical_mm"] / 10.0
        text = f"Uniform: {val_cm:+.1f} cm"
    elif key == "height":
        grad = physical["creep_gradient_mm_m"]
        text = f"{grad:+.3f} mm/m"
        if np.isfinite(max_cm):
            text += f"\nmax: {max_cm:.1f} cm"
    elif key == "tilt":
        tilt_deg = physical["tilt_east_urad"] * 1.0e-6 * 180.0 / np.pi
        text = f"E-W: {tilt_deg:+.3f}°"
        if np.isfinite(max_cm):
            text += f"\nmax: {max_cm:.1f} cm"
    else:
        if not np.isfinite(max_cm):
            return
        text = f"max: {max_cm:.1f} cm"

    ax.text(
        0.97,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.4,
        fontweight="bold",
        linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="0.70", alpha=0.88),
        zorder=20,
    )


def load_all_buildings(buildings, coord_mode, lon_pad_fraction=0.01,
                       lat_pad_fraction=0.08, target_geo_ratio=0.78):
    loaded = []
    for b in buildings:
        print(f"[LOAD] {b['name']}: {b['result_path']}")
        summary = load_building_summary(b["result_path"])
        parts = []
        physicals = []
        for track in ("ASC", "DESC"):
            frame, mode = load_building_track(b["result_path"], track, coord_mode)
            if frame is None:
                continue
            first, last = load_track_timing(b["result_path"], track)
            b[track] = frame
            parts.append(frame)
            physicals.append(physical_track_annotation_values(summary, first, last))
            print(f"       {track} n={len(frame)}, coords={mode}")
        if not parts:
            raise ValueError(f"No track results: {b['result_path']}")
        combined = pd.concat(parts, ignore_index=True)
        extent = padded_extent(combined["_plot_x"], combined["_plot_y"],
                               lon_pad_fraction=lon_pad_fraction,
                               lat_pad_fraction=lat_pad_fraction,
                               target_geo_ratio=target_geo_ratio)
        physical = combine_physical_annotations(physicals[0], physicals[1]) if len(physicals)==2 else physicals[0]
        loaded.append({**b, "summary": summary, "combined": combined,
                       "extent": extent, "combined_physical": physical})
    return loaded


def build_scales(args):
    return {
        "total": (float(args.total_vmin), float(args.total_vmax)),
        "uniform": (float(args.uniform_vmin), float(args.uniform_vmax)),
        "height": (
            float(args.height_vmin if args.height_vmin is not None else args.structural_vmin),
            float(args.height_vmax if args.height_vmax is not None else args.structural_vmax),
        ),
        "tilt": (
            float(args.tilt_vmin if args.tilt_vmin is not None else args.structural_vmin),
            float(args.tilt_vmax if args.tilt_vmax is not None else args.structural_vmax),
        ),
        "residual": (
            float(args.residual_vmin if args.residual_vmin is not None else args.structural_vmin),
            float(args.residual_vmax if args.residual_vmax is not None else args.structural_vmax),
        ),
    }



def _buildings_y_geometry(
    fig_width,
    target_geo_ratio,
    n_buildings,
    n_components,
    header_height,
    row_spacing,
    left=0.028,
    right=0.99,
    top=0.972,
    bottom=0.085,
    left_col_ratio=4.05,
    map_col_ratio=4.20,
):
    """
    Return height ratios and a figure height for the buildings-y layout.

    Matplotlib GridSpec wspace/hspace are fractions of average axes size,
    and set_aspect(..., adjustable="box") can shrink an axes inside its
    GridSpec cell.  To make visible horizontal and vertical gaps comparable,
    size each map row cell so its height/width matches target_geo_ratio.
    """
    usable_w_in = fig_width * (right - left)

    width_units = left_col_ratio + n_components * map_col_ratio
    # GridSpec spacing: total extra width is approximately
    # wspace * average-axis-width * (ncols - 1)
    ncols = 1 + n_components
    avg_w_units = width_units / ncols
    spacing_units_x = row_spacing * avg_w_units * (ncols - 1)
    total_units_x = width_units + spacing_units_x

    map_w_in = usable_w_in * (map_col_ratio / total_units_x)
    map_h_in = map_w_in * target_geo_ratio

    header_h_in = max(1.0, header_height * 0.20)

    # hspace is likewise a fraction of average row height.  Solve
    # approximately for a figure height that yields the same physical gap
    # as the horizontal gap between map columns.
    physical_gap_in = row_spacing * map_w_in
    content_h_in = header_h_in + n_buildings * map_h_in + n_buildings * physical_gap_in
    fig_height = content_h_in / max(top - bottom, 1e-6)

    # Height ratios can be in arbitrary units; use inch-like values.
    height_ratios = [header_h_in] + [map_h_in] * n_buildings
    return height_ratios, fig_height, physical_gap_in

def make_figure(
    buildings,
    output_path,
    provider="esri_satellite",
    zoom=19,
    coord_mode="auto",
    dpi=300,
    fig_width=33.0,
    map_row_height=6.4,
    header_height=6.0,
    point_size=20.0,
    no_basemap=False,
    basemap_alpha=0.38,
    total_vmin=-100.0,
    total_vmax=30.0,
    uniform_vmin=-100.0,
    uniform_vmax=30.0,
    structural_vmin=-30.0,
    structural_vmax=30.0,
    height_vmin=None,
    height_vmax=None,
    tilt_vmin=None,
    tilt_vmax=None,
    residual_vmin=None,
    residual_vmax=None,
    component_label_fontsize=20,
    building_name_fontsize=20,
    tick_label_fontsize=11.5,
    lon_pad_fraction=0.01,
    lat_pad_fraction=0.08,
    target_geo_ratio=0.78,
    column_spacing=0.32,
    row_spacing=0.10,
    show_coordinates=True,
    layout="buildings-x",
):
    class ArgsObj:
        pass

    a = ArgsObj()
    a.total_vmin = total_vmin
    a.total_vmax = total_vmax
    a.uniform_vmin = uniform_vmin
    a.uniform_vmax = uniform_vmax
    a.structural_vmin = structural_vmin
    a.structural_vmax = structural_vmax
    a.height_vmin = height_vmin
    a.height_vmax = height_vmax
    a.tilt_vmin = tilt_vmin
    a.tilt_vmax = tilt_vmax
    a.residual_vmin = residual_vmin
    a.residual_vmax = residual_vmax
    scales = build_scales(a)

    loaded = load_all_buildings(
        buildings,
        coord_mode=coord_mode,
        lon_pad_fraction=lon_pad_fraction,
        lat_pad_fraction=lat_pad_fraction,
        target_geo_ratio=target_geo_ratio,
    )

    print("\n[COLOR SCALES]")
    for key in ["total", "uniform", "height", "tilt", "residual"]:
        vmin, vmax = scales[key]
        print(f"  {key:8s}: {vmin:.3f} .. {vmax:.3f} mm")

    n_buildings = len(loaded)
    n_components = len(COMPONENTS)
    cmap_struct = plt.get_cmap("jet")
    map_axes = {}

    if layout == "buildings-x":
        fig_height = header_height + n_components * map_row_height + 1.1
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False, facecolor="white")
        gs = GridSpec(
            nrows=1 + n_components,
            ncols=1 + n_buildings,
            figure=fig,
            width_ratios=[3.45] + [4.20] * n_buildings,
            height_ratios=[header_height] + [map_row_height] * n_components,
            left=0.035,
            right=0.942,
            top=0.975,
            bottom=0.07,
            wspace=column_spacing,
            hspace=row_spacing,
        )

        ax_blank = fig.add_subplot(gs[0, 0])
        ax_blank.axis("off")

        for bi, b in enumerate(loaded, start=1):
            ax_head = fig.add_subplot(gs[0, bi])
            draw_building_image(ax_head, b["image_path"], b["name"], name_fontsize=building_name_fontsize)

        for ri, (key, title, _, _) in enumerate(COMPONENTS, start=1):
            ax_label = fig.add_subplot(gs[ri, 0])
            ax_label.axis("off")
            ax_label.text(
                0.5, 0.50, title, transform=ax_label.transAxes,
                ha="center", va="center", fontsize=component_label_fontsize,
                fontweight="bold", linespacing=1.0,
            )

            for bi, b in enumerate(loaded, start=1):
                ax = fig.add_subplot(gs[ri, bi])
                map_axes[(key, bi)] = ax
                df = b["combined"]
                extent = b["extent"]
                west, east, south, north = extent

                if not no_basemap:
                    add_web_basemap(
                        ax, extent, provider_key=provider, zoom=zoom,
                        alpha=basemap_alpha,
                        cache_root=Path(output_path).expanduser().parent / "_tile_cache",
                    )
                ax.set_xlim(west, east)
                ax.set_ylim(south, north)
                set_geo_aspect(ax, extent)

                values = component_values(df, key)
                x = df["_plot_x"].to_numpy(float)
                y = df["_plot_y"].to_numpy(float)
                good = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
                vmin, vmax = scales[key]
                norm = make_norm(vmin, vmax)
                ax.scatter(
                    x[good], y[good], c=values[good], s=point_size, cmap=cmap_struct,
                    norm=norm, edgecolors="none", alpha=0.98, rasterized=True, zorder=5,
                )
                annotate_component(ax, key, values[good], b["combined_physical"])
                format_map_ticks(
                    ax,
                    show_yticklabels=True,
                    show_xticklabels=(ri == n_components),
                    max_ticks=4,
                    tick_label_fontsize=tick_label_fontsize,
                    show_coordinates=show_coordinates,
                )

        fig.canvas.draw()

        for ri, (key, _, _, cblabel) in enumerate(COMPONENTS, start=1):
            row_ax = map_axes[(key, n_buildings)]
            pos = row_ax.get_position()
            cax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.012, pos.height])
            sm = ScalarMappable(norm=make_norm(*scales[key]), cmap=cmap_struct)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, orientation="vertical")
            cb.ax.tick_params(labelsize=10, length=2.2, pad=1.5)
            cb.set_label(cblabel, fontsize=10.5, fontweight="bold", labelpad=8, rotation=90)

        for bi in range(2, n_buildings + 1):
            prev_ax = map_axes[(COMPONENTS[0][0], bi - 1)]
            next_ax = map_axes[(COMPONENTS[0][0], bi)]
            p_prev = prev_ax.get_position()
            p_next = next_ax.get_position()
            x = 0.5 * (p_prev.x1 + p_next.x0)
            y0 = map_axes[(COMPONENTS[-1][0], bi - 1)].get_position().y0
            y1 = map_axes[(COMPONENTS[0][0], bi - 1)].get_position().y1
            fig.add_artist(plt.Line2D([x, x], [y0, y1], transform=fig.transFigure, linewidth=0.7, color="0.80", alpha=0.85))

    elif layout == "buildings-y":
        # Use the SAME requested visible gap horizontally and vertically.
        # The map rows are sized from target_geo_ratio so set_aspect(...,
        # adjustable="box") no longer leaves a large amount of unused
        # vertical space inside each GridSpec row.
        left_y = 0.028
        right_y = 0.99
        top_y = 0.972
        bottom_y = 0.085
        left_col_ratio = 4.05
        map_col_ratio = 4.20

        height_ratios_y, fig_height_y, physical_gap_in = _buildings_y_geometry(
            fig_width=fig_width,
            target_geo_ratio=target_geo_ratio,
            n_buildings=n_buildings,
            n_components=n_components,
            header_height=header_height,
            row_spacing=row_spacing,
            left=left_y,
            right=right_y,
            top=top_y,
            bottom=bottom_y,
            left_col_ratio=left_col_ratio,
            map_col_ratio=map_col_ratio,
        )

        # GridSpec's wspace and hspace use different reference dimensions.
        # Set wspace from row_spacing and solve hspace so the physical gaps
        # are approximately equal.
        ncols_y = 1 + n_components
        nrows_y = 1 + n_buildings
        width_units_y = left_col_ratio + n_components * map_col_ratio
        avg_w_units_y = width_units_y / ncols_y
        map_width_fraction = (
            (right_y - left_y) * map_col_ratio /
            (width_units_y + row_spacing * avg_w_units_y * (ncols_y - 1))
        )
        map_width_in = fig_width * map_width_fraction

        # Desired physical gap in inches.
        desired_gap_in = row_spacing * map_width_in

        # hspace is a fraction of average row height in GridSpec.
        avg_row_h_in = (
            fig_height_y * (top_y - bottom_y) /
            nrows_y
        )
        hspace_y = desired_gap_in / max(avg_row_h_in, 1e-9)

        fig = plt.figure(
            figsize=(fig_width, fig_height_y),
            constrained_layout=False,
            facecolor="white",
        )
        gs = GridSpec(
            nrows=1 + n_buildings,
            ncols=1 + n_components,
            figure=fig,
            width_ratios=[left_col_ratio] + [map_col_ratio] * n_components,
            height_ratios=height_ratios_y,
            left=left_y,
            right=right_y,
            top=top_y,
            bottom=bottom_y,
            wspace=row_spacing,
            hspace=hspace_y,
        )

        ax_blank = fig.add_subplot(gs[0, 0])
        ax_blank.axis("off")

        # Place component names just above the first map row.
        for ci, (key, title, _, _) in enumerate(COMPONENTS, start=1):
            ax_label = fig.add_subplot(gs[0, ci])
            ax_label.axis("off")
            ax_label.text(
                0.5,
                0.02,
                title,
                transform=ax_label.transAxes,
                ha="center",
                va="bottom",
                fontsize=component_label_fontsize,
                fontweight="bold",
                linespacing=1.0,
            )

        for bi, b in enumerate(loaded, start=1):
            ax_head = fig.add_subplot(gs[bi, 0])
            draw_building_header_side(
                ax_head,
                b["image_path"],
                b["name"],
                name_fontsize=building_name_fontsize,
            )

            for ci, (key, _, _, _) in enumerate(COMPONENTS, start=1):
                ax = fig.add_subplot(gs[bi, ci])
                map_axes[(key, bi)] = ax
                df = b["combined"]
                extent = b["extent"]
                west, east, south, north = extent

                if not no_basemap:
                    add_web_basemap(
                        ax,
                        extent,
                        provider_key=provider,
                        zoom=zoom,
                        alpha=basemap_alpha,
                        cache_root=Path(output_path).expanduser().parent / "_tile_cache",
                    )

                ax.set_xlim(west, east)
                ax.set_ylim(south, north)
                set_geo_aspect(ax, extent)

                values = component_values(df, key)
                x = df["_plot_x"].to_numpy(float)
                y = df["_plot_y"].to_numpy(float)
                good = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
                vmin, vmax = scales[key]
                norm = make_norm(vmin, vmax)

                ax.scatter(
                    x[good],
                    y[good],
                    c=values[good],
                    s=point_size,
                    cmap=cmap_struct,
                    norm=norm,
                    edgecolors="none",
                    alpha=0.98,
                    rasterized=True,
                    zorder=5,
                )

                annotate_component(
                    ax,
                    key,
                    values[good],
                    b["combined_physical"],
                )
                format_map_ticks(
                    ax,
                    show_yticklabels=True,
                    show_xticklabels=(bi == n_buildings),
                    max_ticks=4,
                    tick_label_fontsize=tick_label_fontsize,
                    show_coordinates=show_coordinates,
                )

        fig.canvas.draw()

        # Put colorbars immediately below the last map row.
        for ci, (key, _, _, cblabel) in enumerate(COMPONENTS, start=1):
            col_bottom_ax = map_axes[(key, n_buildings)]
            pos = col_bottom_ax.get_position()
            cb_h = 0.014
            cb_gap = 0.006
            cax = fig.add_axes(
                [pos.x0, pos.y0 - cb_gap - cb_h, pos.width, cb_h]
            )
            sm = ScalarMappable(
                norm=make_norm(*scales[key]),
                cmap=cmap_struct,
            )
            sm.set_array([])
            cb = fig.colorbar(
                sm,
                cax=cax,
                orientation="horizontal",
            )
            cb.ax.tick_params(
                labelsize=10,
                length=2.2,
                pad=1.5,
            )
            cb.set_label(
                cblabel,
                fontsize=10.5,
                fontweight="bold",
                labelpad=2,
            )

        # Separators centered in the now-equal physical row gaps.
        for bi in range(2, n_buildings + 1):
            prev_ax = map_axes[(COMPONENTS[0][0], bi - 1)]
            next_ax = map_axes[(COMPONENTS[0][0], bi)]
            p_prev = prev_ax.get_position()
            p_next = next_ax.get_position()
            y = 0.5 * (p_prev.y0 + p_next.y1)
            x0 = map_axes[(COMPONENTS[0][0], bi - 1)].get_position().x0
            x1 = map_axes[(COMPONENTS[-1][0], bi - 1)].get_position().x1
            fig.add_artist(
                plt.Line2D(
                    [x0, x1],
                    [y, y],
                    transform=fig.transFigure,
                    linewidth=0.7,
                    color="0.80",
                    alpha=0.85,
                )
            )

    else:
        raise ValueError(f"Unknown layout: {layout}")

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"\n[SAVED] {output_path.resolve()}")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {pdf_path.resolve()}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot cumulative structural building-analysis components for selected buildings in one combined figure.")
    p.add_argument("--results-root", default=".")
    p.add_argument("--results", nargs="+", default=None, help="Specific result folders; otherwise scan --results-root")
    p.add_argument("--image-dir", default=None, help="Optional building photograph folder")
    p.add_argument("--coord-mode", choices=["auto", "geocorr", "original"], default="original", help="Map coordinates. auto prefers X_geocorr/Y_geocorr.")
    p.add_argument("--basemap-provider", choices=sorted(BASEMAP_PROVIDERS.keys()), default="esri_satellite")
    p.add_argument("--basemap-zoom", type=int, default=19)
    p.add_argument("--basemap-alpha", type=float, default=0.38)
    p.add_argument("--no-basemap", action="store_true")

    p.add_argument("--total-vmin", type=float, default=-100.0)
    p.add_argument("--total-vmax", type=float, default=30.0)
    p.add_argument("--uniform-vmin", type=float, default=-100.0)
    p.add_argument("--uniform-vmax", type=float, default=30.0)
    p.add_argument("--structural-vmin", type=float, default=-30.0, help="Fallback lower limit for height, tilt, and residual if component-specific limits are not provided.")
    p.add_argument("--structural-vmax", type=float, default=30.0, help="Fallback upper limit for height, tilt, and residual if component-specific limits are not provided.")

    p.add_argument("--height-vmin", type=float, default=None)
    p.add_argument("--height-vmax", type=float, default=None)
    p.add_argument("--tilt-vmin", type=float, default=None)
    p.add_argument("--tilt-vmax", type=float, default=None)
    p.add_argument("--residual-vmin", type=float, default=None)
    p.add_argument("--residual-vmax", type=float, default=None)

    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fig-width", type=float, default=33.0)
    p.add_argument("--map-row-height", type=float, default=6.4)
    p.add_argument("--header-height", type=float, default=6.0)
    p.add_argument("--point-size", type=float, default=20.0)
    p.add_argument("--component-label-fontsize", type=float, default=20.0)
    p.add_argument("--building-name-fontsize", type=float, default=20.0)
    p.add_argument("--tick-label-fontsize", type=float, default=11.5)
    p.add_argument("--lon-pad-fraction", type=float, default=0.01,
                   help="Tight east-west padding around combined ASC+DESC PS footprint.")
    p.add_argument("--lat-pad-fraction", type=float, default=0.08,
                   help="Initial north-south padding before target-ratio expansion.")
    p.add_argument("--map-target-ratio", type=float, default=0.78,
                   help="Target physical N-S / E-W span ratio; larger values make maps taller.")
    p.add_argument("--column-spacing", type=float, default=0.10,
                   help="Horizontal spacing between columns in the chosen layout.")
    p.add_argument("--row-spacing", type=float, default=0.04,
                   help="For buildings-y, master visual gap control used to match horizontal and vertical spacing.")
    p.add_argument("--hide-coordinates", action="store_true",
                   help="Remove latitude/longitude ticks and labels from all map panels.")
    p.add_argument("--layout", choices=["buildings-x", "buildings-y"], default="buildings-x",
                   help="buildings-x: buildings across X and components down Y; buildings-y: buildings down Y and components across X.")
    p.add_argument("--out", default="SunnyIsles_4buildings_structural_components.png")
    return p.parse_args()


def main():
    args = parse_args()
    buildings = load_building_config(results_root=args.results_root, image_dir=args.image_dir, requested=args.results)
    make_figure(
        buildings=buildings,
        output_path=args.out,
        provider=args.basemap_provider,
        zoom=args.basemap_zoom,
        coord_mode=args.coord_mode,
        dpi=args.dpi,
        fig_width=args.fig_width,
        map_row_height=args.map_row_height,
        header_height=args.header_height,
        point_size=args.point_size,
        no_basemap=args.no_basemap,
        basemap_alpha=args.basemap_alpha,
        total_vmin=args.total_vmin,
        total_vmax=args.total_vmax,
        uniform_vmin=args.uniform_vmin,
        uniform_vmax=args.uniform_vmax,
        structural_vmin=args.structural_vmin,
        structural_vmax=args.structural_vmax,
        height_vmin=args.height_vmin,
        height_vmax=args.height_vmax,
        tilt_vmin=args.tilt_vmin,
        tilt_vmax=args.tilt_vmax,
        residual_vmin=args.residual_vmin,
        residual_vmax=args.residual_vmax,
        component_label_fontsize=args.component_label_fontsize,
        building_name_fontsize=args.building_name_fontsize,
        tick_label_fontsize=args.tick_label_fontsize,
        lon_pad_fraction=args.lon_pad_fraction,
        lat_pad_fraction=args.lat_pad_fraction,
        target_geo_ratio=args.map_target_ratio,
        column_spacing=args.column_spacing,
        row_spacing=args.row_spacing,
        show_coordinates=(not args.hide_coordinates),
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
