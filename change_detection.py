from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyrsgis import raster


CLASS_LABELS: List[str] = [
  "Urban",
  "Grass",
  "Forest",
  "Bare Soil",
  "Water Bodies",
  "Coastal Areas",
  "Wetland",
]


CLASS_COLORS: List[Tuple[int, int, int]] = [
  (165, 42, 42),     # Urban - brownish/red
  (255, 235, 59),    # Grass - yellow
  (34, 139, 34),     # Forest - green
  (210, 180, 140),   # Bare Soil - tan
  (30, 144, 255),    # Water Bodies - blue
  (244, 164, 96),    # Coastal Areas - sandy
  (46, 139, 87),     # Wetland - teal/green
]
UNCHANGED_COLOR: Tuple[int, int, int] = (255, 255, 255) 



def load_label_array(path: str) -> np.ndarray:
  arr = np.load(path)
  if arr.ndim > 1:
    arr = arr.ravel()
  return arr.astype(np.int64, copy=False)


def load_labels(years: List[int], labels_dir: str, pattern: str) -> Dict[int, np.ndarray]:
  data: Dict[int, np.ndarray] = {}
  expected_size = None
  for y in years:
    fname = pattern.format(year=y)
    path = os.path.join(labels_dir, fname)
    if not os.path.isfile(path):
      raise FileNotFoundError(f"Missing label file for {y}: {path}")
    arr = load_label_array(path)
    if expected_size is None:
      expected_size = arr.size
    elif arr.size != expected_size:
      raise ValueError(f"Label size mismatch at {y}: {arr.size} vs {expected_size}")
    data[y] = arr
  return data



def transition_counts(prev_labels: np.ndarray, curr_labels: np.ndarray, n_classes: int) -> np.ndarray:
  idx = prev_labels * n_classes + curr_labels
  mat = np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
  return mat



def make_replacement_rgb(prev_labels: np.ndarray, curr_labels: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
  h, w = shape
  changed = (prev_labels != curr_labels)
  rgb = np.full((h, w, 3), UNCHANGED_COLOR, dtype=np.uint8)
  if changed.any():
    to_classes = curr_labels[changed]
    # Build color map array
    color_lut = np.array(CLASS_COLORS, dtype=np.uint8)
    colors = color_lut[to_classes]
    # Place into image
    yy, xx = np.nonzero(changed.reshape(h, w))
    rgb[yy, xx, :] = colors
  return rgb


def get_extent_and_shape_from_tif(tif_path: str) -> Tuple[List[float], Tuple[int, int]]:
  ds, arr = raster.read(tif_path, bands='all')
  min_x = ds.bbox[0][0]
  max_x = ds.bbox[1][0]
  min_y = ds.bbox[0][1]
  max_y = ds.bbox[1][1]
  extent = [min_x, max_x, min_y, max_y]
  h, w = int(arr.shape[1]), int(arr.shape[2])
  return extent, (h, w)


def save_palette_legend(path: str) -> None:
  fig, ax = plt.subplots(figsize=(10,10))
  ax.axis('off')
  for i, (label, color) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
    ax.barh(i, 1, color=np.array(color)/255.0)
    ax.text(1.05, i, label, va='center')
  ax.set_ylim(-0.5, len(CLASS_LABELS) - 0.5)
  ax.set_yticks([])
  ax.set_xlim(0, 1.5)
  plt.tight_layout()
  plt.savefig(path, dpi=200)
  plt.close()


def main():
  start_year = 2013
  end_year = 2024
  labels_dir = os.path.join("results", "vector_files")
  file_pattern = "classified_{year}.npy"
  output_dir = os.path.join("results", "change_analysis")
  pixel_area = 849.73
  n_classes = len(CLASS_LABELS)
  image_shape = 780,818

  years = list(range(start_year, end_year + 1))

  os.makedirs(output_dir, exist_ok=True)

  # Load labels
  labels_by_year = load_labels(years, labels_dir, file_pattern)

  # Compute transition area matrices for consecutive pairs (no CSV output)
  pair_years_from: List[int] = years[:-1]
  pair_years_to: List[int] = years[1:]
  transition_area_mats: List[np.ndarray] = []

  for y0, y1 in zip(pair_years_from, pair_years_to):
    counts = transition_counts(labels_by_year[y0], labels_by_year[y1], n_classes)
    area_mat = counts.astype(np.float64) * pixel_area
    transition_area_mats.append(area_mat)


  per_class_per_year = np.stack([mat.sum(axis=1) for mat in transition_area_mats], axis=1)  # (n_classes, num_pairs)

  series_rows = []
  for col_idx, y in enumerate(pair_years_to):
    row = {CLASS_LABELS[c]: float(per_class_per_year[c, col_idx]) for c in range(n_classes)}
    row["year"] = y
    series_rows.append(row)
  series_df = pd.DataFrame(series_rows).set_index("year")
  series_csv = os.path.join(output_dir, "area_per_class_per_year.csv")
  series_df.to_csv(series_csv)

  # Plot series using real labels
  plt.figure(figsize=(10, 10))
  for c, name in enumerate(CLASS_LABELS):
    plt.plot(pair_years_to, per_class_per_year[c, :], label=name)
  plt.xlabel("Year")
  plt.ylabel("Area (sq meters)")
  plt.title("Area per Class over Years (from transitions)")
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.legend(loc="best", ncol=2)
  plt.tight_layout()
  plot_path = os.path.join(output_dir, "area_per_class_over_years.png")
  plt.savefig(plot_path, dpi=200)
  plt.close()


  if True:
    ref_tif = os.path.join('data', f'{years[0]}_original.tif')
    extent, (h, w) = get_extent_and_shape_from_tif(ref_tif)
  
    first_urban_year = np.full(h * w, -1, dtype=np.int32)
    prev = labels_by_year[years[0]]
    for y in years[1:]:
      curr = labels_by_year[y]
      became_urban = (prev != 0) & (curr == 0) & (first_urban_year == -1)
      first_urban_year[became_urban] = y
      prev = curr

    # Color by year bucket; unchanged stays white
    valid = first_urban_year > -1
    if valid.any():
      # Build a year-indexed float image for plotting with colormap
      year_img = np.full((h, w), np.nan, dtype=np.float32)
      year_img.reshape(-1)[valid] = first_urban_year[valid]
  

      fig, ax = plt.subplots(figsize=(10, 10))
      ax.tick_params(axis='both', which='major', labelsize=12)
      ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      plt.yticks(rotation=90, va='center')
      ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      # 2013 baseline urban mask (class 0)
      baseline_mask = (labels_by_year[years[0]] == 0).reshape(h, w)
      base_cmap = ListedColormap(['white', '#4d4d4d'])  # non-urban=white, baseline urban=dark gray
      ax.imshow(baseline_mask.astype(np.uint8), cmap=base_cmap, extent=extent, interpolation='nearest')
      # Overlay year-colored expansion
      cmap = plt.get_cmap('YlOrRd')
      im = ax.imshow(year_img, cmap=cmap, vmin=years[1], vmax=years[-1], extent=extent, interpolation='nearest')
      ax.set_xlabel('Longitude')
      ax.set_ylabel('Latitude')
      ax.set_title('Urban expansion: first year becoming Urban')
      cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08, location='bottom')
      cbar.set_label('Year')
      # Baseline (2013) key strip just below colorbar
      # Place an extra small axes below the colorbar using figure coordinates
      bbox = cbar.ax.get_position()
      x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
      height = (y1 - y0) * 0.7
      y = y0 - height - 0.01
      cax2 = fig.add_axes([x0, y, x1 - x0, height])
      cax2.axis('off')
      cax2.add_patch(Rectangle((0.02, 0.2), 0.08, 0.6, transform=cax2.transAxes, color='#4d4d4d', ec='none'))
      cax2.text(0.12, 0.5, 'Urban (2013)', transform=cax2.transAxes, va='center', ha='left')

      plt.grid(False)
      plt.tight_layout()
      fig.savefig(os.path.join(output_dir, 'urban_expension_map.png'), dpi=200)
      plt.close(fig)



if __name__ == "__main__":
  main()
