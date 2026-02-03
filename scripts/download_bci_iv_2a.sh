#!/usr/bin/env bash
# Download BCI Competition IV Dataset 2a (GDF, ~420 MB) into data/BCI_IV_2a/

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT/data/BCI_IV_2a"
ZIP_URL="https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
ZIP_FILE="$DATA_DIR/BCICIV_2a_gdf.zip"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -f "A01T.gdf" ]; then
  echo "BCI IV 2a GDF files already present in $DATA_DIR"
  exit 0
fi

echo "Downloading BCI Competition IV Dataset 2a (~420 MB)..."
curl -L -o "$ZIP_FILE" "$ZIP_URL" || { echo "Download failed. Get the zip manually from: $ZIP_URL"; exit 1; }

echo "Unzipping..."
unzip -o "$ZIP_FILE"
rm -f "$ZIP_FILE"
# If zip contained a subfolder (e.g. BCICIV_2a_gdf/), move .gdf into DATA_DIR
for sub in "$DATA_DIR"/*/; do
  if [ -d "$sub" ] && ls "$sub"/*.gdf 1>/dev/null 2>&1; then
    mv "$sub"/*.gdf "$DATA_DIR/"
    rmdir "$sub" 2>/dev/null || true
    break
  fi
done

echo "Done. GDF files are in $DATA_DIR"
ls -la "$DATA_DIR"/*.gdf 2>/dev/null || true
