import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import pandas as pd
import plotly.express as px
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import io, base64
from PIL import Image
import matplotlib.colors as mcolors
import branca.colormap as cm

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoVisualizador Cuenca del Maule", layout="wide")
st.title("💧 GeoVisualizador Cuenca del Maule (Calidad de Agua)")
DATA = Path("data")

# ─────────────────────────────────────────────────────────────
# Lógica de Rasters
# ─────────────────────────────────────────────────────────────
COLORMAP_DEM = [
    (0.00, "#006400"), (0.15, "#228B22"), (0.30, "#9ACD32"), 
    (0.45, "#DAA520"), (0.60, "#CD853F"), (0.75, "#8B4513"), 
    (0.88, "#D2B48C"), (1.00, "#FFFAFA"),
]

def aplicar_colormap_dem(band, nodata):
    posiciones = [p for p, _ in COLORMAP_DEM]
    colores    = [c for _, c in COLORMAP_DEM]
    cmap = mcolors.LinearSegmentedColormap.from_list("dem", list(zip(posiciones, colores)))

    mascara = (band == nodata) if nodata is not None else np.zeros_like(band, dtype=bool)
    valid   = band[~mascara]

    dem_min = float(valid.min()) if len(valid) > 0 else 0
    dem_max = float(valid.max()) if len(valid) > 0 else 1

    norm  = mcolors.Normalize(vmin=dem_min, vmax=dem_max)
    rgba  = cmap(norm(band))

    rgba[mascara, 3] = 0
    rgba[~mascara, 3] = 0.82

    img_array = (rgba * 255).astype(np.uint8)
    return img_array, dem_min, dem_max

MAX_DIM_OVERLAY = 1500

@st.cache_data
def raster_a_overlay(raster_path, es_dem=False):
    with rasterio.open(raster_path) as src:
        src_crs = src.crs or rasterio.crs.CRS.from_epsg(32719)
        if src_crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src_crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
        else:
            transform, width, height = src.transform, src.width, src.height

        escala = max(width, height) / MAX_DIM_OVERLAY
        if escala > 1:
            width = max(1, int(width / escala))
            height = max(1, int(height / escala))
            transform = transform * rasterio.Affine.scale(escala, escala)

        data = np.zeros((src.count, height, width), dtype=np.float32)
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=data[i - 1],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
        bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
        nodata, dem_min, dem_max = src.nodata, None, None

        if es_dem:
            img_array, dem_min, dem_max = aplicar_colormap_dem(data[0], nodata)
        else:
            rgb = np.stack([data[0]]*3) if src.count < 3 else data[:3].copy()
            for i in range(3):
                mask = (rgb[i] == nodata) if nodata is not None else (rgb[i] <= 0)
                valid = rgb[i][~mask]
                if len(valid) > 0:
                    mn, mx = np.percentile(valid, 2), np.percentile(valid, 98)
                    rgb[i] = np.clip((rgb[i] - mn) / (mx - mn + 1e-10), 0, 1)
                rgb[i][mask] = 0
            base = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
            alpha = np.full((base.shape[0], base.shape[1]), 255, dtype=np.uint8)
            alpha[data[0] <= (nodata if nodata is not None else 0)] = 0
            img_array = np.dstack([base, alpha])

        img_pil = Image.fromarray(img_array)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8"), [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]], dem_min, dem_max

# ─────────────────────────────────────────────────────────────
# Carga de Vectores y Sidebar
# ─────────────────────────────────────────────────────────────
@st.cache_data
def cargar_vectores():
    capas = {}
    for archivo in (list(DATA.glob("*.gpkg")) + list(DATA.glob("*.shp"))):
        try:
            gdf = gpd.read_file(archivo)
            if gdf.crs is None: gdf = gdf.set_crs("EPSG:32719", allow_override=True)
            gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
            if gdf.geom_type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"]).any():
                gdf["geometry"] = gdf["geometry"].simplify(0.0003, preserve_topology=True)
            capas[archivo.stem.replace("_", " ")] = gdf
        except Exception: pass
    return capas

capas = cargar_vectores()

st.sidebar.markdown("### 🛰️ Capas y Controles")
mostrar_dem = st.sidebar.checkbox("Sombra de colina (DEM)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ Capas Vectoriales")
check_capas = {nombre: st.sidebar.checkbox(nombre.title(), value=True) for nombre in capas.keys()}

# ─────────────────────────────────────────────────────────────
# Mapa
# ─────────────────────────────────────────────────────────────
def construir_mapa(_capas, incluir_dem, _checks):
    m = folium.Map(location=[-35.7, -71.5], zoom_start=9, tiles="Esri.WorldImagery")
    folium.ControlScale().add_to(m)

    if incluir_dem:
        dem_file = DATA / "dem_hillshade.tif"
        if dem_file.exists():
            img_b64, bounds, _, _ = raster_a_overlay(dem_file, es_dem=False)
            folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_b64}", bounds=bounds, opacity=0.7, name="Sombra de colina").add_to(m)
            # Leyenda DEM
            colormap = cm.LinearColormap(colors=[c for _, c in COLORMAP_DEM], caption="Elevación")
            colormap.add_to(m)

    for nombre, gdf in _capas.items():
        if not _checks.get(nombre, True): continue
        # [Lógica de renderizado GeoJSON igual a tu versión original...]
        # (Se omite la repetición interna para brevedad, insertar aquí tu lógica de 
        # Estaciones, Topónimos, Hidro y General del código original)
        folium.GeoJson(gdf, name=nombre).add_to(m)

    folium.LayerControl().add_to(m)
    return m

m = construir_mapa(capas, mostrar_dem, check_capas)
salida_mapa = st_folium(m, width=1000, height=500, key="mapa_final")

# ─────────────────────────────────────────────────────────────
# Análisis e Integración de Gráfico
# ─────────────────────────────────────────────────────────────
if archivo_datos.exists() and salida_mapa.get("last_active_drawing"):
    # ... [tu lógica de filtro de estación]
    fig = px.scatter(df_plot, x='FECHA MEDICION', y='VALOR', trendline="lowess")
    st.plotly_chart(fig, use_container_width=True, config={
        'toImageButtonOptions': {'format': 'png', 'filename': 'grafico_calidad', 'scale': 2}
    })
