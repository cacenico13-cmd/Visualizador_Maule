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
import warnings

# ─────────────────────────────────────────────────────────────
# Silenciar advertencias fantasmas en los logs
# ─────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
try:
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
except AttributeError:
    pass

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoVisualizador Cuenca del Maule", layout="wide")
st.title("💧 GeoVisualizador Cuenca del Maule (Calidad de Agua)")
DATA = Path("data")

# ─────────────────────────────────────────────────────────────
# Lógica de Rasters (Optimizada y Segura)
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

# ¡CLAVE! Guardamos el raster ya procesado en la memoria RAM (caché)
@st.cache_data(show_spinner=False)
def procesar_raster_a_memoria(raster_path_str, es_dem=False):
    try:
        with rasterio.open(raster_path_str) as src:
            crs_origen = src.crs
            transform_orig = src.transform
            
            if crs_origen is None:
                crs_origen = "EPSG:32719"

            es_wgs84 = False
            if crs_origen:
                if hasattr(crs_origen, 'to_epsg') and crs_origen.to_epsg() == 4326:
                    es_wgs84 = True
                elif str(crs_origen).upper() == "EPSG:4326":
                    es_wgs84 = True

            if not es_wgs84:
                transform, width, height = calculate_default_transform(
                    crs_origen, "EPSG:4326", src.width, src.height, *src.bounds
                )
                data = np.zeros((src.count, height, width), dtype=np.float32)
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=data[i - 1],
                        src_transform=transform_orig,
                        src_crs=crs_origen,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.bilinear,
                    )
                bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
            else:
                data = src.read().astype(np.float32)
                bounds_wgs84 = src.bounds

            # Protección de coordenadas infinitas
            if bounds_wgs84[0] < -180 or bounds_wgs84[2] > 180 or bounds_wgs84[1] < -90 or bounds_wgs84[3] > 90:
                return None, None

            nodata = src.nodata

            if es_dem:
                img_array, _, _ = aplicar_colormap_dem(data[0], nodata)
            else:
                if src.count >= 3:
                    rgb = data[:3].copy()
                else:
                    rgb = np.stack([data[0]] * 3)
                for i in range(3):
                    band  = rgb[i]
                    mask  = (band == nodata) if nodata is not None else np.zeros_like(band, dtype=bool)
                    valid = band[~mask]
                    if len(valid) > 0:
                        mn, mx = np.percentile(valid, 2), np.percentile(valid, 98)
                        rgb[i] = np.clip((band - mn) / (mx - mn + 1e-10), 0, 1)
                    rgb[i][mask] = 0
                base = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
                alpha = np.full((base.shape[0], base.shape[1]), 200, dtype=np.uint8)
                if nodata is not None:
                    alpha[data[0] == nodata] = 0
                img_array = np.dstack([base, alpha])

            img_pil = Image.fromarray(img_array)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            bounds_final = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
            
            return img_b64, bounds_final
    except Exception:
        return None, None

# ─────────────────────────────────────────────────────────────
# Carga de Vectores (En caché)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_vectores():
    capas = {}
    for archivo in (list(DATA.glob("*.gpkg")) + list(DATA.glob("*.shp"))):
        try:
            gdf = gpd.read_file(archivo)
            if gdf.crs is None: gdf = gdf.set_crs("EPSG:32719", allow_override=True)
            gdf = gdf.to_crs("EPSG:4326")
            for col in gdf.select_dtypes(include=['datetime', 'datetimetz']).columns:
                gdf[col] = gdf[col].astype(str)
            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
            capas[archivo.stem.replace("_", " ")] = gdf
        except Exception: pass
    return capas

capas = cargar_vectores()

# ─────────────────────────────────────────────────────────────
# Armado del Mapa (Rápido, sin cachear el objeto folium)
# ─────────────────────────────────────────────────────────────
m = folium.Map(location=[-35.7, -71.5], zoom_start=9, tiles=None)

# 1. Cargar Hillshade desde la memoria caché
dem_file = DATA / "dem_hillshade.tif"
if dem_file.exists():
    img_b64, bounds_img = procesar_raster_a_memoria(str(dem_file), es_dem=False)
    if img_b64 and bounds_img:
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=bounds_img,
            opacity=0.7,
            name="Sombra de colina"
        ).add_to(m)

# 2. Agregar mapa base
folium.TileLayer('openstreetmap', name="Mapa base").add_to(m)

# 3. Construir vectores
for nombre, gdf in capas.items():
    nombre_lower = nombre.lower()
    
    # Estaciones
    if "estacion" in nombre_lower:
        folium.GeoJson(gdf, name=nombre, 
                       marker=folium.CircleMarker(radius=6, fill=True, color="red"),
                       tooltip=folium.GeoJsonTooltip(fields=[gdf.columns[1]])).add_to(m)
                       
    # Topónimos
    elif "toponimo" in nombre_lower:
        fg_toponimos = folium.FeatureGroup(name=nombre)
        col_nombre = next((c for c in ["NOMBRE", "Nombre", "nombre", "TEXTO", "TextString", "NAME"] if c in gdf.columns), gdf.columns[0])
        for _, row in gdf.iterrows():
            if row.geometry:
                texto = str(row[col_nombre]).strip()
                if texto and texto.lower() not in ["none", "nan", ""]:
                    pt = row.geometry.representative_point()
                    etiqueta = folium.DivIcon(
                        html=f'<div style="font-size: 11px; font-weight: bold; color: #222; text-shadow: 1px 1px 3px white, -1px -1px 3px white, 1px -1px 3px white, -1px 1px 3px white; white-space: nowrap;">{texto}</div>'
                    )
                    folium.Marker(location=[pt.y, pt.x], icon=etiqueta).add_to(fg_toponimos)
        fg_toponimos.add_to(m)
        
    # Hidrología (Ríos)
    elif "hidro" in nombre_lower or "subcuen" in nombre_lower:
        if "Dren_Tipo" in gdf.columns:
            gdf_rios = gdf[gdf["Dren_Tipo"] == "Río"]
        else:
            gdf_rios = gdf
            
        if not gdf_rios.empty:
            fg_hidro = folium.FeatureGroup(name=nombre)
            cols_disp = [c for c in ["Nombre", "Dren_Tipo", "Region", "Provincia"] if c in gdf_rios.columns]
            
            folium.GeoJson(
                gdf_rios, 
                style_function=lambda x: {'color': '#1E88E5', 'weight': 1.5, 'opacity': 0.8},
                tooltip=folium.GeoJsonTooltip(fields=cols_disp) if cols_disp else None
            ).add_to(fg_hidro)
            
            if "Nombre" in gdf_rios.columns:
                nombres_vistos = set()
                for _, row in gdf_rios.iterrows():
                    if row.geometry and not pd.isna(row["Nombre"]):
                        texto = str(row["Nombre"]).strip()
                        if texto and texto.lower() not in ["none", "nan", "sin nombre", ""]:
                            if texto not in nombres_vistos:
                                nombres_vistos.add(texto)
                                pt = row.geometry.representative_point()
                                etiqueta_rio = folium.DivIcon(
                                    html=f'<div style="font-size: 10px; font-style: italic; font-weight: bold; color: #0D47A1; text-shadow: 1px 1px 2px white, -1px -1px 2px white, 1px -1px 2px white, -1px 1px 2px white; white-space: nowrap;">{texto}</div>'
                                )
                                folium.Marker(location=[pt.y, pt.x], icon=etiqueta_rio).add_to(fg_hidro)
            fg_hidro.add_to(m)
        
    else:
        folium.GeoJson(gdf, name=nombre).add_to(m)

folium.LayerControl().add_to(m)

# El motor web de Streamlit ahora solo escuchará el clic exacto en la estación
salida_mapa = st_folium(
    m, 
    width=1000, 
    height=500, 
    key="mapa_final",
    returned_objects=["last_active_drawing"]
)

# ─────────────────────────────────────────────────────────────
# Análisis e Integración de Gráfico
# ─────────────────────────────────────────────────────────────
st.subheader("📊 Análisis de Calidad")
archivo_datos = DATA / "datos_limpios_sin_outliers.xlsx"

if archivo_datos.exists():
    df = pd.read_excel(archivo_datos)
    
    if salida_mapa and salida_mapa.get("last_active_drawing"):
        props = salida_mapa["last_active_drawing"].get("properties", {})
        valor_mapa = list(props.values())[0] if props else None
        
        if valor_mapa:
            st.write(f"### Estación detectada: {valor_mapa}")
            col_codigo = 'COD. ESTACIÓN' 
            
            if col_codigo in df.columns:
                df_est = df[df[col_codigo].astype(str) == str(valor_mapa)]
                
                if not df_est.empty:
                    parametro = st.selectbox("Seleccione el parámetro a graficar:", df_est['PARAMETRO'].unique())
                    df_plot = df_est[df_est['PARAMETRO'] == parametro].copy()
                    
                    df_plot['FECHA MEDICION'] = pd.to_datetime(df_plot['FECHA MEDICION'])
                    df_plot = df_plot.sort_values('FECHA MEDICION')
                    
                    fig = px.scatter(
                        df_plot, x='FECHA MEDICION', y='VALOR',
                        title=f"Serie temporal: {parametro} (Datos limpios)",
                        trendline="lowess",
                        trendline_color_override="blue",
                        opacity=0.7
                    )
                    
                    fig.update_traces(marker=dict(size=6))
                    fig.update_layout(
                        plot_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df_plot[['FECHA MEDICION', 'PARAMETRO', 'VALOR']], use_container_width=True)
                else:
                    st.warning(f"No hay registros en el Excel para el código: {valor_mapa}")
            else:
                st.error(f"No se encontró la columna '{col_codigo}'.")
        else:
            st.info("No se pudo obtener el código del mapa.")
    else:
        st.info("👆 Haz clic en un marcador rojo en el mapa para ver el análisis.")
