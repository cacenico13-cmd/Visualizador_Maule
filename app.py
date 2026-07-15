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
import os
import traceback

# ─────────────────────────────────────────────────────────────
# Silenciar advertencias de consola
# ─────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
try:
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
except AttributeError:
    pass

# ─────────────────────────────────────────────────────────────
# Configuración y Diagnóstico
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoVisualizador Cuenca del Maule", layout="wide")
DATA = Path("data")

st.sidebar.header("🛠️ Diagnóstico del Sistema")
st.sidebar.write(f"**Directorio:** `{os.getcwd()}`")

if DATA.exists():
    archivos_data = list(DATA.iterdir())
    st.sidebar.success(f"Carpeta 'data' encontrada con {len(archivos_data)} archivos.")
    with st.sidebar.expander("Ver archivos encontrados"):
        for f in archivos_data:
            st.sidebar.text(f.name)
else:
    st.sidebar.error("❌ No se encontró la carpeta 'data'. El mapa no funcionará.")

st.title("💧 GeoVisualizador Cuenca del Maule (Calidad de Agua)")

# ─────────────────────────────────────────────────────────────
# Lógica de Rasters con Diagnóstico
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

@st.cache_data(show_spinner=False)
def procesar_raster_a_memoria(raster_path_str, es_dem=False):
    try:
        with rasterio.open(raster_path_str) as src:
            # Validación sugerida por IA: Verificar si no tiene georreferenciación
            if src.transform.is_identity:
                return "ERROR_IDENTITY", None
                
            crs_origen = src.crs
            transform_orig = src.transform
            if crs_origen is None: crs_origen = "EPSG:32719"
            
            es_wgs84 = False
            if crs_origen:
                if hasattr(crs_origen, 'to_epsg') and crs_origen.to_epsg() == 4326: es_wgs84 = True
                elif str(crs_origen).upper() == "EPSG:4326": es_wgs84 = True

            if not es_wgs84:
                transform, width, height = calculate_default_transform(crs_origen, "EPSG:4326", src.width, src.height, *src.bounds)
                data = np.zeros((src.count, height, width), dtype=np.float32)
                for i in range(1, src.count + 1):
                    reproject(source=rasterio.band(src, i), destination=data[i - 1], src_transform=transform_orig, src_crs=crs_origen, dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
                bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
            else:
                data = src.read().astype(np.float32)
                bounds_wgs84 = src.bounds

            if bounds_wgs84[0] < -180 or bounds_wgs84[2] > 180 or bounds_wgs84[1] < -90 or bounds_wgs84[3] > 90: 
                return "ERROR_BOUNDS", None

            nodata = src.nodata
            if es_dem: img_array, _, _ = aplicar_colormap_dem(data[0], nodata)
            else:
                if src.count >= 3: rgb = data[:3].copy()
                else: rgb = np.stack([data[0]] * 3)
                for i in range(3):
                    band, mask = rgb[i], (rgb[i] == nodata) if nodata is not None else np.zeros_like(rgb[i], dtype=bool)
                    valid = band[~mask]
                    if len(valid) > 0:
                        mn, mx = np.percentile(valid, 2), np.percentile(valid, 98)
                        rgb[i] = np.clip((band - mn) / (mx - mn + 1e-10), 0, 1)
                    rgb[i][mask] = 0
                base = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
                alpha = np.full((base.shape[0], base.shape[1]), 200, dtype=np.uint8)
                if nodata is not None: alpha[data[0] == nodata] = 0
                img_array = np.dstack([base, alpha])

            img_pil = Image.fromarray(img_array)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8"), [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
    except Exception as e:
        return f"ERROR_EXCEPTION: {str(e)}", None

# ─────────────────────────────────────────────────────────────
# Carga de Vectores con Manejo de Excepciones
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_vectores():
    capas = {}
    errores_carga = []
    
    if DATA.exists():
        for archivo in (list(DATA.glob("*.gpkg")) + list(DATA.glob("*.shp"))):
            try:
                gdf = gpd.read_file(archivo)
                if gdf.crs is None: gdf = gdf.set_crs("EPSG:32719", allow_override=True)
                gdf = gdf.to_crs("EPSG:4326")
                for col in gdf.select_dtypes(include=['datetime', 'datetimetz']).columns: gdf[col] = gdf[col].astype(str)
                gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
                capas[archivo.stem.replace("_", " ")] = gdf
            except Exception as e:
                errores_carga.append((archivo.name, traceback.format_exc()))
    
    return capas, errores_carga

capas, errores_vectores = cargar_vectores()

st.sidebar.subheader("Carga de Capas")
if errores_vectores:
    for nombre, error in errores_vectores:
        st.sidebar.error(f"❌ Falló: {nombre}")
        with st.sidebar.expander("Ver detalle técnico"):
            st.code(error)
else:
    st.sidebar.success(f"✅ {len(capas)} capas vectoriales listas.")

# ─────────────────────────────────────────────────────────────
# Armado del Mapa
# ─────────────────────────────────────────────────────────────
m = folium.Map(location=[-35.7, -71.5], zoom_start=9, tiles=None)

# 1. Raster
dem_file = DATA / "dem_hillshade.tif"
if dem_file.exists():
    img_b64, bounds_img = procesar_raster_a_memoria(str(dem_file), es_dem=False)
    if bounds_img:
        folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_b64}", bounds=bounds_img, opacity=0.7, name="Sombra de colina").add_to(m)
    else:
        if img_b64 == "ERROR_IDENTITY":
            st.sidebar.warning("⚠️ El TIF de sombra de colina no tiene transformada geográfica. Omitido.")
        elif img_b64 == "ERROR_BOUNDS":
            st.sidebar.warning("⚠️ El TIF generó coordenadas infinitas o inválidas. Omitido.")
        else:
            st.sidebar.warning(f"⚠️ Error cargando TIF: {img_b64}")

folium.TileLayer('openstreetmap', name="Mapa base").add_to(m)

# 2. Vectores
for nombre, gdf in capas.items():
    nombre_lower = nombre.lower()
    
    if "estacion" in nombre_lower:
        folium.GeoJson(
            gdf, name=nombre, 
            marker=folium.CircleMarker(radius=6, fill=True, color="red"),
            tooltip=folium.GeoJsonTooltip(fields=[gdf.columns[1]])
        ).add_to(m)
                       
    elif "toponimo" in nombre_lower:
        col_nombre = next((c for c in ["NOMBRE", "Nombre", "nombre", "TEXTO", "TextString", "NAME"] if c in gdf.columns), gdf.columns[0])
        folium.GeoJson(
            gdf, name=nombre,
            marker=folium.CircleMarker(radius=3, color="gray", fill=True),
            tooltip=folium.GeoJsonTooltip(fields=[col_nombre])
        ).add_to(m)
        
    elif "hidro" in nombre_lower or "subcuen" in nombre_lower:
        gdf_rios = gdf[gdf["Dren_Tipo"] == "Río"] if "Dren_Tipo" in gdf.columns else gdf
        if not gdf_rios.empty:
            cols_disp = [c for c in ["Nombre", "Dren_Tipo", "Region", "Provincia"] if c in gdf_rios.columns]
            folium.GeoJson(
                gdf_rios, name=nombre,
                style_function=lambda x: {'color': '#1E88E5', 'weight': 1.5, 'opacity': 0.8},
                tooltip=folium.GeoJsonTooltip(fields=cols_disp) if cols_disp else None
            ).add_to(m)
        
    else:
        folium.GeoJson(gdf, name=nombre).add_to(m)

folium.LayerControl().add_to(m)

salida_mapa = st_folium(
    m, 
    width=1000, 
    height=600, 
    key="mapa_final",
    returned_objects=["last_active_drawing"]
)

# ─────────────────────────────────────────────────────────────
# Análisis e Integración de Gráfico
# ─────────────────────────────────────────────────────────────
st.subheader("📊 Análisis de Calidad")
archivo_datos = DATA / "datos_limpios_sin_outliers.xlsx"

if archivo_datos.exists():
    try:
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
                            trendline="lowess", trendline_color_override="blue", opacity=0.7
                        )
                        fig.update_traces(marker=dict(size=6))
                        fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(df_plot[['FECHA MEDICION', 'PARAMETRO', 'VALOR']], use_container_width=True)
                    else:
                        st.warning(f"No hay registros en el Excel para el código: {valor_mapa}")
                else:
                    st.error(f"No se encontró la columna '{col_codigo}' en el Excel.")
            else:
                st.info("Haz clic en una estación para iniciar el análisis.")
        else:
            st.info("👆 Haz clic en un marcador rojo en el mapa para ver el análisis.")
            
    except Exception as e:
        st.error("Error al procesar el archivo Excel.")
        st.code(traceback.format_exc())
else:
    st.error("❌ No se encontró el archivo de datos Excel.")
