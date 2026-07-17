import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import MiniMap, MeasureControl
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

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoVisualizador Cuenca del Maule", layout="wide")
st.title("💧 GeoVisualizador Cuenca del Maule (Calidad de Agua)")

st.markdown("""
<style>
.footer-bar{background:#eef2ff;border-radius:8px;padding:14px 20px;margin-top:10px;
    display:flex;justify-content:space-between;font-size:13.5px;color:#333;}
</style>
""", unsafe_allow_html=True)
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

# Un overlay en un mapa web no necesita más resolución que esto (evita picos de RAM)
MAX_DIM_OVERLAY = 1500

@st.cache_data
def raster_a_overlay(raster_path, es_dem=False):
    with rasterio.open(raster_path) as src:
        # Parche de CRS: Si no tiene, forzamos UTM 19S (EPSG:32719) por seguridad
        src_crs = src.crs
        if src_crs is None:
            src_crs = rasterio.crs.CRS.from_epsg(32719)

        if src_crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src_crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
        else:
            transform, width, height = src.transform, src.width, src.height

        # Limitar tamaño de salida ANTES de reproyectar: reproyectar directo a un
        # tamaño chico es mucho más liviano que reproyectar a resolución completa
        # y recién después achicar (eso es lo que estaba causando el consumo de RAM excesivo).
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

        nodata  = src.nodata
        dem_min = dem_max = None

        if es_dem:
            img_array, dem_min, dem_max = aplicar_colormap_dem(data[0], nodata)
        else:
            if src.count >= 3:
                rgb = data[:3].copy()
            else:
                rgb = np.stack([data[0]] * 3)

            # Corrección de NODATA para Hillshade
            for i in range(3):
                band  = rgb[i]
                # Si no hay nodata definido, asumimos que los valores <= 0 son el fondo
                mask  = (band == nodata) if nodata is not None else (band <= 0)
                valid = band[~mask]
                if len(valid) > 0:
                    mn, mx = np.percentile(valid, 2), np.percentile(valid, 98)
                    rgb[i] = np.clip((band - mn) / (mx - mn + 1e-10), 0, 1)
                rgb[i][mask] = 0

            base = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
            alpha = np.full((base.shape[0], base.shape[1]), 255, dtype=np.uint8)
            
            if nodata is not None:
                alpha[data[0] == nodata] = 0
            else:
                alpha[data[0] <= 0] = 0
                
            img_array = np.dstack([base, alpha])

        img_pil = Image.fromarray(img_array)
        buf     = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        bounds = [
            [bounds_wgs84[1], bounds_wgs84[0]],
            [bounds_wgs84[3], bounds_wgs84[2]],
        ]
        return img_b64, bounds, dem_min, dem_max

# ─────────────────────────────────────────────────────────────
# Carga de Vectores
# ─────────────────────────────────────────────────────────────
@st.cache_data
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
            # Simplificar geometrías: reduce vértices sin cambiar la forma visible
            # a escala de cuenca. Clave para capas de líneas densas (ej. red de drenaje),
            # que si no se simplifican pueden colgar el navegador al renderizar el GeoJSON.
            if gdf.geom_type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"]).any():
                gdf["geometry"] = gdf["geometry"].simplify(0.0003, preserve_topology=True)
            capas[archivo.stem.replace("_", " ")] = gdf
        except Exception: pass
    return capas

capas = cargar_vectores()

# Nombres solo para mostrar en el visualizador (no afectan la lógica interna de matching)
NOMBRES_VISIBLES = {
    "Estaciones 2024final": "Estaciones calidad de agua DGA",
    "Hidro subcuen": "Red hidrografica",
    "Toponimos maule": "Toponimos",
    "masas lacustres maule": "Lagos",
}
def nombre_visible(nombre):
    return NOMBRES_VISIBLES.get(nombre, nombre)

# ─────────────────────────────────────────────────────────────
# Mapa
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🛰️ Rasters")
mostrar_dem = st.sidebar.checkbox("Sombra de colina (DEM)", value=False)

st.sidebar.markdown("### 🗺️ Capas vectoriales")
capas_visibles = {
    nombre: st.sidebar.checkbox(nombre_visible(nombre), value=True, key=f"chk_{nombre}")
    for nombre in capas.keys()
}

def construir_mapa(_capas, incluir_dem):
    # Inicializamos el mapa directamente con OpenStreetMap para no tapar el relieve
    m = folium.Map(location=[-35.7, -71.5], zoom_start=9, tiles="OpenStreetMap",
                   control_scale=True, zoomSnap=0.25, zoomDelta=0.5, wheelPxPerZoomLevel=40)

    # Capa satelital de Google (base layer alternativa, seleccionable en LayerControl)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satélite",
        overlay=False,
        control=True,
    ).add_to(m)

    # 1. Agregar Hillshade — SOLO si el usuario lo activa (evita bloquear la carga inicial)
    if incluir_dem:
        dem_file = DATA / "dem_hillshade.tif"
        if dem_file.exists():
            try:
                img_b64, bounds, _, _ = raster_a_overlay(dem_file, es_dem=False)
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{img_b64}",
                    bounds=bounds,
                    opacity=0.7,
                    name="Sombra de colina"
                ).add_to(m)
            except Exception as e:
                st.error(f"Error cargando DEM: {e}")

    # 2. Agregar Vectores y Etiquetas
    # Las estaciones se procesan al final (quedan arriba, con prioridad de clic)
    orden_capas = sorted(_capas.items(), key=lambda kv: "estacion" in kv[0].lower())
    for nombre, gdf in orden_capas:
        nombre_lower = nombre.lower()

        # Capa de Estaciones
        if "estacion" in nombre_lower:
            col_cod_est = next(
                (c for c in ["COD_BNA", "COD. ESTACIÓN", "COD. ESTACION", "COD_ESTACION",
                              "COD ESTACION", "CODIGO", "Codigo", "codigo"]
                 if c in gdf.columns),
                gdf.columns[0],
            )
            col_nombre_est = next(
                (c for c in ["NOMBRE", "Nombre", "nombre"] if c in gdf.columns),
                None,
            )
            campos_tooltip = [c for c in [col_cod_est, col_nombre_est] if c]
            folium.GeoJson(
                gdf, name=nombre_visible(nombre),
                marker=folium.CircleMarker(radius=6, fill=True, color="red"),
                tooltip=folium.GeoJsonTooltip(fields=campos_tooltip),
            ).add_to(m)
                           
        # Capa de Topónimos
        elif "toponimo" in nombre_lower:
            fg_toponimos = folium.FeatureGroup(name=nombre_visible(nombre))
            col_nombre = next((c for c in ["NOMBRE", "Nombre", "nombre", "TEXTO", "TextString", "NAME"] if c in gdf.columns), gdf.columns[0])
            
            for _, row in gdf.iterrows():
                if row.geometry:
                    texto = str(row[col_nombre]).strip()
                    if texto and texto.lower() not in ["none", "nan", ""]:
                        pt = row.geometry.representative_point()
                        etiqueta = folium.DivIcon(
                            html=f'<div style="font-size: 11px; font-weight: bold; color: #222; text-shadow: 1px 1px 3px white, -1px -1px 3px white, 1px -1px 3px white, -1px 1px 3px white; white-space: nowrap;">{texto}</div>'
                        )
                        folium.Marker(
                            location=[pt.y, pt.x], icon=etiqueta,
                            tooltip=texto,
                        ).add_to(fg_toponimos)
            fg_toponimos.add_to(m)
            
        # Capa de Hidrología 
        elif "hidro" in nombre_lower or "subcuen" in nombre_lower:
            
            if "Dren_Tipo" in gdf.columns:
                gdf = gdf[gdf["Dren_Tipo"] == "Río"]
                
            if not gdf.empty:
                fg_hidro = folium.FeatureGroup(name=nombre_visible(nombre))
                
                cols_disp = [c for c in ["Nombre", "Dren_Tipo", "Region", "Provincia"] if c in gdf.columns]
                gdf_liviano = gdf[["geometry"] + cols_disp]  # solo lo necesario, no todo el gpkg
                
                folium.GeoJson(
                    gdf_liviano, 
                    style_function=lambda x: {'color': '#1E88E5', 'weight': 1.5, 'opacity': 0.8},
                    tooltip=folium.GeoJsonTooltip(fields=cols_disp) if cols_disp else None
                ).add_to(fg_hidro)
                
                if "Nombre" in gdf.columns:
                    nombres_vistos = set()
                    MAX_ETIQUETAS_RIO = 300  # límite de seguridad para no saturar el navegador
                    
                    for _, row in gdf.iterrows():
                        if len(nombres_vistos) >= MAX_ETIQUETAS_RIO:
                            break
                        if row.geometry and not pd.isna(row["Nombre"]):
                            texto = str(row["Nombre"]).strip()
                            if texto and texto.lower() not in ["none", "nan", "sin nombre", ""]:
                                if texto not in nombres_vistos:
                                    nombres_vistos.add(texto)
                                    pt = row.geometry.representative_point()
                                    etiqueta_rio = folium.DivIcon(
                                        html=f'<div style="font-size: 10px; font-style: italic; font-weight: bold; color: #0D47A1; text-shadow: 1px 1px 2px white, -1px -1px 2px white, 1px -1px 2px white, -1px 1px 2px white; white-space: nowrap;">{texto}</div>'
                                    )
                                    folium.Marker(
                                        location=[pt.y, pt.x], icon=etiqueta_rio,
                                        tooltip=texto,
                                    ).add_to(fg_hidro)
                fg_hidro.add_to(m)
            
        # Cuerpos de agua (lagos/lagunas) — sí llevan relleno azul, son agua real
        elif "lacustre" in nombre_lower or "masa" in nombre_lower:
            folium.GeoJson(
                gdf,
                name=nombre_visible(nombre),
                style_function=lambda x: {
                    "color": "#1565C0",
                    "weight": 1,
                    "fillColor": "#1E88E5",
                    "fillOpacity": 0.5,
                },
            ).add_to(m)

        # Capas Generales (ej. límite de cuenca) — sin relleno para no tapar el hillshade
        # y SIN interactividad: si no, su relleno invisible captura los clics
        # destinados a las estaciones que están debajo.
        else:
            folium.GeoJson(
                gdf,
                name=nombre_visible(nombre),
                style_function=lambda x: {
                    "color": "#333333",
                    "weight": 1.5,
                    "fillOpacity": 0,
                },
                interactive=False,
            ).add_to(m)

    folium.LayerControl().add_to(m)

    # MiniMap de contexto (esquina inferior) y control de medición de distancia/área
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    MeasureControl(primary_length_unit="kilometers", primary_area_unit="hectares").add_to(m)

    # Leyenda fija con los símbolos de cada capa
    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 10px; z-index: 9999;
        background: rgba(255,255,255,0.92); padding: 10px 14px; border-radius: 8px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.3); font-size: 12.5px; font-family: sans-serif;">
        <b>Leyenda</b>
        <div style="margin-top:6px; display:flex; align-items:center; gap:6px;">
            <span style="width:11px;height:11px;border-radius:50%;background:red;display:inline-block;"></span>
            Estaciones calidad de agua DGA
        </div>
        <div style="margin-top:4px; display:flex; align-items:center; gap:6px;">
            <span style="width:16px;height:3px;background:#1E88E5;display:inline-block;"></span>
            Red hidrografica
        </div>
        <div style="margin-top:4px; display:flex; align-items:center; gap:6px;">
            <span style="font-weight:bold;font-style:italic;color:#222;">Aa</span>
            Toponimos
        </div>
        <div style="margin-top:4px; display:flex; align-items:center; gap:6px;">
            <span style="width:13px;height:13px;background:#1E88E5;border:1.5px solid #1565C0;display:inline-block;"></span>
            Lagos
        </div>
        <div style="margin-top:4px; display:flex; align-items:center; gap:6px;">
            <span style="width:13px;height:13px;border:1.5px solid #333333;display:inline-block;"></span>
            Cuenca Rio Maule
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

capas_a_mostrar = {nombre: gdf for nombre, gdf in capas.items() if capas_visibles.get(nombre, True)}

col_mapa, col_info = st.columns([2, 1])

with col_mapa:
    m = construir_mapa(capas_a_mostrar, mostrar_dem)
    salida_mapa = st_folium(m, width=None, height=500, key="mapa_final")

with col_info:
    st.markdown("""
    <style>
    .info-card{background:#fff;border-radius:10px;padding:14px 18px;
        box-shadow:0 1px 6px rgba(0,0,0,0.08);margin-bottom:12px;border-left:4px solid #2563eb;}
    .info-card h4{margin:0 0 8px 0;font-size:15px;}
    .info-card ul{margin:0;padding-left:18px;font-size:13.5px;color:#333;}
    .info-card p{font-size:13.5px;color:#333;margin:0;}
    .info-card li{margin-bottom:3px;}
    </style>
    <div class="info-card">
        <h4>ℹ️ Descripción</h4>
        <p>Visualizador de la cuenca del río Maule con estaciones de calidad de agua de la DGA
        y registros históricos. Presiona una estación en el mapa para ver sus parámetros
        fisicoquímicos y exportar los gráficos con las tendencias temporales.</p>
    </div>
    <div class="info-card">
        <h4>🎮 Qué puedes hacer</h4>
        <ul>
            <li>Explorar estaciones de calidad de agua en el mapa</li>
            <li>Visualizar series de tiempo de parámetros fisicoquímicos</li>
            <li>Exportar gráficos y datos</li>
            <li>Activar/desactivar capas de información</li>
        </ul>
    </div>
    <div class="info-card">
        <h4>📄 Fuente de los datos</h4>
        <ul>
            <li><b>DGA</b> - Dirección General de Aguas</li>
            <li><b>BCN</b> - Biblioteca del Congreso Nacional</li>
            <li><b>MOP</b> - Ministerio de Obras Públicas</li>
        </ul>
        <p style="margin-top:8px;"><b>Parámetros fisicoquímicos evaluados:</b> pH, Conductividad Eléctrica,
        Oxígeno Disuelto, Nutrientes, metales pesados y más...</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Análisis e Integración de Gráfico
# ─────────────────────────────────────────────────────────────
st.subheader("📊 Análisis de Calidad")
archivo_datos = DATA / "datos_limpios_sin_outliers.xlsx"

if archivo_datos.exists():
    df = pd.read_excel(archivo_datos)
    
    if salida_mapa.get("last_active_drawing"):
        props = salida_mapa["last_active_drawing"].get("properties", {})
        col_cod_est = next(
            (c for c in ["COD_BNA", "COD. ESTACIÓN", "COD. ESTACION", "COD_ESTACION",
                          "COD ESTACION", "CODIGO", "Codigo", "codigo"]
             if c in props),
            None,
        )
        valor_mapa = props.get(col_cod_est) if col_cod_est else (list(props.values())[0] if props else None)
        
        if valor_mapa:
            st.write(f"### Estación detectada: {valor_mapa}")
            col_codigo = next(
                (c for c in ["COD. ESTACIÓN", "COD. ESTACION", "COD_ESTACION", "COD ESTACION"]
                 if c in df.columns),
                'COD. ESTACIÓN',
            )
            
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
                    
                    formato_export = st.radio(
                        "Formato de descarga del gráfico:", ["png", "jpeg"], horizontal=True
                    )
                    config_export = {
                        "toImageButtonOptions": {
                            "format": formato_export,
                            "filename": f"{parametro}_{valor_mapa}",
                            "scale": 2,
                        },
                        "displaylogo": False,
                    }
                    st.plotly_chart(fig, use_container_width=True, config=config_export)
                    st.caption("📷 Usa el ícono de la cámara en la barra del gráfico para descargarlo.")
                    st.dataframe(df_plot[['FECHA MEDICION', 'PARAMETRO', 'VALOR']], use_container_width=True)
                else:
                    st.warning(f"No hay registros en el Excel para el código: {valor_mapa}")
            else:
                st.error(f"No se encontró la columna '{col_codigo}'.")
        else:
            st.info("No se pudo obtener el código del mapa.")
    else:
        st.info("👆 Haz clic en un marcador rojo en el mapa para ver el análisis.")

st.markdown("""
<div class="footer-bar">
    <span>Desarrollado para la gestión y análisis de la calidad del agua en la cuenca del río Maule.</span>
</div>
""", unsafe_allow_html=True)
