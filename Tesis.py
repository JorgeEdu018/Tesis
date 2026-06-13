import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time, os

st.set_page_config(page_title="Tesis – Clusterización Logística Zacatecas", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .titulo-principal { font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700; color: #1a1a2e; letter-spacing: -0.5px; }
    .subtitulo { font-size: 1rem; color: #5a6472; margin-top: -0.4rem; margin-bottom: 1.5rem; }
    .metric-card { background: #f7f9fc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem 1.3rem; text-align: center; }
    .metric-card h3 { font-size: 1.8rem; font-weight: 700; color: #1e3a5f; margin: 0; }
    .metric-card p { font-size: 0.78rem; color: #718096; margin: 0.2rem 0 0 0; text-transform: uppercase; letter-spacing: 0.06em; }
    .seccion-titulo { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: #1a1a2e; border-left: 4px solid #3b82f6; padding-left: 0.6rem; margin-bottom: 0.8rem; }
    .badge { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.18rem 0.55rem; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.05em; }
    .badge-kmeans { background: #dbeafe; color: #1d4ed8; }
    .badge-dbscan { background: #d1fae5; color: #065f46; }
    .badge-raw    { background: #f3f4f6; color: #374151; }
    .info-box  { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.88rem; color: #1e40af; }
    .warn-box  { background: #fef9c3; border: 1px solid #fde047; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.88rem; color: #713f12; }
    .winner-box { background: #d1fae5; border: 1px solid #6ee7b7; border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.92rem; color: #064e3b; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cargar dataset
# ─────────────────────────────────────────────
@st.cache_data
@st.cache_data
def cargar_datos():
    df = pd.read_csv("datos_entrenamiento1.csv")

    # Limpiar coordenadas con puntos de miles (ej: 227.487.769 -> 22.7487769)
    for col in ["Latitud", "Longitud"]:
        df[col] = (
            df[col].astype(str)
                   .str.replace(",", ".", regex=False)   # comas decimales -> punto
                   .apply(lambda x: x[:2] + "." + x[2:].replace(".", "") 
                          if x.count(".") > 1 else x)    # doble punto -> un solo decimal
                   .astype(float)
        )

    excluir = ["sin nombre", "calle no registrada", "no especificada", "no especificado"]
    df = df[~df["Calle"].str.strip().str.lower().isin(excluir)].copy()
    return df.dropna(subset=["Latitud", "Longitud"])

def calcular_codo(coords, k_max=12):
    ks = list(range(2, k_max + 1))
    inercias, siluetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        lbl = km.fit_predict(coords)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(coords, lbl))
    return ks, inercias, siluetas

def sugerir_epsilon(coords, min_samples):
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(coords)
    dist, _ = nbrs.kneighbors(coords)
    k_dist = np.sort(dist[:, -1])
    return float(np.percentile(k_dist, 90)), k_dist

PALETA = [
    [59,130,246],[239,68,68],[34,197,94],[251,191,36],[168,85,247],
    [249,115,22],[20,184,166],[236,72,153],[132,204,22],[99,102,241],
    [245,158,11],[16,185,129],
]
COLOR_NOISE  = [150,150,150,180]
COLOR_BORDER = [255,220,80,200]


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="titulo-principal">🗺️ Clusterización Logística — Última Milla</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">K-means vs DBSCAN · Municipios de Zacatecas y Guadalupe, Zac.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔵 K-means")
    modo_k = st.radio("Asignación de K", ["Manual (flotilla definida)", "Analítico (Método del Codo)"])
    k_valor = st.slider("K (vehículos/zonas)", 2, 12, 5) if modo_k == "Manual (flotilla definida)" else None
    st.markdown("---")
    st.markdown("### 🟢 DBSCAN")
    min_samples = st.slider("MinPts (densidad mínima)", 2, 20, 5)
    auto_eps    = st.checkbox("Sugerir ε automáticamente (k-dist)", value=True)
    eps_ph      = st.empty()
    epsilon     = None if auto_eps else st.slider("ε Radio de vecindad (grados)", 0.001, 0.05, 0.012, step=0.001, format="%.3f")
    st.markdown("---")
    st.markdown("### 🎨 Visualización")
    radio_punto        = st.slider("Radio de puntos (m)", 30, 200, 70)
    mostrar_centroides = st.checkbox("Mostrar centroides K-means", value=True)


# ─────────────────────────────────────────────
# DATOS + EPSILON
# ─────────────────────────────────────────────
df     = cargar_datos()
coords = df[["Latitud","Longitud"]].values

if auto_eps:
    epsilon_auto, k_dist_vals = sugerir_epsilon(coords, min_samples)
    epsilon = epsilon_auto
    eps_ph.markdown(f'<div class="info-box" style="margin-top:0.4rem;">ε sugerido: <strong>{epsilon:.4f}°</strong> (~{epsilon*111:.2f} km)</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MÉTRICAS SUPERIORES
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><h3>{len(df):,}</h3><p>Puntos de entrega</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><h3>{df["Municipio"].nunique()}</h3><p>Municipios</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><h3>{df["Colonia"].nunique()}</h3><p>Colonias únicas</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><h3>📋</h3><p>Datos reales · Zacatecas</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAPA INICIAL — todos los puntos sin cluster
# ─────────────────────────────────────────────
st.markdown('<div class="seccion-titulo">📍 Dataset — Ubicaciones de entrega</div>', unsafe_allow_html=True)
st.markdown('<span class="badge badge-raw">Sin clusterizar · datos crudos</span>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

vista = pdk.ViewState(latitude=df["Latitud"].mean(), longitude=df["Longitud"].mean(), zoom=11.5)
capa_raw = pdk.Layer("ScatterplotLayer", df, get_position=["Longitud","Latitud"],
                     get_color=[30,144,255,160], get_radius=radio_punto, pickable=True)
st.pydeck_chart(pdk.Deck(layers=[capa_raw], initial_view_state=vista,
                         map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                         tooltip={"text": "Municipio: {Municipio}\nColonia: {Colonia}\nCalle: {Calle}"}))

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# MÉTODO DEL CODO
# ─────────────────────────────────────────────
if modo_k == "Analítico (Método del Codo)":
    st.markdown('<div class="seccion-titulo">📐 Método del Codo — Determinación del K óptimo</div>', unsafe_allow_html=True)
    with st.spinner("Calculando inercia y coeficiente de silueta..."):
        ks, inercias, siluetas = calcular_codo(coords)
    mejor_k = ks[int(np.argmax(siluetas))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor("#f7f9fc")
    ax1.plot(ks, inercias, "o-", color="#3b82f6", lw=2, ms=6)
    ax1.set_title("Inercia (WCSS)", fontsize=11, fontweight="bold", color="#1a1a2e")
    ax1.set_xlabel("K"); ax1.set_ylabel("Inercia"); ax1.set_facecolor("#f7f9fc"); ax1.grid(alpha=0.3)
    ax2.plot(ks, siluetas, "o-", color="#10b981", lw=2, ms=6)
    ax2.axvline(mejor_k, color="#ef4444", ls="--", alpha=0.7, label=f"K óptimo = {mejor_k}")
    ax2.set_title("Coeficiente de Silueta", fontsize=11, fontweight="bold", color="#1a1a2e")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silueta"); ax2.set_facecolor("#f7f9fc"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    k_valor = mejor_k
    st.markdown(f'<div class="info-box">✅ K óptimo: <strong>{mejor_k}</strong> · Silueta máxima: {max(siluetas):.3f}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EJECUTAR ALGORITMOS
# ─────────────────────────────────────────────
t0 = time.time()
km = KMeans(n_clusters=k_valor, init="k-means++", n_init=10, random_state=42)
df["km_cluster"] = km.fit_predict(coords)
t_kmeans   = time.time() - t0
centroides = km.cluster_centers_
sil_km     = silhouette_score(coords, df["km_cluster"])
inercia_km = km.inertia_

t0 = time.time()
db = DBSCAN(eps=epsilon, min_samples=min_samples)
df["db_cluster"] = db.fit_predict(coords)
t_dbscan = time.time() - t0

core_idx = set(db.core_sample_indices_)
df["db_tipo"] = df.apply(
    lambda r: "Ruido" if r["db_cluster"] == -1
    else ("Núcleo" if r.name in core_idx else "Borde"), axis=1)

n_clusters_db = int(df[df["db_cluster"] >= 0]["db_cluster"].nunique())
n_ruido_db    = int((df["db_cluster"] == -1).sum())
df_db_valid   = df[df["db_cluster"] >= 0]
sil_db = silhouette_score(df_db_valid[["Latitud","Longitud"]].values, df_db_valid["db_cluster"].values) \
         if n_clusters_db >= 2 else None
sil_txt = f"{sil_db:.4f}" if sil_db is not None else "N/A (< 2 clusters)"

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_km, tab_db, tab_cmp, tab_datos = st.tabs([
    "🔵 K-means", "🟢 DBSCAN", "⚖️ Comparativa de Métricas", "📋 Datos"
])

# ══════════════════════════════════════════════ K-MEANS
with tab_km:
    st.markdown(f'<span class="badge badge-kmeans">K-means · K={k_valor}</span>', unsafe_allow_html=True)
    st.markdown(f"**Silueta:** {sil_km:.4f} &nbsp;|&nbsp; **Inercia:** {inercia_km:,.1f} &nbsp;|&nbsp; **Tiempo:** {t_kmeans*1000:.1f} ms", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    df["km_color"] = df["km_cluster"].apply(lambda c: PALETA[c % len(PALETA)] + [190])
    capas_km = [pdk.Layer("ScatterplotLayer", df, get_position=["Longitud","Latitud"],
                          get_color="km_color", get_radius=radio_punto, pickable=True)]
    if mostrar_centroides:
        df_c = pd.DataFrame(centroides, columns=["Latitud","Longitud"])
        df_c["color"] = [[255,255,255,230]] * len(df_c)
        capas_km.append(pdk.Layer("ScatterplotLayer", df_c, get_position=["Longitud","Latitud"],
                                  get_color="color", get_radius=radio_punto*2.5,
                                  stroked=True, line_width_min_pixels=3))
    st.pydeck_chart(pdk.Deck(layers=capas_km, initial_view_state=vista,
                             map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                             tooltip={"text": "Zona: {km_cluster}\nMunicipio: {Municipio}\nColonia: {Colonia}\nCalle: {Calle}"}))

    ley_cols = st.columns(min(k_valor, 6))
    for i in range(k_valor):
        r, g, b = PALETA[i % len(PALETA)][:3]
        hx = "#{:02x}{:02x}{:02x}".format(r, g, b)
        with ley_cols[i % 6]:
            st.markdown(f'<div style="display:flex;align-items:center;gap:6px;">'
                        f'<div style="width:13px;height:13px;border-radius:50%;background:{hx};"></div>'
                        f'<span style="font-size:0.82rem;">Zona {i+1} ({int((df["km_cluster"]==i).sum())})</span></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="seccion-titulo">Resumen por zona</div>', unsafe_allow_html=True)
    res_km = (df.groupby("km_cluster")
                .agg(Puntos=("Latitud","count"),
                     Municipios=("Municipio", lambda x: ", ".join(sorted(x.unique()))),
                     Colonias=("Colonia","nunique"))
                .reset_index())
    res_km["km_cluster"] = res_km["km_cluster"].apply(lambda x: f"Zona {x+1}")
    res_km.columns = ["Zona","Puntos","Municipios","Colonias únicas"]
    st.dataframe(res_km, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════ DBSCAN
with tab_db:
    st.markdown(f'<span class="badge badge-dbscan">DBSCAN · ε={epsilon:.4f}° · MinPts={min_samples}</span>', unsafe_allow_html=True)
    st.markdown(f"**Clusters:** {n_clusters_db} &nbsp;|&nbsp; **Ruido:** {n_ruido_db} &nbsp;|&nbsp; **Silueta:** {sil_txt} &nbsp;|&nbsp; **Tiempo:** {t_dbscan*1000:.1f} ms", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if n_clusters_db == 0:
        st.markdown('<div class="warn-box">⚠️ DBSCAN no encontró ningún cluster. Reduce ε o MinPts desde el sidebar.</div>', unsafe_allow_html=True)
    else:
        def color_db(row):
            if row["db_cluster"] == -1: return COLOR_NOISE
            if row["db_tipo"] == "Borde": return COLOR_BORDER
            return PALETA[row["db_cluster"] % len(PALETA)] + [200]
        df["db_color"] = df.apply(color_db, axis=1)

        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", df, get_position=["Longitud","Latitud"],
                              get_color="db_color", get_radius=radio_punto, pickable=True)],
            initial_view_state=vista, map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"text": "Tipo: {db_tipo}\nCluster: {db_cluster}\nMunicipio: {Municipio}\nColonia: {Colonia}"}))

        st.markdown("""
        <div style="display:flex;gap:1.5rem;margin-top:0.5rem;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:6px;"><div style="width:13px;height:13px;border-radius:50%;background:#3b82f6;"></div><span style="font-size:0.82rem;">Punto Núcleo (Core)</span></div>
            <div style="display:flex;align-items:center;gap:6px;"><div style="width:13px;height:13px;border-radius:50%;background:#fbbf24;"></div><span style="font-size:0.82rem;">Punto Borde (Border)</span></div>
            <div style="display:flex;align-items:center;gap:6px;"><div style="width:13px;height:13px;border-radius:50%;background:#9ca3af;"></div><span style="font-size:0.82rem;">Ruido / Outlier</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="seccion-titulo">Clasificación de nodos</div>', unsafe_allow_html=True)
        tipos = df["db_tipo"].value_counts().reset_index()
        tipos.columns = ["Tipo de nodo","Cantidad"]
        tipos["%"] = (tipos["Cantidad"]/len(df)*100).round(1).astype(str)+"%"
        st.dataframe(tipos, use_container_width=True, hide_index=True)

        n_nucleo = int((df["db_tipo"]=="Núcleo").sum())
        n_borde  = int((df["db_tipo"]=="Borde").sum())
        st.markdown(f'<div class="info-box" style="margin-top:0.6rem;">'
                    f'<strong>{n_ruido_db} entregas atípicas</strong> — candidatas a paquetería externa.<br>'
                    f'<strong>{n_nucleo}</strong> puntos núcleo · <strong>{n_borde}</strong> puntos borde.</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="seccion-titulo">Resumen por cluster DBSCAN</div>', unsafe_allow_html=True)
        res_db = (df[df["db_cluster"]>=0].groupby("db_cluster")
                    .agg(Puntos=("Latitud","count"),
                         Núcleos=("db_tipo", lambda x: (x=="Núcleo").sum()),
                         Bordes=("db_tipo", lambda x: (x=="Borde").sum()),
                         Municipios=("Municipio", lambda x: ", ".join(sorted(x.unique()))),
                         Colonias=("Colonia","nunique"))
                    .reset_index())
        res_db["db_cluster"] = res_db["db_cluster"].apply(lambda x: f"Cluster {x+1}")
        res_db.columns = ["Cluster","Puntos","Núcleos","Bordes","Municipios","Colonias únicas"]
        st.dataframe(res_db, use_container_width=True, hide_index=True)

        if auto_eps:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="seccion-titulo">Gráfica k-dist — selección de ε</div>', unsafe_allow_html=True)
            fig_kd, ax_kd = plt.subplots(figsize=(8,3))
            fig_kd.patch.set_facecolor("#f7f9fc")
            ax_kd.plot(range(len(k_dist_vals)), k_dist_vals, color="#10b981", lw=1.5)
            ax_kd.axhline(epsilon, color="#ef4444", ls="--", lw=1.5, label=f"ε = {epsilon:.4f}°")
            ax_kd.set_facecolor("#f7f9fc"); ax_kd.grid(alpha=0.3)
            ax_kd.set_title(f"k-dist (MinPts={min_samples})", fontweight="bold", color="#1a1a2e", fontsize=10)
            ax_kd.set_xlabel("Puntos ordenados"); ax_kd.set_ylabel(f"Distancia al vecino {min_samples}")
            ax_kd.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig_kd); plt.close()


# ══════════════════════════════════════════════ COMPARATIVA
with tab_cmp:
    st.markdown('<div class="seccion-titulo">⚖️ Métricas de Evaluación Comparativa</div>', unsafe_allow_html=True)
    st.markdown("Validación interna sobre variables espaciales (Latitud, Longitud) — sin etiquetas reales.")
    st.markdown("<br>", unsafe_allow_html=True)

    st.dataframe(pd.DataFrame({
        "Métrica": ["Clusters encontrados","Puntos de ruido / outliers","Coeficiente de Silueta ↑",
                    "Inercia (WCSS) ↓","Tiempo de cómputo (ms)","Requiere definir K","Detecta outliers"],
        "K-means": [str(k_valor),"0 (todos asignados)",f"{sil_km:.4f}",f"{inercia_km:,.1f}",
                    f"{t_kmeans*1000:.1f}","✅ Sí","❌ No"],
        "DBSCAN":  [str(n_clusters_db),str(n_ruido_db),sil_txt,"N/A",
                    f"{t_dbscan*1000:.1f}","❌ No","✅ Sí"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="seccion-titulo">Coeficiente de Silueta — comparación visual</div>', unsafe_allow_html=True)
    fig_sil, ax_sil = plt.subplots(figsize=(6,3))
    fig_sil.patch.set_facecolor("#f7f9fc")
    algos, valores, colores = ["K-means"], [sil_km], ["#3b82f6"]
    if sil_db is not None:
        algos.append("DBSCAN"); valores.append(sil_db); colores.append("#10b981")
    bars = ax_sil.barh(algos, valores, color=colores, height=0.4, edgecolor="white")
    ax_sil.axvline(0, color="#1a1a2e", lw=0.8)
    ax_sil.axvline(0.5, color="#94a3b8", lw=1, ls="--", label="Umbral buena calidad (0.5)")
    ax_sil.set_xlim(-0.1, 1.0); ax_sil.set_xlabel("Coeficiente de Silueta")
    ax_sil.set_title("Mayor valor = zonas más compactas y separadas", fontsize=10, color="#1a1a2e")
    ax_sil.set_facecolor("#f7f9fc"); ax_sil.grid(axis="x", alpha=0.3); ax_sil.legend(fontsize=8)
    for bar, val in zip(bars, valores):
        ax_sil.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig_sil); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="seccion-titulo">Análisis operativo</div>', unsafe_allow_html=True)
    col_km2, col_db2 = st.columns(2)
    with col_km2:
        st.markdown("""<div style="background:#1e3a5f;border:1px solid #3b82f6;border-radius:8px;padding:1rem;color:#e2e8f0;">
        <strong style="color:#60a5fa;">🔵 K-means</strong><br><br>
        ✅ Asigna <em>todos</em> los puntos a una zona<br>✅ K = tamaño exacto de la flotilla<br>
        ✅ Zonas balanceadas en volumen<br>✅ Rápido y determinístico (k-means++)<br>
        ❌ No detecta entregas atípicas<br>❌ Requiere definir K a priori<br>❌ Sensible a la forma de los clusters
        </div>""", unsafe_allow_html=True)
    with col_db2:
        st.markdown("""<div style="background:#064e3b;border:1px solid #10b981;border-radius:8px;padding:1rem;color:#e2e8f0;">
        <strong style="color:#34d399;">🟢 DBSCAN</strong><br><br>
        ✅ Detecta outliers (rutas no rentables)<br>✅ No requiere definir K<br>
        ✅ Encuentra clusters de forma arbitraria<br>✅ Robusto ante ruido geoespacial<br>
        ❌ Sensible a ε y MinPts<br>❌ Puntos de ruido quedan sin asignar<br>❌ Clusters desbalanceados en volumen
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if sil_db is not None:
        ganador = "K-means" if sil_km >= sil_db else "DBSCAN"
        diff = abs(sil_km - sil_db)
        st.markdown(f'<div class="winner-box">🏆 Con los parámetros actuales, <strong>{ganador}</strong> obtiene mayor coeficiente de silueta '
                    f'(Δ = {diff:.4f}), indicando zonas de entrega más compactas y bien separadas. '
                    f'Consistente con los hallazgos de Yin (2025) sobre K-means vs DBSCAN en logística urbana.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">⚠️ DBSCAN no generó suficientes clusters para calcular la silueta. Ajusta ε o MinPts para una comparativa completa.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════ DATOS
with tab_datos:
    st.markdown('<div class="seccion-titulo">Dataset con etiquetas K-means y DBSCAN</div>', unsafe_allow_html=True)
    cols_drop = [c for c in ["km_color","db_color"] if c in df.columns]
    df_show = df.drop(columns=cols_drop).copy()
    df_show["km_cluster"] = df_show["km_cluster"].apply(lambda x: f"Zona {x+1}")
    df_show["db_cluster"] = df_show["db_cluster"].apply(lambda x: "Ruido" if x==-1 else f"Cluster {x+1}")
    st.dataframe(df_show, use_container_width=True, height=420)
    st.download_button("⬇️ Descargar resultados (.csv)",
                       data=df_show.to_csv(index=False).encode("utf-8"),
                       file_name="resultados_clustering_zacatecas.csv", mime="text/csv")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#94a3b8;font-size:0.78rem;'>Tesis · K-means vs DBSCAN para logística de última milla · Zacatecas y Guadalupe, Zac. · 2025</p>", unsafe_allow_html=True)
