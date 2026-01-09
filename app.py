# app.py
# Streamlit app — Geothermal gradient (fully adjustable)
# UI language: English by default, switchable to French / Italian in a menu.

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import folium
from streamlit_folium import st_folium
#from pyproj import Transformer


# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path(__file__).parent
EMBEDDED_CSV = BASE_DIR / "data" / "gradient_geo.csv"
CONTOUR_GEOJSON = BASE_DIR / "data" / "contours_gradient.geojson"

# UTM 33N -> WGS84 for map (your case)
#TRANSFORMER_UTM33N_TO_WGS84 = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)


# ---------------------------
# i18n (translations)
# ---------------------------
TRANSLATIONS = {
    "en": {
        "lang_name": "English",
        "page_title": "Geothermal gradient",
        "title": "Geothermal gradient — fully adjustable",
        "sidebar_data": "1) Data",
        "upload_csv": "Upload CSV (gradient_geo.csv)",
        "use_uploaded": "Use uploaded CSV (otherwise use embedded file)",
        "separator": "Separator",
        "demo_mode": "Demo mode (ignore file)",
        "demo_info": "Demo mode: synthetic data.",
        "need_file_warning": "Upload a CSV or enable demo mode.",
        "csv_error": "CSV read/format error:",
        "missing_embedded": "Embedded CSV missing. Upload a file or enable demo mode.",
        "using_embedded": "Using embedded data file:",
        "loaded_rows": "Loaded rows",
        "sidebar_refs": "2) Reference & gradients",
        "t_ref": "T_REF (°C at z=0)",
        "z_ref": "Z_REF (m)",
        "g_local": "Ladispoli gradient (°C/km)",
        "sidebar_depth": "3) Depth / resolution",
        "z_max": "Z_MAX (m)",
        "n_z": "Resolution (points)",
        "sidebar_axes": "4) Axes X/Y",
        "fix_xlim": "Fix x-limits (xlim)",
        "xlim_min": "xlim min",
        "xlim_max": "xlim max",
        "force_xticks": "Force X ticks",
        "xtick_start": "X tick start",
        "xtick_end": "X tick end",
        "xtick_step": "X tick step",
        "force_yticks": "Force Y ticks",
        "ytick_start": "Y tick start",
        "ytick_end": "Y tick end",
        "ytick_step": "Y tick step",
        "sidebar_style": "5) Appearance",
        "fig_w": "Figure width",
        "fig_h": "Figure height",
        "plot_title": "Plot title",
        "env_color": "Envelope color",
        "env_alpha": "Envelope alpha",
        "mean_color": "Mean line color",
        "mean_lw": "Mean line width",
        "mean_ls": "Mean line style",
        "show_points": "Show measured mean points",
        "pts_color": "Points color",
        "pts_marker": "Point marker",
        "pts_ms": "Point size",
        "show_err": "Show ±1σ error bars",
        "err_color": "Error bars color",
        "err_lw": "Error bars line width",
        "err_cap": "Cap size",
        "err_alpha": "Error bars alpha",
        "sidebar_grad_style": "6) Gradients (styles)",
        "show_grad_mean": "Show mean gradient",
        "grad_mean_color": "Mean gradient color",
        "grad_mean_ls": "Mean gradient line style",
        "grad_mean_lw": "Mean gradient line width",
        "show_grad_local": "Show Ladispoli gradient",
        "grad_local_color": "Ladispoli gradient color",
        "grad_local_ls": "Ladispoli gradient line style",
        "grad_local_lw": "Ladispoli gradient line width",
        "sidebar_grid": "7) Grid",
        "grid_on": "Show grid",
        "grid_style": "Grid style",
        "grid_lw": "Grid line width",
        "grid_alpha": "Grid alpha",
        "sidebar_legend": "8) Legend (bottom)",
        "legend_on": "Show legend",
        "legend_frame": "Legend box",
        "legend_ncol": "Legend columns",
        "legend_font": "Legend font size",
        "legend_yoffset": "Legend vertical position",
        "sidebar_texts": "9) Text labels",
        "env_label": "Envelope label",
        "mean_label": "Mean label",
        "pts_label": "Measured means label",
        "err_label": "±1σ label",
        "grad_mean_label": "Mean gradient label",
        "grad_local_label": "Ladispoli gradient label",
        "download_png": "Download figure (PNG)",
        "summary": "Stats summary",
        "csv_help": "CSV help",
        "csv_help_text": (
            "- Expected columns: **X, Y, T400, T1000, T2000**\n"
            "- Or: **X, Y, Temp_400m, Temp_1000m, Temp_2000m** (auto-renamed)\n"
            "- Separator can be `;`, `,`, or tab."
        ),
        "xlabel": "Temperature (°C)",
        "ylabel": "Depth (m)",
        "default_plot_title": "Temperature vs Depth — ±1σ envelope and gradients",
        "default_env_label": "±1σ envelope",
        "default_mean_label": "Mean",
        "default_pts_label": "Measured means",
        "default_err_label": "Measured ±1σ",
        "default_grad_mean_label": "Mean gradient",
        "default_grad_local_label": "Ladispoli gradient",
        "map_title": "Basemap & geothermal contours",
        "map_height": "Map height",
        "map_points": "Show points (UTM 33N -> lon/lat)",
        "map_contours": "Show contour lines (GeoJSON) in red",
        "map_labels": "Show labels every 20",
        "map_missing_contours": "Contour GeoJSON not found: data/contours_gradient.geojson",
        "map_level_field": "Contour field name (level)",
    },
    "fr": {
        "lang_name": "Français",
        "page_title": "Gradient géothermique",
        "title": "Gradient géothermique — entièrement paramétrable",
        "sidebar_data": "1) Données",
        "upload_csv": "Charger le CSV (gradient_geo.csv)",
        "use_uploaded": "Utiliser le CSV uploadé (sinon fichier embarqué)",
        "separator": "Séparateur",
        "demo_mode": "Mode démo (ignore le fichier)",
        "demo_info": "Mode démo : données synthétiques.",
        "need_file_warning": "Charge un CSV ou active le mode démo.",
        "csv_error": "Erreur lecture/format CSV :",
        "missing_embedded": "CSV embarqué manquant. Uploade un fichier ou active le mode démo.",
        "using_embedded": "Fichier embarqué utilisé :",
        "loaded_rows": "Lignes chargées",
        "sidebar_refs": "2) Références & gradients",
        "t_ref": "T_REF (°C à z=0)",
        "z_ref": "Z_REF (m)",
        "g_local": "Gradient Ladispoli (°C/km)",
        "sidebar_depth": "3) Profondeur / résolution",
        "z_max": "Z_MAX (m)",
        "n_z": "Résolution (points)",
        "sidebar_axes": "4) Axes X/Y",
        "fix_xlim": "Fixer xlim",
        "xlim_min": "xlim min",
        "xlim_max": "xlim max",
        "force_xticks": "Forcer ticks X",
        "xtick_start": "X tick start",
        "xtick_end": "X tick end",
        "xtick_step": "X tick step",
        "force_yticks": "Forcer ticks Y",
        "ytick_start": "Y tick start",
        "ytick_end": "Y tick end",
        "ytick_step": "Y tick step",
        "sidebar_style": "5) Apparence",
        "fig_w": "Largeur figure",
        "fig_h": "Hauteur figure",
        "plot_title": "Titre du graphe",
        "env_color": "Couleur enveloppe",
        "env_alpha": "Alpha enveloppe",
        "mean_color": "Couleur moyenne",
        "mean_lw": "Épaisseur moyenne",
        "mean_ls": "Style moyenne",
        "show_points": "Afficher points des moyennes mesurées",
        "pts_color": "Couleur points",
        "pts_marker": "Marker points",
        "pts_ms": "Taille marker",
        "show_err": "Afficher barres ±1σ",
        "err_color": "Couleur barres σ",
        "err_lw": "Épaisseur barres σ",
        "err_cap": "Taille caps",
        "err_alpha": "Alpha barres σ",
        "sidebar_grad_style": "6) Gradients (styles)",
        "show_grad_mean": "Afficher gradient moyen",
        "grad_mean_color": "Couleur grad moyen",
        "grad_mean_ls": "Style grad moyen",
        "grad_mean_lw": "Épaisseur grad moyen",
        "show_grad_local": "Afficher gradient Ladispoli",
        "grad_local_color": "Couleur gradient Ladispoli",
        "grad_local_ls": "Style gradient Ladispoli",
        "grad_local_lw": "Épaisseur gradient Ladispoli",
        "sidebar_grid": "7) Grille",
        "grid_on": "Afficher grille",
        "grid_style": "Style grille",
        "grid_lw": "Épaisseur grille",
        "grid_alpha": "Alpha grille",
        "sidebar_legend": "8) Légende (en bas)",
        "legend_on": "Afficher légende",
        "legend_frame": "Boîte légende",
        "legend_ncol": "Colonnes légende",
        "legend_font": "Taille police légende",
        "legend_yoffset": "Position verticale légende",
        "sidebar_texts": "9) Textes (labels)",
        "env_label": "Label enveloppe",
        "mean_label": "Label moyenne",
        "pts_label": "Label moyennes mesurées",
        "err_label": "Label ±1σ",
        "grad_mean_label": "Label gradient moyen",
        "grad_local_label": "Label gradient Ladispoli",
        "download_png": "Télécharger la figure (PNG)",
        "summary": "Résumé des stats",
        "csv_help": "Aide CSV",
        "csv_help_text": (
            "- Colonnes attendues : **X, Y, T400, T1000, T2000**\n"
            "- Ou : **X, Y, Temp_400m, Temp_1000m, Temp_2000m** (renommage auto)\n"
            "- Séparateur : `;`, `,` ou tabulation."
        ),
        "xlabel": "Température (°C)",
        "ylabel": "Profondeur (m)",
        "default_plot_title": "Profil Température vs Profondeur — enveloppe ±1σ et gradients",
        "default_env_label": "Enveloppe ±1σ",
        "default_mean_label": "Moyenne",
        "default_pts_label": "Moyennes mesurées",
        "default_err_label": "±1σ mesuré",
        "default_grad_mean_label": "Gradient moyen",
        "default_grad_local_label": "Gradient Ladispoli",
        "map_title": "Fond de carte & courbes du gradient",
        "map_height": "Hauteur de la carte",
        "map_points": "Afficher les points (UTM 33N -> lon/lat)",
        "map_contours": "Afficher les courbes (GeoJSON) en rouge",
        "map_labels": "Afficher labels tous les 20",
        "map_missing_contours": "GeoJSON des courbes introuvable : data/contours_gradient.geojson",
        "map_level_field": "Nom du champ (niveau)",
    },
    "it": {
        "lang_name": "Italiano",
        "page_title": "Gradiente geotermico",
        "title": "Gradiente geotermico — completamente regolabile",
        "sidebar_data": "1) Dati",
        "upload_csv": "Carica CSV (gradient_geo.csv)",
        "use_uploaded": "Usa CSV caricato (altrimenti file incorporato)",
        "separator": "Separatore",
        "demo_mode": "Modalità demo (ignora il file)",
        "demo_info": "Modalità demo: dati sintetici.",
        "need_file_warning": "Carica un CSV oppure abilita la modalità demo.",
        "csv_error": "Errore lettura/formato CSV:",
        "missing_embedded": "CSV incorporato mancante. Carica un file o abilita demo.",
        "using_embedded": "Uso file incorporato:",
        "loaded_rows": "Righe caricate",
        "sidebar_refs": "2) Riferimento & gradienti",
        "t_ref": "T_REF (°C a z=0)",
        "z_ref": "Z_REF (m)",
        "g_local": "Gradiente Ladispoli (°C/km)",
        "sidebar_depth": "3) Profondità / risoluzione",
        "z_max": "Z_MAX (m)",
        "n_z": "Risoluzione (punti)",
        "sidebar_axes": "4) Assi X/Y",
        "fix_xlim": "Fissa xlim",
        "xlim_min": "xlim min",
        "xlim_max": "xlim max",
        "force_xticks": "Forza ticks X",
        "xtick_start": "X tick start",
        "xtick_end": "X tick end",
        "xtick_step": "X tick step",
        "force_yticks": "Forza ticks Y",
        "ytick_start": "Y tick start",
        "ytick_end": "Y tick end",
        "ytick_step": "Y tick step",
        "sidebar_style": "5) Aspetto",
        "fig_w": "Larghezza figura",
        "fig_h": "Altezza figura",
        "plot_title": "Titolo del grafico",
        "env_color": "Colore inviluppo",
        "env_alpha": "Alpha inviluppo",
        "mean_color": "Colore media",
        "mean_lw": "Spessore media",
        "mean_ls": "Stile media",
        "show_points": "Mostra punti delle medie misurate",
        "pts_color": "Colore punti",
        "pts_marker": "Marker punti",
        "pts_ms": "Dimensione marker",
        "show_err": "Mostra barre ±1σ",
        "err_color": "Colore barre σ",
        "err_lw": "Spessore barre σ",
        "err_cap": "Dimensione cap",
        "err_alpha": "Alpha barre σ",
        "sidebar_grad_style": "6) Gradienti (stili)",
        "show_grad_mean": "Mostra gradiente medio",
        "grad_mean_color": "Colore gradiente medio",
        "grad_mean_ls": "Stile gradiente medio",
        "grad_mean_lw": "Spessore gradiente medio",
        "show_grad_local": "Mostra gradiente Ladispoli",
        "grad_local_color": "Colore gradiente Ladispoli",
        "grad_local_ls": "Stile gradiente Ladispoli",
        "grad_local_lw": "Spessore gradiente Ladispoli",
        "sidebar_grid": "7) Griglia",
        "grid_on": "Mostra griglia",
        "grid_style": "Stile griglia",
        "grid_lw": "Spessore griglia",
        "grid_alpha": "Alpha griglia",
        "sidebar_legend": "8) Legenda (in basso)",
        "legend_on": "Mostra legenda",
        "legend_frame": "Box legenda",
        "legend_ncol": "Colonne legenda",
        "legend_font": "Dimensione font legenda",
        "legend_yoffset": "Posizione verticale legenda",
        "sidebar_texts": "9) Testi (etichette)",
        "env_label": "Etichetta inviluppo",
        "mean_label": "Etichetta media",
        "pts_label": "Etichetta medie misurate",
        "err_label": "Etichetta ±1σ",
        "grad_mean_label": "Etichetta gradiente medio",
        "grad_local_label": "Etichetta gradiente Ladispoli",
        "download_png": "Scarica la figura (PNG)",
        "summary": "Riepilogo statistiche",
        "csv_help": "Aiuto CSV",
        "csv_help_text": (
            "- Colonne attese: **X, Y, T400, T1000, T2000**\n"
            "- Oppure: **X, Y, Temp_400m, Temp_1000m, Temp_2000m** (rinomina automatica)\n"
            "- Separatore: `;`, `,` o tab."
        ),
        "xlabel": "Temperatura (°C)",
        "ylabel": "Profondità (m)",
        "default_plot_title": "Temperatura vs Profondità — inviluppo ±1σ e gradienti",
        "default_env_label": "Inviluppo ±1σ",
        "default_mean_label": "Media",
        "default_pts_label": "Medie misurate",
        "default_err_label": "±1σ misurato",
        "default_grad_mean_label": "Gradiente medio",
        "default_grad_local_label": "Gradiente Ladispoli",
        "map_title": "Basemap & curve del gradiente",
        "map_height": "Altezza mappa",
        "map_points": "Mostra punti (UTM 33N -> lon/lat)",
        "map_contours": "Mostra curve (GeoJSON) in rosso",
        "map_labels": "Mostra etichette ogni 20",
        "map_missing_contours": "GeoJSON curve non trovato: data/contours_gradient.geojson",
        "map_level_field": "Nome del campo (livello)",
    },
}

LANG_OPTIONS = [
    ("en", TRANSLATIONS["en"]["lang_name"]),
    ("fr", TRANSLATIONS["fr"]["lang_name"]),
    ("it", TRANSLATIONS["it"]["lang_name"]),
]


def t(lang, key):
    return TRANSLATIONS[lang][key]


# ---------------------------
# Utils
# ---------------------------
def fig_to_png_bytes(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def standardize_columns(df):
    rename_map = {"Temp_400m": "T400", "Temp_1000m": "T1000", "Temp_2000m": "T2000"}
    return df.rename(columns=rename_map)


def compute_stats(df):
    df = standardize_columns(df)
    required = {"X", "Y", "T400", "T1000", "T2000"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    depths = np.array([400.0, 1000.0, 2000.0])
    T_mean = np.array([df["T400"].mean(), df["T1000"].mean(), df["T2000"].mean()])
    T_std = np.array([df["T400"].std(), df["T1000"].std(), df["T2000"].std()])
    return df, depths, T_mean, T_std


def make_figure(depths, T_mean, T_std, params, lang):
    Z_MAX = params["Z_MAX"]
    N_Z = params["N_Z"]
    z_line = np.linspace(0.0, Z_MAX, int(N_Z))

    # Fit on mean -> slope only (gradient)
    b_mean, _a_mean = np.polyfit(depths, T_mean, 1)
    G_mean_profile = b_mean * 1000.0  # °C/km
    b_mean_m = G_mean_profile / 1000.0  # °C/m

    # Nodes for mean and sigma: point at z=0 (sigma=0), then measured
    z_nodes = np.array([0.0, 400.0, 1000.0, 2000.0])
    T_nodes = np.array([params["T_REF"], T_mean[0], T_mean[1], T_mean[2]])
    S_nodes = np.array([0.0, T_std[0], T_std[1], T_std[2]])

    T_line = np.empty_like(z_line)
    S_line = np.empty_like(z_line)

    mask_obs = z_line <= 2000.0
    mask_ext = z_line > 2000.0

    T_line[mask_obs] = np.interp(z_line[mask_obs], z_nodes, T_nodes)
    S_line[mask_obs] = np.interp(z_line[mask_obs], z_nodes, S_nodes)

    T_2000 = T_nodes[-1]
    S_2000 = S_nodes[-1]
    T_line[mask_ext] = T_2000 + b_mean_m * (z_line[mask_ext] - 2000.0)
    S_line[mask_ext] = S_2000  # constant beyond 2000

    T_env_low = T_line - S_line
    T_env_high = T_line + S_line

    # Gradients referenced at (Z_REF, T_REF)
    Z_REF = params["Z_REF"]
    T_REF = params["T_REF"]
    G_LADISPOLI = params["G_LADISPOLI"]

    T_grad_mean = T_REF + (G_mean_profile / 1000.0) * (z_line - Z_REF)
    T_grad_ladispoli = T_REF + (G_LADISPOLI / 1000.0) * (z_line - Z_REF)

    fig = plt.figure(figsize=params["FIGSIZE"])

    plt.fill_betweenx(
        z_line, T_env_low, T_env_high,
        color=params["ENV_COLOR"], alpha=params["ENV_ALPHA"],
        label=params["ENV_LABEL"]
    )

    plt.plot(
        T_line, z_line,
        color=params["MEAN_COLOR"], linewidth=params["MEAN_LW"],
        linestyle=params["MEAN_LS"],
        label=params["MEAN_LABEL"]
    )

    if params["SHOW_POINTS"]:
        plt.plot(
            T_mean, depths,
            marker=params["PTS_MARKER"], markersize=params["PTS_MS"],
            color=params["PTS_COLOR"], linestyle="none",
            label=params["PTS_LABEL"]
        )

    if params["SHOW_ERR"]:
        plt.errorbar(
            T_mean, depths, xerr=T_std,
            fmt="none",
            ecolor=params["ERR_ECOLOR"],
            elinewidth=params["ERR_ELW"],
            capsize=params["ERR_CAP"],
            alpha=params["ERR_ALPHA"],
            label=params["ERR_LABEL"]
        )

    if params["SHOW_GRAD_MEAN"]:
        plt.plot(
            T_grad_mean, z_line,
            color=params["GRAD_MEAN_COLOR"],
            linestyle=params["GRAD_MEAN_LS"],
            linewidth=params["GRAD_MEAN_LW"],
            label=f"{params['GRAD_MEAN_LABEL']} : {G_mean_profile:.1f} °C/km"
        )

    if params["SHOW_GRAD_LADISPOLI"]:
        plt.plot(
            T_grad_ladispoli, z_line,
            color=params["GRAD_LADISPOLI_COLOR"],
            linestyle=params["GRAD_LADISPOLI_LS"],
            linewidth=params["GRAD_LADISPOLI_LW"],
            label=f"{params['GRAD_LADISPOLI_LABEL']} : {G_LADISPOLI:.0f} °C/km"
        )

    plt.ylim(Z_MAX, 0)
    if params["XLIM_ON"]:
        plt.xlim(params["XLIM_MIN"], params["XLIM_MAX"])

    if params["XTICKS_ON"]:
        plt.xticks(np.arange(params["XTICK_START"], params["XTICK_END"] + 1e-9, params["XTICK_STEP"]))
    if params["YTICKS_ON"]:
        plt.yticks(np.arange(params["YTICK_START"], params["YTICK_END"] + 1e-9, params["YTICK_STEP"]))

    plt.xlabel(t(lang, "xlabel"))
    plt.ylabel(t(lang, "ylabel"))
    plt.title(params["TITLE"])

    if params["GRID_ON"]:
        plt.grid(True, linestyle=params["GRID_STYLE"], linewidth=params["GRID_LW"], alpha=params["GRID_ALPHA"])

    if params["LEGEND_ON"]:
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, params["LEGEND_YOFFSET"]),
            ncol=params["LEGEND_NCOL"],
            frameon=params["LEGEND_FRAME"],
            fontsize=params["LEGEND_FONT"]
        )

    plt.tight_layout()
    return fig

def load_geojson(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _label_point_lonlat(geometry):
    """
    Return a label point (lon, lat) for LineString/MultiLineString/Polygon/MultiPolygon.
    """
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    # --- Lines ---
    if gtype == "LineString" and isinstance(coords, list) and len(coords) > 1:
        lon, lat = coords[len(coords) // 2]
        return float(lon), float(lat)

    if gtype == "MultiLineString" and isinstance(coords, list) and len(coords) > 0:
        # choose the longest line
        best = max(coords, key=lambda line: len(line) if isinstance(line, list) else 0)
        if isinstance(best, list) and len(best) > 1:
            lon, lat = best[len(best) // 2]
            return float(lon), float(lat)

    # --- Polygons (use a crude centroid of the outer ring) ---
    def centroid_of_ring(ring):
        # ring = [[lon,lat], ...]
        xs = [p[0] for p in ring if isinstance(p, list) and len(p) >= 2]
        ys = [p[1] for p in ring if isinstance(p, list) and len(p) >= 2]
        if not xs or not ys:
            return None
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    if gtype == "Polygon" and isinstance(coords, list) and len(coords) > 0:
        outer = coords[0]
        c = centroid_of_ring(outer)
        return c

    if gtype == "MultiPolygon" and isinstance(coords, list) and len(coords) > 0:
        # pick first polygon outer ring
        outer = coords[0][0] if isinstance(coords[0], list) and len(coords[0]) > 0 else None
        if outer:
            c = centroid_of_ring(outer)
            return c

    return None


def add_contours_geojson_layer(
    m: folium.Map,
    contours_geojson: dict,
    level_field: str,
    show_labels_every_20: bool,
):
    def style_fn(feature):
        return {"color": "red", "weight": 2, "opacity": 0.9}

    # Draw contours
    folium.GeoJson(
        contours_geojson,
        name="Contours",
        style_function=style_fn,
    ).add_to(m)

    if not show_labels_every_20:
        return

    def parse_level(val):
        """Robust parse: accepts 20, '20', '20.0', '20,0', '20 m', etc."""
        if val is None:
            return None
        s = str(val).strip()
        s = s.replace(",", ".")
        # keep digits, dot, minus
        cleaned = "".join(ch for ch in s if (ch.isdigit() or ch in ".-"))
        if cleaned in ("", ".", "-", "-."):
            return None
        try:
            return float(cleaned)
        except Exception:
            return None

    feats = contours_geojson.get("features", [])
    for feat in feats:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        if not geom:
            continue

        val_num = parse_level(props.get(level_field))
        if val_num is None:
            continue

        # robust "every 20": round then modulo
        val_int = int(round(val_num))
        if val_int % 20 != 0:
            continue

        pt = _label_point_lonlat(geom)
        if pt is None:
            continue

        lon, lat = pt

        folium.DivIcon(
    html=f"""
    <div style="
        font-size:14px;
        font-weight:700;
        color:#b00000;
        background:rgba(255,255,255,0.95);
        padding:4px 8px;
        border:2px solid #b00000;
        border-radius:6px;
        box-shadow:0 1px 4px rgba(0,0,0,0.35);
        white-space:nowrap;
    ">{val_int}</div>
    """
)
        ).add_to(m)



# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=TRANSLATIONS["en"]["page_title"], layout="wide")

# Language selector
st.sidebar.header("Language / Langue / Lingua")
lang_code = st.sidebar.selectbox(
    "Select UI language",
    options=[code for code, _ in LANG_OPTIONS],
    format_func=lambda c: dict(LANG_OPTIONS)[c],
    index=0
)

st.title(t(lang_code, "title"))

# Data
st.sidebar.header(t(lang_code, "sidebar_data"))
uploaded = st.sidebar.file_uploader(t(lang_code, "upload_csv"), type=["csv"])
sep = st.sidebar.selectbox(t(lang_code, "separator"), options=[";", ",", "\t"], index=0)
use_uploaded = st.sidebar.checkbox(t(lang_code, "use_uploaded"), value=(uploaded is not None))
use_demo = st.sidebar.checkbox(t(lang_code, "demo_mode"), value=False)

if use_demo:
    rng = np.random.default_rng(0)
    n = 500
    df_demo = pd.DataFrame(
        {
            "X": rng.uniform(0, 1000, n),
            "Y": rng.uniform(0, 1000, n),
            "T400": 25 + 0.055 * 400 + rng.normal(0, 5, n),
            "T1000": 25 + 0.055 * 1000 + rng.normal(0, 7, n),
            "T2000": 25 + 0.055 * 2000 + rng.normal(0, 10, n),
        }
    )
    df, depths, T_mean, T_std = compute_stats(df_demo)
    st.info(t(lang_code, "demo_info"))
else:
    try:
        if use_uploaded and uploaded is not None:
            df_raw = pd.read_csv(uploaded, sep=sep)
        else:
            if not EMBEDDED_CSV.exists():
                st.warning(t(lang_code, "missing_embedded"))
                st.stop()
            df_raw = pd.read_csv(EMBEDDED_CSV, sep=sep)
            st.caption(f"{t(lang_code, 'using_embedded')} {EMBEDDED_CSV.name} — {t(lang_code, 'loaded_rows')}: {len(df_raw)}")

        df, depths, T_mean, T_std = compute_stats(df_raw)
    except Exception as e:
        st.error(f"{t(lang_code, 'csv_error')} {e}")
        st.stop()

# Reference & gradients
st.sidebar.header(t(lang_code, "sidebar_refs"))
T_REF = st.sidebar.number_input(t(lang_code, "t_ref"), value=25.0, step=0.5)
Z_REF = st.sidebar.number_input(t(lang_code, "z_ref"), value=0.0, step=10.0)
G_LADISPOLI = st.sidebar.number_input(t(lang_code, "g_local"), value=60.0, step=1.0)

# Depth / resolution
st.sidebar.header(t(lang_code, "sidebar_depth"))
Z_MAX = st.sidebar.slider(t(lang_code, "z_max"), min_value=500, max_value=5000, value=2500, step=100)
N_Z = st.sidebar.slider(t(lang_code, "n_z"), min_value=100, max_value=1200, value=350, step=50)

# Axes
st.sidebar.header(t(lang_code, "sidebar_axes"))
XLIM_ON = st.sidebar.checkbox(t(lang_code, "fix_xlim"), value=False)
if XLIM_ON:
    default_min = float(np.floor(np.min(T_mean - T_std) - 10))
    default_max = float(np.ceil(np.max(T_mean + T_std) + 10))
    XLIM_MIN = st.sidebar.number_input(t(lang_code, "xlim_min"), value=default_min)
    XLIM_MAX = st.sidebar.number_input(t(lang_code, "xlim_max"), value=default_max)
else:
    XLIM_MIN, XLIM_MAX = 0.0, 0.0

XTICKS_ON = st.sidebar.checkbox(t(lang_code, "force_xticks"), value=False)
if XTICKS_ON:
    XTICK_START = st.sidebar.number_input(t(lang_code, "xtick_start"), value=0.0)
    XTICK_END = st.sidebar.number_input(t(lang_code, "xtick_end"), value=200.0)
    XTICK_STEP = st.sidebar.number_input(t(lang_code, "xtick_step"), value=25.0)
else:
    XTICK_START = XTICK_END = XTICK_STEP = 0.0

YTICKS_ON = st.sidebar.checkbox(t(lang_code, "force_yticks"), value=False)
if YTICKS_ON:
    YTICK_START = st.sidebar.number_input(t(lang_code, "ytick_start"), value=0.0)
    YTICK_END = st.sidebar.number_input(t(lang_code, "ytick_end"), value=float(Z_MAX))
    YTICK_STEP = st.sidebar.number_input(t(lang_code, "ytick_step"), value=500.0)
else:
    YTICK_START = YTICK_END = YTICK_STEP = 0.0

# Appearance
st.sidebar.header(t(lang_code, "sidebar_style"))
FIG_W = st.sidebar.slider(t(lang_code, "fig_w"), 4.0, 12.0, 6.0, 0.5)
FIG_H = st.sidebar.slider(t(lang_code, "fig_h"), 4.0, 12.0, 7.0, 0.5)

TITLE = st.sidebar.text_input(t(lang_code, "plot_title"), value=t(lang_code, "default_plot_title"))

ENV_COLOR = st.sidebar.color_picker(t(lang_code, "env_color"), "#1f77b4")
ENV_ALPHA = st.sidebar.slider(t(lang_code, "env_alpha"), 0.0, 1.0, 0.20, 0.01)

MEAN_COLOR = st.sidebar.color_picker(t(lang_code, "mean_color"), "#000000")
MEAN_LW = st.sidebar.slider(t(lang_code, "mean_lw"), 0.5, 5.0, 2.2, 0.1)
MEAN_LS = st.sidebar.selectbox(t(lang_code, "mean_ls"), ["-", "--", ":", "-."], index=0)

SHOW_POINTS = st.sidebar.checkbox(t(lang_code, "show_points"), value=True)
PTS_COLOR = st.sidebar.color_picker(t(lang_code, "pts_color"), "#000000")
PTS_MARKER = st.sidebar.selectbox(t(lang_code, "pts_marker"), ["o", "s", "^", "D", "x", "+"], index=0)
PTS_MS = st.sidebar.slider(t(lang_code, "pts_ms"), 2.0, 12.0, 6.5, 0.5)

SHOW_ERR = st.sidebar.checkbox(t(lang_code, "show_err"), value=True)
ERR_ECOLOR = st.sidebar.color_picker(t(lang_code, "err_color"), "#000000")
ERR_ELW = st.sidebar.slider(t(lang_code, "err_lw"), 0.5, 5.0, 1.4, 0.1)
ERR_CAP = st.sidebar.slider(t(lang_code, "err_cap"), 0, 12, 4, 1)
ERR_ALPHA = st.sidebar.slider(t(lang_code, "err_alpha"), 0.0, 1.0, 0.9, 0.05)

# Gradients styles
st.sidebar.header(t(lang_code, "sidebar_grad_style"))
SHOW_GRAD_MEAN = st.sidebar.checkbox(t(lang_code, "show_grad_mean"), value=True)
GRAD_MEAN_COLOR = st.sidebar.color_picker(t(lang_code, "grad_mean_color"), "#1f77b4")
GRAD_MEAN_LS = st.sidebar.selectbox(t(lang_code, "grad_mean_ls"), ["-", "--", ":", "-."], index=2)
GRAD_MEAN_LW = st.sidebar.slider(t(lang_code, "grad_mean_lw"), 0.5, 5.0, 2.0, 0.1)

SHOW_GRAD_LADISPOLI = st.sidebar.checkbox(t(lang_code, "show_grad_local"), value=True)
GRAD_LADISPOLI_COLOR = st.sidebar.color_picker(t(lang_code, "grad_local_color"), "#d62728")
GRAD_LADISPOLI_LS = st.sidebar.selectbox(t(lang_code, "grad_local_ls"), ["-", "--", ":", "-."], index=1)
GRAD_LADISPOLI_LW = st.sidebar.slider(t(lang_code, "grad_local_lw"), 0.5, 5.0, 2.2, 0.1)

# Grid
st.sidebar.header(t(lang_code, "sidebar_grid"))
GRID_ON = st.sidebar.checkbox(t(lang_code, "grid_on"), value=True)
GRID_STYLE = st.sidebar.selectbox(t(lang_code, "grid_style"), [":", "--", "-", "-."], index=0)
GRID_LW = st.sidebar.slider(t(lang_code, "grid_lw"), 0.1, 2.5, 0.8, 0.1)
GRID_ALPHA = st.sidebar.slider(t(lang_code, "grid_alpha"), 0.0, 1.0, 0.7, 0.05)

# Legend
st.sidebar.header(t(lang_code, "sidebar_legend"))
LEGEND_ON = st.sidebar.checkbox(t(lang_code, "legend_on"), value=True)
LEGEND_FRAME = st.sidebar.checkbox(t(lang_code, "legend_frame"), value=True)
LEGEND_NCOL = st.sidebar.slider(t(lang_code, "legend_ncol"), 1, 4, 2, 1)
LEGEND_FONT = st.sidebar.slider(t(lang_code, "legend_font"), 6, 16, 10, 1)
LEGEND_YOFFSET = st.sidebar.slider(t(lang_code, "legend_yoffset"), -0.40, 0.05, -0.12, 0.01)

# Text labels
st.sidebar.header(t(lang_code, "sidebar_texts"))
ENV_LABEL = st.sidebar.text_input(t(lang_code, "env_label"), value=t(lang_code, "default_env_label"))
MEAN_LABEL = st.sidebar.text_input(t(lang_code, "mean_label"), value=t(lang_code, "default_mean_label"))
PTS_LABEL = st.sidebar.text_input(t(lang_code, "pts_label"), value=t(lang_code, "default_pts_label"))
ERR_LABEL = st.sidebar.text_input(t(lang_code, "err_label"), value=t(lang_code, "default_err_label"))
GRAD_MEAN_LABEL = st.sidebar.text_input(t(lang_code, "grad_mean_label"), value=t(lang_code, "default_grad_mean_label"))
GRAD_LADISPOLI_LABEL = st.sidebar.text_input(t(lang_code, "grad_local_label"), value=t(lang_code, "default_grad_local_label"))

# Map controls
st.sidebar.header(t(lang_code, "map_title"))
MAP_HEIGHT = st.sidebar.slider(t(lang_code, "map_height"), 220, 1200, 550, 10)
#SHOW_MAP_POINTS = st.sidebar.checkbox(t(lang_code, "map_points"), value=False)
SHOW_CONTOURS = st.sidebar.checkbox(t(lang_code, "map_contours"), value=True)
SHOW_LABELS_20 = st.sidebar.checkbox(t(lang_code, "map_labels"), value=True)
LEVEL_FIELD = st.sidebar.text_input(t(lang_code, "map_level_field"), value="ELEV")

params = {
    "Z_MAX": float(Z_MAX),
    "N_Z": int(N_Z),
    "T_REF": float(T_REF),
    "Z_REF": float(Z_REF),
    "G_LADISPOLI": float(G_LADISPOLI),
    "FIGSIZE": (float(FIG_W), float(FIG_H)),
    "TITLE": TITLE,

    "ENV_COLOR": ENV_COLOR,
    "ENV_ALPHA": float(ENV_ALPHA),
    "ENV_LABEL": ENV_LABEL,

    "MEAN_COLOR": MEAN_COLOR,
    "MEAN_LW": float(MEAN_LW),
    "MEAN_LS": MEAN_LS,
    "MEAN_LABEL": MEAN_LABEL,

    "SHOW_POINTS": bool(SHOW_POINTS),
    "PTS_COLOR": PTS_COLOR,
    "PTS_MARKER": PTS_MARKER,
    "PTS_MS": float(PTS_MS),
    "PTS_LABEL": PTS_LABEL,

    "SHOW_ERR": bool(SHOW_ERR),
    "ERR_ECOLOR": ERR_ECOLOR,
    "ERR_ELW": float(ERR_ELW),
    "ERR_CAP": int(ERR_CAP),
    "ERR_ALPHA": float(ERR_ALPHA),
    "ERR_LABEL": ERR_LABEL,

    "SHOW_GRAD_MEAN": bool(SHOW_GRAD_MEAN),
    "GRAD_MEAN_COLOR": GRAD_MEAN_COLOR,
    "GRAD_MEAN_LS": GRAD_MEAN_LS,
    "GRAD_MEAN_LW": float(GRAD_MEAN_LW),
    "GRAD_MEAN_LABEL": GRAD_MEAN_LABEL,

    "SHOW_GRAD_LADISPOLI": bool(SHOW_GRAD_LADISPOLI),
    "GRAD_LADISPOLI_COLOR": GRAD_LADISPOLI_COLOR,
    "GRAD_LADISPOLI_LS": GRAD_LADISPOLI_LS,
    "GRAD_LADISPOLI_LW": float(GRAD_LADISPOLI_LW),
    "GRAD_LADISPOLI_LABEL": GRAD_LADISPOLI_LABEL,

    "GRID_ON": bool(GRID_ON),
    "GRID_STYLE": GRID_STYLE,
    "GRID_LW": float(GRID_LW),
    "GRID_ALPHA": float(GRID_ALPHA),

    "LEGEND_ON": bool(LEGEND_ON),
    "LEGEND_FRAME": bool(LEGEND_FRAME),
    "LEGEND_NCOL": int(LEGEND_NCOL),
    "LEGEND_FONT": int(LEGEND_FONT),
    "LEGEND_YOFFSET": float(LEGEND_YOFFSET),

    "XLIM_ON": bool(XLIM_ON),
    "XLIM_MIN": float(XLIM_MIN),
    "XLIM_MAX": float(XLIM_MAX),

    "XTICKS_ON": bool(XTICKS_ON),
    "XTICK_START": float(XTICK_START),
    "XTICK_END": float(XTICK_END),
    "XTICK_STEP": float(XTICK_STEP),

    "YTICKS_ON": bool(YTICKS_ON),
    "YTICK_START": float(YTICK_START),
    "YTICK_END": float(YTICK_END),
    "YTICK_STEP": float(YTICK_STEP),
}

# Layout
col1, col2 = st.columns([2.2, 1.0], gap="large")

with col1:
    fig = make_figure(depths, T_mean, T_std, params, lang_code)
    st.pyplot(fig, clear_figure=True)

    png_bytes = fig_to_png_bytes(fig, dpi=200)
    st.download_button(
        label=t(lang_code, "download_png"),
        data=png_bytes,
        file_name="figure.png",
        mime="image/png",
    )

    # ---------------------------
    # Embedded "Geoportail-like" basemap (Italy-centered) + contours + labels every 20
    # ---------------------------
    ITALY_CENTER = [41.8719, 12.5674]  # lat, lon
    DEFAULT_ZOOM = 5

    m = folium.Map(
        location=ITALY_CENTER,
        zoom_start=DEFAULT_ZOOM,
        control_scale=True,
        tiles=None,
    )

    # ESRI basemaps
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Streets",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

       # Contours (GeoJSON) + labels every 20
    if SHOW_CONTOURS:
        contours = load_geojson(CONTOUR_GEOJSON)
        if contours is None:
            st.warning(t(lang_code, "map_missing_contours"))
        else:
            add_contours_geojson_layer(
                m=m,
                contours_geojson=contours,
                level_field=LEVEL_FIELD,
                show_labels_every_20=SHOW_LABELS_20,
            )

    folium.LayerControl().add_to(m)
    st_folium(m, width=None, height=int(MAP_HEIGHT))

with col2:
    st.subheader(t(lang_code, "summary"))
    st.write(pd.DataFrame({"z (m)": depths, "T_mean (°C)": T_mean, "sigma (°C)": T_std}))

    st.subheader(t(lang_code, "csv_help"))
    st.write(t(lang_code, "csv_help_text"))
