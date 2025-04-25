import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import plotly.express as px
from tqdm import tqdm
from analyze_participant import analyze_participant

# === Chargement des variables d'environnement (.env) ===
load_dotenv()
url = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
engine = create_engine(url)

# === Fonction Haversine vectorisée ===
def haversine_distance_np(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# === Lecture des données GPS depuis la base PostgreSQL ===
with engine.connect() as conn:
    df = pd.read_sql_query(text("SELECT * FROM clean_gps"), con=conn)

# === Prétraitement global ===
df['timestamp'] = pd.to_datetime(df['time'], utc=True)
df = df.drop(columns=['time'])

# === Génération des rapports individuels ===
for pid in tqdm(df['participant_virtual_id'].unique(), desc="Rapports utilisateurs"):
    df_user = df[df['participant_virtual_id'] == pid].copy()
    if len(df_user) > 200:
        analyze_participant(df_user, participant_id=pid, fast_mode=True)

# === Résumé par utilisateur ===
summary = []
for pid, df_user in df.groupby('participant_virtual_id'):
    df_user = df_user.sort_values('timestamp')
    df_user['time_diff_s'] = df_user['timestamp'].diff().dt.total_seconds()

    lat1 = df_user['lat'].shift()
    lon1 = df_user['lon'].shift()
    lat2 = df_user['lat']
    lon2 = df_user['lon']
    df_user['dist_m'] = haversine_distance_np(lat1, lon1, lat2, lon2)

    df_user['speed_kmh'] = df_user['dist_m'] / df_user['time_diff_s'] * 3.6

    summary.append({
        "participant": pid,
        "distance_km": np.nansum(df_user['dist_m']) / 1000,
        "avg_speed_kmh": np.nanmean(df_user['speed_kmh']),
        "total_time_h": np.nansum(df_user['time_diff_s']) / 3600,
        "nb_points": len(df_user)
    })

# === Visualisation récapitulative ===
df_summary = pd.DataFrame(summary)
print("\n📊 Résumé global :")
print(df_summary)

fig = px.bar(df_summary.sort_values("distance_km", ascending=False),
             x="participant", y="distance_km",
             color="participant",
             title="🚶‍♂️ Distance parcourue par participant",
             labels={"distance_km": "Distance (km)", "participant": "Utilisateur"},
             template="plotly_white")
fig.show()

# === Vitesse moyenne horaire par utilisateur ===
df['hour'] = df['timestamp'].dt.hour
df_clean = df[(df['speed_kmh'].notna()) & (df['speed_kmh'] <= 150)]

hourly_stats = df_clean.groupby(['participant_virtual_id', 'hour'])['speed_kmh'].mean().reset_index()
hourly_stats.rename(columns={'speed_kmh': 'avg_speed_kmh'}, inplace=True)

fig_hour = px.line(hourly_stats,
                   x='hour', y='avg_speed_kmh',
                   color='participant_virtual_id',
                   markers=True,
                   title="📈 Vitesse moyenne horaire par utilisateur",
                   labels={'avg_speed_kmh': 'Vitesse (km/h)', 'hour': 'Heure'},
                   template='plotly_white')
fig_hour.update_layout(xaxis=dict(dtick=1), legend_title_text="Participant")
fig_hour.show()
