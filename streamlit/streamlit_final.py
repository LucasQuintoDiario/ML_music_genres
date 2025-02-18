import librosa
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- Configuración de la Página ---
st.set_page_config(
    page_title="🎶 Predictor de géneros musicales",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título Principal ---
st.title("🎶 Predictor de géneros musicales")
st.markdown("📀 Sube una canción y descubre su genero, tempo y tono con nuestro modelo de Machine Learning.")

# --- Cargar modelo y scaler ---
@st.cache_resource
def load_resources():
    with open("../modelos/xgboost_oversampled_model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)
    with open("../modelos/xgboost_oversampled_scaler.pkl", 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    # Cargar dataset de playlists
    df_playlists = pd.read_csv("../data/playlists_predictions.csv")
    
    return model, scaler, df_playlists

model, scaler, df_playlists = load_resources()

# --- Sidebar: Cargar Archivo de Audio ---
st.sidebar.header("📂 Cargar Archivo")
uploaded_file = st.sidebar.file_uploader("🎵 Sube una pista de audio", type=["wav", "mp3"])


if uploaded_file is not None:
    st.sidebar.audio(uploaded_file, format='audio/wav')

    with st.spinner("⏳ Analizando el audio..."):
        try:
            # --- Cargar y procesar audio ---
            y, sr = librosa.load(uploaded_file, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_means_scaled = scaler.transform(mfcc_means.reshape(1, -1))
            mfcc_means_scaled = mfcc_means_scaled.reshape(1, -1)

            # --- Predicción ---
            pred_probabilities = model.predict_proba(mfcc_means_scaled)  # Predicción de probabilidades
            pred_class =  model.predict(mfcc_means_scaled) # Predicción de la clase más probable

            # Mapear predicción a género
            genre_dict = {
                0: "🎧 Musica Electrónica",
                1: "🎸 Múscia Rock & derivados",
                2: "🎻 Música Clásica & Tradicional",
                3: "🌾 Folk & Indie"
            }
            genre_pred = genre_dict.get(pred_class[0], "❓ Estilo desconocido")

            # Mostrar la probabilidad de pertenencia a cada categoría
            st.success(f"**🎶 Estilo Musical Detectado:** {genre_pred}")
            st.markdown("### Probabilidades de Pertenencia:")
            
            # Mostrar las probabilidades para cada categoría
            categories = list(genre_dict.values())
            for i, category in enumerate(categories):
                st.write(f"**{category}:** {pred_probabilities[0][i]*100:.2f}%")

            # --- Calcular Tempo ---
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # --- Calcular Tono ---
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitches_filtered = pitches[magnitudes > np.median(magnitudes)]

            if len(pitches_filtered) > 0:
                dominant_pitch = np.median(pitches_filtered)
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                estimated_note = note_names[int(round(dominant_pitch) % 12)]
            else:
                estimated_note = "No detectado"

            # --- Mostrar resultados ---
            # --- Cajas con Tempo y Tono ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🥁 Tempo Aproximado")
                st.info(f"**{int(tempo)} BPM** (Beats por Minuto)")

            with col2:
                st.subheader("🎵 Tono Estimado")
                st.warning(f"**{estimated_note}** (Nota dominante)")

            # --- Recomendación de Playlists ---
            st.subheader("🔊 Playlists Recomendadas")
            st.markdown("_Si quieres escuchar más canciones como esta, ¿por qué no le echas un vistazo a estas playlists? 🎧_")
            filtered_playlists = df_playlists[df_playlists['playlist_categories'] == pred_class[0]]
            
            if not filtered_playlists.empty:
                recommended_playlists = filtered_playlists.sample(n=min(5, len(filtered_playlists)))  # Tomar 5 aleatorias
                for _, row in recommended_playlists.iterrows():
                    st.markdown(f"🔹 [{row['playlist_name']}]({row['playlist_url']})")
            else:
                st.warning("No se encontraron playlists para este género.")

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")

else:
    st.sidebar.warning("⚠️ No has subido ningún archivo de audio.")
