import librosa
import streamlit as st
import numpy as np
import pickle

# --- Configuración de la Página ---
st.set_page_config(
    page_title="🎶 Clasificador de Música: Digital vs. Tradicional",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título Principal ---
st.title("🎶 Clasificador de Música: Digital vs. Estudio")
st.markdown("📀 Sube una canción y descubre su estilo, tempo y tono con nuestro modelo de Machine Learning.")

# --- Cargar modelo y scaler ---
@st.cache_resource
def load_model_and_scaler():
    with open("secondmodel.pkl", 'rb') as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model_and_scaler()

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

            # --- Predicción ---
            pred = model.predict(mfcc_means_scaled)

            # --- Mapear predicción a género ---
            genre_dict = {
                0: "🎛️ Música Electrónica / Digital",
                1: "🎸 Música de Estudio"
            }
            genre_pred = genre_dict.get(pred[0], "❓ Estilo desconocido")

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
            st.success(f"**🎶 Estilo Musical Detectado:** {genre_pred}")

            # --- Cajas con Tempo y Tono ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🥁 Tempo Aproximado")
                st.info(f"**{int(tempo)} BPM** (Beats por Minuto)")

            with col2:
                st.subheader("🎵 Tono Estimado")
                st.warning(f"**{estimated_note}** (Nota dominante)")

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")

else:
    st.sidebar.warning("⚠️ No has subido ningún archivo de audio.")
