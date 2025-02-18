import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("../data/processed.csv")
df_genres =  pd.read_csv("../data/genres.csv")
df = df.copy().drop(columns=["song_id","filename", "filepath", 'Chroma_1', 'Chroma_2', 'Chroma_3', 'Chroma_4',
       'Chroma_5', 'Chroma_6', 'Chroma_7', 'Chroma_8', 'Chroma_9', 'Chroma_10',
       'Chroma_11', 'Chroma_12', 'Spectral_contrast_1', 'Spectral_contrast_2',
       'Spectral_contrast_3', 'Spectral_contrast_4', 'Spectral_contrast_5',
       'Spectral_contrast_6', 'Spectral_contrast_7', 'Zero_crossing_rate',
       'Spectral_Rolloff', 'Tempo'])

# Agrupar los generos en 4 grandes categorias
genre_mapping = {
    # 0 - Música Electrónica
    'Electronic': 0,
    'Ambient Electronic': 0,
    'Chiptune / Glitch': 0,


    # 1 - Musica Rock & derivados
    'Rock': 1,
    'Punk': 1,

    # 2 - Música Clásica & Tradicional
    'Classical': 2,
    'Old-Time / Historic': 2,

    # 3 - Folk, Soul, Blues & Country
    'Folk': 3,


    # Géneros descartado
    'Hip-Hop': None,
    'International': None,
    'Spoken': None,
    'Pop': None,
    'Instrumental': None,
    'Experimental': None,
    'Easy Listening': None,
    'Jazz': None,
    'Soul-RnB': None,
    'Blues': None,
    'Country': None

}

df_3 = df.copy().drop(columns="genre_id")
df_3['genre'] = df_3['genre'].replace(genre_mapping).astype("Int32")
df_3 = df_3.dropna(subset=['genre'])

# Definir X e y
X = df_3.drop(columns=["genre"])
y = df_3["genre"]

# Definir el oversampler
oversampler = SMOTE(sampling_strategy='auto', random_state=42)

# Aplicar el oversampling
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=41)

# Escalar los datos
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Definir el modelo XGBoost para clasificación multiclase
model_xgb_3 = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0,
    reg_lambda=1,
    objective='multi:softmax',
    num_class=4,
    eval_metric='mlogloss',
    random_state=42
)

model_xgb_3.fit(X_train_scaled, y_train)





