# ML Music Genres 🎵🎶

Clasificación de géneros musicales utilizando técnicas de **Machine Learning** y **Deep Learning**.

## 📌 Descripción
Este proyecto busca entrenar modelos de Machine Learning para clasificar géneros musicales a partir de características extraídas de archivos de audio.

### 🔹 Características principales:
- Procesamiento de datos de audio con **MFCCs** y otras características relevantes.
- Modelos de clasificación con **XGBoost, CatBoost y Redes Neuronales**.
- Evaluación de rendimiento con métricas como **Matriz de Confusión, Accuracy, Precision, Recall, F1-Score**.

---

## 📂 Estructura del repositorio
```
ML_music_genres/
│── data/               # Datasets con las canciones y características extraidas
│── notebooks/          # Jupyter Notebooks con descarga de datos, limpieza, entrenamiento y evaluación
│── modelos/            # Modelos entrenados
│── src/           # Scripts para preprocesamiento y entrenamiento
│── streamlit/         # Con los scripts para ejecurtar la demo en streamlit
│── README.md          # Este archivo 😃
```

---

## 📊 Modelos Implementados
### ✅ **XGBoost**
- **Descripción:** Algoritmo basado en árboles de decisión optimizados para clasificación multiclase.
- **Parámetros clave:** `max_depth`, `learning_rate`, `n_estimators`.

### ✅ **CatBoost**
- **Descripción:** Algoritmo basado en boosting con soporte para categóricas y optimización GPU.
- **Parámetros clave:** `iterations`, `depth`, `loss_function`.

### ✅ **LightGBM**
- **Descripción:** Algoritmo de boosting eficiente para grandes volúmenes de datos, optimizado para velocidad y rendimiento.
- **Parámetros clave:** `num_leaves`, `learning_rate`, `n_estimators`.

### ✅ **Random Forest**
- **Descripción:** Modelo basado en múltiples árboles de decisión para mejorar la precisión y reducir el sobreajuste.
- **Parámetros clave:** `n_estimators`, `max_depth`, `min_samples_split`.

### ✅ **PCA (Análisis de Componentes Principales)**
- **Descripción:** Técnica de reducción de dimensionalidad que permite seleccionar las características más importantes de los datos.
- **Parámetros clave:** `n_components`.

### ✅ **Red Neuronal (Deep Learning)**
- **Descripción:** Modelo basado en **Keras/TensorFlow** con varias capas densas.
- **Parámetros clave:** `epochs`, `batch_size`, `optimizer`.

---

## 📈 Evaluación y Resultados
Los modelos son evaluados mediante:
- **Matriz de Confusión** 🟦
- **Accuracy, Precision, Recall, F1-Score** ✅

Ejemplo de matriz de confusión:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicciones")
plt.ylabel("Valores Reales")
plt.title("Matriz de Confusión")
plt.show()
```

---

## 📜 Licencia
Este proyecto está bajo la licencia **MIT**. Puedes utilizarlo y modificarlo libremente.

---

## 📬 Contacto
📧 **Lucas Herranz**  
🔗 [LinkedIn](https://www.linkedin.com/in/lucasherranzvicente/)  

---

¡Gracias por visitar este repositorio! 🚀🎧

