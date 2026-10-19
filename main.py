import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Menjaga konsistensi random seed
np.random.seed(42)
tf.random.set_seed(42)

print('=== PIPELINE DEEP LEARNING & IMK: INTERPRETASI PREDIKSI DASBOR ===')

# ================= ================= =================
# 1. SIMULASI DATASET & PELATIHAN MODEL DEEP LEARNING
# ================= ================= =================
n_samples = 2000
n_features = 32

X = np.random.randn(n_samples, n_features)
y = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Arsitektur Neural Network (Deep Learning)
model = models.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Melatih model secara senyap (verbose=0)
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
print('Model Deep Learning berhasil dilatih dan divalidasi.')

# ================= ================= =================
# 2. INFERENSI & PENGELOLAAN BEBAN KOGNITIF (TRANSPARANSI UI)
# ================= ================= =================
# Mendapatkan probabilitas prediksi dari model
y_pred_probs = model.predict(X_test, verbose=0).flatten()

df_results = pd.DataFrame({
    'sample_id': [f'IMG-{i:04d}' for i in range(len(y_test))],
    'actual_label': y_test,
    'prediction_probability': y_pred_probs
})

# Menerapkan ambang batas keputusan (Threshold) & Kategori Keyakinan (Confidence Level)
def categorize_confidence(prob):
    if prob >= 0.80 or prob <= 0.20:
        return 'HIGH_CONFIDENCE (Jelas)'
    elif 0.40 <= prob <= 0.60:
        return 'LOW_CONFIDENCE (Butuh Review Pakar)'
    else:
        return 'MODERATE_CONFIDENCE'

df_results['predicted_class'] = (df_results['prediction_probability'] >= 0.5).astype(int)
df_results['confidence_status'] = df_results['prediction_probability'].apply(categorize_confidence)

# ================= ================= =================
# 3. REKAPITULASI RINGKASAN UNTUK DASBOR INTERAKTIF (SHNEIDERMAN OVERVIEW)
# ================= ================= =================
dashboard_summary = df_results.groupby('confidence_status').agg(
    total_samples=('sample_id', 'count'),
    mean_probability=('prediction_probability', 'mean')
).reset_index()

print('\n--- RINGKASAN ANALITIK PREDIKSI MODEL UNTUK DASBOR IMK ---')
print(dashboard_summary.to_string(index=False))

# Simpan hasil olahan siap pakai untuk visualisasi antarmuka
df_results.to_csv('dl_model_predictions_for_ui.csv', index=False)
print('\n[SUCCESS] Data prediktif transparan berhasil disimpan ke "dl_model_predictions_for_ui.csv"')