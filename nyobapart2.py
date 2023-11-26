import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats
import librosa

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier

st.title("UTS PROYEK SAINS DATA")
st.write("Nama : Firdatul A'yuni")
st.write('NIM : 2104111000144')
st.write('Proyek Sains Data A')

data_abalone, processing, prediksi  = st.columns(3)

with data_abalone:
    st.write("### Deskripsi Dataset :")
    st.write("Usia abalon ditentukan dengan memotong cangkang melalui kerucut, mewarnainya, dan menghitung jumlah cincin melalui mikroskop -- sebuah tugas yang membosankan dan memakan waktu.")
    st.write("Pengukuran lain, yang lebih mudah diperoleh, digunakan untuk memperkirakan umur. Informasi lebih lanjut, seperti pola cuaca dan lokasi (sehingga ketersediaan makanan) mungkin diperlukan untuk mengatasi masalah ini.")
    st.write("Dataset ini diambil dari usi.edu: https://archive.ics.uci.edu/dataset/1/abalone")
    st.write("### Source Aplikasi di Colaboratory :")
    st.write("https://colab.research.google.com/drive/1lieFwq2XPBnQwzGR6oyIfGtqI9oP8xIl?usp=sharing")

with processing:
    st.write("### Dataset Processing")
    st.write("Pada dataset yang diambil dari UCI terdapat 8 fitur yaitu Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight")
    st.write("Berikut adalah tabel dataset Abalone: ")
    df = pd.read_csv('https://raw.githubusercontent.com/Firdatulayuni/PSD_PROJECT2/main/abalone2.csv')
    st.dataframe(df)

    # Memisahkan fitur (X) dan label (y)
    X = df.drop(['Rings'], axis=1)
    y = df['Rings']

    # One-hot encoding untuk fitur "Sex"
    X_encoded = pd.get_dummies(X, columns=['Sex'], prefix=['Sex'])

    # Memisahkan fitur numerik dan kategorikal setelah enkoding
    X_numeric = X_encoded.select_dtypes(include=[np.number])
    X_categorical = X_encoded.select_dtypes(exclude=[np.number])

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=1, test_size=0.2)

    # Define and fit the scaler on the training dataset for numeric features
    scaler = StandardScaler()
    scaler.fit(X_numeric)

    # Simpan scaler menggunakan pickle
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    # Load the scaler using pickle
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Transformasi fitur numerik pada data pelatihan dan pengujian
    X_numeric_train_scaled = scaler.transform(X_numeric.loc[X_train.index])
    X_numeric_test_scaled = scaler.transform(X_numeric.loc[X_test.index])

    # Mengubah X_numeric_train_scaled dan X_numeric_test_scaled menjadi DataFrames
    X_numeric_train_scaled_df = pd.DataFrame(X_numeric_train_scaled, columns=X_numeric.columns)
    X_numeric_test_scaled_df = pd.DataFrame(X_numeric_test_scaled, columns=X_numeric.columns)

    # Menggabungkan fitur numerik yang sudah dinormalisasi dengan fitur kategorikal
    X_train_scaled = pd.concat([X_numeric_train_scaled_df, X_categorical.loc[X_train.index]], axis=1)
    X_test_scaled = pd.concat([X_numeric_test_scaled_df, X_categorical.loc[X_test.index]], axis=1)

    st.write("### Tabel Abalone Setelah Normalisasi")
    st.dataframe(X_test_scaled)

    st.write("### K-Nearest Neighbor")
    # Hitung akurasi KNN dari k = 1 hingga 30
    K = 20
    acc = np.zeros((K - 1))

    for n in range(1, K, 2):
        knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean").fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        acc[n - 1] = accuracy_score(y_test, y_pred)

    best_accuracy = acc.max()
    best_k = acc.argmax() + 1

    # Tampilkan akurasi terbaik dan nilai k
    st.write('Akurasi terbaik adalah', best_accuracy, 'dengan nilai k =', best_k)

    # Simpan model KNN terbaik
    best_knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    best_knn.fit(X_train_scaled, y_train)

    # Save the best KNN model using pickle
    with open('KNN_model.pkl', 'wb') as model_file:
        pickle.dump(best_knn, model_file)

    with open('KNN_model.pkl', 'rb') as knn_model:
        load_knn = pickle.load(knn_model)

    y_pred = load_knn.predict(X_test_scaled)

    # Hitung dan tampilkan akurasi KNN
    st.write('Akurasi KNN:')
    accuracy = accuracy_score(y_test, y_pred)
    st.write(accuracy)

    # Hitung prediksi label KNN
    knn_predictions = load_knn.predict(X_test_scaled)

    # Simpan hasil prediksi KNN ke dalam DataFrame
    knn_results_df = pd.DataFrame({'Actual Label': y_test, 'Predicted Label (KNN)': knn_predictions})

    # Tampilkan tabel prediksi KNN
    st.write("### Tabel Prediksi Label KNN")
    st.dataframe(knn_results_df)

    st.write("### PCA")
    # Lakukan reduksi PCA
    sklearn_pca = sklearnPCA(n_components=5)
    X_train_pca = sklearn_pca.fit_transform(X_train_scaled)
    
    st.write("Principal Components 5:")
    st.write(X_train_pca)

    # Save the PCA model
    with open('PCA5.pkl', 'wb') as pca_model_file:
        pickle.dump(sklearn_pca, pca_model_file)

    # Load the PCA model
    with open('PCA5.pkl', 'rb') as pca_model:
        loadpca = pickle.load(pca_model)

    # Transform test data using the loaded PCA model
    X_test_pca = loadpca.transform(X_test_scaled)

    # Continue with KNN and evaluation as needed
    K = 20
    acc_pca = np.zeros((K - 1))
    for n in range(1, K, 2):
        knn_pca = KNeighborsClassifier(n_neighbors=n, metric="euclidean").fit(X_train_pca, y_train)
        y_pred_pca = knn_pca.predict(X_test_pca)
        acc_pca[n - 1] = accuracy_score(y_test, y_pred_pca)

    best_accuracy_pca = acc_pca.max()
    best_k_pca = acc_pca.argmax() + 1

    # Tampilkan akurasi terbaik dan nilai k dengan PCA
    st.write('Akurasi KNN terbaik dengan PCA adalah', best_accuracy_pca, 'dengan nilai k =', best_k_pca+1)

    # Hitung prediksi label KNN setelah PCA
    knn_pca_predictions = knn_pca.predict(X_test_pca)

    # Simpan hasil prediksi KNN setelah PCA ke dalam DataFrame
    knn_pca_results_df = pd.DataFrame({'Actual Label': y_test, 'Predicted Label (KNN with PCA)': knn_pca_predictions})

    # Tampilkan tabel prediksi KNN setelah PCA
    st.write("Tabel Prediksi Label KNN dengan PCA")
    st.dataframe(knn_pca_results_df)

    # GAUSSIANNB
    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train)

    # Simpan model Gaussian Naive Bayes ke dalam file pkl
    with open('gnb_model.pkl', 'wb') as gnb_model_file:
        pickle.dump(gnb, gnb_model_file)

    with open('gnb_model.pkl', 'rb') as gnb_model:
        load_gnb = pickle.load(gnb_model)

    st.write("### Gaussian Naive Bayes")
    # Melakukan prediksi dengan model Gaussian Naive Bayes
    y_pred_gnb = load_gnb.predict(X_test_scaled)

    # Hitung dan tampilkan akurasi Gaussian Naive Bayes
    gnb_accuracy = accuracy_score(y_test, y_pred_gnb)
    st.write('Akurasi Gaussian Naive Bayes:')
    st.write(gnb_accuracy)

    # Simpan hasil prediksi Gaussian Naive Bayes ke dalam DataFrame
    gnb_results_df = pd.DataFrame({'Actual Label': y_test, 'Predicted Label (Gaussian Naive Bayes)': y_pred_gnb})

    # Tampilkan tabel prediksi Gaussian Naive Bayes
    st.write("Tabel Prediksi Label Gaussian Naive Bayes")
    st.dataframe(gnb_results_df)

    # DECISSION TREE
    # Membuat model Decision Tree
    st.write("### Decision Tree")
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train_scaled, y_train)

    # Simpan model Decision Tree ke dalam file pkl
    with open('decision_tree_model.pkl', 'wb') as decision_tree_model_file:
        pickle.dump(decision_tree, decision_tree_model_file)

    with open('decision_tree_model.pkl', 'rb') as decision_tree_model:
        load_decision_tree = pickle.load(decision_tree_model)

    # Melakukan prediksi dengan model Decision Tree
    y_pred_decision_tree = load_decision_tree.predict(X_test_scaled)

    # Hitung dan tampilkan akurasi Decision Tree
    decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)
    st.write('Akurasi Decision Tree :')
    st.write(decision_tree_accuracy)

    # Simpan hasil prediksi Decision Tree ke dalam DataFrame
    decision_tree_results_df = pd.DataFrame({'Actual Label': y_test, 'Predicted Label (Decision Tree)': y_pred_decision_tree})

    # Tampilkan tabel prediksi Decision Tree
    st.write("Tabel Prediksi Label Decision Tree")
    st.dataframe(decision_tree_results_df)

with prediksi:
    # Fungsi untuk menghitung fitur audio
    def extract_features(audio_path):
        y, sr = librosa.load(audio_path)

        # Fitur 1: Mean
        mean = np.mean(y)

        # Fitur 2: Median
