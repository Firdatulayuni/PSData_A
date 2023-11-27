import streamlit as st
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from io import StringIO
import pickle
from pickle import dump
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

st.title("PREDIKSI PENYAKIT HEPATITIS C")
st.write("Nama : Firdatul A'yuni")
st.write('NIM : 2104111000144')
st.write('Kelas: Proyek Sains Data A')
data_hcv, explore, processing, prediksi  = st.tabs(["Deskripsi Dataset HCV", "Eksplorasi Data", "Processing Dataset", "Prediksi Hasil Lab Donor Darah"])


with data_hcv:
    st.write("### Deskripsi Dataset")
    st.write("Dataset HCV ini merupakan data klasifikasi yang digunakan untuk memprediksi kategori atau status pasien berdasarkan 12 atribut klinis yang diambil dari 615 pasien melalui sampel darah, dengan target utama berfokus pada kategori penyakit Hepatitis C dan perkembangannya. Ini membantu dalam diagnosis dini dan pengelolaan penyakit Hepatitis C serta pemantauan kemungkinan perkembangan penyakit ini pada pasien.")
    st.write("Dataset ini diambil dari uci.edu: https://archive.ics.uci.edu/dataset/571/hcv+data")
    st.write("### Tujuan Dataset ###")
    st.write("Untuk klasifikasi kategori atau status pasien berdasarkan berbagai fitur atau atribut klinis. Fokus utama adalah pada kategori penyakit hepatitis, termasuk perkembangannya ke tahap-tahap yang lebih serius seperti fibrosis dan sirosis hati.")
    st.write("### Jumlah Data")
    st.write("Dataset ini memiliki jumlah data sebanyak 615 data dengan 5 kelas yaitu:")
    kelas = """
    - Blood Donors: 526 data
    - Fitur: 12
    - Suspect Blood Donors: 24 data
    - Hepatitis C: 20 data
    - Fibrosis: 12 data
    - Chirrosis: 7 data
    - Missing Value: 26"""
    st.markdown(kelas)
    st.write("### Penjelasan Target ###")
    target = """ 
    - Blood Donors (Tidak terifeksi Hepatitis C)

    Kategori ini mencakup pasien yang tidak terinfeksi Hepatitis C

    - Suspect Blood Donors

    Merujuk pada pasien yang dalam konteks penyakit Hepatitis C menunjukkan tanda-tanda atau karakteristik tertentu yang menimbulkan kecurigaan terhadap kemungkinan infeksi, namun belum sepenuhnya dikonfirmasi sebagai pasien Hepatitis C

    - Hepatitis C (Pasien dengan infeksi Hepatitis C tanpa perkembangan lebih lanjut)

    Pasien dalam kategori ini telah terinfeksi Hepatitis C, tetapi infeksi tersebut belum berkembang menjadi tahap lebih lanjut seperti fibrosis atau sirosis hati. Ini mencakup pasien yang masih dalam tahap awal infeksi Hepatitis C

    - Fibrosis (Hepatitis C menjadi fibrosis hati)

    Fibrosis adalah tahap dimana jaringan parut mulai terbentuk di hati sebagai respons terhadap kerusakan. Kategori ini mencakup pasien dengan infeksi Hepatitis C yang telah berkembang menjadi tahap fibrosis

    - Cirrhosis (Hepatitis C menjadi sirosis hati)

    Sirosis hati adalah tahap yang lebih parah dari fibrosis, di mana jaringan parut yang luas menggantikan jaringan hati yang sehat. Pasien dalam kategori ini telah mengalami perkembangan penyakit menjadi sirosis hati, yang dapat menyebabkan gangguan fungsi hati yang signifikan

    """
    st.markdown(target)
    st.write("### Jumlah Fitur  ###")
    st.write("Pada dataset yang HCV diambil dari UCI terdapat 12 fitur yaitu Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT dan 1 Label yaitu Category. Data diperoleh dari hasil laboratorium donor darah dan pasien Hepatitis C serta nilai demografi seperti usia.")
    st.write("### Penjelasan Fitur ###")
    # Teks yang ingin ditampilkan
    text = """
    **Age**

    Usia pasien saat pengambilan sampel darah. Penyakit hepatitis, terutama Hepatitis C, cenderung lebih sering terjadi pada orang yang lebih tua. Data pada fitur ini memiliki tipe data numerik.

    **Sex**

    Jenis kelamin pasien, beberapa studi menunjukkan bahwa pria memiliki risiko lebih tinggi terkena fibrosis atau sirosis hati dibandingkan wanita dalam kasus Hepatitis C. Data pada fitur ini memiliki tipe data string.

    **ALB (Albumin)**

    Albumin adalah protein yang diproduksi oleh hati dan merupakan komponen utama dalam serum protein. Penurunan konsentrasi albumin dapat terjadi pada pasien dengan gangguan hati yang parah, termasuk sirosis hati. Fitur ini memiliki nilai kosong sebanyak 1 data. Data pada fitur ini memiliki tipe data numerik.

    **ALP (Alkaline Phosphatase)**

    Alkaline Phosphatase adalah enzim yang terdapat dalam sel-sel hati dan tulang, konsentrasi alkaline phosphatase yang tinggi dalam darah dapat menjadi tanda adanya masalah pada hati atau saluran empedu. Ini dapat terkait dengan penyakit hati seperti sirosis. Fitur ini memiliki nilai kosong sebanyak 18 data. Data pada fitur ini memiliki tipe data numerik.
   
    **ALT (Alanine Aminotransferase)**

     ALT adalah enzim yang terdapat dalam sel-sel hati, konsentrasi alanine aminotransferase yang tinggi adalah indikasi kerusakan sel hati. Ini bisa menjadi tanda penyakit hati, termasuk Hepatitis C. Fitur ini memiliki nilai kosong sebanyak 1 data. Data pada fitur ini memiliki tipe data numerik.

    **AST (Aspartate Aminotransferase)**

     AST adalah enzim yang terdapat dalam sel-sel hati, jantung, otot, dan organ lainnya. Konsentrasi aspartate aminotransferase yang tinggi juga dapat mencerminkan kerusakan hati. Data pada fitur ini memiliki tipe data numerik.

    **BIL (Billirubin)**

    Bilirubin adalah pigmen kuning yang terbentuk ketika sel darah merah dihancurkan, konsentrasi bilirubin yang tinggi dalam darah bisa menjadi tanda masalah hati, terutama pada sirosis hati. Data pada fitur ini memiliki tipe data numerik.
 
    **CHE (Cholinesterase)**

    holinesterase adalah enzim yang terlibat dalam pemecahan asetilkolin. Konsentrasi cholinesterase yang rendah dalam darah dapat terjadi pada pasien dengan penyakit hati yang parah, seperti sirosis hati. Data pada fitur ini memiliki tipe data numerik.

    **CHOL (Kolesterol)**

    Cholesterol adalah lemak yang penting untuk pembentukan membran sel dan produksi hormon. Kolesterol rendah dapat terjadi pada pasien dengan sirosis hati karena hati yang rusak mungkin tidak mampu memetabolisme kolesterol dengan baik. Fitur ini memiliki nilai kosong sebanyak 10 data. Data pada fitur ini memiliki tipe data numerik. 

    **CREA (Kreatinin)**

    Creatinine adalah produk buangan metabolisme otot yang diekskresikan melalui ginjal. Kadar creatinine dalam darah dapat memberikan informasi tentang fungsi ginjal. Data pada fitur ini memiliki tipe data numerik.

    **GGT (Gamma-Glutamyl Transferase)**

    GGT adalah enzim yang terdapat dalam hati dan saluran empedu. Konsentrasi gamma-glutamyl transferase yang tinggi dapat menjadi indikasi kerusakan hati. Ini adalah tanda umum penyakit hati, termasuk Hepatitis C. Data pada fitur ini memiliki tipe data numerik.

    **PROT (Protein Total)**

     Total Protein mencakup jumlah albumin dan globulin dalam serum. Penurunan konsentrasi protein total dapat terjadi pada sirosis hati. Fitur ini memiliki nilai kosong sebanyak 1 data. Data pada fitur ini memiliki tipe data numerik.
    """
    st.markdown(text)
    st.write("### Source Aplikasi di Colaboratory :")
    st.write("https://colab.research.google.com/drive/1QBN0L3cBtQUtYJD_Z6blpn2oepPKCTbU?usp=sharing")
    
with explore:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load data
    df = pd.read_csv('https://raw.githubusercontent.com/Firdatulayuni/PSData_A/main/hcv.csv')

    # Statistik deskriptif
    descriptive_stats = df.describe()
    st.write("### Statistik Deskriptif: ###")
    st.dataframe(descriptive_stats)

   # Setel warna untuk setiap kelas
    palette_colors = sns.color_palette("Set2", n_colors=5)

    # Grafik distribusi kategori
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Category', data=df, palette=palette_colors)
    st.write('### Distribusi Kategori ###')
    st.pyplot(plt)

    # Grafik distribusi umur (Age) berdasarkan kategori
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Category', kde=True, bins=20)
    st.write('### Distribusi Umur berdasarkan Kategori ###')
    st.pyplot(plt)
with processing:
    st.write("### Dataset Preprocessing")
    st.write("Preprosesing pada data berikut adalah:")
    preprocessing = """ 
    1. Membaca Data
    
    Tahap ini mengambil data dari file CSV dengan URL github dan menampilkannya menggunakan Pandas DataFrame

    2. Menghapus missing value

    Penghapusan nilai yang hilang ini menggunakan dropna, metode ini merupakan metode dari DataFrame yang digunakan untuk menghapus baris dengan nilai yang hilang. Metode ini secara default akan menghapus semua baris yang mengandung setidaknya satu nilai yang hilang (NaN).
    Karena pada dataset ini memiliki missing values maka dilakukan penghapusan data (baris). Sebelum dilakukan penghapusan data, data terdiri dari 615 baris, setelah dilakukan penghapusan data tersisa sebanyak 589 baris. Maka data berkurang sebanyak 4,24% (kekurangan data/data awal x 100%).

    3. Mengubah fitur Sex menjadi numerik

    Fitur Sex merupakan fitur yang berupa string 'm' dan 'f',  maka diubah menjadi nilai biner 0 dan 1 yang dapat lebih mudah digunakan dalam analisis atau pemodelan, karena nilai biner seringkali lebih mudah diolah.

    4. Normalisasi

    Normalisasi yang digunakan adalah StandarScaler dengan rumus:
    
    y = (x – mean)/ deviasi_standar.
    
    Data akan diubah sedemikian rupa sehingga rata-ratanya menjadi 0 dan deviasi standarnya menjadi 1. Hal ini berguna dalam analisis statistik dan pemodelan statistik karena memungkinkan perbandingan yang lebih baik antara berbagai fitur atau variabel dalam dataset, terutama jika variabel-variabel tersebut memiliki skala yang berbeda"""

    st.markdown(preprocessing)

    df = pd.read_csv('https://raw.githubusercontent.com/Firdatulayuni/PSData_A/main/hcv.csv')
    st.dataframe(df)

    # Menghapus baris dengan nilai yang hilang
    df.dropna(inplace=True)

    # Mengubah fitur "Sex" menjadi nilai biner (0 dan 1)
    df['Sex'] = df['Sex'].replace(['m', 'f'], [0, 1])

    df.to_csv('data_cleaned.csv', index=False)
    cleaned_df = pd.read_csv('data_cleaned.csv')

    # Split data menjadi set pelatihan dan pengujian
    X = cleaned_df.drop(columns=['Category'])  # Pastikan 'Class' adalah kolom target
    y = cleaned_df['Category']

    st.write("### Tabel Baru")
    st.dataframe(cleaned_df)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

    # Identifikasi kolom yang merupakan data biner (jika ada)
    binary_columns = ['Sex']  # Gantilah dengan kolom-kolom biner Anda

    # Define and fit the scaler on the training dataset
    scaler = StandardScaler()
    
    # Hanya fit scaler pada fitur yang bukan data biner
    non_binary_columns = [col for col in X_train.columns if col not in binary_columns]
    scaler.fit(X_train[non_binary_columns])

    # Save the scaler using pickle
    scaler_file_path = r'C:\psd_project_2\scaler.pkl'
    with open(scaler_file_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    # Transformasi data pelatihan (hanya pada fitur non-biner)
    X_train_scaled = X_train.copy()
    X_train_scaled[non_binary_columns] = scaler.transform(X_train[non_binary_columns])
   
    # Tampilkan hasil normalisasi
    normalized_data_train = pd.DataFrame(X_train_scaled, columns=X.columns)

    # Menyeimbangkan data menggunakan RandomOverSampler
    ros = RandomOverSampler(random_state=1)
    X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)

    # Simpan data yang telah di-resample
    resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_data['Category'] = y_resampled
    resampled_data.to_csv('balanced_data_train.csv', index=False)

    # Tampilkan data yang telah di-resample
    balanced_data_train = pd.read_csv('balanced_data_train.csv')
    st.write("### Balanced Data ###")
    balanced_data_train

    with open('scaler.pkl', 'rb') as standarisasi:
        loadscal= pickle.load(standarisasi)

    # Normalisasi data pengujian (X_test) menggunakan scaler yang telah dimuat
    X_test_scaled = X_test.copy()
    X_test_scaled[non_binary_columns] = loadscal.transform(X_test[non_binary_columns])

    # Tampilkan hasil normalisasi
    normalized_data_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    st.write("### Tabel HCV Setelah Normalisasi")
    st.dataframe(normalized_data_test)


    st.write("### SVM ###")
    st.write("Support Vector Machine (SVM) merupakan salah satu metode machine learning dengan pendekatan supervised learning yang paling populer digunakan.Metode SVM mengkelaskan data baru, mengelompokan data-data dengan memisahkannya berdasarkan hyperplane dengan ruang N-dimensi (N - jumlah fitur) yang secara jelas mengklasifikasikan titik data.")
    st.write("Pada modelling ini kernel yang digunakan adalah svm kernel linear. Kernel linear adalah fungsi kernel yang paling sederhana. Kernel linear digunakan ketika data yang dianalisis sudah terpisah secara linear. Kernel linear cocok ketika terdapat banyak fitur dikarenakan pemetaan ke ruang dimensi yang lebih tinggi tidak benar-benar meningkatkan kinerja.")

    # Pelatihan model SVM
    svm_model = SVC(kernel='linear', C=1.0, random_state=42, class_weight='balanced')
    svm_model.fit(X_train_scaled, y_train)

    # Simpan model SVM ke dalam file pickle
    svm_model_file_path = 'svm_model.pkl'
    with open(svm_model_file_path, 'wb') as svm_model_file:
        pickle.dump(svm_model, svm_model_file)

    # Memuat model SVM dari file pickle
    with open(svm_model_file_path, 'rb') as svm_model_file:
        loaded_svm_model = pickle.load(svm_model_file)

    # Prediksi menggunakan model SVM
    y_pred_svm = loaded_svm_model.predict(X_test_scaled)

    # Membuat DataFrame untuk menampilkan nilai sebenarnya dan hasil prediksi
    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_svm})

    st.write("### Tabel Prediksi ###")
    st.dataframe(result_df)

    # Menghitung akurasi SVM
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    st.write("Akurasi SVM:", svm_accuracy)

    # Menghitung jumlah prediksi yang sesuai dan tidak sesuai
    correct_predictions = result_df['Actual'] == result_df['Predicted']
    incorrect_predictions = ~correct_predictions

    # Menampilkan jumlah prediksi yang sesuai dan tidak sesuai
    st.write("Jumlah prediksi yang sesuai:", correct_predictions.sum())
    st.write("Jumlah prediksi yang tidak sesuai:", incorrect_predictions.sum())


with prediksi:
    st.write("### Prediksi Hasil Lab Donor Darah")
    st.write("Masukkan nilai fitur berikut untuk melakukan prediksi:")

    # Membuat input fields untuk fitur
    age = st.number_input('Age:', value=30)
    sex = st.radio('Sex:', ['1', '0'])
    alb = st.number_input('ALB:', value=3.0)
    alp = st.number_input('ALP:', value=80)
    alt = st.number_input('ALT:', value=40)
    ast = st.number_input('AST:', value=30)
    bil = st.number_input('BIL:', value=0.3)
    che = st.number_input('CHE:', value=120)
    chol = st.number_input('CHOL:', value=200)
    crea = st.number_input('CREA:', value=0.7)
    ggt = st.number_input('GGT:', value=30)
    prot = st.number_input('PROT:', value=70)
    tombol_prediksi = st.button('Prediksi')
    output = st.empty()

    # Membuat fungsi untuk melakukan prediksi
    def prediksi():
        # Membuat DataFrame dengan nama fitur dan nilai
        input_features = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ALB': [alb],
            'ALP': [alp],
            'ALT': [alt],
            'AST': [ast],
            'BIL': [bil],
            'CHE': [che],
            'CHOL': [chol],
            'CREA': [crea],
            'GGT': [ggt],
            'PROT': [prot]
        })

        # Normalisasi input features
        #input_features_scaled = loadscal.transform(input_features)

        # define scaler
        scaler = StandardScaler()

        # Hanya fit scaler pada fitur yang bukan data biner
        non_binary_columns = [col for col in input_features.columns if col not in binary_columns]
        scaler.fit(input_features[non_binary_columns])

        # Transformasi data pelatihan (hanya pada fitur non-biner)
        input_features_scaled = input_features.copy()
        input_features_scaled[non_binary_columns] = scaler.transform(input_features[non_binary_columns])

        # Melakukan prediksi dengan model KNN
        hasil_prediksi = loaded_svm_model.predict(input_features_scaled)[0]

        output.markdown("Hasil Prediksi Kategori:")
        output.write(hasil_prediksi)

    if tombol_prediksi:
        prediksi()
