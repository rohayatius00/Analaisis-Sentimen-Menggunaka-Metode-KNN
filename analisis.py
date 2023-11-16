import pandas as pd
import numpy as np
import nltk
import string
import re
print(" ")
print("=======================================================")
print(" ==== Analisis Sentimen Menggunakan Metode KNN =======")
print("=======================================================")
print(" ")
# Mengambil input dari pengguna
nama_file = input("Masukkan nama file CSV: ")

# Membaca file CSV
try:
    #membaca data
    df = pd.read_csv(nama_file)
    
    #penghapusan tabel yg tidak penting
    df = df.drop('Username', axis= 1)
    df = df.drop('Date', axis= 1)
    #CASEFOLDING - lower case & cleaning
    def cleaning(Text):
        Text = re.sub(r'http\S+', '', str(Text)) #menghapus URL 
        Text = re.sub(r"\d+", " ", str(Text)) #menghapus angka 
        Text = re.sub(r"\b[a-zA-Z]\b", "", str(Text)) #menghapus kata tunggal dalam teks yang disimpan dalam kolom `text`.
        Text = re.sub(r"[^\w\s]", " ", str(Text)) #menggantikan karakter non-alphanumerik dan non-spasi dalam teks yang disimpan dalam variabel `content` dengan spasi kosong.
        Text = re.sub(r'(.)\1+', r'\1\1', Text) #mengganti dua atau lebih karakter berulang dalam teks dengan hanya dua karakter yang berulang. ex: karakter berulang "eeeee" dalam teks, maka akan digantikan dengan "ee".
        Text = re.sub(r"\s+", " ", str(Text)) #menggantikan satu atau lebih spasi berturut-turut dalam teks
        Text = re.sub("#[A-Za-z0-9_]+","", Text)  #menghapus tanda pagar (#) dalam teks
        Text = re.sub("@[A-Za-z0-9_]+","", Text) #menghapus mention
        Text = re.sub(r'\s\s+', ' ', Text) #menggantikan dua atau lebih spasi berturut-turut dalam teks dengan satu spasi tunggal.
        Text = re.sub(r'^RT[\s]+', '', Text) #menghapus RT(retweet)
        Text = re.sub(r'^b[\s]+', '', Text) #menghapus spasi di awal teks
        Text = re.sub("[^a-z0-9]"," ", Text) # menghapus emotiocon
        return Text
    df['case_folding'] = df['Tweet'].apply(cleaning).str.lower()

    #TOKENIZING - membagi kalimat jadi perkata (dipisah)
    from nltk.tokenize import word_tokenize

    def tokenizing_text(Text):
        tokens = nltk.tokenize.word_tokenize(Text)
        return tokens

    df['tokenizing'] = df['case_folding'].apply(tokenizing_text) #menerapkan tokenizing ke data

    #Normalisasi-menormalisasikan kata yang non formal menjadi formal sesuai dengan kamus colloquial-indonesian-lexicon
    def normalization (Text):
        df_slang = pd.read_csv('colloquial-indonesian-lexicon.csv')
        dict_slang ={}
        for i in range(df_slang.shape[0]):
            dict_slang[df_slang["slang"][i]]=df_slang["formal"][i]

        drop_slang = []
        for teks in Text:
            normalisasi_teks = [dict_slang[word] if word in dict_slang.keys() else word for word in teks]
            drop_slang.append(normalisasi_teks)

        return drop_slang
    df['normalisasi'] = normalization(df['tokenizing'])

    #STOPWORD REMOVAL - menghapus kata sesuai dengan kamus indonesia
    from nltk.corpus import stopwords

    list_stopwords = stopwords.words('indonesian')

    list_stopwords.extend(['yg', 'dg', 'rt', 'dgn', 'ny', 'gt', 'klo',
                        'kalo', 'amp', 'biar', 'xad', 'xef',
                        'gak', 'xbc', 'krn', 'nya', 'nih', 'sih',
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                        'jd', 'jgn', 'sdh', 'xae', 'xa', 'xe', 'xa', 'xf', 'n', 't',
                        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                        '&', 'yah', 'no', 'je', 'xbb', 'xb', 'sch',
                        'injirrr', 'ah', 'oena', 'bu', 'eh', 'xac', 'anjir']) #tambahan kata

    list_stopwords = set(list_stopwords)

    def stopwords_removal(Text):
        return [word for word in Text if word not in list_stopwords]
    df['stopword_removal'] = df['normalisasi'].apply(stopwords_removal)

    #STEMMING
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()

    def stemming_text(tokens):
        hasil = [stemmer.stem(token) for token in tokens]
        return hasil

    df['stemming'] = df['stopword_removal'].apply(stemming_text) #menerapkan stemming ke data

    #buat stemming bebas dari kurung siku
    stemming = df[['stemming']]

    def fit_stemming(text):
        text = np.array(text)
        text = ' '.join(text)

        return text

    df['stemming'] = df['stemming'].apply(lambda x: fit_stemming(x))

    #menghapus kalimat duplikat dari kolom stemming
    df.drop_duplicates(subset = "stemming", keep = 'first', inplace = True)
    #simpan kedalam csv
    df.to_csv('PreProcessing-baru.csv', sep=',', index=False)
    #!pip install googletrans==3.1.0a0
    import pandas as pd
    import googletrans
    from googletrans import Translator
    translator = Translator()
    pd.set_option('max_colwidth', 300)
    #import modul yang dibuthkan labelin vader
    import nltk
    #nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    #masukan data hasil preprocessing
    df = pd.read_csv('PreProcessing-baru.csv')

    # cari bahasa
    lang_df = pd.DataFrame.from_dict(googletrans.LANGUAGES,  orient='index', columns=['Language'])

    # cari code indonesia, dan inggris
    lang_df[lang_df.Language.isin(['english', 'indonesian'])]
    #translate ke bahasa inggris
    translate = pd.DataFrame(df['stemming']) #labelin menggunakan kolom stemming/hasil akhir dari preprocessing data
    translate['english_stemming'] = translate['stemming'].apply(lambda x: translator.translate(x, src='id', dest='en').text)

    #proses pengskoran/pembobotan kata
    sid.polarity_scores(translate.loc[0]['english_stemming'])
    translate['scores'] = translate['english_stemming'].apply(lambda x : sid.polarity_scores(str(x)))

    #penjumlahan skor
    translate['compound']  = translate['scores'].apply(lambda score_dict: score_dict['compound'])


    #pelabelan skor dengan 3 kelas
    def condition(c):
        if c >= 0.05:
            return "positif"
        elif c <= -0.05:
            return "negatif"
        else:
            return "neutral"
    translate['sentimen'] = translate['compound'].apply(condition)

    #konversi sentiment ke polaritas
    def convert(polarity):
        if polarity == 'positif':
            return 1
        elif polarity == 'neutral':
            return 0
        else:
            return -1
    translate['polarity'] = translate['sentimen'].apply(convert)

    #penghapusan kolong yg tidak diperlukan 
    del(translate["english_stemming"])
    del(translate["scores"])

    #pemisahan data (tokenizing)
    #!pip install nltk
    import nltk
    #nltk.download('punkt')
    from nltk.tokenize import word_tokenize

    def tokenize_with_quotes(text):
        tokens = nltk.tokenize.word_tokenize(str(text))
        return tokens
    translate['tokenized_stemming'] = translate['stemming'].apply(tokenize_with_quotes) 
    #simpan kedalam csv
    translate.to_csv('labeling-baru.csv', sep=',', index=False)
    #panggil data
    df = pd.read_csv('labeling-baru.csv')
    print("-------------------------------------------------------")
    positif = (df['sentimen'].value_counts()['positif'] / df['sentimen'].count()) * 100
    neutral = (df['sentimen'].value_counts()['neutral'] / df['sentimen'].count()) * 100
    negatif = (df['sentimen'].value_counts()['negatif'] / df['sentimen'].count()) * 100
    print("Sentimen positif:", positif, "%")
    print("Sentimen netral :", neutral, "%")
    print("Sentimen negatif:", negatif, "%")
    print("-------------------------------------------------------")
    print(" ")

    #KLASIFIKASI KNN 
    #Set nilai X dan Y
    X = df['tokenized_stemming']
    y = df['polarity']

    #spliting data untuk data train dan data test
    from sklearn.model_selection import train_test_split
    #pengujian dengan perbandingan 10:90
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=105) #random_state 10% dari jml data = 0.1x 1046

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, classification_report

    # Inisialisasi vektorisasi TF-IDF dan mengubah data pelatihan
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # untuk menentukan nilai k yang akan dievaluasi
    k_values = range(1, 100)
    accuracy_terbaik = 0
    k_tinggi = 0

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)

        # melakukan validasi silang k-fold
        scores = cross_val_score(knn, X_train_tfidf, y_train, cv=5, scoring='accuracy')

        # Hitung akurasi rata-rata dari validasi silang
        avg_score = scores.mean()

        # memilih nilai k terbaik berdasarkan hasil evaluasi
        if avg_score > accuracy_terbaik:
            accuracy_terbaik = avg_score
            k_tinggi = k
    
    # klasifikasi  KNN dengan nilai k terbaik pada subset pelatihan
    knn = KNeighborsClassifier(n_neighbors=k_tinggi)
    knn.fit(X_train_tfidf, y_train)

    # Ubah data pengujian menggunakan vektorisasi TF-IDF yang telah dilatih
    X_test_tfidf = vectorizer.transform(X_test)

    # Uji model pada subset pengujian dan evaluasi
    y_pred = knn.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    # Cetak nilai k tertinggi dan akurasi pada subset pengujian
    print(f"Nilai k tertinggi: {k_tinggi}")
    print(f"Akurasi pengujian: {accuracy}")
    print("-------------------------------------------------------")
    #melihat hasil classification report pada Knn
    classification_rep = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(classification_rep)
    print("=======================================================")
    print(" ")

except FileNotFoundError:
    print("File tidak ditemukan.")
except Exception as e:
    print("Terjadi kesalahan:", e)
