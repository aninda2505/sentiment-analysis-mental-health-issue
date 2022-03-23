from ast import mod
from cgitb import text
from math import ceil
from operator import pos
import re
import string
from flask import Flask, app, render_template, request, url_for, flash
import pickle
from itsdangerous import exc
from nltk.util import pr
import tweepy
import csv
import nltk
from scipy.sparse import csr_matrix
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
import googletrans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import urllib.request
import pandas as pd
from googletrans import Translator
from textblob import TextBlob
from werkzeug.utils import redirect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from collections import Counter

app = Flask(__name__, static_folder="templates/static")

hasil_crawling =[]
hasiL_pre =[]
hasil_labelling=[]

app.config['SECRET_KEY'] = 'anin'
def prepropecossing_twitter():
    
    # Membuat File CSV
    file = open('templates/static/files/Data Preprocessing Twitter.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)

    hasiL_pre.clear()

    with open("templates/static/files/Data Mentah Twitter.csv", "r",encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter =',')
        hasil_labelling.clear()
        for row in readCSV:
        # proses clean
            #simbol, hastag, url
            clean = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", row[2]).split()) 

            #single-char
            clean = re.sub(r"\b[a-zA-Z]\b", "", clean)

            #number
            clean = re.sub("\d+", "", clean)
            
            # proses casefold
            casefold = clean.casefold()

            # proses tokenize
            tokenizing = nltk.tokenize.word_tokenize(casefold)

            # proses stop removal
            # mengambil data stop word dari library
            stop_factory = StopWordRemoverFactory().get_stop_words()
            # menambah stopword sendiri
            more_stop_word = [ 'apa', 'yg']
            # menggabungkan stopword library + milik sendiri
            data = stop_factory + more_stop_word
            
            dictionary = ArrayDictionary(data)
            str = StopWordRemover(dictionary)
            stop_wr = nltk.tokenize.word_tokenize(str.remove(casefold))

            
            # proses stemming
            kalimat = ' '.join(stop_wr)
            factory = StemmerFactory()
            # normalisasi
            
            # mamanggil fungsi stemming
            stemmer = factory.create_stemmer()
            stemming = stemmer.stem(kalimat)
            

            tweets =[row[1], row[2], clean, casefold, tokenizing, stop_wr,  stemming]
            hasiL_pre.append(tweets)

            writer.writerow(tweets)
            flash('Berhasil Preprocessing data ', 'preprocessing')


df= None
df2 = None
akurasi = 0

positive = 0
negative = 0
neutral = 0



def klasifikasi_data():
    global df
    global df2
    global akurasi
    global positive
    global neutral
    global negative
    # membca csv
    
    # data = pd.read_csv("templates/static/files/Data Labelling Twitter.csv",error_bad_lines=False)
    # tweet = data.iloc[:, 1]
    # label =  data.iloc[:, 2]

   
    # split data training dan testing
        # ambil data tweet dan sentimen
    tweet = []
    y = []

    with open("templates/static/files/Data Labelling Twitter.csv", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            tweet.append(row[1])
            y.append(row[2])
    
    x_train, x_test, y_train, y_test = train_test_split(tweet, y, test_size=0.2, random_state=42)


    vec = CountVectorizer()
    x = vec.fit_transform(x_train)
    x2 = vec.transform(x_test)
    # tfidf

    # tfidf
    tf_transform = TfidfTransformer().fit(x)
    x = tf_transform.transform(x)

    tf_transform_test = TfidfTransformer().fit(x2)
    x2 = tf_transform_test.transform(x2)
    # naive bayes
    clf = MultinomialNB()
    clf.fit(x, y_train)

    pickle.dump(vec, open('templates/static/files/vec.pkl', 'wb'))
    pickle.dump(tf_transform, open('templates/static/files/tfidf.pkl', 'wb'))
    pickle.dump(clf, open('templates/static/files/model.pkl', 'wb'))
    
    predict = clf.predict(x2)
    
    report = classification_report(y_test, predict, output_dict=True)
    # simpan ke csv
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv("templates/static/files/Hasil Klasifikasi.csv", index= True)

    # menyimpan tfidf
    
    # names = vec.get_feature_names()
    # df_tfidf = pd.DataFrame(data=csr_matrix.todense(x_train))
    # df_tfidf.index = y_train
    # df_tfidf.columns = names
    # df_tfidf.to_csv('templates/static/files/TFIDF TRAINING.csv')

    # df_tfidf_test = pd.DataFrame(data=csr_matrix.todense(x_test))
    # df_tfidf_test.index = y_test
    # df_tfidf_test.columns = names
    # df_tfidf_test.to_csv('templates/static/files/TFIDF TESTING.csv')

    fileFinal = open('templates/static/files/Data Hasil Pengujian.csv', 'w', newline='', encoding='utf-8')
    writerFinal = csv.writer(fileFinal)
    for i in range(len(predict)):
        data = [x_test[i],y_test[i],predict[i] ]
        writerFinal.writerow(data)

    print(len(predict), flush=True)
    #y_test : labelling-data testing
    #predict : hasil prediksi klasifikasi

    unique_label = np.unique([y_test, predict])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, predict, labels=unique_label), 
        index=['true:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
    )


    
    cmtx.to_csv("templates/static/files/Hasil Confusion Matrix.csv", index= True)

    df = pd.read_csv("templates/static/files/Hasil Confusion Matrix.csv", sep=",")
    df.rename( columns={'Unnamed: 0':''}, inplace=True )

    df2 = pd.read_csv("templates/static/files/Hasil Klasifikasi.csv", sep=",")
    df2.rename( columns={'Unnamed: 0':''}, inplace=True )

    akurasi = round(accuracy_score(y_test, predict)  * 100, 2)

    
    kalimat = ""

    for i in tweet:
        s =("".join(i))
        kalimat += s

    urllib.request.urlretrieve("https://firebasestorage.googleapis.com/v0/b/skripsi-29e1a.appspot.com/o/love.jpg?alt=media&token=1cc23af7-f275-4c05-9eee-d911b65741f8", 'love.jpg')
    mask = np.array(Image.open("love.jpg"))
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color="#a8b4e8" , mask = mask)
    wordcloud.generate(kalimat)
    plt.figure(figsize=(12,10))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.savefig('templates/static/files/wordcloud.png', transparent=True)

    plt.figure()
    # diagram
    numbers_list = y_test
    
    counter = dict((i, numbers_list.count(i)) for i in numbers_list)

    # cek apakah key ada
    isPositive = 'Positif' in counter.keys()
    isNegative = 'Negatif' in counter.keys()
    isNeutral = 'Netral' in counter.keys()
    
    # value 
    negatif2 = counter["Negatif"] if isNegative == True  else 0
    positif2 = counter["Positif"] if isPositive == True  else 0
    neutral2 = counter["Netral"] if isNeutral == True  else 0
    

    sizes = [positif2, neutral2, negatif2]
    labels = ['Positif', 'Netral', 'Negatif']
    plt.pie(sizes, labels=labels, autopct='%1.0f%%', shadow=True, textprops={'fontsize': 14}, colors=["#429EBD", "#053F5C", "#84A59D"])
    plt.savefig('templates/static/files/pie-chart.png', transparent=True)

    # menghitung persen
    total  = positif2 + neutral2 + negatif2
    positive = round((positif2 / total) * 100)
    neutral = round((neutral2/ total) * 100)
    negative = round((negatif2 / total) * 100)
        

    # diagram batang
    # creating the bar plot

    plt.figure()

    plt.hist(numbers_list, histtype='step')
    
    plt.xlabel("Tweet")
    plt.ylabel("Jumlah")
    plt.title("Presentase Sentimen Isu Kesehatan Mental")
    plt.savefig('templates/static/files/grafik.png', transparent=True)

            
def crawling_twitter(query, count, since, until):
    api_key = "5t4ZehrDahkfTqhQsM9dplmih"
    api_secret_key = "jAIPsxDYCtLg9AA9WYq6E5N1vBxCupo2EqS9xJKPr1MMxV9WCV"
    access_token = "726397221550194688-7omVPL8iy7sbexGjMnGBbPQLvLR8jzX"
    access_token_secret = "lCQnI0Fqs7PricXL0nsXjq2aoKuwFRuWCErPZve8XjIOM"


    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    filterKey = " -filter:retweets"

    # Membuat File CSV
    file = open('templates/static/files/Data Mentah Twitter.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)

    
    hasil_crawling.clear()

    for tweet in tweepy.Cursor(api.search, q=query + filterKey, lang='id', since=since, until=until, tweet_mode="extended").items(int(count)):
        tweet_properties = {}
        tweet_properties["tanggal_tweet"] = tweet.created_at
        tweet_properties["username"] = tweet.user.screen_name
        tweet_properties["tweet"] =  tweet.full_text.replace('\n', '')      
 
        # Menuliskan data ke csv
        tweets =[tweet.created_at, tweet.user.screen_name, tweet.full_text.replace('\n', '')]
        if tweet.retweet_count > 0:
            if tweet_properties not in hasil_crawling:
                hasil_crawling.append(tweets)
        else:
            hasil_crawling.append(tweets)

        writer.writerow(tweets)

def labelling_process():

    # Membuat File CSV
    file = open('templates/static/files/Data Labelling Twitter.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    translator = Translator()

    with open("templates/static/files/Data Preprocessing Twitter.csv", "r",encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter =',')
        hasil_labelling.clear()
        for row in readCSV:
            tweet = {}
            try:
                value = translator.translate(row[6], dest='en')
            except:
                print("Terjadi Error", flush=True)



            terjemahan = value.text
            data_label = TextBlob(terjemahan)


            if data_label.sentiment.polarity > 0.0 :
                tweet['sentiment'] = "Positif"
            elif data_label.sentiment.polarity == 0.0 :
                tweet['sentiment'] = "Netral"
            else : 
                tweet['sentiment'] = "Negatif"

            labelling = tweet['sentiment']
            tweets =[row[0], row[6], labelling]
            hasil_labelling.append(tweets)

            writer.writerow(tweets)

            flash('Success!!', 'labelling')
            
        

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/crawling',  methods = ['POST', 'GET'])
def crawling():
    if request.method == 'POST':
        if request.form.get('cleaned') == 'Cleaned':
            prepropecossing_twitter()
            return redirect(url_for('preprocessing'))
        if request.form.get('generate') == 'Generate Data':
            query = request.form.get('query')
            jumlah = request.form.get('jumlah')
            since = request.form.get('since')
            until = request.form.get('until')
            
            
            crawling_twitter(query, jumlah, since, until)
            return render_template('crawling.html', data=hasil_crawling)

    return render_template('crawling.html')




#untuk upload file
ALLOWED_EXTENSION = set(['csv'])

def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION
    

@app.route('/preprocessing_upload', methods=['GET', 'POST'])
def preprocessing_upload():
    if request.method == 'POST':
        if request.form.get('upload') == 'upload':
            file = request.files['uploadfile']
            
            if not allowed_files(file.filename):
                flash('Failed!!', 'upload_category')
                return render_template('preprocessing_upload.html', value=hasiL_pre)
            if file and allowed_files(file.filename):
                flash('Success!!', 'upload_category')
                file.save("templates/static/files/Data Mentah Twitter.csv")
                return render_template('preprocessing_upload.html')
    
        hasiL_pre.clear()
        if request.form.get('preprocessing') == 'Pre-processing Data':
            prepropecossing_twitter()
            return render_template('preprocessing_upload.html', value=hasiL_pre)
        if request.form.get('labelling') == 'Labelling':
            labelling_process()
            return redirect(url_for('labelling'))

    return render_template("preprocessing_upload.html")
    
@app.route('/preprocessing', methods = ['POST', 'GET'])
def preprocessing():
    if request.method == 'POST':
        if request.form.get('labelling') == 'Labelling':
            labelling_process()
            return redirect(url_for('labelling'))
            
    return render_template('preprocessing.html', value=hasiL_pre)

@app.route('/classification', methods=['POST', 'GET'])
def classification():
    if request.method == 'POST':
        if request.form.get('visualization') == 'visualization':
            klasifikasi_data()
            return redirect(url_for('visualisasi'))

    if akurasi == 0:
        return render_template('classification.html')
    else:
        return render_template('classification.html', accuracy=akurasi, tables=[df.to_html(classes='myTable', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='myTable', index=False, justify='left')], titles2=df2.columns.values)



@app.route('/classification_upload', methods=['POST', 'GET'])
def classification_upload():
    if request.method == 'POST':
        if request.form.get('classification') == 'Classification':
            klasifikasi_data()
            return render_template('classification_upload.html', accuracy=akurasi, tables=[df.to_html(classes='myTable', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='myTable', index=False, justify='left')], titles2=df2.columns.values)
        if request.form.get('visualization') == 'visualization':
            return redirect(url_for('visualisasi'))
        if request.form.get('upload') == 'upload':
            file = request.files['uploadfile']
            
            if not allowed_files(file.filename):
                flash('Failed!!', 'upload_category')
                return render_template('classification_upload.html')
            if file and allowed_files(file.filename):
                flash('Success!!', 'upload_category')
                file.save("templates/static/files/Data Labelling Twitter.csv")
                return render_template('classification_upload.html')
    
    if akurasi == 0:
        return render_template('classification_upload.html')
    else:
        return render_template('classification_upload.html', accuracy=akurasi, tables=[df.to_html(classes='myTable', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='myTable', index=False, justify='left')], titles2=df2.columns.values)





@app.route('/labelling', methods=['GET', 'POST'])
def labelling():
    if request.method == 'POST':
        if request.form.get('classification') == 'Classification':
            klasifikasi_data()
            return redirect(url_for('classification'))
    return render_template('labelling.html', value=hasil_labelling)

@app.route('/labelling_upload', methods=['GET', 'POST'])
def labelling_upload():
    if request.method == 'POST':
        if request.form.get('upload') == 'upload':
            file = request.files['uploadfile']
            
            if not allowed_files(file.filename):
                flash('Failed!!', 'upload_category')
                return render_template('labelling_upload.html', value=hasil_labelling)
            if file and allowed_files(file.filename):
                flash('Success!!', 'upload_category')
                file.save("templates/static/files/Data Preprocessing Twitter.csv")
                return render_template('labelling_upload.html')
    
        hasiL_pre.clear()
        if request.form.get('labelling') == 'Labelling':
            labelling_process()
            return render_template('labelling_upload.html', value=hasil_labelling)
        if request.form.get('classification') == 'Classification':
            klasifikasi_data()
            return redirect(url_for('classification'))


    return render_template('labelling_upload.html')

@app.route('/visualisasi')
def visualisasi():
    return render_template('visualisasi.html',pos=positive,neu=neutral, neg=negative)


hasil_predict = []
def model_predict():
    global positive
    global  neutral
    global negative
    # membca csv
    data = pd.read_csv("templates/static/files/Data Labelling Twitter untuk Model Predict.csv")
    tweet = data.iloc[:, 1]
    y =  data.iloc[:, 2]

   

    file = open('templates/static/files/Data Model Predict.csv', 'w', newline='', encoding='utf-8')
    with open('templates/static/files/model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    with open('templates/static/files/vec.pkl', 'rb') as h:
        vec = pickle.load(h)
   
    with open('templates/static/files/tfidf.pkl', 'rb') as t:
        tfidf = pickle.load(t)

    writer = csv.writer(file)
    for i, line in data.iterrows():
        isi = line[1]
        # # transform cvector & tfidf
        transform_cvec = vec.transform([isi])
        transform_tfidf = tfidf.transform(transform_cvec)
        
        # predict start
        predic_result = model.predict(transform_tfidf)
      

        data = [isi , predic_result[0]]
        hasil_predict.append(data)
        writer.writerow(data)
        
    
    kalimat = ""

    for i in tweet.tolist():
        s =("".join(i))
        kalimat += s

    urllib.request.urlretrieve("https://firebasestorage.googleapis.com/v0/b/skripsi-29e1a.appspot.com/o/love.jpg?alt=media&token=1cc23af7-f275-4c05-9eee-d911b65741f8", 'love.jpg')
    mask = np.array(Image.open("love.jpg"))
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color="#a8b4e8" , mask = mask)
    wordcloud.generate(kalimat)
    plt.figure(figsize=(12,10))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.savefig('templates/static/files/wordcloud.png', transparent=True)

    plt.figure()
    # diagram
    numbers_list = y.tolist()
    counter = dict((i, numbers_list.count(i)) for i in numbers_list)

    # cek apakah key ada
    isPositive = 'Positif' in counter.keys()
    isNegative = 'Negatif' in counter.keys()
    isNeutral = 'Netral' in counter.keys()
    
    # value 
    negatif2 = counter["Negatif"] if isNegative == True  else 0
    positif2 = counter["Positif"] if isPositive == True  else 0
    neutral2 = counter["Netral"] if isNeutral == True  else 0
    

    sizes = [positif2, neutral2, negatif2]
    labels = ['Positif', 'Netral', 'Negatif']
    plt.pie(sizes, labels=labels, autopct='%1.0f%%', shadow=True, textprops={'fontsize': 14}, colors=["green", "white", "red"])
    plt.savefig('templates/static/files/pie-chart.png', transparent=True)

    # menghitung persen
    total  = positif2 + neutral2 + negatif2
    positive = round((positif2 / total) * 100)
    neutral = round((neutral2/ total) * 100)
    negative = round((negatif2 / total) * 100)


    

    # diagram batang
    # creating the bar plot

    plt.figure()

    plt.hist(numbers_list, histtype='step')
    
    plt.xlabel("Tweet")
    plt.ylabel("Jumlah")
    plt.title("Presentase Sentimen Isu Kesehatan Mental")
    plt.savefig('templates/static/files/grafik.png', transparent=True)
        
        
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        if request.form.get('upload') == 'upload':
            file = request.files['uploadfile']
            
            if not allowed_files(file.filename):
                flash('Failed!!', 'upload_category')
                return render_template('predict.html', value=hasil_labelling)
            if file and allowed_files(file.filename):
                flash('Success!!', 'upload_category')
                file.save("templates/static/files/Data Labelling Twitter untuk Model Predict.csv")
                return render_template('predict.html')

        if request.form.get('predict') == 'Model Predict':
        
            model_predict()
            return redirect(url_for('predict'))
        if request.form.get('visualization') == 'visualization':
            return redirect(url_for('visualisasi'))
   
    return render_template('predict.html', value=hasil_predict)





@app.route('/aboutme')
def aboutme():
    return render_template('aboutme.html')




if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=3000, debug=True)