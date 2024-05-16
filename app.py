from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from urllib.parse import urlparse
import ipaddress
import re
from joblib import load
import label_data

app = Flask(__name__, template_folder='templates')

# Load models
modelLSTM = tf.keras.models.load_model('LSTM/model_LSTM.h5')
modelbiLSTM = tf.keras.models.load_model('biLSTM/model_biLSTM.h5')
modelSVM = load('SVM/svm_reduced.joblib')
modelDT = load('Decision Tree/tree_reduced.joblib')
modelMLP = load('MLP/mlp_reduced.joblib')
modelRandomForest = load('Random Forest/forest_reduced.joblib')

# Shortening services regex
shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"

def getDomain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain

def havingIP(url):
    try:
        ipaddress.ip_address(url)
        return 1
    except:
        return 0

def haveAtSign(url):
    return 1 if "@" in url else 0

def getLength(url):
    return 0 if len(url) < 54 else 1

def getDepth(url):
    s = urlparse(url).path.split('/')
    return sum(len(part) != 0 for part in s)

def redirection(url):
    pos = url.rfind('//')
    return 1 if pos > 6 and (pos > 7 or pos == 7) else 0

def httpDomain(url):
    return 1 if 'https' in urlparse(url).netloc else 0

def tinyURL(url):
    match = re.search(shortening_services, url)
    return 1 if match else 0

def prefixSuffix(url):
    return 1 if '-' in urlparse(url).netloc else 0

def preprocess_url_ML(url):
    url =  getDomain(url)
    features = [
        havingIP(url),
        haveAtSign(url),
        getLength(url),
        getDepth(url),
        redirection(url),
        httpDomain(url),
        tinyURL(url),
        prefixSuffix(url),
    ]
    return features


# vocab = sorted(set("".join(X)), reverse=True)
# # Inserting a space at index 0, since it is not used in url and will be used for padding the examples.
# vocab.insert(0, " ")
# vocab_size = len(vocab)
# char2idx = {u:i for i, u in enumerate(vocab)}

# def text_to_int(text):
#   return np.array([char2idx[c] for c in text])

def preprocess_url_DL(url):
    url = url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
    urlz = label_data.main()

    samples = list(urlz.keys())

    maxlen = 40
    max_words = 20000

    # encoded_text = sequence.pad_sequences([text_to_int(url)], max_seq_len)
    # result = model.predict(encoded_text)

    tokenizer = Tokenizer(num_words=max_words, char_level=True)
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(url)
    domain_name = pad_sequences(sequences, maxlen=maxlen)

    return domain_name

def predict_result(model, url_prepped_DL, url_prepped_ML):
    if model == modelLSTM or model == modelbiLSTM:
        prediction = model.predict(url_prepped_DL)
        result = "URL is phishing." if prediction[0][0] > 0.5 else "URL is  NOT phishing."
    else:
        prediction = model.predict([url_prepped_ML])
        result = "PHISHING" if prediction[0] == 1 else "NOT-PHISHING"

    return result, float(prediction[0])

@app.route('/')
def index():
    return render_template('index.html', result=None, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        url = request_data["url"]
        selected_model = request_data["selected_model"]

        url_prepped_DL = preprocess_url_DL(url)
        url_prepped_ML = preprocess_url_ML(url)

        if selected_model == "LSTM":
            result, prediction = predict_result(modelLSTM, url_prepped_DL, None)
        elif selected_model == "biLSTM":
            result, prediction = predict_result(modelbiLSTM, url_prepped_DL, None)
        elif selected_model in ['SVM', 'Decision Tree', 'mlp', 'rf']:
            model = {
                'SVM': modelSVM,
                'Decision Tree': modelDT,
                'mlp': modelMLP,
                'rf': modelRandomForest
            }[selected_model]
            result, prediction = predict_result(model, None, url_prepped_ML)
        else:
            return jsonify({'error': 'Invalid model selection'}), 400

        return jsonify({'result': result, 'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
