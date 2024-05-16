# **Phishing Website Detection Using RNN** #

## **Overview** ##

This is my Final year project, Phishing Website Detection. It used RNN to detect whether the website is safe or not-safe. Two RNNs models, LSTM and BiLSTM where chosen. However, I added another four ML models which were MLP, Decision Tree, Random Forest, and SVM. Experimental results indicate that the deep learning models achieved an accuracy of 97.97% for Bi-LSTM, and 97.79% for LSTM, respectively. Among the machine learning models, MLP achieved the highest accuracy of 80%, followed by Decision Tree with 79.8% accuracy, Random Forest with 78.08% accuracy, and SVM with 77.07% accuracy.

**Features used for ML models**

* **Domain:** The domain name of the URL (e.g., "google.com" in "https://www.google.com").
* **Have_IP:** Indicates the presence of an IP address in the text (1 or 0).
* **Have_AtSign (@):** Indicates the presence of an "@" symbol (1 or 0).
* **URL_Length:** The length of the URL string.
* **URL_Depth:** The number of subdirectories within the URL path (e.g., 2 in "https://www.google.com/search").
* **Redirection:** Whether the URL redirects to another location (1 or 0).
* **https_Domain:** Indicates if the URL uses HTTPS protocol (1 or 0).
* **TinyURL:** Indicates if the URL is a shortened URL using a service like TinyURL (1 or 0).
* **Prefix/Suffix:** Presence of specific prefixes or suffixes in the URL (e.g., "http://" or ".com").
* **Label:** The target variable used for classification or regression tasks in the model (e.g., Phishing/not-Phishing).

**Features used for RNN models**

*Only domain was extracted from the URL, no other features extracted like ML models.*

## **Getting Started** ##
*Run this commands using CMD/ Powershell from the directory. It will start the server, automatically redirect to index.html.*

	python app.py
 
## **Resources** ##

### Datasets ###
* PhishTank
* Zenodo

## **Languages & Tools used** ##

![Python](https://img.shields.io/badge/language-Python_Notebook-yellow)
![HTML](https://img.shields.io/badge/language-HTML-yellow)
![JavaScript](https://img.shields.io/badge/language-JavaScript-yellow)
![Static Badge](https://img.shields.io/badge/tools-Flask_API-red)
![Static Badge](https://img.shields.io/badge/library-Tensorflow-orange)
![Static Badge](https://img.shields.io/badge/library-Numpy-orange)
