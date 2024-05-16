#!/usr/bin/env python
"""
This file gathers data to be used for pre-processing in training and prediction.
"""
import pandas as pd

def main():
    blacklist_file = 'Datasets/cleaned_rnn_phishtank.csv'
    whitelist_file = 'Datasets/cleaned_rnn_zenodo.csv'

    urls = {}
    
    # Read blacklist URLs from CSV
    blacklist = pd.read_csv(blacklist_file)

    # Set oversampling_rate to 1 to have the positive samples match the phishing samples. Set to greater than 1 to use more negative samples.
    # oversampling_rate = 1.5

    # Getting the array of all phishing domain names.
    # phishing_domains = blacklist["domain_names"].values
    
    # Assign 1 for malicious URLs for supervised learning
    for url in blacklist['domain_names']:
        urls[url] = 1
    
    # Read whitelist URLs from CSV
    whitelist = pd.read_csv(whitelist_file)
    
    # Randomly sample a number of safe urls, sice the ratio of classes in the training data should not be too much out of balance.
    # whitelist_domains = np.random.choice(whitelist["domain_names"].values, size=int(oversampling_rate*len(phishing_domains)), replace=True)
    
    # Assign 0 for non-malicious URLs
    for url in whitelist['domain_names']:
        urls[url] = 0

    return urls

if __name__ == "__main__":
    main()
