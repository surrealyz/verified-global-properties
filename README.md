# Learning Security Classifiers with Verified Global Robustness Properties (CCS'21)
Paper: https://arxiv.org/pdf/2105.11363.pdf

We will finish releasing all the code by November 15, 2021.

## Datasets

| Dataset | Training set size  | Test set size  | Validation set size  | # of features  |
|---|---|---|---|---|
| Cryptojacking [[Kharraz et al.](https://dl.acm.org/doi/pdf/10.1145/3308558.3313665?casa_token=tIEYZgkTcskAAAAA:fNtttzlY6d93ScDwC2dQdB4PgDkuVqvctmiLW7NxgkV8HkpPpLy2-kR1_ItFoR1Gastc5lzSc0zorw)]  | 2,800 | 1,200  | Train set  | 7  |
| Twitter Spam Accounts [[Lee et al.](https://people.engr.tamu.edu/caverlee/pubs/lee11icwsm.pdf)]  | 36,000  | 4,000  | Train set  | 15  |
| Twitter Spam URLs [[Kwon et al.](https://www3.cs.stonybrook.edu/~heekwon/papers/17-pakdd-urlspam.pdf)]  | 295,870  | 63,401  | 63,401  | 25  |

We evaluated over three datasets under the `data/` directory. The Twitter Spam Accunts dataset is named as `social_honeypot.*`, and the Twitter Spam URLs dataset is named as `unnormalized_twitter_spam.*`. The zip files need to be unzipped under the `data/` directory.

The first field of each dataset is the label, where 1 means malicious and 0 means benign. From the second field, the names of the fields are available in the `${dataset_name}_header.csv` files.

## Installation

* We recommend using a virtualenv to install this
* After activating your virtualenv, install the required packages ```pip install -r requirements.txt```
* CLN library

## Train the Models



## Pre-trained Models
