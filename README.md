# Learning Security Classifiers with Verified Global Robustness Properties (CCS'21)
Paper: https://arxiv.org/pdf/2105.11363.pdf

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
* Clone the [cln_tutorial](https://github.com/gryan11/cln_tutorial) repo, and install in the virtualenv `python setup.py build; python setup.py install`
* Obtain [an academic license from Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/)

## Train the Models

Train a Cryptojacking classifier with all five properties:
```
model='cryptojacker_all'; \
time python train_cln_xgb_property.py \
--train data/cryptojacker.train.csv \
--test data/cryptojacker.test.csv \
--validation data/cryptojacker.train.csv \
--nlabels 2 --nfeat 7 \
--intfeat data/cryptojacker_integer_indices.json -z \
--save_model_path models/cln_models/${model}.pth \
--init -e 300 --header data/cryptojacker_header.csv \
--size 1 --add tree --num_boost_round 4 --max_depth 4 \
--robust \
--monotonicity "[0, 1, 2, 3, 4, 5, 6]" \
--monotonicity_dir "[1, 1, 1, 1, 1, 1, 1]" \
--stability "[0, 1, 2, 3, 4, 5, 6]" --stability_th 0.1 \
--lowcost "{2:(None, None)}" --lowcost_th 0.98 \
--eps 0.2 --C 0.5 \
--featmax data/cryptojacker_featmax.csv \
>! log/${model}.log 2>&1&
```

If you'd like to train one property, or a subset of properties, change the parameters after `--robust` accordingly.

Tain a Twitter spam account classifier with four properties:
```
model="social_honeypot_all"; \
time python train_cln_xgb_property_all.py \
-train data/social_honeypot.train.csv \
--test data/social_honeypot.test.csv \
--validation data/social_honeypot.train.csv \
--nlabels 2 --nfeat 15 \
--intfeat data/social_honeypot_integer_indices.json -z \
--structure models/cln_models/social_honeypot_all_nt4d5_w0.2_ex11.json \
--save_model_path models/cln_models/${model}.pth \
--init --min_atoms 1 --max_atoms 1 -e 100000 \
--header data/social_honeypot_header.csv \
--size 1024 --add tree --num_boost_round 7 --max_depth 5 \
--robust \
--monotonicity "[2,4,0,3,10,11]" \
--monotonicity_dir "[-1,-1,1,1,1,1]" \
--stability "[0, 1, 8, 9, 10, 11, 12, 13]" \
--stability_th 8 \
--redundancy "[{0:(5, None), 1:(None, None)}, {8:(None, None), 9:(None, None)}, {10:(None, None), 11:(None, None)}, {12:(None, None), 13:(None, None)}]" \
--lowcost_th 0.98 \
--scale_pos_weight 0.2 --loss_weight \
--randfree
>! twitter_spam/log/${model}.log 2>&1&
```

## Pre-trained Models

Unzip the `cln_models.zip` under the `models/` directory. The models trained with one property is named with the property. `cryptojacker_all` is trained with five properties. `social_honeypot_all` is trained with four properties.

## Verify the Global Properties

### Example Usages for Monotonicity Property

Crytpojacker

```
model='cryptojacker_stability'; \
time python attack_ilp.py \
--model_type cln \
--model_path models/cln_models/${model}.pth \
--intfeat data/cryptojacker_integer_indices.json \
--default_lo 0 --nfeat 7 --nlabels 2 \
--int_var --no_timeout \
--monotonicity "[0, 1, 2, 3, 5, 6, 7]" \
--monotonicity_dir "[1, 1, 1, 1, 1, 1, 1]"
```

Twitter spam account

```
model='social_honeypot_stability'; \
time python attack_ilp.py \
--model_type cln \
--model_path models/cln_models/${model}.pth \
--intfeat data/social_honeypot_integer_indices.json \
--default_lo 0 --nfeat 15 --nlabels 2 \
--int_var --no_timeout \
--monotonicity "[2,4,0,3,10,11]" \
--monotonicity_dir "[-1,-1,1,1,1,1]"
```

Twitter spam URL

```
model='twitter_spam_highconfidence'; \
time python attack_ilp.py \
--model_type cln \
--model_path models/cln_models/${model}.pth \
--intfeat data/unnormalized_twitter_spam_integer_indices.csv \
--default_lo 0 --nfeat 25 --nlabels 2 \
--int_var --no_timeout \
--monotonicity '[0,1,2,3,4,5,6]' \
--monotonicity_dir '[1,1,1,1,1,1,1]'
```
