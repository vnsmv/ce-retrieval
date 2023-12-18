# Efficient Nearest Neighbor Search for Cross-Encoder Models using Matrix Factorization
This repository contains code used in experiments for our EMNLP 2022 paper titled  "[Efficient Nearest Neighbor Search for Cross-Encoder Models using Matrix Factorization](https://arxiv.org/pdf/2210.12579.pdf)". 

## Setup ##

* Clone the repository and install the dependencies (optionally) in a separate conda environment.
```
conda create -n <env_name> -y python=3.7 && conda activate <env_name>
pip install -r requirements.txt
```

* Setup some enviroment variables

```
source bin/setup.sh
```

## Dependensies ##
Add current folder to `PATH` and define `CUDA` device
```
export PYTHONPATH=/home/.../ce-retrieval
export CUDA_VISIBLE_DEVICES=0
```

## Data Setup ##
1. Download [ZeShEL](https://paperswithcode.com/dataset/zeshel) data from [here](https://github.com/lajanugen/zeshel).
2. Preprocess data into the required format using `utils/preprocess_zeshel.py` in order to train dual-encoder and 
cross-encoder models on this dataset. We will use standard train/test/dev splits as defined [here](https://aclanthology.org/P19-1335.pdf).

## Tokenization ##
Data tokenized with  word-piece tokenization [Wu et al., 2016](https://arxiv.org/abs/1609.08144) for with a maximum of 128 tokens including special tokens for tokenizing entities and mentions. 
We used [bert-base_uncased](https://huggingface.co/bert-base-uncased) for tokenization:
```
python utils/tokenize_entities.py --ent_file data/zeshel/documents/star_trek.json --out_file data/zeshel/tokenized_entities/star_trek_128_bert_base_uncased.npy --bert_model_type bert-base-uncased --max_seq_len 128 --lowercase 0
```

## Cross-Encoder ##
CE model embeds special tokens amongst query and item tokens, and computes the
query-item score using contextualixed query and item embeddings extracted using the special tokens (see tokenization step) after jointly encoding the query-item pair:

<img width="324" alt="image_2023-12-18_14-30-08" src="https://github.com/justfollowthesun/ce-retrieval/assets/74874227/e55123f3-ca8e-4943-87c9-5977912004f6">


## Query-item score matrix computation ##
We compute cross-encoder scores for all item in the data. The approach selects a fixed set of anchor queries and anchor items, and uses scores between anchor queries and
all items to generate latent embeddings for indexing the item set. At test time, we generate latent embedding for the query using cross-encoder scores for the test query and anchor items, and use it to approximate scores of all items for the given query
and/or retrieve top-k items according to the approximate scores. In contrast to distillation-based approaches, our proposed approach does not involve any additional compute-intensive training of a student model such as dual-encoder via distillation.
Query-item score matrix computed via executing (example is `star_track` data)
```
python eval/run_cross_encoder_for_ment_ent_matrix_zeshel.py --data_name star_trek --cross_model_ckpt checkpoints/cls_crossencoder_zeshel/cls_crossenc_zeshel.ckpt --layers final --res_dir results/ --disable_wandb 1
```
We used different NLA approaches for query-item score matrix factorixation

## CUR decomposition ##
CUR was implemented followed by [(Mahoney and Drineas,
2009](https://www.pnas.org/doi/10.1073/pnas.0803205106)
![image](https://github.com/justfollowthesun/ce-retrieval/assets/74874227/06af3a7c-46a9-42da-adf6-287396980888)

In code this is `CURApprox` class in `eval/matrix_approx_zeshel.py`

## SVD decomposition ##
We used classical [SVD] (https://en.wikipedia.org/wiki/Singular_value_decomposition). 
In code this is`SVDApprox` class in `eval/matrix_approx_zeshel.py`

## Pretrained models download ##

* [Dual-Encoder Model](https://huggingface.co/nishantyadav/dual_encoder_zeshel)
* [Cross-Encoder Model w/ [CLS] token pooling](https://huggingface.co/nishantyadav/cls_crossencoder_zeshel)
* [Cross-Encoder Model w/ proposed special token based pooling (see paper for details)](https://huggingface.co/nishantyadav/emb_crossenc_zeshel)
```
mkdir checkpoints
cd checkpoints

git clone https://huggingface.co/nishantyadav/dual_encoder_zeshel
git clone https://huggingface.co/nishantyadav/cls_crossencoder_zeshel
```

## Metrics ## 
* In the first setting, we retrieve $k_r$ items for a given query, re-rank them using exact CE scores and keep top-k items. We evaluate each method using $Top-k-Recall@k_r$ which is thepercentage of $top-k$ items according to the CE model present in the $k_r$ retrieved items. In this project we used $k=10$.
We plotted Top-10-Recall@k_r$ vs t cost (the number of CE calls made during inference for re-ranking retrieved items).
* In the second setting, we operate under a fixed test-time cost budget where the cost is defined as the number of CE calls made during inference. Baselines such as DE and TF-IDF will use the entire cost budget for re-ranking items using exact CE scores while our proposed approach will have to split the budget between the number of anchor items ($k_i$) used for embedding the query and the number of items ($k_r$) retrieved for final re-ranking.

## Experimental pipeline ##
We run extensive experiments with crossencoder models trained for the downstream task
of entity linking. The query and item in this case correspond to a mention of an entity in text and a document with an entity description respectively.

* Download data
* Tokenize and compute score matrix via `tokenize.sh`
* Evaluate cross encoder model. Here is the command for evaluation on `pro_wrestling` data, but you can choose any dataset from downloaded zeshel folder as well:
```
python eval/run_retrieval_eval_wrt_exact_crossenc.py --res_dir results --data_name pro_wrestling --bi_model_file checkpoints/dual_encoder_zeshel/dual_encoder_zeshel.ckpt
```
* We measured quality and time for both **CUR** and **SVD** approaches in score matrix factorizarion (see `eval/matrix_approx_zeshel.py`)
## Results and conclusions ##
We compared SVD in CUR decomposition under the different metrics:
![image](https://github.com/justfollowthesun/ce-retrieval/assets/74874227/44039818-4054-4e05-b2e6-a280660e4a52)

![image](https://github.com/justfollowthesun/ce-retrieval/assets/74874227/1f1301e5-afde-48bc-978c-b2af63a19076)

SVD shows the highest decomposition quality (which additionally follows from the Eckart-Young theorem), but is not quite optimal in terms of time complexity

## Our contribution ##
We have repeated the results of the paper with CUR decomposition, implemented SVD factorization, and compared both of these approaches in terms of approximation quality and speed of performance.
