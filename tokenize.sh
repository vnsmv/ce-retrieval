# star_track
export PYTHONPATH=/home/ansafronov/Yandex.Disk/Studies/NLA/ce-retrieval
export CUDA_VISIBLE_DEVICES=0

python utils/tokenize_entities.py --ent_file data/zeshel/documents/star_trek.json --out_file data/zeshel/tokenized_entities/star_trek_128_bert_base_uncased.npy --bert_model_type bert-base-uncased --max_seq_len 128 --lowercase 0
python eval/run_cross_encoder_for_ment_ent_matrix_zeshel.py --data_name star_trek --cross_model_ckpt checkpoints/cls_crossencoder_zeshel/cls_crossenc_zeshel.ckpt --layers final --res_dir results/ --disable_wandb 1