# Variational Template Machine

Code for [Variational Template Machine for Data-to-text generation](https://openreview.net/forum?id=HkejNgBtPB) (Ye et al., ICLR 2020). 

## Requirements
All dependencies (Python 3) can be installed via:
```
pip install -r requirements.txt
```

## Running
- **Datasets**:

Two datasets (SPNLG and Wiki) can be downloaded from: https://drive.google.com/drive/folders/1bX3VWzQosm0p2DlEnp7T5jQLI79Sl2nN?usp=sharing

Details info: [HERE](./data/README.md)

- **Training**:

```
DATASET_PATH=<path_to_dataset>
MODEL_PATH=<path_to_model>

python train.py -data ${DATASET_PATH} -max_vocab_cnt 40000 -emb_size 786 -hid_size 512 -table_hid_size 256 -pool_type max -sent_represent last_hid -z_latent_size 128 -c_latent_size 256 -dec_attention -drop_emb -add_preserving_content_loss -pc_weight 1.0 -add_preserving_template_loss -pt_weight 1.0 -anneal_function_z const -anneal_k_z 0.8 -anneal_function_c const -anneal_k_c 0.8 -add_mi_z -mi_z_weight 0.5 -add_mi_c -mi_c_weight 0.5 -lr 0.001 -clip 5.0 -cuda -log_interval 500 -bsz 16 -paired_epochs 5 -raw_epochs 2 -epochs 20 -cuda -save ${MODEL_PATH}
```
Notice that the arguments may be different.
- **Generation**:
```
OUTPUT_PATH=<path_to_result>
python generate.py -data ${DATASET_PATH} -max_vocab_cnt 50000 -load ${MODEL_PATH} -various_gen 5 -mask_prob 0.0 -cuda -decode_method temp_sample -sample_temperature 0.2 -gen_to_fi ${OUTPUT_PATH}
``` 