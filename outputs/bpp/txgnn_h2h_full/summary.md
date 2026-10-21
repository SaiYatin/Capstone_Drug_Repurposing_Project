# TxGNN head-to-head

```
{
 "tx_commit": "0e0203c3939f7d12da27aaf10e55daaff5424023",
 "split": "autoimmune",
 "split_random_seed": 0,
 "tx_split_seed": 1,
 "tx_seeds": [
  0,
  1,
  2
 ],
 "our_seeds": [
  0,
  1,
  2
 ],
 "coverage": "full",
 "pretrain_max_h": 3.0,
 "finetune_epochs": 500,
 "dgl_ok": true,
 "jobs": {
  "txgnn_s0": "done",
  "txgnn_s1": "done",
  "ours_s0": "done",
  "txgnn_s2": "done",
  "ours_s1": "done",
  "ours_s2": "done"
 },
 "ours_finished": {
  "0": true,
  "1": true,
  "2": true
 },
 "convert_stats": {
  "coverage": "full",
  "nodes": 129312,
  "context_edges": 3430198,
  "relations": 27,
  "therapeutic": 34302,
  "train_diseases": 1139,
  "valid_diseases": 63,
  "held_out_diseases": 411,
  "held_out_with_train_indications": 127,
  "area_diseases": 18,
  "eval_diseases": 13,
  "test_pairs": 75,
  "candidate_drugs": 1798,
  "dropped_test_edges": 3248,
  "dropped_test_pairs": 3248,
  "diseases_with_dropped_test_edges": 851
 }
}
```

```
[ours_s0.log] epoch 0: loss 7.0153  valid MRR 0.2223  H@50 0.451  [0.19 h]
[ours_s0.log] epoch 1: loss 6.0796  valid MRR 0.2527  H@50 0.460  [0.38 h]
[ours_s0.log] epoch 2: loss 5.9629  valid MRR 0.2583  H@50 0.459  [0.56 h]
[ours_s0.log] epoch 3: loss 5.9003  valid MRR 0.2647  H@50 0.488  [0.75 h]
[ours_s0.log] epoch 4: loss 5.5984  valid MRR 0.2951  H@50 0.523  [0.94 h]
[ours_s0.log] epoch 5: loss 5.4207  valid MRR 0.3018  H@50 0.521  [1.12 h]
[ours_s0.log] epoch 6: loss 5.2970  valid MRR 0.3071  H@50 0.534  [1.31 h]
[ours_s0.log] epoch 7: loss 5.2689  valid MRR 0.3085  H@50 0.511  [1.50 h]
[ours_s0.log] epoch 8: loss 5.2152  valid MRR 0.2944  H@50 0.517  [1.69 h]
[ours_s0.log] epoch 9: loss 5.1759  valid MRR 0.3080  H@50 0.561  [1.87 h]
[ours_s0.log] epoch 10: loss 5.0945  valid MRR 0.3032  H@50 0.562  [2.06 h]
[ours_s0.log] epoch 11: loss 5.0404  valid MRR 0.3044  H@50 0.580  [2.25 h]
[ours_s0.log] epoch 12: loss 5.0308  valid MRR 0.3262  H@50 0.583  [2.43 h]
[ours_s0.log] epoch 13: loss 4.9461  valid MRR 0.2994  H@50 0.578  [2.62 h]
[ours_s0.log] epoch 14: loss 4.8994  valid MRR 0.3068  H@50 0.575  [2.81 h]
[ours_s0.log] epoch 15: loss 4.9295  valid MRR 0.3061  H@50 0.588  [3.00 h]
[ours_s0.log] epoch 16: loss 4.8950  valid MRR 0.3091  H@50 0.582  [3.18 h]
[ours_s0.log] epoch 17: loss 4.8596  valid MRR 0.3091  H@50 0.571  [3.37 h]
[ours_s0.log] epoch 18: loss 4.8888  valid MRR 0.3082  H@50 0.599  [3.56 h]
[ours_s0.log] early stopping
[ours_s1.log] epoch 0: loss 7.1414  valid MRR 0.2123  H@50 0.449  [0.19 h]
[ours_s1.log] epoch 1: loss 6.1009  valid MRR 0.2576  H@50 0.464  [0.38 h]
[ours_s1.log] epoch 2: loss 5.6612  valid MRR 0.2875  H@50 0.504  [0.56 h]
[ours_s1.log] epoch 3: loss 5.4175  valid MRR 0.2964  H@50 0.513  [0.75 h]
[ours_s1.log] epoch 4: loss 5.3289  valid MRR 0.3054  H@50 0.517  [0.94 h]
[ours_s1.log] epoch 5: loss 5.2908  valid MRR 0.3018  H@50 0.520  [1.12 h]
[ours_s1.log] epoch 6: loss 5.2102  valid MRR 0.3029  H@50 0.527  [1.31 h]
[ours_s1.log] epoch 7: loss 5.1485  valid MRR 0.2966  H@50 0.529  [1.50 h]
[ours_s1.log] epoch 8: loss 5.1266  valid MRR 0.2949  H@50 0.553  [1.69 h]
[ours_s1.log] epoch 9: loss 5.0976  valid MRR 0.2949  H@50 0.544  [1.87 h]
[ours_s1.log] epoch 10: loss 5.0708  valid MRR 0.2913  H@50 0.526  [2.06 h]
[ours_s1.log] early stopping
[ours_s2.log] epoch 0: loss 7.0315  valid MRR 0.2271  H@50 0.452  [0.19 h]
[ours_s2.log] epoch 1: loss 6.1192  valid MRR 0.2508  H@50 0.458  [0.38 h]
[ours_s2.log] epoch 2: loss 5.8189  valid MRR 0.2846  H@50 0.540  [0.56 h]
[ours_s2.log] epoch 3: loss 5.4266  valid MRR 0.3091  H@50 0.548  [0.75 h]
[ours_s2.log] epoch 4: loss 5.3539  valid MRR 0.2902  H@50 0.522  [0.94 h]
[ours_s2.log] epoch 5: loss 5.2450  valid MRR 0.2991  H@50 0.545  [1.13 h]
[ours_s2.log] epoch 6: loss 5.0972  valid MRR 0.2931  H@50 0.534  [1.31 h]
[ours_s2.log] epoch 7: loss 5.0352  valid MRR 0.2898  H@50 0.559  [1.50 h]
[ours_s2.log] epoch 8: loss 4.9968  valid MRR 0.3047  H@50 0.586  [1.69 h]
[ours_s2.log] epoch 9: loss 4.9456  valid MRR 0.2923  H@50 0.574  [1.88 h]
[ours_s2.log] early stopping
[txgnn_s0.log] ### split seed 1, training seed 0
[txgnn_s0.log] ### pretrain 1.24 h
[txgnn_s0.log] ### finetune 0.14 h
[txgnn_s0.log] ### TxGNN done
[txgnn_s1.log] ### split seed 1, training seed 1
[txgnn_s1.log] ### pretrain 1.24 h
[txgnn_s1.log] ### finetune 0.14 h
[txgnn_s1.log] ### TxGNN done
[txgnn_s2.log] ### split seed 1, training seed 2
[txgnn_s2.log] ### pretrain 1.23 h
[txgnn_s2.log] ### finetune 0.14 h
[txgnn_s2.log] ### TxGNN done
[eval.log] aligned 13 diseases, 0 missing in TxGNN output
[eval.log] aligned 13 diseases, 0 missing in TxGNN output
[eval.log] aligned 13 diseases, 0 missing in TxGNN output
[eval.log] ours   ours_s0                n=13  TxGNN-protocol Recall@100 0.657 AUROC 0.787 AUPRC 0.352 | full MRR 0.264
[eval.log] ours   ours_s1                n=13  TxGNN-protocol Recall@100 0.611 AUROC 0.776 AUPRC 0.338 | full MRR 0.273
[eval.log] ours   ours_s2                n=13  TxGNN-protocol Recall@100 0.573 AUROC 0.768 AUPRC 0.338 | full MRR 0.283
[eval.log] txgnn  txgnn_s0               n=13  TxGNN-protocol Recall@100 0.494 AUROC 0.906 AUPRC 0.124 | full MRR 0.019
[eval.log] txgnn  txgnn_s1               n=13  TxGNN-protocol Recall@100 0.499 AUROC 0.897 AUPRC 0.205 | full MRR 0.104
[eval.log] txgnn  txgnn_s2               n=13  TxGNN-protocol Recall@100 0.613 AUROC 0.901 AUPRC 0.153 | full MRR 0.020
```
