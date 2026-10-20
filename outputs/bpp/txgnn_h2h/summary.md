# TxGNN head-to-head

```
{
 "tx_commit": "0e0203c3939f7d12da27aaf10e55daaff5424023",
 "split": "autoimmune",
 "split_random_seed": 0,
 "tx_seed": 1,
 "our_seed": 0,
 "pretrain_max_h": 3.0,
 "finetune_epochs": 500,
 "dgl_ok": true,
 "ours_finished": true
}
```

```
[txgnn_train.log] ### pretrain 1.27 h
[txgnn_train.log] ### finetune 0.14 h
[txgnn_train.log] ### TxGNN done
[ours_train.log] epoch 0: loss 7.4139  valid MRR 0.0775  H@50 0.324  [0.11 h]
[ours_train.log] epoch 1: loss 6.5098  valid MRR 0.1357  H@50 0.354  [0.21 h]
[ours_train.log] epoch 2: loss 6.2729  valid MRR 0.1411  H@50 0.369  [0.32 h]
[ours_train.log] epoch 3: loss 6.2133  valid MRR 0.1641  H@50 0.369  [0.42 h]
[ours_train.log] epoch 4: loss 6.1942  valid MRR 0.1616  H@50 0.374  [0.53 h]
[ours_train.log] epoch 5: loss 6.1811  valid MRR 0.1648  H@50 0.366  [0.63 h]
[ours_train.log] epoch 6: loss 6.1711  valid MRR 0.1620  H@50 0.374  [0.74 h]
[ours_train.log] epoch 7: loss 6.1653  valid MRR 0.1595  H@50 0.372  [0.84 h]
[ours_train.log] epoch 8: loss 6.1254  valid MRR 0.1738  H@50 0.366  [0.95 h]
[ours_train.log] epoch 9: loss 6.1542  valid MRR 0.1593  H@50 0.366  [1.06 h]
[ours_train.log] epoch 10: loss 6.1752  valid MRR 0.1608  H@50 0.369  [1.16 h]
[ours_train.log] epoch 11: loss 6.1273  valid MRR 0.1743  H@50 0.372  [1.27 h]
[ours_train.log] epoch 12: loss 6.1227  valid MRR 0.1743  H@50 0.372  [1.37 h]
[ours_train.log] epoch 13: loss 5.8878  valid MRR 0.1864  H@50 0.458  [1.48 h]
[ours_train.log] epoch 14: loss 5.6044  valid MRR 0.1850  H@50 0.470  [1.58 h]
[ours_train.log] epoch 15: loss 5.6168  valid MRR 0.1837  H@50 0.474  [1.69 h]
[ours_train.log] epoch 16: loss 5.5116  valid MRR 0.1709  H@50 0.458  [1.79 h]
[ours_train.log] epoch 17: loss 5.3725  valid MRR 0.1772  H@50 0.431  [1.90 h]
[ours_train.log] epoch 18: loss 5.4057  valid MRR 0.1760  H@50 0.424  [2.00 h]
[ours_train.log] epoch 19: loss 5.2810  valid MRR 0.1568  H@50 0.465  [2.11 h]
[ours_train.log] early stopping
[eval.log] aligned 13 diseases, 0 missing in TxGNN output
[eval.log] ours   ours_s0                n=13  TxGNN-protocol Recall@100 0.475 AUROC 0.746 AUPRC 0.101 | full MRR 0.061
[eval.log] txgnn  txgnn_s1               n=13  TxGNN-protocol Recall@100 0.474 AUROC 0.909 AUPRC 0.147 | full MRR 0.026
```
