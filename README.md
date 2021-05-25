# *CLCC*: Contrastive Learning for Color Constancy (CVPR 2021)
**(Poster & video will be released before 5/31!)**

[Yi-Chen Lo*](https://scholar.google.com/citations?user=EPYQ48sAAAAJ), [Chia-Che Chang*](https://scholar.google.com.tw/citations?user=FK1RcpoAAAAJ), [Hsuan-Chao Chiu](https://scholar.google.com/citations?user=9gisBUMAAAAJ), [Yu-Hao Huang](https://www.linkedin.com/in/yu-hao-huang-72821060), [Chia-Ping Chen](https://www.linkedin.com/in/chia-ping-chen-81674078/), [Yu-Lin Chang](https://scholar.google.com/citations?user=0O9rukQAAAAJ), Kevin Jou

MediaTek Inc., Hsinchu, Taiwan

(*) indicates equal contribution.

## Paper
TL;DR: 
 
## Code
**CLCC** is a Python 3 & TensorFlow 1.x implementation based on [**FC4**](https://github.com/yuanming-hu/fc4) codebase.
* **Dataset preparation**: Download dataset from here. Please make sure your dataset folder is structured as `<DATA_DIR>/<DATA_NAME>/<FOLD_ID>` (e.g., `data/gehler/0`).
* **Training**: Modify `config.py` (rename `EXP_NAME` and specify training data `DATA_NAME`, `TRAIN_FOLDS`, `TEST_FOLDS`) and execute `train.py`. Checkpoints will be saved under `ckpts/EXP_NAME` during training.
* **Evaluation**: Once training is done, you can evaluate checkpoint with `eval.py` on a specific test fold. We recommend to refer to `scripts/eval_squeezenet_clcc_gehler.sh` for 3-fold cross-validation.

## Dataset
Since the [original data preprocessing code](https://github.com/yuanming-hu/fc4/blob/master/datasets.py) and procedure are quite tedious, we preprocess each fold of dataset and stored in `.pkl` format for each sample. Each sample contains:
* Raw image: Mask color checker; Subtract black level; Convert to **uint16 [0, 65535] BGR** numpy array with shape (h, w, 3).
* RGB label: L2-normalized numpy vector with shape (3,).
* Color checker: **[0, 4095] BGR** numpy array with shape (24, 3) for raw-to-raw mapping presented in our paper (see `util/raw2raw.py` and also **section 4.3** in our paper). **A few of them are stored in all zeros due to the failure of color checker detection.** **Note that we convert it into RGB format during preprocessing in `dataloader.py`, and our raw-to-raw mapping algorithm also manipulates it in RGB format.**
