# *CLCC*: Contrastive Learning for Color Constancy (CVPR 2021)

[Yi-Chen Lo*](https://scholar.google.com/citations?user=EPYQ48sAAAAJ), [Chia-Che Chang*](https://scholar.google.com.tw/citations?user=FK1RcpoAAAAJ), [Hsuan-Chao Chiu](https://scholar.google.com/citations?user=9gisBUMAAAAJ), [Yu-Hao Huang](https://www.linkedin.com/in/yu-hao-huang-72821060), [Chia-Ping Chen](https://www.linkedin.com/in/chia-ping-chen-81674078/), [Yu-Lin Chang](https://scholar.google.com/citations?user=0O9rukQAAAAJ), Kevin Jou

MediaTek Inc., Hsinchu, Taiwan

(*) indicates equal contribution.

## [Paper](https://arxiv.org/abs/2106.04989) | [Poster](https://drive.google.com/file/d/1CMQc4UNz3u7YNzRIndhv9JZOsnyJXWja/view?usp=sharing) | [5-min Video](https://drive.google.com/file/d/1X1r-Tdpg9muDIuL0KQhToDVCj8DktWkj/view?usp=sharing) | [5-min Slides](https://drive.google.com/file/d/1B5XjoIUgMD-zUngdNjXgUhJPTwSb-Ekv/view?usp=sharing) | [10-min Slides](https://drive.google.com/file/d/1WdNwoSzwu-FV9AD3YWogO2A4e3otz2j5/view?usp=sharing)

<img src="https://github.com/howardyclo/CLCC-CVPR21/blob/master/fig/poster.png" />

## [Dataset](https://mega.nz/folder/G9JUQRja#Nnd40DVW41M_lNCW5f0ZGg)
We preprocess each fold of dataset and stored in `.pkl` format for each sample. Each sample contains:
* Raw image: Mask color checker; Subtract black level; Convert to uint16 [0, 65535] BGR numpy array with shape (H, W, 3).
* RGB label: L2-normalized numpy vector with shape (3,).
* Color checker: [0, 4095] BGR numpy array with shape (24, 3) for raw-to-raw mapping presented in our paper (see `util/raw2raw.py` and also *section 4.3* in our paper). A few of them are stored in all zeros due to the failure of color checker detection. Note that we convert it into RGB format during preprocessing in `dataloader.py`, and our raw-to-raw mapping algorithm also manipulates it in RGB format.
 
## Training and Evaluation
**CLCC** is a Python 3 & TensorFlow 1.x implementation based on [**FC4**](https://github.com/yuanming-hu/fc4) codebase.
* **Dataset preparation**: [Download preprocessed dataset here.](https://mega.nz/folder/G9JUQRja#Nnd40DVW41M_lNCW5f0ZGg) Please make sure your dataset folder is structured as `<DATA_DIR>/<DATA_NAME>/<FOLD_ID>` (e.g., `data/gehler/0`, just like how it is structured in download source).
 
* **Pretrained weights preparation**: [Download ImageNet-pretrained weights here.](https://mega.nz/folder/O0wAjSQb#hUN2CgxrrwrFQHQ-9Iz3Qw) Place pretrained weight files under `pretrained_models/imagenet/`.
 
* **Training**: Modify `config.py` (i.e., you may want to rename `EXP_NAME` and specify training data `DATA_NAME`, `TRAIN_FOLDS`, `TEST_FOLDS`) and execute `train.py`. Checkpoints will be saved under `ckpts/EXP_NAME` during training.

* **Evaluation**: Once training is done, you can evaluate checkpoint with `eval.py` on a specific test fold. We recommend to refer to `scripts/eval_squeezenet_clcc_gehler.sh` for 3-fold cross-validation.

## Acknowledgments
* **FC4**: https://github.com/yuanming-hu/fc4.
* **Color checker detection**: https://github.com/colour-science/colour-checker-detection. To increase detection accuracy, performing homography with color checker coordinates provided by the original dataset can help a lot.
