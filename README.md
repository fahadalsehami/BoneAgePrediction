# BoneAgePrediction: age estimation from adult pubic symphysis images

Forensic and biological anthropologists estimate age at death from the surface of the **pubic symphysis**. This project frames that as image classification with transfer learning: a **VGG16** network pretrained on ImageNet turns each image into a 25,088-dimensional feature vector, and a **logistic-regression** classifier is trained on those features to sort images into three classes (`GOOD`, `MEDIUM`, `POOR`, set in `settings/config.py`).

> The images are not included. The code expects you to supply your own labelled data, as described below.

## Pipeline

| Step | Script | What it does |
| --- | --- | --- |
| 1. Split | `dataset.py` | Copies images from `MyData/{training,validation,evaluation}/` into `dataset/<split>/<class>/`, taking each image's class from the number before the first `_` in its file name (`0_…` is `GOOD`, `1_…` is `MEDIUM`, `2_…` is `POOR`). |
| 2. Features | `features.py` | Runs every image (resized to 224 × 224) through VGG16 without its top layers, in batches of 180, and writes one CSV per split to `output/` (class index, then 7 × 7 × 512 features), plus the label encoder. |
| 3. Train and evaluate | `train.py` | Fits logistic regression on the training CSV, prints a per-class report on the evaluation CSV, and saves the model to `output/model.cpickle`. |

## Run it

```bash
git clone https://github.com/fahadalsehami/BoneAgePrediction.git
cd BoneAgePrediction
python -m venv .venv && source .venv/bin/activate   # Python 3.10 – 3.12
pip install -r requirements.txt

# add your images, named <class>_<anything>.jpg, where <class> is 0 (GOOD), 1 (MEDIUM) or 2 (POOR)
mkdir -p MyData/training MyData/validation MyData/evaluation
python dataset.py
python features.py   # the first run downloads the VGG16 ImageNet weights (about 60 MB)
python train.py
```

All paths, class names and the batch size are set in `settings/config.py`. The pipeline runs with TensorFlow 2.15 (Keras 2) and TensorFlow 2.21 (Keras 3).

## Stack

Python, TensorFlow and Keras (VGG16), scikit-learn, NumPy, Pillow

## License

MIT, see [LICENSE](LICENSE).
