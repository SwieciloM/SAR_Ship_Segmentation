<h1 align="center">SAR_ship_segmentation</h1>

As part of the project, 4 U-Net neural network models were developed and tested. These models were applied to segment ships in images obtained from synthetic aperture radar (SAR). The thesis involved selection of a suitable dataset (HRSID), its preprocessing, creation of a data generator, application of augmentation techniques, and training and testing of the neural network models. Finaly project included development of the ShipPaste augmentation algorithm, which increased the number of vessels in the images, allowing the models to perform better in tests. To accelerate work, a transfer learning technique was applied using ResNet34 neural network as well as cloud computing in form of Google Colab platform. Tests carried out showed high performance of the models in vessel segmentation in open water, and satisfactory results in complex port areas.

## Models performance

| Model         | Dice Loss | Precision | Recall | Mean IoU |
|---------------|-----------|-----------|--------|----------|
| UNetAug       | 0.231     | 0.679     | 0.576  | 0.723    |
| UNetResAug    | 0.200     | 0.831     | 0.448  | 0.639    |
| UNetResSP1    | 0.154     | 0.884     | 0.624  | 0.722    |
| UNetResSP2    | 0.097     | 0.907     | 0.861  | 0.800    |

<div align="center">
    <img src="https://github.com/user-attachments/assets/c71a9a3a-b13b-4030-b8d1-711c3c6cebe8" width="779" height="594" />
    <img src="https://github.com/user-attachments/assets/8c4a220c-193b-4a80-ab89-6a08a7061af8" width="779" height="594" />
</div>

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.

    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
