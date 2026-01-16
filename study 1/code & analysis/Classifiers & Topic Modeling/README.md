# Study 1 Classification & Topic Modeling Scripts

## Setup Environment

Python 3.10 is required.

```bash
python -m venv venv # create a virtual environment named venv
source venv/bin/activate # activate the environment
pip install -r requirements.txt # install the required packages
```

## Data

Data is not pushed to the GitHub because of its size and security concerns. You can request the dataset from the authors. After you have the data:

- Create a folder named `data` in the root of this folder.
- Put the data in the `data` folder.

## Scripts and Notebooks (What They Do)

### Top-level scripts

- `main.py`: Training entry point (loads config, builds dataloaders, trains models with W&B support, evaluates, saves best checkpoints).
- `apply_binary_morality_model_to_data.py`: Loads the best binary morality model and adds `binary_morality_predicted` to the 500 annotated tweets CSV.
- `apply_multilabel_morality_model_to_data.py`: Loads the best multilabel morality model, predicts moral foundation labels for moral tweets, and writes the expanded CSV.
- `apply_prolife_prochoice_model_to_data.py`: Loads the best prolife/prochoice model and adds `prolife_prochoice_predicted` to the annotated tweets CSV.
- `tweet_data_to_csv.py`: Pulls tweet IDs from `data/firstpass.json`, fetches them from local MongoDB, and writes per-year/hashtag CSVs.
- `sample_isi_pickle_file.py`: Samples 10% of the ISI pickle dataset evenly across months and writes `sampled_tweets_equal_per_month.csv`.

### Top-level notebooks

- `compare_models_performance.ipynb`: Compares locally trained models against LLMs using `preprocessed_500_annotated_tweets_with_predictions.csv`.
- `run_topic_modeling.ipynb`: BERTopic pipeline for prolife/prochoice/neutral subsets; saves topic models, visualizations, and topic info CSVs.

### Data scripts

- `data/xlsx_to_csv.py`: Converts abortion sentiment annotation `.xlsx` files to `.csv`.
- `data/combine_annotated_data.py`: Combines annotated abortion1 CSVs into `wave1_annotated.csv` (drops empty annotations, renames columns).
- `data/combine_unannotated_data.py`: Combines unannotated abortion2 CSVs into `wave2_not_annotated.csv`.
- `data/loaders/abortion_dataset_loader.py`: Dataset/dataloader for single-label classification with tokenization and class weights.
- `data/loaders/moral_multilabel_dataset_loader.py`: Dataset/dataloader for multilabel classification with label vector parsing.
- `data/preprocessors/preprocess_abortion_dataset.py`: Cleans abortion1/abortion2 data, maps labels, concatenates, and splits train/val/test.
- `data/preprocessors/preprocess_abortion1_dataset.py`: Cleans abortion1 data, merges neutral/throw out into `throw_out`, and splits train/val/test.
- `data/preprocessors/preprocess_binary_morality_dataset.py`: Builds a balanced binary morality dataset from MFTC/MFRC and splits train/val/test.
- `data/preprocessors/preprocess_multilabel_morality_dataset.py`: Builds a multilabel morality dataset from MFTC/MFRC and splits train/val/test.
- `data/expand_the_predicted_data.ipynb`: Merges predicted labels back into large abortion datasets (e.g., reproduction tweets, geotagged sentiment).

### Preprocessing notebooks

- `data/preprocessors_notebooks/preprocess_500_annotated_tweets.ipynb`: Cleans the 500 annotated tweets dataset for model/LLM comparison.
- `data/preprocessors_notebooks/preprocess_abortion1_dataset.ipynb`: Notebook-based preprocessing/cleanup for the abortion1 dataset.
- `data/preprocessors_notebooks/preprocess_abortion_dataset.ipynb`: Notebook-based preprocessing for the abortion2 dataset.
- `data/preprocessors_notebooks/preprocess_isi_raw_abortion_dataset.ipynb`: Cleans and prepares the before/after Roe ISI CSVs.
- `data/preprocessors_notebooks/preprocess_mftc_mfrc_binary_dataset.ipynb`: Notebook version of binary MFTC/MFRC preprocessing.
- `data/preprocessors_notebooks/preprocess_mftc_mfrc_multilabel_dataset.ipynb`: Notebook version of multilabel MFTC/MFRC preprocessing.
- `data/preprocessors_notebooks/preprocess_mortezas_and_isi_abortion_datasets.ipynb`: Combines/cleans Morteza’s dataset with ISI abortion data.
- `data/preprocessors_notebooks/preprocess_mortezas_raw_abortion_dataset.ipynb`: Preprocesses Morteza’s raw 15GB abortion dataset.

### Training and evaluation helpers

- `train/trainer.py`: Training loop with early stopping, LR scheduling, W&B logging, and model/config saving.
- `evaluation/metrics.py`: Metric helpers for binary, multiclass, and multilabel evaluation.
- `evaluation/evaluate.py`: Runs inference over a dataloader and produces classification report/confusion matrix inputs.
- `evaluation/visualizations.py`: Plots confusion matrices.
- `utils/config_utils.py`: Config loading, logging setup, path helpers, device selection, and model save/load utilities.
