### Main Script to run the entire pipeline

import pandas as pd
from helper_functions.preprocessing_functions import filter_targets, get_feature_length
from preprocess import PreprocessingClass
from trainer_predictor import TrainerClass
import numpy as np

# set base path
base_path = "./data/Seeds/"
dataset_name = "aligned_XY"
# build file path
file_path = f"{base_path}{dataset_name}.gz"

# Load the dataframe
df = pd.read_parquet(file_path)

# get the feature length
feature_length = get_feature_length(dataset_name)

## get sample distribution
sample_distribution = df.index.value_counts()
## get mean sample distribution
mean_sample_distribution = sample_distribution.mean()
print("Mean sample distribution:", mean_sample_distribution)



# filter the targets
df_red, microbes_left = filter_targets(df, feature_length)
print("Number Microbes left:", len(microbes_left))

# average the samples over the species
df_avg = df_red.groupby(df_red.index).mean()


# initialize an empty dataframe to store the predictions
predictions_all_species = pd.DataFrame()

# set the algorithm to use
ALGO_NAME = "random_forest"
# set augmentation to use
AUGMENTATION = False
# set the path to save according to the augmentation
AUGMENTATION_PATH = "non-augmentation" if not AUGMENTATION else "augmentation"

# Iterate over the all subspecies as holdouts
# for each subspecies in subspecies we give it to the preprocessor
for subspecies in df_avg.index.unique():
    # store the ordering of the columns for later use
    index_trues = df_avg.iloc[:, feature_length:].loc[[subspecies]].columns
    # copy the dataframe
    df_input = df_avg.copy()
    print("Hold out subspecies:", subspecies)
    preprocessor = PreprocessingClass(hold_out_species=subspecies, data=df_input, use_augmentation=AUGMENTATION, feature_length=feature_length)
    preprocessor.run_all_methods()

    # get X_train and Y_train for training and prediction
    X = preprocessor.X_final
    Y = preprocessor.Y_candidates_final
    # get X_hold_out and Y_hold_out to make predictions
    X_hold_out = preprocessor.X_hold_out
    Y_hold_out = preprocessor.Y_hold_out

    # give to trainer class
    trainer = TrainerClass(x_matrix=X, y_abundance_matrix=Y, x_hold_out=X_hold_out, y_hold_out=Y_hold_out, algorithm=ALGO_NAME)
    trainer.run_train_predict_based_on_algorithm()
    # retrieve predictions and cv_results from trainer
    predictions = trainer.predictions
    cv_results = trainer.cv_results
    print("Predictions done for subspecies:", subspecies)
    print(predictions.shape)

    # sort the predictions
    predictions_sorted = predictions.reindex(index_trues)
    # rename the column
    predictions_sorted.columns = [f"predicted_{subspecies}"]

    # append to the predictions_all_species
    predictions_all_species = pd.concat([predictions_all_species, predictions_sorted], axis=1)

# save the predictions based on the algorithm name
predictions_all_species.to_csv(f"./predictions/{AUGMENTATION_PATH}/{ALGO_NAME}_predictions_aligned.csv")
print("Predictions saved")
print("Process done!")








