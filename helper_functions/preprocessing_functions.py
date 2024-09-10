import  numpy as np
import pandas as pd



# Function to calculate the coefficient of variation
def calculate_cv(data):
    if np.mean(data) == 0:
        return np.nan  # Avoid division by zero
    return np.std(data, ddof=1) / np.mean(data)

def filter_targets_on_quantile(df_raw, n_features, quantile=0.85):
    df_quantile = df_raw.iloc[:, n_features:].groupby(df_raw.index).quantile(quantile)
    candidates = df_quantile.drop(columns=[col for col in df_quantile.columns if df_quantile[col].eq(0).all()]).columns
    df_interim = df_raw[candidates]

    return df_interim

def filter_for_low_variance(df_raw, df_interim, n_features, cv=0.5):
    # Initialize an empty dictionary to store core microbiome data
    core_microbiome = {}

    for species in df_interim.index.unique():
        # Subset DataFrame by species
        subset = df_interim.loc[[species]]

        # Calculate variability (CV) for each microorganism within this species
        variability = subset.apply(calculate_cv)

        # Filter based on a threshold, e.g., CV < 0.5 for low variability
        core_microbes = variability[variability < cv].index.tolist()

        # Store the core microorganisms in the dictionary
        core_microbiome[species] = core_microbes

        # Flatten the list of all microorganisms from the core microbiome of all species
        all_core_microorganisms = [microbe for microbes in core_microbiome.values() for microbe in microbes]

        # Convert the list to a set to remove duplicates, getting the unique set of core microorganisms
        unique_core_microorganisms = set(all_core_microorganisms)

        df_output = pd.concat([df_raw.iloc[:, :n_features], df_raw[list(unique_core_microorganisms)]], axis=1)

    return df_output, unique_core_microorganisms



def get_feature_length(dataset_name):
    """
    Determines the feature length based on dataset name
    """
    if "60_features" in dataset_name:
        return 60
    elif "68_features" in dataset_name:
        return 68
    elif "aligned" in dataset_name:
        return 813
    else:
        raise ValueError("Feature length not found for dataset name: ", dataset_name)
