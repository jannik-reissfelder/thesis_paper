# This is the main module of the program. It is responsible for the preprocessing of the data, the training
# and the evaluation of the model.


# handle imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import time


# Additional helper functions

def find_closest_species(feature_vector, n_neighbors=5):
    """
    Finds the closest species based on phylogenetic distance.
    """

    closest_species = feature_vector.sort_values()[:n_neighbors].index.tolist()
    closest_species = pd.Series(closest_species).str.strip().tolist()
    return closest_species


def mixup_data(X, Y, alpha=1, mix_up_x=False, degree=1):
    '''Helper Function to apply Mixup augmentation to the dataset multiple times.'''
    # Initialize empty lists to store augmented data
    augmented_x_dfs = []
    augmented_y_dfs = []

    for _ in range(degree):
        # Convert X and Y to numpy arrays
        x = X.values
        y = Y.values
        if alpha > 0:
            # Sample λ from a Beta distribution
            lam = np.random.beta(alpha, alpha, x.shape[0])
        else:
            # No Mixup is applied if alpha is 0 or less
            lam = np.ones(x.shape[0])

        # Reshape λ to allow element-wise multiplication with x and y
        lam_y = lam.reshape(-1, 1)

        # Randomly shuffle the data
        index = np.random.permutation(x.shape[0])

        # Create mixed outputs
        mixed_y = lam_y * y + (1 - lam_y) * y[index, :]
        mixed_y_df = pd.DataFrame(mixed_y, columns=Y.columns)
        mixed_y_df.index = Y.index

        # If mix_up_x is True, also mix the inputs
        if mix_up_x:
            lam_x = lam.reshape(-1, 1)
            mixed_x = lam_x * x + (1 - lam_x) * x[index, :]
            mixed_x_df = pd.DataFrame(mixed_x, columns=X.columns)
        else:
            # If mix_up_x is False, use the original inputs
            mixed_x_df = X

        # Append the mixed data to the lists
        augmented_x_dfs.append(mixed_x_df)
        augmented_y_dfs.append(mixed_y_df)

    # Concatenate augmented dataframes
    augmented_x_df_final = pd.concat(augmented_x_dfs, ignore_index=False)
    augmented_y_df_final = pd.concat(augmented_y_dfs, ignore_index=False)

    return augmented_x_df_final, augmented_y_df_final


# initialize the class

class PreprocessingClass:
    def __init__(self,
                 hold_out_species: str,
                 data: pd.DataFrame,
                 feature_length: int,
                 use_augmentation: bool = False,
                 use_normalized_data: str = "absolute",
                 candidate_n_neighbors: int = 80,
                 mix_up_x=False
                 ):

        self.hold_out_species = hold_out_species
        self.data = data
        self.feature_length = feature_length
        self.use_augmentation = use_augmentation
        self.use_normalized_data = use_normalized_data
        self.candidate_n_neighbors = candidate_n_neighbors
        self.mix_up_x = mix_up_x
        self.X = None
        self.Y = None
        self.closest = None

    def set_hold_out_set(self):
        """
        This function sets the hold out set.
        """
        # remove the hold out species from the distance matrix columns
        # self.data.drop(columns=[self.hold_out_species], inplace=True)
        # removing the  hold out  species from the row data
        # first assign it to the hold_out_set
        self.hold_out_set = self.data.loc[[self.hold_out_species]]
        # removing the hold out species from the data
        self.data.drop(self.data.loc[[self.hold_out_species]].index, inplace=True)
        print("Hold out set shape:", self.hold_out_set.shape)
        print("Data shape after hold out:", self.data.shape)

    def preprocesses_seed_data(self, data: pd.DataFrame = None):
        """
        This function preprocesses the data.
        It assigns the target matrix and the feature matrix.
        """

        # applying the hold out strategy
        self.set_hold_out_set()

        # Splitting the dataframe into features and targets
        self.X = self.data.iloc[:, :(self.feature_length )]
        self.Y = self.data.iloc[:, (self.feature_length ):]
        print("X-shape:", self.X.shape)
        print("Y-shape:", self.Y.shape)

        # do same for hold out set
        self.X_hold_out = self.hold_out_set.iloc[:, :(self.feature_length)]
        self.Y_hold_out = self.hold_out_set.iloc[:, (self.feature_length):]
        print("X_hold_out-shape:", self.X_hold_out.shape)
        print("Y_hold_out-shape:", self.Y_hold_out.shape)


    def mixup_by_subspecies(self):
        '''Applies Mixup augmentation to the dataset within each subspecies.'''

        # Initialize empty DataFrames to hold the augmented data
        augmented_X = pd.DataFrame(columns=self.X.columns)
        augmented_Y = pd.DataFrame(columns=self.Y.columns)

        # Get unique subspecies from the index of Y_candidates
        subspecies = self.Y.index.unique()

        # Iterate through each subspecies (without the holdout species)
        for species in subspecies:

            # Locate the rows for the current subspecies
            species_mask = self.Y.index == species
            self.X_sub = self.X[species_mask]
            self.Y_sub = self.Y[species_mask]

            # Apply mixup_data function to the current subspecies
            mixed_X_sub, mixed_Y_sub = mixup_data(self.X_sub, self.Y_sub, mix_up_x=self.mix_up_x)

            # Append the mixed data to the augmented DataFrames
            augmented_X = pd.concat([augmented_X, mixed_X_sub])
            augmented_Y = pd.concat([augmented_Y, mixed_Y_sub])

        # set X and Y to augmented data
        self.mixed_X = augmented_X
        self.mixed_Y = augmented_Y

        # create a column called 'source' and set it to 'artificial'
        self.mixed_X['source'] = 'artificial'
        self.mixed_Y['source'] = 'artificial'

        # create a column called 'source' and set it to 'original'
        self.X['source'] = 'original'
        self.Y['source'] = 'original'

    def augment_data_if_needed(self):
        '''
        Executes mixup_by_subspecies if use_augmentation is True and
        concatenates the original data with the augmented data. Reassigns them to X_final and Y_candidates_final.
        If use_augmentation is False,  X and Y_candidates are copied and reassigned to X_final and Y_candidates_final.
        '''
        if self.use_augmentation:
            # Call the mixup_by_subspecies method
            self.mixup_by_subspecies()
            print("Data augmentation applied.")
            # concatenate the original data with the augmented data
            self.X_final = pd.concat([self.X, self.mixed_X])
            self.Y_candidates_final = pd.concat([self.Y, self.mixed_Y])
            # drop the source columns for both
            self.X_final.drop(columns=['source'], inplace=True)
            self.Y_candidates_final.drop(columns=['source'], inplace=True)
            # print final shapes
            print("X_final-shape:", self.X_final.shape)
            print("Y_candidates_final-shape:", self.Y_candidates_final.shape)

        else:
            print("Data augmentation not applied; use_augmentation is set to False.")
            # reassing X_final and Y_candidates_final to X and Y_candidates
            self.X_final = self.X
            self.Y_candidates_final = self.Y

    def run_all_methods(self):
        """
        Runs all methods in the correct order.
        """

        self.preprocesses_seed_data()
        self.augment_data_if_needed()
