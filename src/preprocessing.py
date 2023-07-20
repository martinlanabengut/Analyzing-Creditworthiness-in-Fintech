from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.
    binary_cols = working_train_df.nunique()[working_train_df.nunique() == 2].keys().tolist()
    multi_cols = [i for i in working_train_df.select_dtypes(include=[object]).columns.tolist() if i not in binary_cols]

    # Binary encoding
    ord_enc = OrdinalEncoder()
    working_train_df[binary_cols] = ord_enc.fit_transform(working_train_df[binary_cols])
    working_val_df[binary_cols] = ord_enc.transform(working_val_df[binary_cols])
    working_test_df[binary_cols] = ord_enc.transform(working_test_df[binary_cols])

    # One hot encoding
    one_hot_enc = OneHotEncoder()
    one_hot_cols_train = one_hot_enc.fit_transform(working_train_df[multi_cols]).toarray()
    one_hot_cols_val = one_hot_enc.transform(working_val_df[multi_cols]).toarray()
    one_hot_cols_test = one_hot_enc.transform(working_test_df[multi_cols]).toarray()

    # Create DataFrame from one-hot encoded columns with feature names converted to strings
    one_hot_df_train = pd.DataFrame(one_hot_cols_train, columns=one_hot_enc.get_feature_names_out(input_features=multi_cols).astype(str), index=working_train_df.index)
    one_hot_df_val = pd.DataFrame(one_hot_cols_val, columns=one_hot_enc.get_feature_names_out(input_features=multi_cols).astype(str), index=working_val_df.index)
    one_hot_df_test = pd.DataFrame(one_hot_cols_test, columns=one_hot_enc.get_feature_names_out(input_features=multi_cols).astype(str), index=working_test_df.index)


# Append the one-hot encoded DataFrame to the original working DataFrame
    working_train_df = pd.concat([working_train_df, one_hot_df_train], axis=1)
    working_val_df = pd.concat([working_val_df, one_hot_df_val], axis=1)
    working_test_df = pd.concat([working_test_df, one_hot_df_test], axis=1)

    # Drop the original multi-category columns
    working_train_df.drop(multi_cols, axis=1, inplace=True)
    working_val_df.drop(multi_cols, axis=1, inplace=True)
    working_test_df.drop(multi_cols, axis=1, inplace=True)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    working_train_df[:] = imp.fit_transform(working_train_df)
    working_val_df[:] = imp.transform(working_val_df)
    working_test_df[:] = imp.transform(working_test_df)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.
    scaler = MinMaxScaler()
    working_train_df[:] = scaler.fit_transform(working_train_df)
    working_val_df[:] = scaler.transform(working_val_df)
    working_test_df[:] = scaler.transform(working_test_df)

    return working_train_df.values, working_val_df.values, working_test_df.values
