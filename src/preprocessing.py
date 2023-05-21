from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
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

   # 2. Encode string categorical features (dtype `object`):
    obj_cols = list(working_train_df.select_dtypes("object").columns)
    low_card_cols = [
        col
        for col in working_train_df[obj_cols].columns
        if working_train_df[col].nunique() <= 2
    ]
    
    high_card_cols = [
        col
        for col in working_train_df[obj_cols].columns
        if working_train_df[col].nunique() > 2
    ]

    # Train the encoders
    ohe = OneHotEncoder(handle_unknown="ignore",   sparse_output=False)
    oe = OrdinalEncoder(handle_unknown="error")

    ohe.fit(working_test_df[high_card_cols])
    oe.fit(working_test_df[low_card_cols])

    # Apply the encoders
    dfs = {
        x: y 
        for x, y in enumerate([working_train_df, working_val_df, working_test_df])
    }

    dfs_hc_econded = {}
    dfs_lc_encoded = {}

    for index, value in dfs.items():
        dfs_hc_econded[index] = pd.DataFrame(
            ohe.transform(value[high_card_cols])
        )
        dfs_hc_econded[index].index = value[high_card_cols].index

        dfs_lc_encoded[index] = pd.DataFrame(
            oe.transform(value[low_card_cols])
        )

        dfs_lc_encoded[index].index = value[low_card_cols].index

        init_col = dfs_hc_econded[index].shape[1]

        fin_col = dfs_hc_econded[index].shape[1] + dfs_lc_encoded[index].shape[1]
        dfs_lc_encoded[index].columns = [
            col_n for col_n in range(init_col, fin_col)
        ]

        dfs[index] = (
            value.drop(columns=[*high_card_cols, *low_card_cols])
            .join(dfs_hc_econded[index])
            .join(dfs_lc_encoded[index])
        )

        dfs[index].columns = dfs[index].columns.astype(str)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    imputer = SimpleImputer(strategy="median")
    imputer.fit(dfs[0])

    for index, value in dfs.items():
        dfs[index] = imputer.transform(value)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    minmax = MinMaxScaler()
    minmax.fit(dfs[0])

    for index, value in dfs.items():
        dfs[index] = minmax.transform(value)

    return dfs[0], dfs[1], dfs[2]
