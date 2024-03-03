
import os

import numpy as np
import pandas as pd


def evaluate_correctness(out):
    # TODO: TEST_FILE WOULD BE RELATIVE TO INPUT SUBDIRECTORY NOW-
    TEST_FILE = 'inputs/OMRDataset.csv'
    if(os.path.exists(TEST_FILE)):
        print("\nStarting evaluation for: " + TEST_FILE)

        TEST_COLS = ['file_id'] + out.respCols
        y_df = pd.read_csv(
            TEST_FILE, dtype=str)[TEST_COLS].replace(
            np.nan, '', regex=True).set_index('file_id')

        if(np.any(y_df.index.duplicated)):
            y_df_filtered = y_df.loc[~y_df.index.duplicated(keep='first')]
            print(
                "WARNING: Found duplicate File-ids in file %s. Removed %d rows from testing data. Rows remaining: %d" %
                (TEST_FILE, y_df.shape[0] - y_df_filtered.shape[0], y_df_filtered.shape[0]))
            y_df = y_df_filtered

        x_df = pd.DataFrame(
            out.OUTPUT_SET,
            dtype=str,
            columns=TEST_COLS).set_index('file_id')
        # print("x_df",x_df.head())
        # print("\ny_df",y_df.head())
        intersection = y_df.index.intersection(x_df.index)

        # Checking if the merge is okay
        if(intersection.size == x_df.index.size):
            y_df = y_df.loc[intersection]
            x_df['TestResult'] = (x_df == y_df).all(axis=1).astype(int)
            print(x_df.head())
            print("\n\t Accuracy on the %s Dataset: %.6f" %
                  (TEST_FILE, (x_df['TestResult'].sum() / x_df.shape[0])))
        else:
            print(
                "\nERROR: Insufficient Testing Data: Have you appended MultiMarked data yet?")
            print("Missing File-ids: ",
                  list(x_df.index.difference(intersection)))

