import pandas as pd

def df_lookup(df, row_labels, col_labels, index='rows'):
    """Look up a series of locations in a data frame df, with the row and
    column indicies given by row_labels and col_labels. This replaces the
    DataFrame.lookup() that was part of previous versions of pandas.

    Returns
    A series containing the looked-up values

    Parameters
    df
    row_labels: the row indicies of the values to be fetched
    col_labels: the column indicies of the values to be fetched
    eg. if row_labels=[1,3] and col_labels=['b','e'], the function would return a Series
    consisting of df.loc[1,'b'] and df.loc[3,'e']

    Taken from https://stackoverflow.com/questions/71156253/alternative-to-df-lookuprow-col?noredirect=1&lq=1
    """
    if index == 'cols':
        return pd.Series(
            df.to_numpy()[
                df.index.get_indexer(row_labels),
                df.columns.get_indexer(col_labels),
            ],
            index=col_labels,
        )
    else:
        return pd.Series(
            df.to_numpy()[
                df.index.get_indexer(row_labels),
                df.columns.get_indexer(col_labels),
            ],
            index=row_labels,
        )