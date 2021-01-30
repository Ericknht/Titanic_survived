import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


def main():
    df_train = pd.read_csv('data/titanic-train.csv')
    y_target_func = _get_y(df_train, 'Survived')
    x_features_one_func = _get_x(df_train, 'Age', 'Embarked', 'S',
                                    ['PassengerId', 'Survived',
                                    'Name', 'Ticket', 'Cabin']
                                )
    print(y_target_func)
    print(x_features_one_func)


def _encoder_columns(df, column):
    label_encoder = preprocessing.LabelEncoder()
    encoder_column = label_encoder.fit_transform(train_df[column])
    df[column] = encoder_column

    return df


def _fillna_for_median(df, column):
    df[column].fillna(df[column].median())

    return df


def _fillna_for_string(df, column, character):
    df[column].fillna(character)

    return df


def _drop_columns(df, columns = []):
    df_drop_columns = df.drop(columns, axis=1)

    return df_drop_columns


def _detect_categorical_cols(df):
    categorical_cols = [cname for cname in df.columns if
                            df[cname].nunique() < 10 and
                            df[cname].dtype == 'object'
                        ]

    return categorical_cols


def _detect_numerical_cols(df):
    numerical_cols = [cname for cname in df.columns if
                        df[cname].dtype in ['int64', 'float64', 'int32']
                    ]

    return numerical_cols


def _reorder_df(df):
    categorical_cols = _detect_categorical_cols(df)
    numerical_cols = _detect_numerical_cols(df)

    selected_cols = categorical_cols + numerical_cols
    return df[selected_cols]


def _get_dummies_df(df):
    dummy_encoded_df = pd.get_dummies(df)

    return dummy_encoded_df


def _get_y(df, column):
    return df[column].values


def _get_x(df, col_median, col_str, char_str, cols_drop, col_encoder = None):
    if col_encoder:
        df = _encoder_columns(df, col_encoder)
    df = _fillna_for_median(df, col_median)
    df = _fillna_for_string(df, col_str, char_str)
    df = _drop_columns(df, cols_drop)
    df = _reorder_df(df)
    df = _get_dummies_df(df)

    return df.values


if __name__ == '__main__':
    main()