from transformers import PreTrainedTokenizer
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import datasets
import tensorflow as tf

def get_tfds(
    train_file: str,
    eval_file: str,
    test_file: str,
    tokenizer: PreTrainedTokenizer,
    label_column_id: int,
    max_seq_length: Optional[int] = None,
    delimiter=',',
    autogenerate_column_names=False
):
    files = {}

    if train_file is not None:
        files[datasets.Split.TRAIN] = [train_file]
    if eval_file is not None:
        files[datasets.Split.VALIDATION] = [eval_file]
    if test_file is not None:
        files[datasets.Split.TEST] = [test_file]

    ds = datasets.load_dataset("csv", delimiter=delimiter, autogenerate_column_names=autogenerate_column_names, data_files=files)
    # ds = datasets.load_dataset("json", data_files=files, field=['user_id', 'label', 'alpha', 'text'])
    features_name = list(ds[list(files.keys())[0]].features.keys())
    label_name = features_name.pop(label_column_id)
    label_list = list(set(ds[list(files.keys())[0]][label_name]))
    label2id = {label: i for i, label in enumerate(label_list)}
    input_names = ["input_ids"] + tokenizer.model_input_names
    transformed_ds = {}

    if len(features_name) == 1:
        for k in files.keys():
            transformed_ds[k] = ds[k].map(
                lambda example: tokenizer.batch_encode_plus(
                    example[features_name[0]], truncation=True, max_length=max_seq_length, padding="max_length"
                ),
                batched=True,
            )
    elif len(features_name) == 2:
        for k in files.keys():
            transformed_ds[k] = ds[k].map(
                lambda example: tokenizer.batch_encode_plus(
                    (example[features_name[0]], example[features_name[1]]),
                    truncation=True,
                    max_length=max_seq_length,
                    padding="max_length",
                ),
                batched=True,
            )

    def gen_train():
        for ex in transformed_ds[datasets.Split.TRAIN]:
            d = {k: v for k, v in ex.items() if k in input_names}
            label = label2id[ex[label_name]]
            yield (d, label)

    def gen_val():
        for ex in transformed_ds[datasets.Split.VALIDATION]:
            d = {k: v for k, v in ex.items() if k in input_names}
            label = label2id[ex[label_name]]
            yield (d, label)

    def gen_test():
        for ex in transformed_ds[datasets.Split.TEST]:
            d = {k: v for k, v in ex.items() if k in input_names}
            label = label2id[ex[label_name]]
            yield (d, label)

    train_ds = (
        tf.data.Dataset.from_generator(
            gen_train,
            ({k: tf.int32 for k in input_names}, tf.int64),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )
        if datasets.Split.TRAIN in transformed_ds
        else None
    )

    if train_ds is not None:
        train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(len(ds[datasets.Split.TRAIN])))

    val_ds = (
        tf.data.Dataset.from_generator(
            gen_val,
            ({k: tf.int32 for k in input_names}, tf.int64),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )
        if datasets.Split.VALIDATION in transformed_ds
        else None
    )

    if val_ds is not None:
        val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(len(ds[datasets.Split.VALIDATION])))

    test_ds = (
        tf.data.Dataset.from_generator(
            gen_test,
            ({k: tf.int32 for k in input_names}, tf.int64),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )
        if datasets.Split.TEST in transformed_ds
        else None
    )

    if test_ds is not None:
        test_ds = test_ds.apply(tf.data.experimental.assert_cardinality(len(ds[datasets.Split.TEST])))

    return train_ds, val_ds, test_ds, label2id