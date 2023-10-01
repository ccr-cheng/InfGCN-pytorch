# Modifying the config files

By Chaoran Cheng, Oct 1, 2023



This folder contains the configuration files in YAML format.

Most arguments in the YAML files are self-explanatory. There are a few more things to note:

- In the `datasets` field, `type` field specifies the dataset type. All other fields are passed to the dataset class. Note that all train, validation, and test datasets are constructed. You may also pass a `validation` or `test` field to overwrite the arguments for the corresponding datasets (like adding rotation during inference). If not provided, the arguments will be the same as the train dataset.
- In the `model` field, `type` field specifies the model type. All other fields are passed to the model class. See each model class docstring for more details of each argument.
- To expand the model class, make sure the fields under the `model` field in the YAML file match the arguments of your model class's `__init__` function (except for the `type` field).

