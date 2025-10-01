from datasets import load_dataset


def get_train_test_dataset(*args, **kwargs):
    num_train_data = kwargs.get("num_train_data", None)
    if num_train_data:
        test_dataset = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{num_train_data}]")
    else:
        test_dataset = load_dataset("HuggingFaceH4/MATH-500", split='test')

    train_dataset = None
    return train_dataset, test_dataset
