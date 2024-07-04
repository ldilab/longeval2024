from pathlib import Path

from dataloader import LongEvalLoader


longeval_dataset_dir = Path("/workspace/longeval/datasets")

dataset_2022_dir = longeval_dataset_dir / "2022"
dataset_2022_train_dir = dataset_2022_dir / "publish" / "English"
dataset_2022_train_heldout_loader = LongEvalLoader(dataset_2022_train_dir, "heldout.tsv")
dataset_2022_train_main_loader = LongEvalLoader(dataset_2022_train_dir, "train.tsv")

dataset_2022_test_short_dir = dataset_2022_dir / "test-collection" / "A-Short-July" / "English"
dataset_2022_test_short_loader = LongEvalLoader(dataset_2022_test_short_dir, "test07.tsv", "a-short-july.txt")

dataset_2022_test_long_dir = dataset_2022_dir / "test-collection" / "B-Long-September" / "English"
dataset_2022_test_long_loader = LongEvalLoader(dataset_2022_test_long_dir, "test09.tsv", "b-long-september.txt")

dataset_2023_dir = longeval_dataset_dir / "2023"
dataset_2023_train_dir = dataset_2023_dir / "train_data" / "2023_01" / "English"
dataset_2023_train_main_loader = LongEvalLoader(dataset_2023_train_dir, "train.tsv")

dataset_2023_test_short_dir = dataset_2023_dir / "test_data" / "2023_06" / "English"
dataset_2023_test_short_loader = LongEvalLoader(dataset_2023_test_short_dir, "test.tsv")

dataset_2023_test_long_dir = dataset_2023_dir / "test_data" / "2023_08" / "English"
dataset_2023_test_long_loader = LongEvalLoader(dataset_2023_test_long_dir, "test.tsv")

loaders = {
    "2022": {
        "train": dataset_2022_train_main_loader,
        "test": {
            "short": dataset_2022_test_short_loader,
            "long": dataset_2022_test_long_loader
        }
    },
    "2023": {
        "train": dataset_2023_train_main_loader,
        "test": {
            "short": dataset_2023_test_short_loader,
            "long": dataset_2023_test_long_loader
        }
    }
}
