import argparse

import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--dataset_name", type=str, default=None)
args = parser.parse_args()

if args.dataset_name is None:
    dataset = datasets.load_dataset(
        path=args.dataset_path,
        trust_remote_code=True,
    )
    print(f"Cached dataset from {args.dataset_path}")
else:
    for name in args.dataset_name.split(","):
        dataset = datasets.load_dataset(
            path=args.dataset_path,
            name=name.strip(),
            trust_remote_code=True,
        )
        print(f"Cached dataset {name} from {args.dataset_path}")
