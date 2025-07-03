import random
import os
from datasets import Dataset

# Constants
MIN_VAL = -10**5
MAX_VAL = 10**5
SEED = 1234
DATASET_SIZE = 10000

random.seed(1234)

SYSTEM_PROMPT = \
'''
You are a number multiplier. Your goal is to output the result of two numbers, prefixed with ####.

Here is an example:

v1 = 1232.213
v2 = 4123.452

#### 5080971.1593
'''

def random_rows():
    for i in range(DATASET_SIZE):
        v1 = round(random.uniform(MIN_VAL, MAX_VAL), 4)
        v2 = round(random.uniform(MIN_VAL, MAX_VAL), 4)
        res = {
            "v1": v1,
            "v2": v2,
            "product": round(v1 * v2, 4)
        }
        yield res

def make_map_fn(split):
    def process_fn(example, idx):
        v1 = example.pop('v1')
        v2 = example.pop('v2')
        query = f"v1 = {v1}\nv2 = {v2}"
        solution = str(example.pop('product'))
        data = {
            "data_source": "numbers",
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": query,
                }
            ],
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    local_dir = "~/data/"
    numbers = Dataset.from_generator(random_rows).train_test_split(test_size=0.2)
    numbers_train = numbers['train']
    numbers_test = numbers['test']
    train_dataset = numbers_train.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = numbers_test.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(local_dir, "numbers_train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "numbers_test.parquet"))

