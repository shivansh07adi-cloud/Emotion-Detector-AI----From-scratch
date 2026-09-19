"""Download the dair-ai/emotion dataset into data/train.csv, validation.csv, test.csv.

It has 16,000 training, 2,000 validation and 2,000 test tweets labelled with
sadness, joy, love, anger, fear or surprise.

    pip install datasets
    python download_data.py
"""
import csv
import os
import sys

try:
    from datasets import load_dataset  # pylint: disable=import-error
except ImportError:
    load_dataset = None


def main():
    """Fetch the dataset and save each split as a CSV file."""
    if load_dataset is None:
        sys.exit('Install the downloader first: pip install datasets')
    dataset = load_dataset('dair-ai/emotion', 'split')
    names = dataset['train'].features['label'].names
    os.makedirs('data', exist_ok=True)
    for split in ('train', 'validation', 'test'):
        path = os.path.join('data', f'{split}.csv')
        with open(path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['text', 'label'])
            for row in dataset[split]:
                writer.writerow([row['text'], names[row['label']]])
        print(f'Saved {path} ({len(dataset[split]):,} rows)')


if __name__ == '__main__':
    main()
