import csv
import sys


def get_accuracy(corrupted_file, cleaned_file, original_file):
    corrupted_data = []
    cleaned_data = []
    original_data = []

    with open(corrupted_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            corrupted_data.append(list(row))

    with open(cleaned_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned_data.append(list(row))

    with open(original_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            original_data.append(list(row))

    zipped = zip(corrupted_data, cleaned_data, original_data)

    total = 0
    correct = 0

    for corrupted, cleaned, original in zipped:
        for corr, clean, orig in zip(corrupted, cleaned, original):
            if corr != orig:
                total += 1
                if clean == orig:
                    correct += 1

    print(f"Corrupted: {total}, Correct: {correct}")
    print(f"Accuracy: {round(correct / total * 100, 2)}%")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python get_accuracy.py <corrupted_file> <cleaned_file> <original_file>"
        )
        sys.exit(1)

    corrupted_file = sys.argv[1]
    cleaned_file = sys.argv[2]
    original_file = sys.argv[3]

    get_accuracy(corrupted_file, cleaned_file, original_file)
