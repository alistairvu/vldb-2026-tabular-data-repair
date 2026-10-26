import re
import sys


def remove_non_ascii(text):
    # Matches any character that is NOT in the ASCII range \x00-\x7F
    return re.sub(r"[^\x00-\x7F]+", "", text)


def clean_file(input_path, output_path):
    with open(input_path, "r") as f:
        text = f.read()
    cleaned_text = remove_non_ascii(text)
    with open(output_path, "w") as f:
        f.write(cleaned_text)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_file.py <input_path> <output_path>")
        sys.exit(1)
    clean_file(sys.argv[1], sys.argv[2])
