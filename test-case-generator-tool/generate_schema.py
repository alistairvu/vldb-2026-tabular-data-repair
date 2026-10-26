import json
import sys


def generate_schema(metadata) -> dict[str, str]:
    record_set = metadata["recordSet"]

    fields = {}

    for item in record_set:
        if "field" not in item:
            continue

        item_fields = item["field"]

        for item in item_fields:
            if "name" not in item:
                continue

            name = item["name"]

            if name in fields and fields[name] != name:
                continue

            fields[name] = item.get("description", name)

    return fields


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_schema.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    metadata_file = f"./test-case-generator-tool/datasets/{dataset}/metadata.json"

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    columns = generate_schema(metadata)

    schema = {"schema_version": 1, "columns": columns}
    print(json.dumps(schema, indent=2))
