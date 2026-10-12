import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
METADATA = ROOT / "artifacts/image_metadata.csv"
OUTPUT = ROOT / "splits.csv"
SEED = 42
VAL_FRACTION = 0.15

if not METADATA.exists():
    raise FileNotFoundError(METADATA)

with METADATA.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    required = {"path", "label", "patient_id", "split_original"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing columns: {required - set(reader.fieldnames or [])}")
    rows = list(reader)

if len(rows) != 5856:
    raise ValueError(f"Expected 5856 metadata rows, got {len(rows)}")

for row in rows:
    row["label"] = row["label"].strip().upper()
    row["patient_id"] = row["patient_id"].strip()
    row["split_original"] = row["split_original"].strip().lower()

    if not row["patient_id"] or row["label"] not in {"NORMAL", "PNEUMONIA"}:
        raise ValueError(f"Invalid metadata row: {row}")

if len({r["path"] for r in rows}) != len(rows):
    raise ValueError("Duplicate image paths found")

test = [r for r in rows if r["split_original"] == "test"]
pool = [r for r in rows if r["split_original"] in {"train", "val"}]

if len(test) != 624:
    raise ValueError(f"Expected 624 test images, got {len(test)}")

if len(pool) + len(test) != len(rows):
    raise ValueError("Unexpected original split values")

# Patient identifiers are filename-derived proxies, not verified identities.
# Test membership is preserved exactly as supplied by the original metadata.
test_ids = {r["patient_id"] for r in test}
original_train_ids = {
    r["patient_id"] for r in rows if r["split_original"] == "train"
}
clean_test_ids = test_ids - original_train_ids

# Group all original train+validation images by patient ID.
groups = defaultdict(list)
for row in pool:
    # Never place a test-overlapping identifier in train or validation.
    if row["patient_id"] not in test_ids:
        groups[row["patient_id"]].append(row)

# A patient group must have one label for this stratified group split.
for patient_id, group in groups.items():
    labels = {r["label"] for r in group}
    if len(labels) != 1:
        raise ValueError(
            f"Patient proxy {patient_id!r} has multiple labels: {labels}. "
            "Inspect before splitting."
        )

by_label = defaultdict(list)
for patient_id, group in groups.items():
    by_label[group[0]["label"]].append((patient_id, group))

rng = random.Random(SEED)
val_ids = set()

# Select whole patient groups, targeting 15% of images per class.
for label, label_groups in sorted(by_label.items()):
    rng.shuffle(label_groups)
    target = round(
        sum(len(group) for _, group in label_groups) * VAL_FRACTION
    )

    # Greedily approach the target image count without splitting groups.
    selected = []
    count = 0
    for patient_id, group in label_groups:
        if abs((count + len(group)) - target) < abs(count - target):
            selected.append(patient_id)
            count += len(group)

    val_ids.update(selected)

# Build output rows without changing original paths or split labels.
output_rows = []
for row in rows:
    new = dict(row)

    if row["split_original"] == "test":
        new["split_new"] = "test"
        new["test_cohort"] = (
            "test_full;test_clean"
            if row["patient_id"] in clean_test_ids
            else "test_full"
        )
    else:
        if row["patient_id"] in test_ids:
            new["split_new"] = "excluded_overlap"
        else:
            new["split_new"] = (
                "val" if row["patient_id"] in val_ids else "train"
            )
        new["test_cohort"] = ""

    output_rows.append(new)

train_ids = {
    r["patient_id"] for r in output_rows if r["split_new"] == "train"
}
val_ids_actual = {
    r["patient_id"] for r in output_rows if r["split_new"] == "val"
}
new_test_ids = {
    r["patient_id"] for r in output_rows if r["split_new"] == "test"
}

assert not (train_ids & val_ids_actual), "Train/validation leakage"
assert not (train_ids & new_test_ids), "Train/test identifier overlap"
assert not (val_ids_actual & new_test_ids), "Validation/test identifier overlap"

assert sum(r["split_new"] == "test" for r in output_rows) == 624
assert sum("test_clean" in r["test_cohort"].split(";")
           for r in output_rows) == 304

# Ensure original test rows and paths are unchanged.
original_test_paths = {r["path"] for r in test}
new_test_paths = {
    r["path"] for r in output_rows if r["split_new"] == "test"
}
assert original_test_paths == new_test_paths

# Write only after all assertions pass.
fields = [
    "path", "label", "patient_id",
    "split_original", "split_new", "test_cohort"
]
with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in output_rows)

print("Manifest created:", OUTPUT)
print("Rows:", len(output_rows))
print("Split image counts:", dict(Counter(r["split_new"] for r in output_rows)))
print("Split class counts:")
for split in ("train", "val", "test"):
    print(split, dict(Counter(
        r["label"] for r in output_rows if r["split_new"] == split
    )))

train_counts = Counter(
    r["label"] for r in output_rows if r["split_new"] == "train"
)
if set(train_counts) != {"NORMAL", "PNEUMONIA"}:
    raise ValueError(
        f"Expected both classes in training data, got: {dict(train_counts)}"
    )
total_train = sum(train_counts.values())
class_weights = {
    label: round(total_train / (len(train_counts) * count), 6)
    for label, count in sorted(train_counts.items())
}
print("Training class weights:", class_weights)
print("Clean test images:", sum(
    "test_clean" in r["test_cohort"].split(";") for r in output_rows
))
print("All leakage assertions passed.")
print("Note: identifiers are filename-derived proxies.")
