# ChestScan — Original Split Leakage Audit

## Dataset verification
- Total images: 5856
- Unique parsed identifiers: 3118
- Missing identifiers: 0

## Original split counts
|                        |   count |
|:-----------------------|--------:|
| ('test', 'NORMAL')     |     234 |
| ('test', 'PNEUMONIA')  |     390 |
| ('train', 'NORMAL')    |    1341 |
| ('train', 'PNEUMONIA') |    3875 |
| ('val', 'NORMAL')      |       8 |
| ('val', 'PNEUMONIA')   |       8 |

## Train-test overlap
- Overlapping parsed patient IDs: 170
- Test images belonging to overlapping IDs: 320
- Test images belonging to test-only IDs: 304
- Test-only NORMAL images: 234
- Test-only PNEUMONIA images: 70

## Interpretation and limitations
- The original test set contains 624 images.
- 320 test images have parsed identifiers also present in training.
- 304 test images have identifiers absent from training.
- Restricting evaluation to test-only identifiers changes the test cohort and class balance.
- Filename-derived identifiers are proxies; patient identity has not been independently verified.
- Preserve the original dataset and test files. Do not silently remove overlapping images.
- ChestScan is a screening aid, not a diagnostic tool.
