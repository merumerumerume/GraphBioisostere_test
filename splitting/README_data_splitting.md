# Data Splitting Tool - Integrated Version

## Overview

An optimized data splitting tool that performs data splitting based on TID-based 5-fold CV, integrating TID frequency filtering and leak exclusion.

## Three Main Approaches

### Approach 1: Leak Exclusion Only
```
Basic split → Exclude pairs appearing in entire original test from train
```
- Test data: All data
- Train data: Exclude pairs from entire test

### Approach 2: TID Filtering + Full Leak Exclusion
```
Basic split → Filter test to TID3+ → Exclude pairs appearing in entire original test from train
```
- Test data: Filtered to TID3+
- Train data: Exclude pairs from entire original test (same exclusion amount as Approach 1)

### Approach 3: TID Filtering + Optimized Leak Exclusion (**Recommended**)
```
Basic split → Filter test to TID3+ → Exclude pairs appearing only in filtered test from train
```
- Test data: Filtered to TID3+
- Train data: Exclude only pairs from filtered test (**improved train data retention rate**)

**Important**: In Approach 3, as the test becomes smaller, fewer pairs are excluded from train, allowing maximum utilization of training data.

## Key Features

### 1. Target-Based 5-Fold CV
- Perform fold splitting by TID units to prevent assays from the same target from spanning train/test sets
- Prevents major causes of data leakage

### 2. TID Frequency Filtering (Optional)
- Only MMPs appearing in the specified number or more TIDs across the entire dataset are used as test subjects
- Enables evaluation of prediction accuracy for more generalizable transformation patterns
- **Applied only to test data** (train/validation remain unchanged)

### 3. Leak Exclusion (Optional)
- When the same MMP appears across different targets, exclude it from train data
- Selectable exclusion criteria:
  - `full_test`: Exclude based on entire original test (Approaches 1, 2)
  - `filtered_test`: Exclude based only on TID-filtered test (Approach 3, **Recommended**)
- **Applied only to train data** (validation is retained)

### 4. Optimized Processing Flow
```
Input data
  ↓
TID-based 5-fold split
  ↓
[Optional] TID frequency filtering (test → becomes smaller)
  ↓
[Optional] Leak exclusion (train → excluded)
  ・full_test mode: Exclude from entire original test
  ・filtered_test mode: Exclude only from filtered test (recommended)
  ↓
Output data
```

**Important**: Using `--leak_removal_mode filtered_test` improves train data retention rate (approximately 5% improvement).

## Usage

### Basic Usage

```bash
python data_splitting.py \
    --csv_path <CSV_FILE> \
    --data_path <PT_FILE> \
    --output_dir <OUTPUT_DIR> \
    --pkl_output <PKL_FILE> \
    [options]
```

### Required Parameters

- `--csv_path`: Path to input CSV file
- `--data_path`: Path to input .pt file (must have matching number of rows with CSV)
- `--output_dir`: Output directory for dataset_cv{i}.pt files
- `--pkl_output`: Path to pkl file for saving CV split information

### Optional Parameters

- `--n_folds`: Number of folds (default: 5)
- `--seed`: Random seed (default: 40)
- `--min_tid_count`: Threshold for TID frequency filtering (e.g., 3 for TID3+)
  - No filtering if not specified
- `--remove_mmp_leak`: Exclude MMP leak (recommended)
- `--remove_frag_leak`: Exclude Fragment leak (optional)
- `--leak_removal_mode`: Leak exclusion mode (`full_test` or `filtered_test`)
  - `full_test`: Exclude based on entire original test (default)
  - `filtered_test`: Exclude based only on TID-filtered test (recommended, improved train data retention)

## Usage Examples

### Case 1: Basic Split Only (Baseline)
```bash
python data_splitting.py \
    --csv_path /path/to/dataset.csv \
    --data_path /path/to/dataset.pt \
    --output_dir ./output/basic \
    --pkl_output ./split_basic.pkl \
    --seed 41
```

### Case 2: Approach 1 - MMP Leak Exclusion Only
```bash
python data_splitting.py \
    --csv_path /path/to/dataset.csv \
    --data_path /path/to/dataset.pt \
    --output_dir ./output/no_leak \
    --pkl_output ./split_no_leak.pkl \
    --seed 41 \
    --remove_mmp_leak \
    --leak_removal_mode full_test
```

### Case 3: Approach 2 - TID3+ + Full Leak Exclusion
```bash
python data_splitting.py \
    --csv_path /path/to/dataset.csv \
    --data_path /path/to/dataset.pt \
    --output_dir ./output/tid3_no_leak_fulltest \
    --pkl_output ./split_tid3_no_leak_fulltest.pkl \
    --seed 41 \
    --min_tid_count 3 \
    --remove_mmp_leak \
    --leak_removal_mode full_test
```

### Case 4: Approach 3 - TID3+ + Optimized Leak Exclusion (**Recommended**)
```bash
python data_splitting.py \
    --csv_path /path/to/dataset.csv \
    --data_path /path/to/dataset.pt \
    --output_dir ./output/tid3_no_leak \
    --pkl_output ./split_tid3_no_leak.pkl \
    --seed 41 \
    --min_tid_count 3 \
    --remove_mmp_leak \
    --leak_removal_mode filtered_test
```

### Case 5: Approach 3 - TID5+ + Optimized Leak Exclusion (More Strict)
```bash
python data_splitting.py \
    --csv_path /path/to/dataset.csv \
    --data_path /path/to/dataset.pt \
    --output_dir ./output/tid5_no_leak \
    --pkl_output ./split_tid5_no_leak.pkl \
    --seed 41 \
    --min_tid_count 5 \
    --remove_mmp_leak \
    --leak_removal_mode filtered_test
```

## Output Files

### 1. PKL File
Saved to the path specified by `pkl_output`.
- Information for 5-fold split (train/val/test record lists)
- Can be used in subsequent experiments

### 2. Dataset PT Files
`output_dir/dataset_cv{i}.pt` (i=0,1,2,3,4)
- Dataset in torch.Data format for each fold
- Dictionary format with train/valid/test keys

### 3. Statistics Files
`output_dir/split_statistics.json` and `.txt`
- Detailed statistics for each processing step
- Data size, exclusion rate, retention rate, etc.

## Statistics Content

### Initial Split Statistics
- Size after basic TID-based split

### TID Frequency Filter (Optional)
- Test data size before and after filtering
- Retention rate for each fold

### MMP Leak Removal (Optional)
- Train data size before and after leak exclusion
- Number of exclusions and exclusion rate for each fold

### Fragment Leak Removal (Optional)
- Train data size before and after fragment leak exclusion
- Number of exclusions and exclusion rate for each fold

### Final Split Statistics
- Final sizes of train/val/test

## Recommended Settings

### General Use
```bash
--min_tid_count 3 --remove_mmp_leak
```
- Focus on highly generalizable transformation patterns with TID3+
- Clean evaluation with MMP leak exclusion

### More Strict Evaluation
```bash
--min_tid_count 5 --remove_mmp_leak
```
- Only more generalizable transformation patterns with TID5+
- Test data is reduced but provides more reliable evaluation

### Maximum Data Utilization
```bash
--remove_mmp_leak
```
- No TID filtering, only leak exclusion
- When you want to maximize data utilization

## Batch Execution

To execute multiple configurations at once, use `run_data_splitting_examples.sh`:

```bash
chmod +x run_data_splitting_examples.sh
./run_data_splitting_examples.sh
```

This script executes data splitting with the following 5 configurations:
1. Basic split
2. MMP leak exclusion only
3. TID3+ filtering only
4. TID3+ + MMP leak exclusion (recommended)
5. TID5+ + MMP leak exclusion

## Conducting Comparative Experiments

Using datasets generated with different settings, you can compare:

1. **Data Retention Rate**: Retention rate of train/test data
2. **Prediction Accuracy**: Prediction performance with each setting
3. **Generalizability**: Prediction accuracy for MMPs with high TID frequency
4. **Leak Impact**: Accuracy changes due to leak exclusion

## Notes

1. **CSV and .pt File Consistency**
   - Always ensure the number of rows in CSV and .pt files match
   - Automatically checked during script execution

2. **Processing Order**
   - TID filtering → Leak exclusion order is optimal
   - This order improves train data retention rate

3. **Memory Usage**
   - For large datasets, ensure sufficient memory is available

4. **Reproducibility**
   - Using the same seed will produce the same split results

## Troubleshooting

### CSV and .pt File Row Count Mismatch
```
✗ ERROR: Length mismatch!
```
→ Confirm you are using the correct CSV and .pt file pair

### Out of Memory Error
→ Consider processing with smaller batch sizes or increase memory

### Statistics Files Not Generated
→ Check write permissions to the output directory

## Example for Paper Publication

### Methods 2.3 Data Splitting and Leak Evaluation

In this study, we implemented target-based 5-fold cross-validation. Fold splitting was performed by TID (Target ID) units to prevent assay data from the same target from spanning train/test sets.

Furthermore, to prevent data leakage across different targets, we implemented the following procedures:

1. **TID Frequency Filtering**: Only MMPs (Matched Molecular Pairs) appearing in three or more targets across the entire dataset were used as test subjects. This enables evaluation of prediction accuracy for more generalizable transformation patterns.

2. **MMP Leak Exclusion**: Molecular pairs identical to MMPs appearing in test data were excluded from train data. This prevents leakage due to identical MMPs across different targets.

Through this process, approximately 40% of test data and 97% of train data were retained, enabling clean and reliable evaluation.
