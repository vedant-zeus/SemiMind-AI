# SemiMind AI – System Architecture

## Module Dependency Graph (Phase 1)

```
main.py
  |-- src/utils/config.py          (load YAML configuration)
  |-- src/utils/logger.py          (setup rotating file + console logging)
  |-- src/ingestion/dataset_loader.py
  |       |-- load_ai4i()          (CSV loader)
  |       |-- load_secom()         (space-delimited dual-file loader)
  |       |-- load_ai4i_from_config()
  |       \-- load_secom_from_config()
  |-- src/validation/dataset_validator.py
  |       |-- DatasetValidator
  |       |       |-- validate_ai4i()
  |       |       \-- validate_secom()
  |       \-- ValidationReport (dataclass)
  \-- src/inspection/dataset_profiler.py
          |-- DatasetProfiler
          |       |-- profile_ai4i()
          |       \-- profile_secom()
          \-- writes JSON + CSV to datasets/reports/
```

## Configuration Flow

```
configs/dataset_config.yaml
  -> src/utils/config.py::load_config()
  -> Config dataclass (paths, metadata, log settings)
  -> consumed by dataset_loader, profiler, main
```

## Data Flow

```
Raw files (datasets/raw/)
  -> dataset_loader.py  (read, parse, rename columns)
  -> pandas.DataFrame
  -> dataset_validator.py  (quality checks -> ValidationReport)
  -> dataset_profiler.py   (statistics -> JSON + CSV reports)
  -> datasets/reports/
```

## Logging Architecture

```
src/utils/logger.py::setup_logging()
  |-- StreamHandler (console, configurable level)
  \-- RotatingFileHandler (logs/semimind_phase1.log, 10 MB, 5 backups)

All modules call:
  from src.utils.logger import get_logger
  logger = get_logger(__name__)
```

## Test Architecture

```
tests/
  |-- test_dataset_loader.py
  |       |-- TestLoadAI4I      (7 tests, synthetic CSV fixtures)
  |       \-- TestLoadSECOM     (10 tests, synthetic space-delimited fixtures)
  \-- test_dataset_validator.py
          |-- TestAI4IValidation    (9 tests)
          |-- TestSECOMValidation   (6 tests)
          |-- TestValidationReport  (3 tests)
          \-- TestConfigLoading     (3 tests)

Total: 38 tests | All pass as of Phase 1 completion
```

## Phase 2 Architecture Additions (Planned)

```
src/
  preprocessing/
    feature_selector.py     (variance threshold, correlation filter)
    imputer.py              (median / KNN / model-based imputation)
    scaler.py               (StandardScaler, MinMaxScaler wrappers)
    splitter.py             (train/val/test split with temporal awareness)
    imbalance_handler.py    (SMOTE, class weighting)
  pipelines/
    ai4i_pipeline.py
    secom_pipeline.py
```
