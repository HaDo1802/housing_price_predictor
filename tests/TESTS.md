Test strategy

Why these tests exist
- This project is an MLOps system, not just a Python package.
- The main failure modes are config drift, training-serving schema drift, stale artifacts, and backward-incompatible API changes.
- The test suite should protect contracts first, then behavior inside those contracts.

What each test file protects
- `tests/unit/config/test_config_manager.py`
  Protects the single-file config contract. If config loading or key sections drift, training should fail in test before runtime.
- `tests/unit/features/test_training_schema.py`
  Protects the canonical training feature contract. It ensures model features are internally consistent and never overlap with dropped columns.
- `tests/unit/features/test_preprocessor_contract.py`
  Protects the preprocessing boundary. It verifies current required columns fit successfully, missing required columns fail loudly, and extra columns are ignored safely.
- `tests/unit/api/test_feature_map.py`
  Protects the serving alias layer. API field aliases and display metadata must always map back to real training features.
- `tests/unit/api/test_schemas.py`
  Protects request validation and backward compatibility. New optional fields must not break older clients, while invalid core inputs must still fail.
- `tests/unit/data/test_splitter.py`
  Protects split shapes and target alignment.
- `tests/unit/models/test_evaluator.py`
  Protects metric output shape for downstream logging and reporting.
- `tests/integration/test_train_serve_contract.py`
  Protects the most important end-to-end path: training artifacts produced by the training code can be loaded and used by the production inference pipeline.

Maintenance rules
- When adding a new training feature:
  update `training_schema.py`, `conf/config.yaml`, and any affected API mappings, then update the schema, preprocessor, and API tests in the same change.
- When changing request payloads:
  add or update a schema test proving both the intended new payload and any supported legacy payloads.
- When changing artifact format:
  update the train-to-serve integration test so saved artifacts still load through `InferencePipeline`.

Rules:
- Keep unit tests small, local, and deterministic.
- Prefer synthetic data over warehouse extracts.
- Prefer contract assertions over brittle exact-metric assertions.
- Add deeper workflow tests only when the external contract is already stable.
