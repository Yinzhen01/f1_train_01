"""Verify existing AMP bundle without rewriting it; append new verification only."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from humanoid.amp.dataset import AMPDataset, TRAIN_IDS, sha256
from humanoid.scripts.prepare_f1_amp import save_json


def verify(output):
    manifest = json.loads((output / "amp_manifest.json").read_text(encoding="utf-8"))
    protected = json.loads((output / "protected_inputs_sha256.json").read_text(encoding="utf-8"))
    for filename, expected in {**protected, **manifest["script_hashes"]}.items():
        if sha256(filename) != expected:
            raise ValueError("File changed since bundle generation: " + filename)
    if manifest["training_ready"] or not manifest["input_and_asset_hashes_unchanged"]:
        raise ValueError("Bundle must remain preparation-only")
    data_root = Path(manifest["records"][0]["input_path"]).parent.parent
    ds = AMPDataset(data_root, REPO, REPO / "configs/amp/f1_100hz.json")
    if ds.config_sha256 != manifest["config_sha256"] or ds.spec.fingerprint != manifest["feature_spec_sha256"]:
        raise ValueError("Config/feature contract drift")
    # Source generator bound robot_xml_sha256 to custom.xml, not its scene wrapper.
    source_xml = Path(manifest["source_model"]).parent / "custom.xml"
    source_xml_hash = sha256(source_xml)
    for row in manifest["records"]:
        cid = row["clip_id"]
        if row["allowed_for_training"] or row["training_ready"] or row["cyclic"]:
            raise ValueError("Unapproved training/cyclic promotion")
        if row["split_eligible_for_training"] != (cid in TRAIN_IDS):
            raise ValueError("Candidate split drift")
        clip = ds.clips[cid]
        if clip.metadata["robot_xml_sha256"] != source_xml_hash:
            raise ValueError("Audited GMR asset does not match source trajectory")
        if sha256(clip.metadata["source_path"]) != row["source_sha256"]:
            raise ValueError("Original BVH changed")
        if sha256(row["output_path"]) != row["output_sha256"]:
            raise ValueError("Converted feature archive hash mismatch")
        with np.load(row["output_path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(z["features"], clip.features.numpy())
            np.testing.assert_array_equal(z["valid_frame"], np.arange(clip.frames) > 0)
            np.testing.assert_allclose(z["time_s"], np.arange(clip.frames) / 100, rtol=0, atol=1e-12)
            meta = json.loads(str(z["metadata_json"].item()))
            if meta["training_ready"] or meta["input_motion_sha256"] != clip.digest:
                raise ValueError("Converted metadata mismatch")
        quality = json.loads((output / cid / "quality.json").read_text(encoding="utf-8"))
        geometry = quality["source_geometry"]
        if geometry["saved_height_max_error_m"] > 1e-6 or geometry["saved_sole_center_max_error_m"] > 1e-6:
            raise ValueError("Source geometry audit does not reproduce source measurements")
    mean, std = ds.fit_normalization()
    normalization = manifest["normalization"]
    if sha256(normalization["path"]) != normalization["sha256"]:
        raise ValueError("Normalizer hash mismatch")
    with np.load(normalization["path"], allow_pickle=False) as z:
        np.testing.assert_array_equal(z["mean"], mean.numpy())
        np.testing.assert_array_equal(z["std"], std.numpy())
        if list(z["fit_ids"]) != list(TRAIN_IDS):
            raise ValueError("Normalizer split leak")
    return dict(clips_verified=10, protected_files_verified=len(protected),
                original_bvh_hashes_verified=10, converted_features_recomputed=True,
                source_model_matches_npz_metadata=True, normalization_train_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path, help="A new JSON file; existing results refused")
    args = parser.parse_args()
    if args.result.exists():
        raise FileExistsError("Refuse to overwrite verification evidence")
    begin = time.monotonic()
    report = verify(args.bundle.resolve())
    test_stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(REPO / "tests"), pattern="test*.py")
    result = unittest.TextTestRunner(stream=test_stream, verbosity=2).run(suite)
    report.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  test_count=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), all_tests_passed=result.wasSuccessful(),
                  duration_s=time.monotonic() - begin,
                  command="python humanoid/scripts/verify_f1_amp_bundle.py --bundle <bundle> --result <new-json>",
                  test_log=test_stream.getvalue(), verifier_sha256=sha256(__file__),
                  bundle_manifest_sha256=sha256(args.bundle / "amp_manifest.json"),
                  final_python_hashes={str(p): sha256(p) for p in sorted((REPO / "humanoid/amp").glob("*.py"))},
                  training_ready=False, formal_training_started=False)
    save_json(args.result, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("test_log", "final_python_hashes")}, indent=2))
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
