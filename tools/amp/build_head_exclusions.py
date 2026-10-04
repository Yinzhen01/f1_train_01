"""Read-only source110 reconstruction -> exclusive exact SHA256 input index.

No simulation, policy forward, fitting, cloud request or source rewrite. This
tool verifies the new allow_pickle=False file against a SECOND full independent
call of the existing CPU source reconstruction before reporting its hash.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from humanoid.amp.head_exclusions import (SOURCE_ENDPOINT_SHA, load_exclusion_index,
                                         write_exclusion_index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-endpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.source_endpoint.resolve() == args.output.resolve():
        raise ValueError('Source and derived output must differ')
    # Import torch only in the local offline builder, never in the native loader.
    import torch
    from humanoid.scripts import collect_amp_head_cohort as collector
    torch.set_num_threads(2)
    if collector.file_sha(args.source_endpoint) != SOURCE_ENDPOINT_SHA:
        raise ValueError('Not the immutable original110 regression endpoint')
    manifest, initial_arrays = collector._read_bundle(args.source_endpoint, torch)
    identity = copy.deepcopy(manifest['identity'])
    del manifest, initial_arrays
    index, proof, source_manifest = collector.source_exclusions(args.source_endpoint, identity, torch)
    proof['source_environment'] = copy.deepcopy(source_manifest['environment'])
    proof['source_runtime'] = copy.deepcopy(source_manifest['runtime'])
    collector_sha = collector.file_sha(Path(collector.__file__))
    digest = write_exclusion_index(args.output, index, proof, identity=identity, builder_sha=collector_sha)
    loaded, loaded_proof = load_exclusion_index(args.output, expected_file_sha=digest, identity=identity)
    restored_proof = dict(loaded_proof['source_regression'],
        source_environment=loaded_proof['source_environment'], source_runtime=loaded_proof['source_runtime'])
    if loaded != index or restored_proof != proof:
        raise ValueError('Derived index readback differs from exact source reconstruction')
    rebuilt, reproof, remanifest = collector.source_exclusions(args.source_endpoint, identity, torch)
    reproof['source_environment'] = copy.deepcopy(remanifest['environment'])
    reproof['source_runtime'] = copy.deepcopy(remanifest['runtime'])
    if rebuilt != loaded or reproof != proof:
        raise ValueError('Second independent source reconstruction differs from saved index/proof')
    if (collector.file_sha(args.source_endpoint) != SOURCE_ENDPOINT_SHA or
            collector.file_sha(Path(collector.__file__)) != collector_sha):
        raise ValueError('Source endpoint/builder changed during reconstruction')
    print(json.dumps(dict(output=str(args.output.resolve()), sha256=digest,
        bytes=args.output.stat().st_size, source_endpoint_sha256=SOURCE_ENDPOINT_SHA,
        counts=loaded_proof['derived_index']['counts'], complete_episode_labels=32,
        readback_exact=True, second_source_reconstruction_exact=True,
        source_preserved=True, no_policy_forward=True, no_training=True,
        effectiveness_verified=False, dr_unlocked=False), sort_keys=True))


if __name__ == '__main__':
    main()
