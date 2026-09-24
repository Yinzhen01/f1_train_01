"""Prepare a NEW AMP audit bundle. Never overwrite input data or start training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from humanoid.amp.dataset import AMPDataset, TRAIN_IDS, HOLDOUT_IDS, sha256
from humanoid.amp.dry_run import run_cpu_dry_run
from humanoid.amp.quality import SourceModelAudit, clip_quality


def snapshot(paths):
    result = {}
    for root in paths:
        root = Path(root)
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        for p in files:
            result[str(p.resolve())] = sha256(p)
    return result


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO / "configs/amp/f1_100hz.json")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data_root, output = args.data_root.resolve(), args.output.resolve()
    protected_roots = [data_root, args.source_model.resolve().parent, REPO / "resources/robots"]
    if output.exists():
        raise FileExistsError("Refuse to overwrite existing output: " + str(output))
    if any(output == p or p in output.parents for p in protected_roots):
        raise ValueError("Output must not be inside input data or robot assets")
    before = snapshot(protected_roots)
    dataset = AMPDataset(data_root, REPO, args.config)
    source = SourceModelAudit(args.source_model, dataset.spec)
    differences = source.model_differences(dataset.kinematics)
    output.mkdir(parents=True, exist_ok=False)
    rows, reports = [], []
    for cid, clip in dataset.clips.items():
        quality = clip_quality(clip, dataset.kinematics, source)
        original_source = Path(clip.metadata["source_path"])
        source_verified = original_source.is_file() and sha256(original_source) == clip.metadata["source_sha256"]
        quality["original_source_sha_verified"] = bool(source_verified)
        reports.append(quality)
        destination = output / cid
        destination.mkdir()
        np.savez_compressed(destination / "amp_features.npz", features=clip.features.numpy(),
                            time_s=clip.arrays["time_s"], fps=100, valid_frame=np.arange(clip.frames) > 0,
                            feature_spec_json=json.dumps(dataset.spec.description()),
                            metadata_json=json.dumps(dict(input_motion_sha256=clip.digest,
                                training_ready=False, cyclic=False, split=clip.record["split"],
                                keypoint_model="training_urdf_fk", urdf_lf_sha256=dataset.kinematics.lf_sha256)))
        save_json(destination / "quality.json", quality)
        row = dict(clip_id=cid, split=clip.record["split"], fps=100, frames=clip.frames,
                   duration_s=clip.duration, source_sha256=clip.metadata["source_sha256"],
                   original_source_sha_verified=bool(source_verified), motion_npz_sha256=clip.digest,
                   input_path=str(clip.path), output_path=str(destination / "amp_features.npz"),
                   output_sha256=sha256(destination / "amp_features.npz"),
                   split_eligible_for_training=cid in TRAIN_IDS, allowed_for_training=False,
                   allowed_for_evaluation=cid in HOLDOUT_IDS, evaluation_scope="offline final evaluation only; never tuning" if cid in HOLDOUT_IDS else "diagnostic only",
                   manual_postprocess_review_hold=clip.record["manual_postprocess_review_hold"],
                   allowed_for_normalization=cid in TRAIN_IDS, training_ready=False, cyclic=False,
                   legal_start_frames=[clip.first_start, clip.last_start])
        rows.append(row)
        print(cid, "training-limit samples:", quality["training_joint_limit_violations"], flush=True)
    mean, std = dataset.fit_normalization()
    np.savez_compressed(output / "normalization.npz", mean=mean.numpy(), std=std.numpy(), fit_ids=np.array(TRAIN_IDS))
    dry_run = run_cpu_dry_run(dataset, args.seed)
    save_json(output / "cpu_dry_run.json", dry_run)
    save_json(output / "effective_config.json", dataset.cfg)
    dataset.verify_unchanged()
    after = snapshot(protected_roots)
    if before != after:
        raise RuntimeError("Protected input or asset changed during run")
    save_json(output / "protected_inputs_sha256.json", before)
    manifest = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip(),
                    branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=str(REPO), text=True).strip(),
                    config_sha256=dataset.config_sha256, source_manifest_sha256=dataset.manifest_sha256,
                    feature_spec=dataset.spec.description(), feature_spec_sha256=dataset.spec.fingerprint,
                    training_urdf_sha256=dataset.kinematics.sha256, training_urdf_lf_sha256=dataset.kinematics.lf_sha256,
                    keypoint_reconstruction="Recomputed from unchanged training URDF and source qpos leg angles; not copied from mismatched GMR body positions",
                    source_model=str(args.source_model.resolve()), model_differences=differences,
                    input_and_asset_hashes_unchanged=True, protected_file_count=len(before),
                    normalization=dict(path=str(output / "normalization.npz"), sha256=sha256(output / "normalization.npz"),
                                       fit_ids=list(TRAIN_IDS), valid_frames_only=True, frozen=True),
                    script_hashes=snapshot(sorted((REPO / "humanoid/amp").glob("*.py")) + [Path(__file__), REPO / "tests/test_f1_amp.py"]),
                    training_ready=False, records=rows,
                    physical_gates=dict(joint_limits_pass=all(r["training_joint_limit_violations"] == 0 for r in reports if r["clip_id"] in TRAIN_IDS),
                                        dynamics="not_tested", self_collision="not_tested", friction="not_tested", torque="not_tested",
                                        acceleration_jerk_acceptance="no approved physical thresholds", root_z="geometric lift remains",
                                        isaac_gym="not_run", license_use="requires project use review"))
    save_json(output / "amp_manifest.json", manifest)
    write_report(output, manifest, reports, dry_run)
    print("Prepared audit bundle:", output, flush=True)


def write_report(output, manifest, reports, dry_run):
    lines = ["# F1 100 Hz AMP 数据接入质量报告", "",
             "结论：数据链路和 CPU 合成前向/单步优化已验证；正式训练、动力学和真机均未验证。training_ready=false。", "",
             "## 模型不一致：新增的明确阻塞", "",
             "源 GMR MJCF 与训练 URDF 关节同名，但限位和右腿部分几何不同。没有修改资产、裁剪/平滑关节或重跑 IK。", "",
             "示范关键点由训练 URDF 对原 qpos 腿角重新 FK；因此与未来策略侧同一模型的刚体原点对应。几何差异仍单独报告，不能说原重定向已适配训练限位。", "",
             "|动作|训练限位超限（帧×关节）|速度峰值 rad/s|加速度峰值 rad/s²|jerk峰值 rad/s³|根Z加速度峰值 m/s²|源网格最低高度 mm|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        j = r["derivatives"]["qpos"]["joints"]
        lines.append("|%s|%d|%.3f|%.2f|%.1f|%.2f|%.3f|" % (r["clip_id"], r["training_joint_limit_violations"], j["velocity"]["abs_peak"], j["acceleration"]["abs_peak"], j["jerk"]["abs_peak"], r["derivatives"]["qpos"]["root_z"]["acceleration"]["abs_peak"], r["source_geometry"]["source_mesh_min_height_mm"]))
    lines += ["", "## 计算口径与边界", "",
              "- 速度/加速度/jerk：逐阶 np.diff / 0.01，长度 N−1/N−2/N−3，不使用跨片段或循环差分。",
              "- 源网格高度和足底中心逐帧在原 MJCF 复算；它们不证明训练 URDF 的接触安全。支撑窗口为 source_support > 0.5 的连续段，不是真实接触力；与旧平足子窗口报告不可直接比较。",
              "- 39 维：根本体系角速度 3、按名映射腿角 12、腿速 12、左右膝/踝 link 原点的本体系位置 12。无根绝对位置、高度、竖向速度、航向角或接触标签。",
              "- 窗口 [B,10,39] → [B,390] → [B,1]；100 Hz、0.09 秒。第0示范帧不采样，策略 reset 状态只用于速度初始化，前9步无效。",
              "- 归一化只拟合01–04有效帧，冻结并对两侧共用。15不用于拟合/归一化/调参；08/09/10储备，10永久hold；站立单组。",
              "- 所有 allowed_for_training=false；split_eligible_for_training 只表示分组候选资格，不是物理放行。",
              "- LSGAN demo +1、policy −1、梯度惩罚；风格奖励无梯度。task_weight/style_weight 可配，准备配置 style_weight=0，CPU测试的非零混合不代表最终选择。",
              "- 既有Actor 66帧、短历史5帧、Critic3帧不变。未注册新训练任务，没有自动启动入口。", "", "## CPU 实际验证", "",
              "```json", json.dumps({k: v for k, v in dry_run.items() if k not in ("sampled_provenance", "reward_mix_for_test_only")}, ensure_ascii=False, indent=2), "```", "",
              "## 仍需解决", "",
              "1. 先决定以哪个模型为物理基准，解决01–04对训练URDF的超限及源/目标几何差异。不得简单改限位或截断角度。",
              "2. 独立处理高加速度/jerk、WALK_10后处理尖峰、地滑和逐帧抬根；若需要新轨迹另行授权输出到新目录。",
              "3. 未来Isaac Gym任务须在自动reset前采集刚体状态，reset后prime，并回读URDF/刚体坐标、控制周期。当前CPU合成状态不能替代该验证。",
              "4. 模型/数据通过后，下一步仅建议小规模数值smoke，先检查有效窗口与判别器统计，再讨论正式训练。当前不启动。", "",
              "## 可追溯文件", "",
              "amp_manifest.json 记录逐条输入/输出SHA、分组、FK模型、质量门槛；每动作quality.json含完整微分、根Z、支撑代理与边界指标。", "",
              "protected_inputs_sha256.json 是运行前后校验一致的输入/机器人文件哈希；cpu_dry_run.json 是真实CPU执行结果。代码与接口说明见仓库 docs/f1-amp-100hz.md。", ""]
    with (output / "质量报告.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


if __name__ == "__main__":
    main()
