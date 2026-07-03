"""Segment matching pipeline 디버깅용 시각화 스크립트."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import FontSkeletonDataset
from losses import LossConfig, compute_losses, hungarian_match_indices, segment_lengths_torch
from model import FontSkeletonModel


def draw_segments(ax, segments, color: str, linewidth: float = 2.0, alpha: float = 1.0, label: str | None = None) -> None:
    """matplotlib axis 위에 [N,4] segment를 그린다."""
    first = True
    for seg in segments:
        ax.plot([seg[0], seg[2]], [seg[1], seg[3]], color=color, linewidth=linewidth, alpha=alpha, label=label if first else None)
        first = False


def load_model(args: argparse.Namespace, device: torch.device) -> FontSkeletonModel:
    """checkpoint가 있으면 가중치를 로드하고 없으면 랜덤 초기화 모델을 반환한다."""
    model = FontSkeletonModel(args.k_segments, args.image_size).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        print(f"checkpoint 로드 완료: {args.checkpoint}")
    else:
        print("checkpoint가 지정되지 않아 랜덤 초기화 모델로 시각화합니다.")
    model.eval()
    return model


def select_dataset(args: argparse.Namespace) -> FontSkeletonDataset:
    """filename 또는 codepoint 기준으로 한 샘플 Dataset을 만든다."""
    filenames = [args.filename] if args.filename else None
    codepoints = [args.codepoint] if args.filename is None and args.codepoint is not None else None
    return FontSkeletonDataset(
        args.outline_dir,
        args.skeleton_dir,
        max_segments=args.k_segments,
        codepoints=codepoints,
        filenames=filenames,
        image_size=args.image_size,
        svg_size=args.svg_size,
        num_target_points=args.num_target_points,
        sigma=args.sigma,
    )


def make_visualization(sample: dict, logits: torch.Tensor, segments: torch.Tensor, losses: dict, args: argparse.Namespace, save_path: Path) -> None:
    """target/predicted segment를 한 장의 figure로 저장한다."""
    exist_prob = torch.sigmoid(logits[0]).detach().cpu()
    pred_segments = segments[0].detach().cpu()
    active = exist_prob > args.exist_threshold
    target_mask = sample["target_segment_mask"].bool()
    target_segments = sample["target_segments"][target_mask].cpu()
    active_segments = pred_segments[active]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    axes[0].imshow(sample["input_image"][0], cmap="gray", origin="upper"); axes[0].set_title("outline image")
    axes[1].imshow(sample["target_heatmap"], cmap="viridis", origin="upper"); axes[1].set_title("target heatmap")
    axes[2].imshow(sample["target_skeleton"], cmap="gray", origin="upper"); draw_segments(axes[2], target_segments, "lime", 2.0); axes[2].set_title("target segments")
    axes[3].imshow(sample["input_image"][0], cmap="gray", origin="upper"); draw_segments(axes[3], active_segments, "red", 2.0); axes[3].set_title("pred active segments")
    axes[4].imshow(sample["target_skeleton"], cmap="gray", origin="upper"); draw_segments(axes[4], target_segments, "lime", 2.0, label="target"); draw_segments(axes[4], active_segments, "red", 1.5, 0.8, label="pred"); axes[4].set_title("target + pred overlay")

    # matched pair를 midpoint 연결선으로 표시한다.
    batch = {k: v.unsqueeze(0).to(segments.device) if torch.is_tensor(v) and k != "codepoint" else v for k, v in sample.items()}
    try:
        matches = hungarian_match_indices(segments, batch["target_segments"], batch["target_segment_mask"], LossConfig(image_size=args.image_size))[0]
        axes[5].imshow(sample["target_skeleton"], cmap="gray", origin="upper")
        draw_segments(axes[5], target_segments, "lime", 2.0, 0.8)
        matched_pred = pred_segments[matches[0].cpu()]
        draw_segments(axes[5], matched_pred, "orange", 1.5, 0.9)
        valid_targets = target_segments[matches[1].cpu()]
        for p, t in zip(matched_pred, valid_targets):
            pm = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
            tm = ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)
            axes[5].plot([pm[0], tm[0]], [pm[1], tm[1]], "c--", linewidth=0.8)
        axes[5].set_title("matched pairs")
    except Exception as exc:
        axes[5].text(0.05, 0.5, f"matching 실패: {exc}", transform=axes[5].transAxes)
        axes[5].set_title("matched pairs")

    for ax in axes:
        ax.set_xlim(0, args.image_size)
        ax.set_ylim(args.image_size, 0)
        ax.axis("off")
    fig.suptitle(f"{sample['filename']} | total loss={float(losses['total'].detach().cpu()):.4f}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def format_segment(seg: torch.Tensor) -> str:
    """segment tensor [4]를 보기 좋은 문자열로 변환한다."""
    s = seg.detach().cpu().tolist()
    return f"[{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}, {s[3]:.2f}]"


### exist 정보 출력 함수 
def print_segment_table(
    title: str,
    indices: torch.Tensor,
    exist_prob: torch.Tensor,
    pred_segments: torch.Tensor,
    lengths: torch.Tensor,
    matched_pred_set: set[int] | None = None,
    active_set: set[int] | None = None,
) -> None:
    """pred segment 목록을 index, exist, length, 좌표와 함께 출력한다."""
    matched_pred_set = matched_pred_set or set()
    active_set = active_set or set()

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'idx':>4} | {'exist':>8} | {'length':>8} | {'active':>6} | {'matched':>7} | segment [x0,y0,x1,y1]")
    print("-" * 100)

    for idx_tensor in indices:
        idx = int(idx_tensor)
        active_mark = "Y" if idx in active_set else "-"
        matched_mark = "Y" if idx in matched_pred_set else "-"

        print(
            f"{idx:4d} | "
            f"{float(exist_prob[idx]):8.4f} | "
            f"{float(lengths[idx]):8.2f} | "
            f"{active_mark:>6} | "
            f"{matched_mark:>7} | "
            f"{format_segment(pred_segments[idx])}"
        )


def print_debug_info(
    sample: dict,
    logits: torch.Tensor,
    segments: torch.Tensor,
    losses: dict,
    threshold: float,
    image_size: int,
) -> None:
    """시각화 대상 샘플의 핵심 수치와 segment 상세 정보를 출력한다."""
    device = segments.device

    exist_prob = torch.sigmoid(logits[0]).detach().cpu()
    pred_segments = segments[0].detach().cpu()
    active = exist_prob > threshold
    active_idx = torch.where(active)[0]

    lengths = segment_lengths_torch(pred_segments).detach().cpu()
    pred_total_length = float((exist_prob * lengths).sum())

    target_mask = sample["target_segment_mask"].bool()
    target_segments = sample["target_segments"][target_mask].detach().cpu()

    print()
    print("#" * 100)
    print("기본 정보")
    print("#" * 100)
    print(f"filename: {sample['filename']}")
    print(f"codepoint: {int(sample['codepoint'])}")
    print(f"target_num_segments: {float(sample['target_num_segments']):.0f}")
    print(f"pred_active_segments threshold={threshold}: {int(active.sum())}")
    print(
        "exist_prob min/max/mean: "
        f"{float(exist_prob.min()):.4f} / "
        f"{float(exist_prob.max()):.4f} / "
        f"{float(exist_prob.mean()):.4f}"
    )
    print(
        "pred segment length min/max/mean: "
        f"{float(lengths.min()):.4f} / "
        f"{float(lengths.max()):.4f} / "
        f"{float(lengths.mean()):.4f}"
    )
    print(f"target_total_length: {float(sample['target_total_length']):.4f}")
    print(f"pred_total_length: {pred_total_length:.4f}")

    print()
    print("#" * 100)
    print("Loss values")
    print("#" * 100)
    for name, value in losses.items():
        print(f"{name}: {float(value.detach().cpu()):.6f}")

    # Hungarian matching 결과 계산
    batch = {
        k: v.unsqueeze(0).to(device)
        if torch.is_tensor(v) and k != "codepoint"
        else v
        for k, v in sample.items()
    }

    matched_pred_idx = torch.empty(0, dtype=torch.long)
    matched_target_idx = torch.empty(0, dtype=torch.long)

    try:
        matches = hungarian_match_indices(
            segments,
            batch["target_segments"],
            batch["target_segment_mask"],
            LossConfig(image_size=image_size),
        )[0]

        matched_pred_idx = matches[0].detach().cpu()
        matched_target_idx = matches[1].detach().cpu()

    except Exception as exc:
        print()
        print("[WARN] Hungarian matching 출력 실패:", exc)

    matched_pred_set = set(int(i) for i in matched_pred_idx.tolist())
    active_set = set(int(i) for i in active_idx.tolist())

    # 1. exist_prob 높은 순서대로 전체 pred segment 출력
    sorted_idx = torch.argsort(exist_prob, descending=True)

    print_segment_table(
        title="Pred segments sorted by exist_prob descending",
        indices=sorted_idx,
        exist_prob=exist_prob,
        pred_segments=pred_segments,
        lengths=lengths,
        matched_pred_set=matched_pred_set,
        active_set=active_set,
    )

    # 2. active segment만 따로 출력
    if len(active_idx) > 0:
        active_sorted_idx = active_idx[torch.argsort(exist_prob[active_idx], descending=True)]

        print_segment_table(
            title=f"Active pred segments only, threshold={threshold}",
            indices=active_sorted_idx,
            exist_prob=exist_prob,
            pred_segments=pred_segments,
            lengths=lengths,
            matched_pred_set=matched_pred_set,
            active_set=active_set,
        )
    else:
        print()
        print("=" * 100)
        print(f"Active pred segments only, threshold={threshold}")
        print("=" * 100)
        print("active segment가 없습니다.")

    # 3. Hungarian matched pair 상세 출력
    print()
    print("=" * 100)
    print("Hungarian matched pairs")
    print("=" * 100)

    if len(matched_pred_idx) == 0:
        print("matched pair가 없습니다.")
    else:
        print(
            f"{'pair':>4} | {'target_idx':>10} | {'pred_idx':>8} | "
            f"{'pred_exist':>10} | {'pred_len':>8} | pred segment | target segment"
        )
        print("-" * 100)

        for pair_i, (p_idx_tensor, t_idx_tensor) in enumerate(zip(matched_pred_idx, matched_target_idx)):
            p_idx = int(p_idx_tensor)
            t_idx = int(t_idx_tensor)

            pred_seg = pred_segments[p_idx]
            target_seg = target_segments[t_idx]

            print(
                f"{pair_i:4d} | "
                f"{t_idx:10d} | "
                f"{p_idx:8d} | "
                f"{float(exist_prob[p_idx]):10.4f} | "
                f"{float(lengths[p_idx]):8.2f} | "
                f"{format_segment(pred_seg)} | "
                f"{format_segment(target_seg)}"
            )

    # 4. target segment 자체도 출력
    print()
    print("=" * 100)
    print("Target segments")
    print("=" * 100)

    for i, seg in enumerate(target_segments):
        target_len = torch.sqrt(((seg[2:] - seg[:2]) ** 2).sum() + 1e-8)
        print(
            f"target {i:3d} | "
            f"length={float(target_len):8.2f} | "
            f"{format_segment(seg)}"
        )


def run_visualization(args: argparse.Namespace) -> None:
    dataset = select_dataset(args)
    sample = dataset[0]
    device = torch.device(args.device)
    model = load_model(args, device)
    batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) and k != "codepoint" else v for k, v in sample.items()}
    with torch.no_grad():
        # logits, segments = model(batch["input_image"])
        logits, segments = model(batch["input_image"], codepoint=batch["codepoint"],)
        losses = compute_losses(logits, segments, batch, LossConfig(image_size=args.image_size, render_sigma=args.sigma))
    print_debug_info(sample, logits, segments, losses, args.exist_threshold, args.image_size)
    save_dir = Path(args.save_dir)
    save_path = save_dir / f"{Path(sample['filename']).stem}_segmatch_debug.png"
    make_visualization(sample, logits, segments, losses, args, save_path)
    print(f"시각화 저장 완료: {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="segment matching pipeline 샘플 시각화")
    p.add_argument("--outline_dir", "--outline-dir", required=True)
    p.add_argument("--skeleton_dir", "--skeleton-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--save_dir", "--save-dir", default="visualizations")
    p.add_argument("--codepoint", type=int, default=65)
    p.add_argument("--filename", default=None)
    p.add_argument("--image_size", "--image-size", type=int, default=128)
    p.add_argument("--svg_size", "--svg-size", type=float, default=50.0)
    p.add_argument("--k_segments", "--num-segments", type=int, default=48)
    p.add_argument("--num_target_points", "--num-target-points", type=int, default=1000)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--exist_threshold", "--exist-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    run_visualization(parse_args())
