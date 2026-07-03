"""Hungarian segment matching 기반 학습 스크립트."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

from dataset import FontSkeletonDataset
from losses import LossConfig, compute_losses
from model import FontSkeletonModel

LOSS_KEYS = ["total", "match_segment", "exist_bce", "exist_count", "total_length", "render", "pred_to_target", "target_to_pred"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="outline SVG에서 skeleton segment를 예측하는 Hungarian matching 학습 루프")
    p.add_argument("--outline_dir", "--outline-dir", required=True)
    p.add_argument("--skeleton_dir", "--skeleton-dir", required=True)
    p.add_argument("--output_dir", "--output-dir", default="runs/segmatch")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", "--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--k_segments", "--num-segments", type=int, default=48)
    p.add_argument("--num_target_points", "--num-target-points", type=int, default=1000)
    p.add_argument("--pred_sample_points", "--samples-per-segment", type=int, default=16)
    p.add_argument("--image_size", "--image-size", type=int, default=128)
    p.add_argument("--svg_size", "--svg-size", type=float, default=50.0)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--val_ratio", "--val-ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", "--test-ratio", type=float, default=0.1)
    p.add_argument("--checkpoint_every", "--checkpoint-every", type=int, default=20)
    p.add_argument("--val_every", "--val-every", type=int, default=1)
    p.add_argument("--log_every", "--log-every", type=int, default=1)
    p.add_argument("--w_match_segment", "--w-match-segment", type=float, default=5.0)
    p.add_argument("--w_exist_bce", "--w-exist-bce", type=float, default=1.0)
    p.add_argument("--w_exist_count", "--w-exist-count", type=float, default=0.1)
    p.add_argument("--w_total_length", "--w-total-length", type=float, default=0.5)
    p.add_argument("--w_render", "--w-render", type=float, default=5.0)
    p.add_argument("--w_pred_to_target", "--w-pred-to-target", type=float, default=0.5)
    p.add_argument("--w_target_to_pred", "--w-target-to-pred", type=float, default=0.5)
    p.add_argument("--render_fg_weight", "--render-fg-weight", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    """문자열 filename은 유지하고 tensor만 device로 이동한다."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    """재현 가능한 train/val/test index split을 만든다."""
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    return train_idx, val_idx, test_idx


def make_loss_config(args: argparse.Namespace) -> LossConfig:
    return LossConfig(
        image_size=args.image_size,
        render_sigma=args.sigma,
        samples_per_segment=args.pred_sample_points,
        w_match_segment=args.w_match_segment,
        w_exist_bce=args.w_exist_bce,
        w_exist_count=args.w_exist_count,
        w_total_length=args.w_total_length,
        w_render=args.w_render,
        w_pred_to_target=args.w_pred_to_target,
        w_target_to_pred=args.w_target_to_pred,
        render_fg_weight=args.render_fg_weight,
    )


def run_epoch(model: FontSkeletonModel, loader: DataLoader, cfg: LossConfig, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> Dict[str, float]:
    """한 split을 실행하고 평균 loss dictionary를 반환한다."""
    training = optimizer is not None
    model.train(training)
    sums = {k: 0.0 for k in LOSS_KEYS}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        # with torch.set_grad_enabled(training):
        #     logits, segments = model(batch["input_image"])
        #     losses = compute_losses(logits, segments, batch, cfg)
        #     if training:
        #         losses["total"].backward()
        #         optimizer.step()
        with torch.set_grad_enabled(training):
            logits, segments = model(
                batch["input_image"],
                codepoint=batch["codepoint"],
            )
            losses = compute_losses(logits, segments, batch, cfg)
            if training:
                losses["total"].backward()
                optimizer.step()
        bsz = int(batch["input_image"].shape[0])
        count += bsz
        for k in LOSS_KEYS:
            sums[k] += float(losses[k].detach()) * bsz
    return {k: sums[k] / max(count, 1) for k in LOSS_KEYS}


def save_checkpoint(path: Path, model: FontSkeletonModel, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace, best_val: float) -> None:
    """학습 재개/시각화를 위한 checkpoint를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "args": vars(args), "best_val": best_val}, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset = FontSkeletonDataset(
        args.outline_dir,
        args.skeleton_dir,
        max_segments=args.k_segments,
        image_size=args.image_size,
        svg_size=args.svg_size,
        num_target_points=args.num_target_points,
        sigma=args.sigma,
    )
    counts = [dataset.get_target_segment_count(i) for i in range(len(dataset))]
    print(f"total dataset size: {len(dataset)}")
    print(f"target segment count min/max/mean: {min(counts)} / {max(counts)} / {sum(counts) / len(counts):.2f}")
    too_many = [(dataset.samples[i]["filename"], c) for i, c in enumerate(counts) if c > args.k_segments]
    if too_many:
        print(f"WARNING: K={args.k_segments}보다 target segment가 많은 파일이 {len(too_many)}개 있습니다. 예: {too_many[:10]}")

    train_idx, val_idx, test_idx = split_indices(len(dataset), args.val_ratio, args.test_ratio, args.seed)
    print(f"train / val / test sizes: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    loaders = {
        "train": DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0) if val_idx else None,
        "test": DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0) if test_idx else None,
    }

    device = torch.device(args.device)
    model = model = FontSkeletonModel(
        num_segments=args.k_segments,
        image_size=args.image_size,
        fc_channel=1024,
        char_categories=94,
        codepoint_min=33,
        use_class_condition=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cfg = make_loss_config(args)

    log_path = log_dir / "losses.csv"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("epoch,split," + ",".join(LOSS_KEYS) + "\n")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_losses = run_epoch(model, loaders["train"], cfg, device, optimizer)
        rows = [("train", train_losses)]
        if loaders["val"] is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            val_losses = run_epoch(model, loaders["val"], cfg, device)
            rows.append(("val", val_losses))
            if val_losses["total"] < best_val:
                best_val = val_losses["total"]
                save_checkpoint(checkpoint_dir / "best_val_model.pt", model, optimizer, epoch, args, best_val)
        if loaders["test"] is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            rows.append(("test", run_epoch(model, loaders["test"], cfg, device)))

        with log_path.open("a", encoding="utf-8") as f:
            for split, losses in rows:
                f.write(f"{epoch},{split}," + ",".join(f"{losses[k]:.8f}" for k in LOSS_KEYS) + "\n")

        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, args, best_val)
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, args, best_val)
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            text = " | ".join(f"{split}: " + ", ".join(f"{k}={v:.5f}" for k, v in losses.items()) for split, losses in rows)
            print(f"epoch {epoch:04d} | {text}")

    save_checkpoint(checkpoint_dir / "font_skeleton_model.pt", model, optimizer, args.epochs, args, best_val)
    print(f"final checkpoint 저장 완료: {checkpoint_dir / 'font_skeleton_model.pt'}")
    print(f"loss CSV 저장 완료: {log_path}")


if __name__ == "__main__":
    main()
