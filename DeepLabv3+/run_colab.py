"""Google Colab에서 DeepLabv3+ 학습·튜닝·평가·시각화·외부평가를 한 번에 실행하는 메인 스크립트."""

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch


def parse_args():
    """Colab/CLI 환경에 맞게 경로와 하이퍼파라미터를 인자로 받습니다."""
    parser = argparse.ArgumentParser(description="DeepLabv3+ 학습·평가 원클릭 실행 스크립트")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="main, DeepLabv3+, Kvasir-SEG 폴더가 있는 프로젝트 루트 경로",
    )
    parser.add_argument("--epochs", type=int, default=30, help="학습 에폭 수")
    parser.add_argument("--batch-size", type=int, default=16, help="배치 크기 (BatchNorm 때문에 2 이상)")
    parser.add_argument("--lr", type=float, default=1e-4, help="초기 학습률")
    parser.add_argument(
        "--cvc-dir",
        type=Path,
        default=None,
        help="외부 평가용 CVC-ClinicDB 폴더 경로 (미지정 시 PROJECT_ROOT/CVC-ClinicDB 확인)",
    )
    parser.add_argument(
        "--skip-improved",
        action="store_true",
        help="개선 실험(BCE+Dice)을 건너뛰고 1차 베이스라인만 실행",
    )
    return parser.parse_args()


def _load_best_model(checkpoint_path: str | Path, encoder_name: str, device: torch.device):
    """저장된 최고 검증 Dice 체크포인트 가중치를 모델에 불러옵니다."""
    from model import build_deeplabv3plus

    # 추론 시에는 로컬 체크포인트를 덮어쓰므로 사전학습 가중치 다운로드 생략(None)
    model = build_deeplabv3plus(encoder_name=encoder_name, encoder_weights=None).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _run_experiment(
    name: str,
    loss_name: str,
    args,
    dataset_dir: Path,
    main_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict, torch.nn.Module]:
    """단일 실험(학습 -> 검증 임계값 튜닝 -> 테스트 평가 -> 성공/실패 시각화)을 실행합니다."""
    from config import TrainConfig
    from evaluate import (
        evaluate_kvasir_test,
        save_qualitative_figure,
        select_success_and_failure_indices,
    )
    from train import build_dataloaders, find_best_threshold, train_model

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        loss_name=loss_name,
    )
    print(f"\n=== [{name}] 학습 시작 (loss={loss_name}, epochs={args.epochs}) ===")
    train_summary = train_model(config, dataset_dir, main_dir, output_dir)

    # 검증 Dice가 가장 높았던 체크포인트 로드 후, 검증 세트에서만 최적 임계값 탐색
    model = _load_best_model(train_summary["checkpoint_path"], config.encoder_name, device)
    _, val_loader, _ = build_dataloaders(dataset_dir, main_dir, config)
    thresh_info = find_best_threshold(model, val_loader, main_dir, device)
    best_thresh = thresh_info["best"]["threshold"]
    print(f"[{name}] 검증 최적 임계값: {best_thresh} (val_Dice={thresh_info['best']['val_dice']:.4f})")

    # 확정된 모델과 임계값으로 테스트 세트 150장 정량 평가 및 벤치마크 수행
    eval_result = evaluate_kvasir_test(
        model=model,
        dataset_dir=dataset_dir,
        main_dir=main_dir,
        device=device,
        threshold=best_thresh,
        output_dir=output_dir,
    )

    # 정성 평가용 성공 사례 Top 3, 실패 사례 Bottom 3 이미지 생성 및 저장
    success_idx, failure_idx = select_success_and_failure_indices(
        eval_result["summary"]["per_image"], top_k=3
    )
    save_qualitative_figure(
        indices=success_idx,
        images_rgb=eval_result["images_rgb"],
        targets=eval_result["targets"],
        predictions=eval_result["predictions"],
        per_image_records=eval_result["summary"]["per_image"],
        title=f"DeepLabv3+ ({name}) - Success Top 3",
        save_path=output_dir / "success_cases.png",
    )
    save_qualitative_figure(
        indices=failure_idx,
        images_rgb=eval_result["images_rgb"],
        targets=eval_result["targets"],
        predictions=eval_result["predictions"],
        per_image_records=eval_result["summary"]["per_image"],
        title=f"DeepLabv3+ ({name}) - Failure Bottom 3",
        save_path=output_dir / "failure_cases.png",
    )
    return eval_result["summary"], model


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    main_dir = project_root / "main"
    deeplab_dir = project_root / "DeepLabv3+"
    dataset_dir = project_root / "Kvasir-SEG"
    output_root = deeplab_dir / "outputs"

    # DeepLabv3+ 내부 모듈과 main 공통 모듈을 import 경로에 추가
    for path in (str(deeplab_dir), str(main_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from evaluate import evaluate_cvc_clinicdb
    from model import select_device

    device = select_device()
    print(f"실행 장치: {device}")

    rows = []
    # 1차 실험: BCEWithLogitsLoss 베이스라인
    baseline_summary, best_model = _run_experiment(
        name="baseline_bce",
        loss_name="bce",
        args=args,
        dataset_dir=dataset_dir,
        main_dir=main_dir,
        output_dir=output_root / "baseline_bce",
        device=device,
    )
    best_thresh = baseline_summary["threshold"]
    rows.append(
        {
            "Model": "DeepLabv3+ (Baseline BCE)",
            "Threshold": baseline_summary["threshold"],
            "Test IoU": round(baseline_summary["mean_iou"], 4),
            "Test Dice": round(baseline_summary["mean_dice"], 4),
            "ms/img": round(baseline_summary["mean_ms_per_image"], 2),
            "Device": baseline_summary["benchmark_device"],
            "Params": baseline_summary["total_params"],
        }
    )

    # 2차 실험: 실패 분석 기반 개선 실험 (BCEDiceLoss)
    if not args.skip_improved:
        improved_summary, best_model = _run_experiment(
            name="improved_bce_dice",
            loss_name="bce_dice",
            args=args,
            dataset_dir=dataset_dir,
            main_dir=main_dir,
            output_dir=output_root / "improved_bce_dice",
            device=device,
        )
        best_thresh = improved_summary["threshold"]
        rows.append(
            {
                "Model": "DeepLabv3+ (Improved BCE+Dice)",
                "Threshold": improved_summary["threshold"],
                "Test IoU": round(improved_summary["mean_iou"], 4),
                "Test Dice": round(improved_summary["mean_dice"], 4),
                "ms/img": round(improved_summary["mean_ms_per_image"], 2),
                "Device": improved_summary["benchmark_device"],
                "Params": improved_summary["total_params"],
            }
        )

    # 개선 전·후 비교표 CSV 저장 및 콘솔 출력
    comparison_df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = output_root / "deeplabv3plus_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("\n=== 최종 DeepLabv3+ 비교표 ===")
    print(comparison_df.to_string(index=False))

    # CVC-ClinicDB 폴더가 있으면 외부 일반화 성능 평가 자동 실행
    cvc_dir = args.cvc_dir or (project_root / "CVC-ClinicDB")
    if cvc_dir.exists():
        cvc_result = evaluate_cvc_clinicdb(
            model=best_model,
            cvc_dir=cvc_dir,
            main_dir=main_dir,
            device=device,
            threshold=best_thresh,
            output_dir=output_root,
        )
        print(
            f"\n[CVC-ClinicDB 외부 평가] {cvc_result['num_images']}장 | "
            f"IoU: {cvc_result['mean_iou']:.4f} | Dice: {cvc_result['mean_dice']:.4f}"
        )


if __name__ == "__main__":
    main()
