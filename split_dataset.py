"""Create the fixed train, val and test split for Kvasir-SEG."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import random

SEED = 42
SPLIT_COUNTS = (("train", 700), ("val", 150), ("test", 150))
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("splits.csv")


def jpg_names(folder):
    return {
        path.name
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }


def build_rows(dataset):
    images = jpg_names(dataset / "images")
    masks = jpg_names(dataset / "masks")
    if images != masks:
        raise ValueError(
            f"이미지·마스크 파일명이 일치하지 않습니다. "
            f"마스크가 없는 이미지 {len(images - masks)}장, "
            f"이미지가 없는 마스크 {len(masks - images)}장."
        )

    expected = sum(count for _, count in SPLIT_COUNTS)
    if len(images) != expected:
        raise ValueError(f"이미지·마스크가 {expected}쌍이어야 합니다. 확인된 수량은 {len(images)}쌍입니다.")

    # 정렬한 파일명을 섞어 운영체제의 디렉터리 순서와 무관하게 분할한다.
    names = sorted(images)
    random.Random(SEED).shuffle(names)

    rows = []
    start = 0
    for split, count in SPLIT_COUNTS:
        end = start + count
        rows.extend((f"images/{name}", f"masks/{name}", split) for name in names[start:end])
        start = end
    return rows


def main():
    parser = argparse.ArgumentParser(description="Kvasir-SEG의 이미지·마스크 쌍을 고정 시드로 분할합니다.")
    parser.add_argument("dataset", type=Path, help="images와 masks 폴더가 들어 있는 Kvasir-SEG 경로")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="저장할 CSV 경로")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".csv":
        parser.error("출력 파일 이름은 .csv로 끝나야 합니다.")

    try:
        rows = build_rows(args.dataset)
        with args.output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image_path", "mask_path", "split"))
            writer.writerows(rows)
    except FileExistsError:
        parser.exit(1, f"이미 존재하는 출력 파일입니다. 다른 출력 경로를 지정하세요. {args.output}\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"분할에 실패했습니다. {error}\n")

    counts = Counter(row[2] for row in rows)
    for split, _ in SPLIT_COUNTS:
        print(f"{split}: {counts[split]}쌍")
    print(f"저장 위치: {args.output.resolve()}")


if __name__ == "__main__":
    main()
