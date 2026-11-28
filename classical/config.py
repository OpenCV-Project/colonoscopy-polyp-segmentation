"""고전 기법 스크립트가 함께 읽는 설정. 실행 환경마다 다른 경로만 명령줄 인자로 덮어쓴다."""

import argparse
import json
from pathlib import Path


def read_config():
    """명령줄 인자로 설정 파일을 읽고, 데이터·결과 경로를 덮어쓴 최종 설정을 반환한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("classical/config.json"))
    parser.add_argument("--data-root", required=True, help="압축을 푼 Kvasir-SEG 폴더")
    parser.add_argument("--output-dir", help="결과 폴더 (기본값은 설정 파일의 output_dir)")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["data_root"] = args.data_root
    if args.output_dir:
        config["output_dir"] = args.output_dir
    return config


def save_config(config, output_dir):
    """실제로 쓴 설정을 결과와 함께 저장한다. 데이터 경로는 컴퓨터마다 달라서 저장하지 않는다."""
    saved_config = {key: value for key, value in config.items() if key != "data_root"}
    (output_dir / "config_used.json").write_text(
        json.dumps(saved_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
