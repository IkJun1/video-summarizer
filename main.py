from __future__ import annotations

from pathlib import Path

from pipeline import run_pipeline_with_default_models
from config import IOConfig, PipelineConfig, ClusterScoreConfig


def main() -> None:
    input_path = Path("examples/input_video.mp4")
    io_cfg = IOConfig(input_video=input_path)
    k_ratio = 0.1  # adjust summary length
    pipe_cfg = PipelineConfig(cluster=ClusterScoreConfig(k_ratio=k_ratio))
    print(f"Input video: {input_path.resolve()}")
    try:
        output_path = run_pipeline_with_default_models(io_cfg, pipe_cfg)
    except Exception as exc:
        print(f"[ERROR] Pipeline failed: {exc}")
        raise
    print(f"Summary saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
