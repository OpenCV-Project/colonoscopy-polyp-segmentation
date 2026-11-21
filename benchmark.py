"""Measure the mean elapsed time per image for a segmentation pipeline."""

from functools import partial
from itertools import chain
from statistics import fmean
from time import perf_counter_ns


def benchmark_image_time(process_image, images, *, device="cpu", warmup=5):
    """Time one decoded image at a time and return mean milliseconds per image.

    Pass original, decoded images in the order used for every method.
    process_image should perform preprocessing, segmentation, and mask
    postprocessing. Keep disk I/O and metric calculation outside it.
    Set the model to inference mode before benchmarking if applicable.
    """
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup은 0 이상의 정수여야 합니다.")

    device = str(device)
    synchronize = _cuda_synchronizer(device)

    images = iter(images)
    try:
        first_image = next(images)
    except StopIteration:
        raise ValueError("시간을 측정할 이미지가 없습니다.") from None

    for _ in range(warmup):
        process_image(first_image)

    times_ms = []
    for image in chain((first_image,), images):
        if synchronize is not None:
            synchronize()
        start = perf_counter_ns()
        process_image(image)
        if synchronize is not None:
            synchronize()
        times_ms.append((perf_counter_ns() - start) / 1_000_000)

    return {
        "device": device,
        "num_images": len(times_ms),
        "mean_ms_per_image": fmean(times_ms),
    }


def _cuda_synchronizer(device):
    if device == "cpu":
        return None
    if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("device는 'cpu', 'cuda', 또는 'cuda:0' 같은 CUDA 장치여야 합니다.")

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("CUDA 시간 측정에는 PyTorch가 필요합니다.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("현재 환경에서 CUDA GPU를 사용할 수 없습니다.")

    device_index = int(device[5:]) if device.startswith("cuda:") else None
    return partial(torch.cuda.synchronize, device_index)
