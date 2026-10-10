#!/usr/bin/env python3
"""
plot_results.py - 读取 results/*.csv，生成最终的GFLOPS对比图。

用法（在项目根目录下执行，需要先跑过 bash bench/bench.sh 生成数据）：
    pip install matplotlib --break-system-packages   # 如果还没装
    python3 bench/plot_results.py

输出两张图，放在 results/ 下：
    gflops_fp32_comparison.png  — V0-V3 (FP32, CUDA Core) 对比，一条曲线一个版本，
                                   每个版本在每个N下取所有tile/config里的最优值
    gflops_fp16_tensorcore.png  — V4 (FP16, Tensor Core) 对比，含cuBLAS基准

注：V4是FP16精度，V0-V3是FP32精度，理论峰值不在同一量级（T4的FP16 Tensor Core
理论峰值约65 TFLOPS，FP32 CUDA Core约8.1 TFLOPS），所以分两张图，不混在一起比较。
这一点在README的"对比口径提醒"里也强调过，这里的脚本设计是对那条原则的落实。
"""
import csv
import os
import sys

try:
    import matplotlib

    matplotlib.use("Agg")  # 无显示环境也能跑（服务器/Colab终端常见情况）
    import matplotlib.pyplot as plt
except ImportError:
    print("需要先安装 matplotlib: pip install matplotlib --break-system-packages")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FP32_CSV = os.path.join(RESULTS_DIR, "results.csv")
FP16_CSV = os.path.join(RESULTS_DIR, "results_v4_fp16.csv")


def load_fp32_data(path):
    """读取V0-V3的csv，每个version在每个N下取所有tile/config里的最优GFLOPS。

    results.csv的tile_size列对V0/V3是"NA"（只有一个配置），对V1是tile size
    (8/16/32)，对V2是TM/TN组合(4x4/8x8)——取最优值能公平地代表"这个版本能
    达到的最好成绩"，而不是任选一个tile size。
    """
    if not os.path.exists(path):
        print(f"[跳过] 找不到 {path}，先跑 bash bench/bench.sh 生成数据")
        return {}

    best = {}  # (version, N) -> gflops
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            version = row.get("version")
            try:
                N = int(row["N"])
                gflops = float(row["gflops"])
            except (ValueError, KeyError, TypeError):
                continue
            key = (version, N)
            if key not in best or gflops > best[key]:
                best[key] = gflops

    data = {}
    for (version, N), gflops in best.items():
        data.setdefault(version, {})[N] = gflops
    return data


def plot_fp32(data, out_path):
    if not data:
        print("[跳过] 没有FP32数据，不生成对比图")
        return

    order = ["V0", "V1", "V2", "V3"]
    labels = {
        "V0": "V0 Naive",
        "V1": "V1 Shared Memory Tiling",
        "V2": "V2 Register Blocking",
        "V3": "V3 Vectorized + Double Buffer",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted_any = False
    for version in order:
        if version not in data:
            print(f"[提示] 数据里没有 {version}，跳过这条曲线")
            continue
        points = sorted(data[version].items())
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=2, label=labels.get(version, version))
        plotted_any = True

    if not plotted_any:
        print("[跳过] 没有任何已知版本的数据，不生成对比图")
        plt.close(fig)
        return

    ax.set_xlabel("矩阵规模 N (方阵 M=N=K=N)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Progressive Performance Matrix: V0-V3 (FP32, CUDA Core)")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[完成] 已生成 {out_path}")


def load_fp16_data(path):
    if not os.path.exists(path):
        print(f"[跳过] 找不到 {path}，先跑 bash bench/bench.sh 生成数据")
        return {}

    data = {}  # kernel -> {N: gflops}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel = row.get("kernel")
            try:
                N = int(row["N"])
                gflops = float(row["gflops"])
            except (ValueError, KeyError, TypeError):
                continue
            data.setdefault(kernel, {})[N] = gflops
    return data


def plot_fp16(data, out_path):
    if not data:
        print("[跳过] 没有FP16/Tensor Core数据，不生成对比图")
        return

    order = ["naive", "tiled", "tiled_dbuf", "cublas"]
    labels = {
        "naive": "V4 WMMA naive",
        "tiled": "V4 WMMA tiled (TILE_K=16)",
        "tiled_dbuf": "V4 WMMA tiled+dbuf",
        "cublas": "cuBLAS (Tensor Core)",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted_any = False
    for kernel in order:
        if kernel not in data:
            print(f"[提示] 数据里没有 {kernel}，跳过这条曲线")
            continue
        points = sorted(data[kernel].items())
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        style = "--" if kernel == "cublas" else "-"
        ax.plot(
            xs, ys, style, marker="o", linewidth=2, label=labels.get(kernel, kernel)
        )
        plotted_any = True

    if not plotted_any:
        print("[跳过] 没有任何已知kernel的数据，不生成对比图")
        plt.close(fig)
        return

    ax.set_xlabel("矩阵规模 N (方阵 M=N=K=N)")
    ax.set_ylabel("GFLOPS (log scale)")
    ax.set_title("V4: FP16 Tensor Core kernel vs cuBLAS")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")  # cuBLAS比我们的kernel高出一个数量级，log轴才看得清楚
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[完成] 已生成 {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fp32_data = load_fp32_data(FP32_CSV)
    plot_fp32(fp32_data, os.path.join(RESULTS_DIR, "gflops_fp32_comparison.png"))

    fp16_data = load_fp16_data(FP16_CSV)
    plot_fp16(fp16_data, os.path.join(RESULTS_DIR, "gflops_fp16_tensorcore.png"))


if __name__ == "__main__":
    main()
