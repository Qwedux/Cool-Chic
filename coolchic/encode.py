import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["MKL_THREADING_LAYER"] = "GNU"

# python3 ./lossless_encode.py --image-index=12 --encoder-gain=64 --color-space=YCoCg --use-image-arm --experiment-name=2026_02_18_smol_test --multiarm_setup=2x2
encoding_fmt_1 = 'CUDA_VISIBLE_DEVICES={} ' \
          'python lossless_encode.py ' \
          '--image-index 14 ' \
          '--encoder-gain=64 ' \
          '--color-space=YCoCg ' \
          '--use-image-arm ' \
          '--experiment-name=2026_03_08_clic2024_three_by_three ' \
          '--multiarm_setup=3x3 ' \

encoding_fmt_2 = 'CUDA_VISIBLE_DEVICES={} ' \
          'python lossless_encode.py ' \
          '--image-index 14 ' \
          '--encoder-gain=64 ' \
          '--color-space=YCoCg ' \
          '--use-image-arm ' \
          '--experiment-name=2026_03_08_clic2024_two_by_two ' \
          '--multiarm_setup=2x2 ' \

def run_command(cmd):
    """ Utility function to run a shell command. """
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

def run_encode(gpu, job_id_range, max_concurrent_tasks):
    futures = []
    with ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        encoding_cmd_1 = encoding_fmt_1.format(gpu, job_id_range)
        encoding_cmd_2 = encoding_fmt_2.format(gpu, job_id_range)
        future_1 = executor.submit(run_command, encoding_cmd_1)
        future_2 = executor.submit(run_command, encoding_cmd_2)
        futures.append(future_1)
        futures.append(future_2)
    return futures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--job_id_range', type=str, default="0")
    parser.add_argument('--max_concurrent_tasks', type=int, default=2)
    args = parser.parse_args()
    run_encode(args.gpu, args.job_id_range, args.max_concurrent_tasks)
