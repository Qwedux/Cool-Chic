import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["MKL_THREADING_LAYER"] = "GNU"

encoding_fmt_1 = 'CUDA_VISIBLE_DEVICES={} ' \
          'python lossless_encode.py ' \
          '--image-index {} ' \
          '--encoder-gain=64 ' \
          '--color-space=YCoCg ' \
          '--use-image-arm ' \
          '--experiment-name=2026_03_15_kodak_4x4 ' \
          '--multiarm_setup=4x4 ' \

def run_command(cmd):
    """ Utility function to run a shell command. """
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

def run_encode(gpu, job_id_range, max_concurrent_tasks):
    futures = []
    with ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        for job_id in range(21, job_id_range):
            encoding_cmd_1 = encoding_fmt_1.format(gpu, job_id)
            future_1 = executor.submit(run_command, encoding_cmd_1)
            futures.append(future_1)
    return futures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--job_id_range', type=int, default=24)
    parser.add_argument('--max_concurrent_tasks', type=int, default=1)
    args = parser.parse_args()
    run_encode(args.gpu, args.job_id_range, args.max_concurrent_tasks)
