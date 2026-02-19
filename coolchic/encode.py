import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["MKL_THREADING_LAYER"] = "GNU"

# python3 ./lossless_encode.py --image-index=12 --encoder-gain=64 --color-space=YCoCg --use-image-arm --experiment-name=2026_02_18_smol_test --multiarm_setup=2x2
encoding_fmt = 'CUDA_VISIBLE_DEVICES={} ' \
          'python lossless_encode.py ' \
          '--image-index {} ' \
          '--encoder-gain=64 ' \
          '--color-space=YCoCg ' \
          '--use-image-arm ' \
          '--experiment-name=2026_02_19_preset_test ' \
          '--multiarm_setup=2x2 ' \

def run_command(cmd):
    """ Utility function to run a shell command. """
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

def run_encode(gpu, max_job_id, max_concurrent_tasks):
    futures = []
    with ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        for job_id in range(max_job_id):
            encoding_cmd = encoding_fmt.format(gpu, job_id)
            future = executor.submit(run_command, encoding_cmd)

            futures.append(future)
    return futures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_job_id', type=int, default=24)
    parser.add_argument('--max_concurrent_tasks', type=int, default=4)
    args = parser.parse_args()
    run_encode(args.gpu, args.max_job_id, args.max_concurrent_tasks)
