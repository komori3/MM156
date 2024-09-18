import os
import sys
import shutil
import tempfile
import subprocess
from datetime import datetime

timestamp = datetime.now()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUBMISSIONS_DIR = os.path.join(ROOT_DIR, 'submissions')
BESTS_TXT = os.path.join(SUBMISSIONS_DIR, 'bests.txt')

SOLVER_DIR = os.path.join(ROOT_DIR, 'vs', 'solver')
PROBLEM_NAME = 'Reversi'
SOURCE_FILE = os.path.join(SOLVER_DIR, 'src', f'{PROBLEM_NAME}.cpp')

TESTER_BIN = os.path.join(ROOT_DIR, 'tester', 'tester.jar')


if __name__ == '__main__':

    tag = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    if len(sys.argv) >= 2:
        tag += '_' + sys.argv[1]

    submission_dir = os.path.join(SUBMISSIONS_DIR, tag)
    assert not os.path.exists(submission_dir)
    print(f'submission dir: {submission_dir}')

    assert os.path.exists(SOURCE_FILE)

    with tempfile.TemporaryDirectory() as temp_dir:

        exec_bin = os.path.join(temp_dir, 'solver')
        assert not os.path.isfile(exec_bin)
        print(f"binary path: {exec_bin}")

        try:
            cmd = f'g++-9 -std=gnu++17 -Wall -Wextra -O3 {SOURCE_FILE} -o {exec_bin}'
            print(f'compiling {SOURCE_FILE} ...')
            print(cmd)
            res = subprocess.run(cmd, shell=True)
            assert res.returncode == 0
            print('finished.')
        except Exception:
            print('failed!')
            raise

        assert os.path.isfile(exec_bin)

        try:
            os.makedirs(submission_dir)
            output_dir = os.path.join(submission_dir, 'out')
            error_dir = os.path.join(submission_dir, 'err')
            scores_txt = os.path.join(submission_dir, 'scores.txt') 
            cmd = f'java -jar {TESTER_BIN} -ex {exec_bin} -sd 1+100 -th 8 -nv -no -pr -so {output_dir} -se {error_dir} -ss {scores_txt} -bs {BESTS_TXT} | tee {os.path.join(submission_dir, "execution_log.txt")}'
            print(cmd)
            subprocess.run(cmd, shell=True)
            shutil.copy2(SOURCE_FILE, os.path.join(submission_dir, f'{PROBLEM_NAME}.cpp'))
        except:
            shutil.rmtree(submission_dir)
            raise

    print()
    cmd = f"python3 {os.path.join(ROOT_DIR, 'scripts', 'evaluate.py')}"
    subprocess.run(cmd, shell=True)