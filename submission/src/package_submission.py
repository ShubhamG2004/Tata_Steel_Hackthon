import os
import shutil
import zipfile


def make_submission_archive(output_zip='submission.zip'):
    submission_dir = 'submission'
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir, exist_ok=True)

    # required files
    candidates = [
        'expected_submission.csv',
        'requirements.txt',
        'README.md',
        'run.sh',
        'run.ps1',
    ]

    for f in candidates:
        if os.path.exists(f):
            shutil.copy(f, submission_dir)

    # copy source
    if os.path.exists('src'):
        shutil.copytree('src', os.path.join(submission_dir, 'src'))

    # create zip
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(submission_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, submission_dir)
                z.write(full, arcname)

    print(f'Created submission archive: {output_zip}')


if __name__ == '__main__':
    make_submission_archive()
