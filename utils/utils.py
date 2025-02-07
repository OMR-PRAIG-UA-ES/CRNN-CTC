import os


def read_samples(samples_file):
    with open(samples_file, "r") as f:
        samples = f.read().splitlines()

    # check if samples exists
    for s in samples:
        if not os.path.exists(s):
            samples.remove(s)

    return samples


def list_files_recursively(directory):
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return sorted(files)
