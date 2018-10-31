import os

def get_files(dir, ext, recursive=True):
    for sub in [os.path.join(dir, sub) for sub in os.listdir(dir)]:
        if os.path.isdir(sub):
            yield from get_files(sub, ext)
        elif os.path.splitext(sub)[1] == ext:
            yield sub