import os


def make_folder_if_does_not_exist(folder: str):
    print(f"Making folder in case it doesnt exist: {folder}")
    os.system(f"mkdir -p {folder}")


def get_filenames_in_a_folder(folder: str):
    """
    returns the list of paths to all the files in a given folder
    """
    if folder[-1] == "/":
        folder = folder[:-1]

    files = os.listdir(folder)
    files = [f"{folder}/" + x for x in files]
    return files
