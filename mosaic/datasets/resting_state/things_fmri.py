from ...constants import resting_state_folder, BASE_URL
from ...utils.download import download_file
import requests
import os

# Valid subjects and sessions
valid_subjects = list(range(1, 9))
valid_runs = [1, 2]
VALID_LOCALIZERS = [f"localizer{i}" for i in range(1, 3)]
VALID_THINGS = [f"things{i:02d}" for i in range(1, 13)]
valid_sessions = VALID_LOCALIZERS + VALID_THINGS

def get_filename(subject: int, session: str, run: int) -> str:
    """
    Construct the filename for a given subject, session, and run.
    """
    return (
        f"sub-{subject:02d}_ses-{session}_task-rest_"
        f"acq-reversePE_space-fsLR_den-91k_bold_clean.dtseries.nii"
    )

def download_things_resting_state_data(subject: int, session: str, run: int, folder: str) -> str:
    """
    Download the NSD resting-state fMRI data for a specific subject, session, and run.
    """
    # Input validation
    if subject not in valid_subjects:
        raise ValueError(f"Invalid subject {subject}. Must be one of {valid_subjects}")
    if session not in valid_sessions:
        raise ValueError(f"Invalid session {session}. Must be one of {valid_sessions}")
    if run not in valid_runs:
        raise ValueError(f"Invalid run {run}. Must be one of {valid_runs}")

    # Build filename and remote path
    filename = get_filename(subject, session, run)
    file_in_s3 = os.path.join(resting_state_folder, "THINGS", filename)
    url = os.path.join(BASE_URL, file_in_s3)

    # Check if URL exists
    response = requests.head(url)
    if response.status_code != 200:
        raise FileNotFoundError(f"URL {url} is not valid or reachable.")

    # Download file
    save_path = os.path.join(folder, filename)
    download_file(base_url=BASE_URL, file=file_in_s3, save_as=save_path)

    return save_path
