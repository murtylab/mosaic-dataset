from ...constants import resting_state_folder, BASE_URL
from ...utils.download import download_file
import requests
import os


valid_subjects = list(
    range(1,11)
)
valid_runs = list(
    range(1,5)
)
valid_sessions = [1]

def get_filename(
    subject: int,
    session: int,
    run: int,
) -> str:
    return f"sub-{subject:02d}_ses-{session:02d}_task-rest_run-{run}_space-fsLR_den-91k_bold_clean.dtseries.nii"

def download_load_bold_moments_resting_state_data(
    subject: int,
    session: int,
    run: int,
    folder: str
):
    assert subject in valid_subjects, f"Invalid subject {subject}. Must be one of {valid_subjects}"
    assert session in valid_sessions, f"Invalid session {session}. Must be one of {valid_sessions}"
    assert run in valid_runs, f"Invalid run {run}. Must be one of {valid_runs}"
    filename = get_filename(
        subject=subject,
        session=session,
        run=run,
    )
    file_in_s3 = os.path.join(
        resting_state_folder,
        "BOLDMomentsDataset",
        filename,
    )

    url = os.path.join(BASE_URL, file_in_s3)
    response = requests.head(url)
    assert response.status_code == 200, f"URL {url} is not valid or reachable."
    download_file(
        base_url=BASE_URL,
        file=file_in_s3,
        save_as=os.path.join(folder, filename),
    )

    return os.path.join(folder, filename)