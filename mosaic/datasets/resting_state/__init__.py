from .bold_moments import download_load_bold_moments_resting_state_data
from .nsd import download_load_nsd_resting_state_data
from .things_fmri import download_things_resting_state_data

def download_resting_state_data(
    dataset: str,
    subject: int,
    session: int,
    run: int,
    folder: str
):
    assert dataset in ["NSD", "BMD", "THINGS"], f"Invalid dataset {dataset}. Must be one of ['NSD', 'BMD', 'THINGS']"
    if dataset == "NSD":
        return download_load_nsd_resting_state_data(
            subject=subject,
            session=session,
            run=run,
            folder=folder
        )
    elif dataset == "BMD":
        return download_load_bold_moments_resting_state_data(
            subject=subject,
            session=session,
            run=run,
            folder=folder
        )
    elif dataset == "THINGS":
        return download_things_resting_state_data(
            subject=subject,
            session=session,
            run=run,
            folder=folder
        )
    else:
        raise ValueError(f"Invalid dataset {dataset}. Must be one of ['NSD', 'BMD', 'THINGS']")
    