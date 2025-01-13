from pathlib import Path
from typing import Literal

import pandas as pd

from .utils import _fetch_dataset


def _get_urls_df():
    return pd.read_csv(Path(__file__).parent / "eegeyenet_urls.csv")


def _get_params(subject, run):
    df = _get_urls_df()
    row = df.loc[(df.subject == subject.upper()) & (df.run == int(run))]
    assert len(row) == 1
    row = row.T.squeeze()
    task = row["task"]
    return dict(
                url=row["url"],
                archive_name=f"{subject}_{task}{run}_EEG.mat",
                folder_name=f"EEGEYENET-Data/{task}/{subject}",
                hash=row["hash"],
                dataset_name="EEGEYENET")


def get_subjects_runs(task: Literal["DOTS", "AS"] = "DOTS"):
    """Get dictionary of {subject: [lists of runs]}.

    Parameters
    ----------
    task :
        Which EEGEYENET task task to extract the subject ID's and runs for. Can be
        ``"DOTS"``, or ``"AS"`` (antisaccade). Defaults to ``'DOTS'``.

    Returns
    -------
    dict
        Dictionary of subjects with the runs as values.
    """
    df = _get_urls_df()
    df = df.loc[df["task"] == task].copy()
    return {subject: df.run.values[df.subject == subject]
            for subject in df.subject.unique()}


def fetch_eegeyenet(subject="EP10", run=1, fetch_dataset_kwargs=None):
    """Fetch a sample file from the EEG Eyenet dataset.

    Parameters
    ----------
    subject : str
        Subject identifier. Defaults to ``'EP10'``.
    run : int | str
        Which run to download. Most Participants completed 6 runs of the task,
        saved to 6 different files. Defaults to ``1``.
    fetch_dataset_kwargs : dict | None
        Keyword arguments to pass to :func:`~mne.datasets.fetch_dataset`.
        if ``None``, no keyword arguments are passed. Defaults to ``None``.

    Returns
    -------
    pathlib.Path
        Path to the downloaded file.
    """
    task = _get_task_from_subject_id(subject)
    if not fetch_dataset_kwargs:
        fetch_dataset_kwargs = dict()
    run = int(run)
    runs = get_subjects_runs(task=task)
    if subject not in runs or run not in runs[subject]:
        raise ValueError(f"subject {subject} and run {run} not available. "
                         "See get_subjects_runs() for information on "
                         "available subjects and runs.")

    fetch_dataset_kwargs["dataset_params"] = _get_params(subject, run)
    fpath = _fetch_dataset(fetch_dataset_kwargs=fetch_dataset_kwargs)
    fpath = fpath / fetch_dataset_kwargs["dataset_params"]["archive_name"]

    # mne.datasets.fetch_dataset will not download a new run if the subject
    # folder already exist
    if not fpath.exists():
        fetch_dataset_kwargs["force_update"] = True
        _fetch_dataset(fetch_dataset_kwargs=fetch_dataset_kwargs)
    return fpath


def _get_task_from_subject_id(subject):
    if  subject.startswith("EP"):
        return "DOTS"
    if subject.startswith(("A", "B")):
        return "AS"
    raise ValueError(
        f"Can't determine task for {subject}. Is this subject in eegeyenet_urls.csv?"
        )
