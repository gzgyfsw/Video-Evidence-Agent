"""Explicit recovery locations; prior partial attempts remain at their paths."""
import json
from pathlib import Path
import re


def group_folder(trial,index,variant='fp32'):
    trial=Path(trial);mapping=trial/'GROUP_OUTPUT_OVERRIDES.json'
    name=f'zero_update_group{index:02d}_{variant}'
    if mapping.exists():
        override=json.loads(mapping.read_text())['folders'].get(str(index))
        if override:
            if not re.fullmatch(rf'zero_update_group{index:02d}_{variant}_recovery[1-9][0-9]*',override):
                raise ValueError('Recovery folder must be a new local folder for the same frozen group')
            name=override
    return trial/name


def numerical_folder(folder):
    """A diagnosed numerical-only recovery never resamples or hides the prior run."""
    folder=Path(folder);mapping=folder.parent/'NUMERICAL_OUTPUT_OVERRIDES.json'
    name='train_mode_probability_replay_v1'
    if mapping.exists():
        override=json.loads(mapping.read_text())['folders'].get(folder.name)
        if override:
            if not re.fullmatch(r'train_mode_probability_replay_recovery[1-9][0-9]*',override):
                raise ValueError('Numerical recovery must use a new local replay directory')
            name=override
    return folder/name


def report_folder(folder):
    folder=Path(folder);mapping=folder.parent/'REPORT_OUTPUT_OVERRIDES.json'
    name=folder.name+'_report_review'
    if mapping.exists():
        override=json.loads(mapping.read_text())['folders'].get(folder.name)
        if override:
            if not re.fullmatch(re.escape(name)+r'_recovery[1-9][0-9]*',override):
                raise ValueError('Report recovery must be a new local directory for the same sampled group')
            name=override
    return folder.parent/name
