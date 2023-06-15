#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from neuromast3d.run_neuromast3d import validate_steps, POSSIBLE_STEPS


def test_validate_steps_no_steps():
    with pytest.raises(ValueError):
        validate_steps([])


def test_validate_steps_one_step(step=[POSSIBLE_STEPS[0]]):
    # Running one step is fine, nothing should happen
    validate_steps(step)


@pytest.mark.parametrize(
        'desired_steps', [
            [POSSIBLE_STEPS[0], POSSIBLE_STEPS[2]],
            [POSSIBLE_STEPS[0], POSSIBLE_STEPS[3]],
            [POSSIBLE_STEPS[0], POSSIBLE_STEPS[1], POSSIBLE_STEPS[3]]
        ]
    )
def test_validate_steps_skipped_step(desired_steps):
    with pytest.raises(ValueError):
        validate_steps(desired_steps)
