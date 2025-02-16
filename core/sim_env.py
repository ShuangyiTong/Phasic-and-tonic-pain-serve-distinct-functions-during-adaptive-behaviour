# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

from typing import List, Tuple, Callable, Union, Type

import numpy as np
import random

from core.utils import FixSizeOrderedDict

from collections import deque

def eye_tracked_obs(pineapple_map: dict,
                    gaze_objects: List[str],
                    agent_eye_pos: List[int],
                    memory: Type[deque]) -> Tuple[Union[Tuple[str, float], None], Union[Tuple[str, float], None], Type[deque]]:
    candidate_pineapples = []
    for pineapple in memory:
        try:
            location = pineapple_map[pineapple] # maybe it is already picked up
        except KeyError:
            continue
        np_location = np.array([location['x'], location['y'], location['z']])
        np_eye_pos = np.array(agent_eye_pos)
        distance_vector = np_location - np_eye_pos
        norm_distance = np.linalg.norm(distance_vector)
        candidate_pineapples.append((pineapple, norm_distance))

    for gaze_object in gaze_objects:
        if gaze_object.startswith('PA'):
            try:
                location = pineapple_map[gaze_object] # sometimes people will still look at the pineapple once they release it
            except KeyError:
                continue
            np_location = np.array([location['x'], location['y'], location['z']])
            np_eye_pos = np.array(agent_eye_pos)
            distance_vector = np_location - np_eye_pos
            norm_distance = np.linalg.norm(distance_vector)
            if gaze_object not in memory:
                memory.append(gaze_object)
            candidate_pineapples.append((gaze_object, norm_distance))

    sorted_memorized_pineapples = sorted(list(candidate_pineapples), key=lambda x: x[1])
    painful = list(filter(lambda x: x[0].endswith('G'), sorted_memorized_pineapples))
    non_painful = list(filter(lambda x: not x[0].endswith('G'), sorted_memorized_pineapples))

    return (painful[0] if len(painful) > 0 else None, non_painful[0] if len(non_painful) > 0 else None, memory)

def action_step(pineapple_map: dict,
                action: Union[str, None],
                agent_eye_pos: List[int], 
                agent_eye_rotation: List[int]) -> Tuple[dict, List[int], List[int]]:
    new_agent_eye_pos = agent_eye_pos
    # TODO: Restrict region
    new_agent_eye_pos[0] += random.uniform(-1, 1)
    new_agent_eye_pos[2] += random.uniform(-1, 1)
    # TODO: implement rotation
    return { n: l for n, l in pineapple_map.items() if n != action }, new_agent_eye_pos, agent_eye_rotation

def eye_tracked_obs_v2(gaze_object: str,
                       memory: Type[FixSizeOrderedDict],
                       reference_pos: List[float],
                       pineapple_map: dict) -> Tuple[Type[FixSizeOrderedDict], str]:
    new_pineapple = ""
    if gaze_object.startswith("PA") and gaze_object not in memory.keys():
        memory[gaze_object] = 0
        new_pineapple = gaze_object
    
    to_pop = []
    for pineapple in memory.keys():
        try:
            location = pineapple_map[pineapple]
        except:
            '''This is possible due to unresponsive eye tracking callback
               so gaze object was not updated promptly
            '''
            print("Warning: ", pineapple, "redeleted")
            to_pop.append(pineapple) # direct pop here can result in RuntimeError: OrderedDict mutated during iteration if there are subsequent elements in the memory
            continue
        np_location = np.array([location['x'], location['y'], location['z']])
        np_ref_pos = np.array(reference_pos)
        distance_vector = np_location - np_ref_pos
        memory[pineapple] = distance_vector.tolist()

    for pineapple in to_pop:
        memory.pop(pineapple)

    return memory, new_pineapple