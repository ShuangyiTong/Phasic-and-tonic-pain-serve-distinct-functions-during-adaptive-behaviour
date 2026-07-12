# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

from typing import List, Callable, Type
from core.utils import a, a1

from core.individual_subject import make_individual_data
from datasets.manage import get_subject_names
from datasets.manage import set_expt as manage_set_expt

def set_expt(expt_name: str) -> None:
    manage_set_expt(expt_name)

def make_experiment_data(exclusive_participants: List[str] = [],
                         exclude_participants: List[str] = [],
                         exclude_device_data: List[str] = [],
                         lazy_closure: bool = False) -> dict:
    exp_data = {}
    for subject_name in get_subject_names():
        if subject_name in exclude_participants:
            continue
        if exclusive_participants != []:
            if subject_name not in exclusive_participants:
                continue
        if lazy_closure:
            # why this doesn't work: lambda: make_individual_data(subject_name, exclude_device_data)
            # stupid python local binding: check https://stackoverflow.com/a/49302326/7733679
            # you can however create a mutable list and put it in the closure, but that is even more cumbersome
            # just iterate this with get_multiple_series_lazy solves all problems, at least on the surface
            exp_data[subject_name] = lambda subject_name: make_individual_data(subject_name, exclude_device_data)
        else:
            exp_data[subject_name] = make_individual_data(subject_name, exclude_device_data)

    return exp_data

def get_multiple_series(experiment_data: dict,
                        series_maker: Callable[[dict], List[a]], 
                        subjects: List[str]) -> List[List[a]]:
    if subjects == []:
        return []
    else:
        return [series_maker(experiment_data[subjects[0]])] + get_multiple_series(experiment_data, series_maker, subjects[1:])

def get_multiple_series_lazy(experiment_data: dict,
                             series_maker: Callable[[dict], List[a]], 
                             subjects: List[str]) -> List[List[a]]:
    if subjects == []:
        return []
    else:
        return [series_maker(experiment_data[subjects[0]](subjects[0]))] + get_multiple_series_lazy(experiment_data, series_maker, subjects[1:])

def get_multiple_series_lazy_subject_indexed(experiment_data: dict,
                             series_maker: Callable[[dict], List[a]], 
                             subjects: List[str]) -> List[List[a]]:
    if subjects == []:
        return []
    else:
        return [series_maker(experiment_data[subjects[0]](subjects[0]), subjects[0])] + get_multiple_series_lazy_subject_indexed(experiment_data, series_maker, subjects[1:])

# TODO: Use Pool instead
from multiprocessing import Process, Queue

def queue_wrapper_subject_indexed(q, series_maker, subject_name, args):
    ret = series_maker(args, subject_name)
    q.put((subject_name, ret))

def queue_wrapper(q, series_maker, subject_name, args):
    ret = series_maker(args)
    q.put((subject_name, ret))

def submit_to_subprocess(res_queue: Type[Queue],
                         series_maker: Callable[[dict], List[a]],
                         individual_data: dict,
                         subject_name: str,
                         subject_index: bool) -> None:
    if subject_index:
        p = Process(target=queue_wrapper_subject_indexed, args=(res_queue, series_maker, subject_name, individual_data))
        p.start()
    else:
        p = Process(target=queue_wrapper, args=(res_queue, series_maker, subject_name, individual_data))
        p.start()

def get_multiple_series_subprocessed_lazy(experiment_data: dict,
                                     series_maker: Callable[[dict], List[a]], 
                                     subjects: List[str],
                                     procs: int=8,
                                     subject_index: bool=False) -> List[List[a]]:
    results = {}
    current_pending = 0
    res_queue = Queue()
    for subject in subjects:
        if current_pending >= procs:
            res = res_queue.get()
            current_pending -= 1
            results[res[0]] = res[1]
    
        submit_to_subprocess(res_queue, series_maker, experiment_data[subject](subject), subject, subject_index)
        current_pending += 1

    while len(subjects) != len(results):
        res = res_queue.get()
        results[res[0]] = res[1]

    return [results[s] for s in subjects]