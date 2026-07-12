from old_model_fast import *

from core.experiment_data import get_multiple_series_lazy

from statistics import mean

def check_green(d):
    return True if list(filter(lambda x: x['target'].endswith('G'), d)) != [] else False

def check_1green(d):
    return True if len(list(filter(lambda x: x['target'].endswith('G'), d))) == 1 else False

def get_individual_statistics(individual_data):
    pineapple_maps = get_fruit_position_map(individual_data)[start_trial_for_analysis:end_trial_for_analysis]
    behavioural_data = [get_abstract_action_v2(individual_data, ts) for ts in get_trial_start_timestamps(individual_data)][start_trial_for_analysis:end_trial_for_analysis]
    print(get_trial_start_timestamps(individual_data))
    print(len(pineapple_maps), len(behavioural_data))
    if USE_PAIN_RATINGS:
        pain_conditions = list(zip(apply_corrections_natural_number_indexing(individual_data, get_end_of_trial_pain_ratings(individual_data), 'ratings_amendment')[start_trial_for_analysis:end_trial_for_analysis],
                                   list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis]))
    else:
        pain_conditions = list(zip([get_pain_cond_val(x) for x in list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis]],
                                list(map(lambda msg: msg.split('-')[-1], get_series_from_control(individual_data, 'log', 'msg', 'Main task session start', 'msg')))[start_trial_for_analysis:end_trial_for_analysis]))

    in_block_simulation_dumps = [in_block_simulation([1, 0, -1, 0.5, 0.5], pineapple_maps[x], pain_conditions[x], *(behavioural_data[x]), local_dump=True) 
                                    for x in range((end_trial_for_analysis if end_trial_for_analysis else 0) - start_trial_for_analysis)]

    in_block_simulation_dumps_trimmed = [trim_dump_to_first_green(d) for d in in_block_simulation_dumps]

    total_trials = 0
    total_trimed_trials = 0
    trimed_green_blocks = 0
    trimed_green_blocks_verify = 0
    total_blocks = 0

    for new_d, old_d in zip(in_block_simulation_dumps_trimmed, in_block_simulation_dumps):
        if old_d != old_d:
            continue
        else:
            total_blocks += 1
            if check_1green(old_d):
                trimed_green_blocks_verify += 1
            if new_d != new_d:
                total_trimed_trials += len(old_d)
                total_trials += len(old_d)
                if check_green(old_d):
                    trimed_green_blocks += 1
            else:
                total_trimed_trials += (len(old_d) - len(new_d))
                total_trials += len(old_d)
                if check_green(old_d):
                    if not check_green(new_d):
                        trimed_green_blocks += 1
    
    assert(trimed_green_blocks == trimed_green_blocks_verify)

    return total_trimed_trials / total_trials, trimed_green_blocks / total_blocks

trim_stats = get_multiple_series_lazy(exp_data, get_individual_statistics, subjects)

print(mean([x for x, y in trim_stats]), mean([y for x, y in trim_stats]))