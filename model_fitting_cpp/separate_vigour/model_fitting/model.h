/*
 * Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
 * Licensed under the MIT License.
 */

#ifndef MODEL_H
#define MODEL_H

#include "frame_defs.h"
#include "cuda_defs.h"
#include "csv_frame_reader.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

#define MODEL_OPTIMAL_MIN -INFINITY
#define GET_ITEM_NONE 0xffffffffu

namespace model
{
    enum PARAMETER
    {
        PAIN_FUNC_X_SCALE,
        PAIN_FUNC_X_TRANSLATE,
        VIGOUR_CONSTANT_NO_PAIN,
        VIGOUR_CONSTANT_LOW_PAIN,
        VIGOUR_CONSTANT_HIGH_PAIN,
#ifdef FIVE_LEVELS
        VIGOUR_CONSTANT_FOURTH,
        VIGOUR_CONSTANT_FIFTH,
#endif
        AVERAGE_REWARD,
        PAIN_FUNC_SCALE,
        HORIZONTAL_DISTANCE_COF, // vertical coefficient is normalized by horizontal
        NUM_FREE_PARAMETER
    };

    float sum(int size, float* array)
    {
        float acc = 0;
        for (int i = 0; i < size; i++)
        {
            acc += array[i];
        }
        return acc;
    }

#ifdef CUDA_ENABLED
    __device__
#endif
    void combinedModel(uint32_t& optimal_choice,
                       float& Q_value,
                       uint32_t* memory,
                       float pain_intensity,
                       float* free_parameter,
                       float avg_reward,
                       float avg_speed,
                       float current_delay,
                       uint32_t get_item=frame_defs::OBJ_BOTTOM_TYPE)
    {
        avg_speed *= 1000;
        current_delay /= 1000;

        uint32_t current_optimal_choice = frame_defs::OBJ_BOTTOM_TYPE;
        float current_optimal_Q = MODEL_OPTIMAL_MIN;
        uint32_t optimal_choice_is_less_optimal = 0;
    #ifdef DEBUG
        float debug_optimal_tau = -1;
        int debug_mem_size = 0;
    #endif

        for (int i = 0; i < frame_defs::MEM_SIZE; i++)
        {
            uint32_t less_optimal = 0;
            uint32_t choice_name = memory[i * 4];
            if (get_item != GET_ITEM_NONE &&
                choice_name != get_item)
            {
                continue;
            }
            if (choice_name == frame_defs::OBJ_BOTTOM_TYPE) break;
        #ifdef DEBUG
            debug_mem_size++;
        #endif
        
        #ifdef FIVE_LEVELS
            int vigour_constant_to_use = VIGOUR_CONSTANT_NO_PAIN + int(round(pain_intensity / 0.25f));
        #else
            int vigour_constant_to_use = VIGOUR_CONSTANT_LOW_PAIN;
            if (fabs(1 - pain_intensity) < 1e-10f)
            {
                vigour_constant_to_use = VIGOUR_CONSTANT_HIGH_PAIN;
            }
            else if (pain_intensity < 1e-10f)
            {
                vigour_constant_to_use = VIGOUR_CONSTANT_NO_PAIN;
            }
        #endif

            float x = frame_defs::INTEGER_DWORD_TO_FLOAT(memory[i * 4 + 1]);
            float y = frame_defs::INTEGER_DWORD_TO_FLOAT(memory[i * 4 + 2]);
            float z = frame_defs::INTEGER_DWORD_TO_FLOAT(memory[i * 4 + 3]);
            float beta_dot_distance_times_C_v = free_parameter[vigour_constant_to_use] * 
                (free_parameter[HORIZONTAL_DISTANCE_COF] * sqrt(x * x + z * z) +
                 sqrt(1 - (free_parameter[HORIZONTAL_DISTANCE_COF] 
                         * free_parameter[HORIZONTAL_DISTANCE_COF])) * abs(y));
            
            float optimal_tau = sqrt(beta_dot_distance_times_C_v / avg_reward);
            float optimal_current_delay = optimal_tau - (sqrt(x * x + y * y + z * z) / avg_speed);

            if (optimal_current_delay > current_delay) 
            {
                less_optimal = 1;
            }

            float choice_value = (choice_name&frame_defs::GREEN_MASK)?(free_parameter[PAIN_FUNC_SCALE] / 
                (1 + exp(-free_parameter[PAIN_FUNC_X_SCALE] * (pain_intensity - free_parameter[PAIN_FUNC_X_TRANSLATE]))))
                :0;
#ifdef DEBUG_SINGLE_TEST
            std::cout << "pain value = " << choice_value;
#endif
            // float choice_value = (choice_name&frame_defs::GREEN_MASK)?free_parameter[PAIN_FUNC_SCALE] * (pain_intensity + free_parameter[PAIN_FUNC_X_TRANSLATE]):0;
            float current_tau = (sqrt(x * x + y * y + z * z) / avg_speed) + current_delay;
            choice_value -= beta_dot_distance_times_C_v / current_tau;
            choice_value -= avg_reward * current_tau;

#ifdef DEBUG_SINGLE_TEST
            std::cout << " beta_dot_vigour = " << beta_dot_distance_times_C_v << " v_constant =" << free_parameter[vigour_constant_to_use] 
            << " current_tau = " << current_tau << " choice_value = " << choice_value << " avg_reward = " << avg_reward << std::endl;
#endif

            if (choice_name == get_item)
            {
                optimal_choice = choice_name;
                Q_value = choice_value;
                return;
            }

        #ifdef DEBUG
            // std::cout << choice_value << " " << current_tau << std::endl;
            // debug_optimal_tau = optimal_tau;
        #endif
            if (choice_value > current_optimal_Q)
            {
                current_optimal_choice = choice_name;
                current_optimal_Q = choice_value;
                optimal_choice_is_less_optimal = less_optimal;
            }
        }

        if (get_item != GET_ITEM_NONE)
        {
            // Not found return None
            optimal_choice = frame_defs::OBJ_BOTTOM_TYPE;
            Q_value = MODEL_OPTIMAL_MIN;
            return;
        }

    #ifdef DEBUG
        assert (current_optimal_choice != frame_defs::OBJ_BOTTOM_TYPE || !debug_mem_size);
    #endif

        if (current_optimal_choice != frame_defs::OBJ_BOTTOM_TYPE && 
            !optimal_choice_is_less_optimal)
        {
            optimal_choice = current_optimal_choice;
            Q_value = current_optimal_Q;
        }
        else
        {
            optimal_choice = frame_defs::OBJ_BOTTOM_TYPE;
            Q_value = MODEL_OPTIMAL_MIN;
        }
        return;
    }

#ifdef CUDA_ENABLED
    __device__
#endif
    float blockSimulation(float* free_parameters,
                            uint32_t num_frames,
                            uint32_t* frame_block,
                            float pain_intensity,
                            uint32_t num_choices,
                            uint32_t* pickup_choices)
    {
    // assert(1 == free_parameters[1]);
    // assert(0x80000074u == pickup_choices[18]);
    // assert(0x88888888u == frame_block[2]);
    // assert(2631 == num_frames);
    // assert(19 == num_choices);
    #ifdef DEBUG
        for (int i = 0; i < model::NUM_FREE_PARAMETER; i++)
        {
            std::cout << free_parameters[i] << "    ";
        }
        std::cout << std::endl;     
    #endif
        uint32_t in_pickup = 0;
        uint32_t skip_current_pickup = 0;

        float correct_prediction = 0;
        float total_actions = 0;

        uint32_t pickup_choices_counter = 0;
        uint32_t last_picked_pineapple_ts = 0;

        uint32_t choice = frame_defs::OBJ_BOTTOM_TYPE;
        float choice_value;
    #ifdef USE_LAST_OPTIMAL_VALUE
        float last_non_choice_optimal_value = MODEL_OPTIMAL_MIN;
        uint32_t last_non_optimal_choice = frame_defs::OBJ_BOTTOM_TYPE;
        uint32_t need_reevaluation = 1;
        uint32_t prediction_is_correct_for_now = 0;
    #endif
        for (int i = 0; static_cast<uint32_t>(i) < num_frames; i++)
        {
            uint32_t* current_frame = frame_block + i * frame_defs::FRAME_SIZE;
            if (!in_pickup)
            {
                if (pickup_choices_counter < num_choices)
                {
                    if (!skip_current_pickup)
                    {
                        if (current_frame[frame_defs::TIMESTAMP] > 0)
                        {
                        #ifdef USE_LAST_OPTIMAL_VALUE
                            // cut duplicate computation
                            if (current_frame[frame_defs::GAZE_OBJ] == frame_defs::OBJ_BOTTOM_TYPE)
                            {
                                goto SKIP_VALUE_UPDATE;
                            }
                            else
                            {
                                if (current_frame[frame_defs::GAZE_OBJ] == pickup_choices[pickup_choices_counter])
                                {
                                    if (!need_reevaluation)
                                    {
                                        goto SKIP_VALUE_UPDATE;
                                    }
                                }
                            }
                            // Bug (fitting logic) fixed here: all frames for non optimal choice should be checked
                            // This makes opportunity-vigour trade-off weights more

                            combinedModel(choice, choice_value, current_frame + frame_defs::MEM_START,
                            pain_intensity, free_parameters, free_parameters[AVERAGE_REWARD],
                            frame_defs::INTEGER_DWORD_TO_FLOAT(current_frame[frame_defs::AVG_SPEED]),
                            current_frame[frame_defs::TIMESTAMP] - last_picked_pineapple_ts,
                            current_frame[frame_defs::GAZE_OBJ]);

                            if (choice != frame_defs::OBJ_BOTTOM_TYPE)
                            {
                                if (choice != pickup_choices[pickup_choices_counter])
                                {
                                    if (last_non_choice_optimal_value < choice_value)
                                    {
                                        last_non_choice_optimal_value = choice_value;
                                        last_non_optimal_choice = choice;
                                        // No need to compare with previous correct choice's optimal value
                                        // Either they
                                        // - need to look at the right pineapple again to pick it up, which will be reevaluated
                                        // - or they don't look at it to pick up which means they almost picked it up
                                    #ifdef DEBUG
                                        std::cout << "Update non choice value: " << frame_defs::OBJECT_IDENTIFIER_TO_READABLE_PINEAPPLE_NAME(choice)
                                        << " -> " << choice_value << std::endl;
                                    #endif
                                    }

                                    need_reevaluation = 1;
                                }
                                else
                                {
                                    if (need_reevaluation)
                                    {
                                        if (last_non_choice_optimal_value <= choice_value)
                                        {
                                            prediction_is_correct_for_now = 1;
                                        #ifdef DEBUG
                                            std::cout << "CORRECT: " << frame_defs::OBJECT_IDENTIFIER_TO_READABLE_PINEAPPLE_NAME(choice) << " -> "
                                            << choice_value << " > " << last_non_choice_optimal_value << std::endl;
                                        #endif
                                        }
                                        else
                                        {
                                            prediction_is_correct_for_now = 0;
                                        #ifdef DEBUG                                            
                                            std::cout << "INCORRECT: " << frame_defs::OBJECT_IDENTIFIER_TO_READABLE_PINEAPPLE_NAME(choice) << " -> "
                                            << choice_value << " < " << last_non_choice_optimal_value << std::endl;
                                        #endif                                        
                                        }
                                        need_reevaluation = 0;
                                    }
                                }
                            }
                        #else
                            combinedModel(choice, choice_value, current_frame + frame_defs::MEM_START,
                            pain_intensity, free_parameters, free_parameters[AVERAGE_REWARD],
                            frame_defs::INTEGER_DWORD_TO_FLOAT(current_frame[frame_defs::AVG_SPEED]),
                            current_frame[frame_defs::TIMESTAMP] - last_picked_pineapple_ts);

                            if (choice != frame_defs::OBJ_BOTTOM_TYPE)
                            {
                            #ifdef DEBUG
                                std::cout << "Predicted: " << frame_defs::OBJECT_IDENTIFIER_TO_READABLE_PINEAPPLE_NAME(choice) <<
                                "  Actual: " << frame_defs::OBJECT_IDENTIFIER_TO_READABLE_PINEAPPLE_NAME(pickup_choices[pickup_choices_counter]) << std::endl;
                            #endif
                                if (pickup_choices[pickup_choices_counter] == choice)
                                {
                                    correct_prediction += 1;
                                }
                                skip_current_pickup = 1;
                            }
                        #ifdef DEBUG
                            else
                            {
                                //std::cout << "Less Optimal Tau: " << choice_value << std::endl;
                            }
                        #endif
                        #endif
                        }
                    }
                }

// Don't worry, we have python code cross validating the correctness of the code, let's just goto!
SKIP_VALUE_UPDATE:

                if (current_frame[frame_defs::PICKED_OBJ] != frame_defs::OBJ_BOTTOM_TYPE)
                {
                #ifdef USE_LAST_OPTIMAL_VALUE
                    if (prediction_is_correct_for_now)
                    {
                        correct_prediction += 1;
                    #ifdef DEBUG
                        std::cout << "==========Correct, Expected: " << frame_defs::OBJECT_IDENTIFIER_TO_READABLE_PINEAPPLE_NAME(current_frame[frame_defs::PICKED_OBJ])
                        << " =========" << std::endl;

                    #endif
                    }
                    #ifdef DEBUG
                    else
                    {
                        std::cout << "==========Incorrect, non_optimal -> " << last_non_choice_optimal_value
                        << " Expected: " << frame_defs::OBJECT_IDENTIFIER_TO_READABLE_PINEAPPLE_NAME(current_frame[frame_defs::PICKED_OBJ])
                        << " =========" << std::endl;
                    }
                    #endif
                    // Reset for next pickup action
                    last_non_choice_optimal_value = MODEL_OPTIMAL_MIN;
                    last_non_optimal_choice = frame_defs::OBJ_BOTTOM_TYPE;
                    need_reevaluation = 1;
                    prediction_is_correct_for_now = 0;
                #endif
                    pickup_choices_counter += 1;
                    in_pickup = 1;
                    last_picked_pineapple_ts = current_frame[frame_defs::TIMESTAMP];
                    total_actions += 1;
                }
            }
            else
            {
                if (current_frame[frame_defs::PICKED_OBJ] == frame_defs::OBJ_BOTTOM_TYPE)
                {
                    in_pickup = 0;
                    skip_current_pickup = 0;
                }
            }

            if (current_frame[frame_defs::IN_BASKET_OBJ] != frame_defs::OBJ_BOTTOM_TYPE)
            {
                if (skip_current_pickup && current_frame[frame_defs::IN_BASKET_OBJ] == choice)
                {
                    choice = frame_defs::OBJ_BOTTOM_TYPE;
                    skip_current_pickup = 0;
                }
                if (last_non_optimal_choice == current_frame[frame_defs::IN_BASKET_OBJ])
                {
                    last_non_choice_optimal_value = MODEL_OPTIMAL_MIN;
                    last_non_optimal_choice = frame_defs::OBJ_BOTTOM_TYPE;
                    need_reevaluation = 1;
                }
            }
        }

    assert(pickup_choices_counter == num_choices);
    #ifdef DEBUG
        std::cout << correct_prediction << "/" << total_actions << std::endl;
    #endif
        return 1 - (correct_prediction / total_actions);
    }
}

#endif // MODEL_H