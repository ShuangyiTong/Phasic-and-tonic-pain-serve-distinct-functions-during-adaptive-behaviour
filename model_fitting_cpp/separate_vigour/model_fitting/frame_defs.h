/*
 * Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
 * Licensed under the MIT License.
 */

#ifndef FRAME_DEFS_H
#define FRAME_DEFS_H

#include <string>
#include <stdint.h>
#include <stdexcept>
#include <tuple>

namespace frame_defs 
{
    enum FRAME : int
    {
        No = 0,
        TIMESTAMP,
        GAZE_OBJ,
        PICKED_OBJ,
        IN_BASKET_OBJ,
        NEW_PINEAPPLE,
        AVG_SPEED,
        MEM_START
    };

    constexpr int MEM_SIZE = 7;
    constexpr uint32_t OBJ_BOTTOM_TYPE = 0x88888888u;
    constexpr uint32_t GREEN_MASK = 0x80000000u;
    constexpr uint32_t UNGREEN_MASK = 0x7fffffffu;
    
    constexpr int FRAME_SIZE = MEM_START + MEM_SIZE * 4;

#ifdef CUDA_ENABLED
    __host__ __device__
#endif
    float INTEGER_DWORD_TO_FLOAT(uint32_t v)
    {
        void* pv = &v;
        return *reinterpret_cast<float*>(pv);
    }
    
#ifdef CUDA_ENABLED
    __host__ __device__
#endif
    uint32_t FLOAT_TO_INTEGER_DWORD(float v)
    {
        void* pv = &v;
        return *reinterpret_cast<uint32_t*>(pv);
    }

    uint32_t OBJ_IDENTIFIER_CONVERTER(std::string field_string, int column_counter)
    {
        if (field_string.length() < 1)
        {
            return OBJ_BOTTOM_TYPE;
        }

        if (field_string.starts_with("PA"))
        {
            uint32_t ret_val = std::stoi(field_string.substr(2));
            // stoi ignores G, we need to detect ourselves and save in the highest bit
            if (field_string.ends_with("G"))
            {
                ret_val |= GREEN_MASK;
            }

            if (ret_val == OBJ_BOTTOM_TYPE)
            {
                throw std::invalid_argument("OBJ_BOTTOM_TYPE collides with object name");
            }
            return ret_val;
        }

        if (column_counter == FRAME::TIMESTAMP)
        {
            return std::stoi(field_string);
        }

        try
        {
            // store 4 byte float in uint32 (our WORD)
            float v = std::stof(field_string);
            
            // no judgement should be made between OBJ_BOTTOM_TYPE against float
            return FLOAT_TO_INTEGER_DWORD(v);
        }
        catch (const std::invalid_argument& ex)
        {
            return OBJ_BOTTOM_TYPE;
        }
    }

    std::string OBJECT_IDENTIFIER_TO_READABLE_PINEAPPLE_NAME(uint32_t object_id)
    {
        if (object_id == OBJ_BOTTOM_TYPE)
        {
            return "";
        }
        else
        {
            // Get number
            int green_flag = 0;
            if (object_id & GREEN_MASK)
            {
                green_flag = 1;
            }

            // Remove initial green bit
            object_id &= UNGREEN_MASK;
            
            return std::string("PA") + std::to_string(object_id) + (green_flag ? std::string("G") : std::string(""));
        }
    }

    uint32_t getCellFromFrames(uint32_t* frames, int current_frame, int column_id)
    {
        return frames[FRAME_SIZE * current_frame + column_id];
    }

    std::tuple<uint32_t, uint32_t*> generatePickedChoicesList(uint32_t num_frames, uint32_t* frames)
    {
        int current_frame = 0;
        uint32_t picked_count = 0;
        uint32_t* picked_choices = new uint32_t[num_frames]; // can't exceed this number, don't worry we have enough memory
        uint32_t prev_frame_picked = OBJ_BOTTOM_TYPE;
        while (current_frame < static_cast<int>(num_frames))
        {
            uint32_t picked_name = getCellFromFrames(frames, current_frame, PICKED_OBJ);

            if (picked_name != OBJ_BOTTOM_TYPE && picked_name != prev_frame_picked)
            {
                picked_choices[picked_count] = picked_name;
                picked_count++;
            }
            prev_frame_picked = picked_name;
            current_frame++;
        }
        
        return std::tuple<uint32_t, uint32_t*>(picked_count, picked_choices);
    }
}

#endif // FRAME_DEFS_H