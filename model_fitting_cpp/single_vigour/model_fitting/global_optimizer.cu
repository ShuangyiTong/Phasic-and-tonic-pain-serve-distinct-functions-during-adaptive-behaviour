/*
 * Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
 * Licensed under the MIT License.
 */

#include "csv_frame_reader.h"
#include "model.h"
#include "cuda_defs.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <fstream>

#ifndef CUDA_ENABLED
    #include <omp.h>
#endif

namespace fs = std::filesystem;

constexpr int PLAIN_TICKS = 0;
constexpr int SLICED_TICKS = 1;

#ifdef CUDA_ENABLED
__host__ __device__
#endif
void getParameters(uint64_t i, float* free_parameter, uint32_t* num_ticks, float** parameter_ticks)
{
    uint64_t remainder = i;
    for (int k = 0; k < model::NUM_FREE_PARAMETER; k++)
    {
        uint64_t nx = num_ticks[model::NUM_FREE_PARAMETER - k - 1];
        free_parameter[model::NUM_FREE_PARAMETER - k - 1] = parameter_ticks[model::NUM_FREE_PARAMETER - k - 1][remainder % nx];
        remainder = (remainder - remainder % nx) / nx;
    }
    assert(remainder == 0);
    return;
}

#ifdef CUDA_ENABLED
__global__
#endif
void singleRun(uint64_t i, float& best_acc, float* best_parameters, uint64_t grid_size, uint32_t blocks,
               uint32_t* num_ticks, float** parameter_ticks, float* pain_intensity, 
               uint32_t* num_choices, uint32_t** pickup_choices, uint32_t* num_frames, uint32_t** frames,
               float* result_grid)
{
    float acc_sum = 0;

#ifdef CUDA_ENABLED
    uint64_t parameter_index = blockDim.x * blockIdx.x + threadIdx.x + i;
    if (parameter_index >= grid_size)
    {
        return;
    }
#endif
    float free_parameter[model::NUM_FREE_PARAMETER];
#ifdef CUDA_ENABLED
    getParameters(parameter_index, free_parameter, num_ticks, parameter_ticks);
#else
    getParameters(i, free_parameter, num_ticks, parameter_ticks);
#endif
    for (int j = 0; static_cast<uint32_t>(j) < blocks; j++)
    {
        float block_acc = model::blockSimulation(free_parameter, num_frames[j], frames[j],
            pain_intensity[j], num_choices[j], pickup_choices[j]);
        acc_sum += block_acc;
    #ifdef DEBUG_SINGLE_TEST
        std::cout << pain_intensity[j] << "======= ACC:" << block_acc << std::endl;
    #endif
    }
    acc_sum /= blocks;

#ifdef CUDA_ENABLED
    result_grid[blockDim.x * blockIdx.x + threadIdx.x] = acc_sum;
#else
#ifndef DEBUG
    #pragma omp critical
#endif
    {
        if (acc_sum < best_acc)
        {
            best_acc = acc_sum;
            memcpy(best_parameters, free_parameter, model::NUM_FREE_PARAMETER * sizeof(float));

        #ifdef DEBUG
            for (int i = 0; i < model::NUM_FREE_PARAMETER; i++)
            {
                std::cout << best_parameters[i] << "    ";
            }
            std::cout << std::endl;
        #endif
        }

    #ifndef DEBUG_SINGLE_TEST
        if (i % (grid_size / 1000) == 0)
        {
            std::cout<<i / (float)grid_size * 100 << "\r";
        }
    #endif
    }
#endif
}

void gridSearch(uint64_t grid_size, uint32_t blocks, float* fitted_result, uint32_t* num_ticks, float** parameter_ticks, float* pain_intensity, 
                uint32_t* num_choices, uint32_t** pickup_choices, uint32_t* num_frames, uint32_t** frames)
{
    float best_acc = 1;
    float best_parameters[model::NUM_FREE_PARAMETER];

#ifndef DEBUG_SINGLE_TEST
    // too small grid size causes division by zero when not enable DEBUG_SINGLE_TEST
    assert(grid_size >= 1000); 
#endif

#ifdef CUDA_ENABLED
    cudaError_t ret = cudaSuccess;

    uint32_t* devptr_num_ticks = nullptr;
    CUDA_DEVICE_ALLOC_AND_COPY(devptr_num_ticks, num_ticks, model::NUM_FREE_PARAMETER, sizeof(uint32_t), ret)
    float** devptr_parameter_ticks = nullptr;
    float** host_ptr_to_devptr_parameter_ticks = nullptr;
    CUDA_DEVICE_ALLOC_AND_COPY_NESTED_ARRAY(float, devptr_parameter_ticks, host_ptr_to_devptr_parameter_ticks, parameter_ticks, model::NUM_FREE_PARAMETER, sizeof(float*), num_ticks, sizeof(float), ret)

    float* devptr_pain_intensity = nullptr;
    CUDA_DEVICE_ALLOC_AND_COPY(devptr_pain_intensity, pain_intensity, blocks, sizeof(float), ret)

    uint32_t* devptr_num_choices = nullptr;
    CUDA_DEVICE_ALLOC_AND_COPY(devptr_num_choices, num_choices, blocks, sizeof(uint32_t), ret)
    uint32_t** devptr_pickup_choices = nullptr;
    uint32_t** host_ptr_to_devptr_pickup_choices = nullptr;
    CUDA_DEVICE_ALLOC_AND_COPY_NESTED_ARRAY(uint32_t, devptr_pickup_choices, host_ptr_to_devptr_pickup_choices, pickup_choices, blocks, sizeof(uint32_t*), num_choices, sizeof(uint32_t), ret)

    uint32_t* devptr_num_frames = nullptr;    
    CUDA_DEVICE_ALLOC_AND_COPY(devptr_num_frames, num_frames, blocks, sizeof(uint32_t), ret)    
    uint32_t** devptr_frames = nullptr;
    uint32_t frame_size = sizeof(uint32_t) * frame_defs::FRAME_SIZE;
    uint32_t** host_ptr_to_devptr_frames = nullptr;
    CUDA_DEVICE_ALLOC_AND_COPY_NESTED_ARRAY(uint32_t, devptr_frames, host_ptr_to_devptr_frames, frames, blocks, sizeof(uint32_t*), num_frames, frame_size, ret)

    float* devptr_result_grid = nullptr;
    ret = cudaMalloc(&devptr_result_grid, MAX_LAUNCH_SIZE * sizeof(float));
    CUDA_CHECKS(ret)

    uint64_t current_count = 0;
    float* result_grid = new float[MAX_LAUNCH_SIZE];
    uint64_t argmin = 0;
    while (current_count < grid_size)
    {
        uint64_t launched_kernels = grid_size - current_count > MAX_LAUNCH_SIZE ? MAX_LAUNCH_SIZE : grid_size - current_count;
        singleRun<<<ceil((double)launched_kernels / (double)THREADSPERBLOCK), THREADSPERBLOCK, 0, 0>>>(current_count, best_acc, nullptr, grid_size, blocks, 
        devptr_num_ticks, devptr_parameter_ticks, devptr_pain_intensity,
        devptr_num_choices, devptr_pickup_choices,
        devptr_num_frames, devptr_frames, devptr_result_grid);
        ret = cudaDeviceSynchronize();
        CUDA_CHECKS(ret)

        ret = cudaMemcpy(result_grid, devptr_result_grid, launched_kernels * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECKS(ret)

        for (int i = 0; i < launched_kernels; i++)
        {
            if (result_grid[i] < best_acc)
            {
                argmin = i + current_count;
                best_acc = result_grid[i];
            }
        }
        current_count += launched_kernels;
        std::cout<<current_count / (double)grid_size * 100 << "\r";
    }
    getParameters(argmin, best_parameters, num_ticks, parameter_ticks);

    CUDA_FREE_WITH_CHECK(devptr_result_grid, ret);
#else
#ifndef DEBUG
    #pragma omp parallel for schedule(dynamic)
#endif
    for (uint64_t i = 0; i < grid_size; i++)
    {
        singleRun(i, best_acc, best_parameters, grid_size, blocks, num_ticks, parameter_ticks, pain_intensity, num_choices, pickup_choices, num_frames, frames, nullptr);
    }
#endif

#ifdef CUDA_ENABLED

    CUDA_FREE_WITH_CHECK_NESTED_ARRAY(devptr_parameter_ticks, host_ptr_to_devptr_parameter_ticks, model::NUM_FREE_PARAMETER, ret);
    CUDA_FREE_WITH_CHECK(devptr_num_ticks, ret);
    CUDA_FREE_WITH_CHECK(devptr_pain_intensity, ret);
    CUDA_FREE_WITH_CHECK_NESTED_ARRAY(devptr_pickup_choices, host_ptr_to_devptr_pickup_choices, blocks, ret)
    CUDA_FREE_WITH_CHECK(devptr_num_choices, ret);
    CUDA_FREE_WITH_CHECK_NESTED_ARRAY(devptr_frames, host_ptr_to_devptr_frames, blocks, ret)
    CUDA_FREE_WITH_CHECK(devptr_num_frames, ret);
#endif
    std::cout<< best_acc << std::endl;
    fitted_result[model::NUM_FREE_PARAMETER] = best_acc;
    memcpy(fitted_result, best_parameters, model::NUM_FREE_PARAMETER * sizeof(float));
    for (int i = 0; i < model::NUM_FREE_PARAMETER; i++)
    {
        std::cout << fitted_result[i] << "    ";
    }
    std::cout << std::endl;
}

uint64_t generateGrid(uint32_t*& num_ticks, float**& parameter_ticks,
                  std::vector<int> slice_type, std::vector<std::vector<float>> slice)
{
    assert(slice_type.size() == slice.size());
    assert(slice_type.size() == model::NUM_FREE_PARAMETER);

    uint64_t total_grid_size = 1;
    parameter_ticks = new float*[model::NUM_FREE_PARAMETER];
    num_ticks = new uint32_t[model::NUM_FREE_PARAMETER];
    for (int i = 0; i < model::NUM_FREE_PARAMETER; i++)
    {
        if (slice_type[i] == PLAIN_TICKS)
        {
            total_grid_size *= slice[i].size();
            num_ticks[i] = slice[i].size();
            parameter_ticks[i] = new float[slice[i].size()];
            std::copy(slice[i].begin(), slice[i].end(), parameter_ticks[i]);
        }
        else if (slice_type[i] == SLICED_TICKS)
        {
            int slice_size = round((slice[i][1] - slice[i][0]) / slice[i][2]) + 1;
            total_grid_size *= slice_size;
            num_ticks[i] = slice_size;
            parameter_ticks[i] = new float[slice_size];
            for (int j = 0; j < slice_size; j++)
            {
                parameter_ticks[i][j] = slice[i][0] + j * slice[i][2];
            }
        }
        else
        {
            throw std::invalid_argument("No such slice type");
        }
    }

    std::cout << "total grid size: " << total_grid_size << std::endl;

    return total_grid_size;
}

void exportFittedResult(const std::unordered_map<std::string, float*>& res, std::string file_prefix)
{
    std::cout << "Writing results for " << res.size() << " fitted subjects data to " << file_prefix + std::string("_fitted.json") << std::endl;
    std::ofstream fs(file_prefix + std::string("_fitted.json"));
    int first_subject = 1;
    fs << "{";
    for (auto it = res.begin(); it != res.end(); ++it)
    {
        const auto& pair = *it;
        int first_element = 1;
        if (first_subject)
        {
            fs << "\"" << pair.first << "\": [";
            first_subject = 0;
        }
        else
        {
            fs << ",\"" << pair.first << "\": [";
        }
        for (auto i = 0; i < model::NUM_FREE_PARAMETER + 1; i++) // include accuracy in the last element
        {
            if (first_element)
            {
                fs << pair.second[i];
                first_element = 0;
            }
            else
            {
                fs << ", " << pair.second[i];
            }
        }
        fs << "]";
    }
    fs << "}" << std::endl;
    fs.close();
}

int main(int argc, char* argv[])
{
    std::string in_path(argv[1]);
    std::unordered_map<std::string, 
                       std::vector<
                            std::tuple<float, // pain intensity
                                        std::string, // pain condition name
                                        uint32_t, // length of pickups
                                        uint32_t*, // pickup choices
                                        uint32_t, // total frame count
                                        uint32_t*>>*> frame_blocks;

#if (!defined DEBUG) && (!defined CUDA_ENABLED)
    omp_set_num_threads(16);
#endif

    try 
    {
        for (const auto& entry : fs::directory_iterator(in_path)) 
        {
            if (entry.is_regular_file()) 
            {
                std::cout << "Reading block data: " << entry.path().string() << std::endl;
                std::string filename = entry.path().filename().string();
                auto parsed_block_name = csv_frame_reader::blockFileNameParser(filename);
                auto [subject_name, pain_intensity, pain_condition] = parsed_block_name;

                std::tuple<float,
                           std::string,
                           uint32_t,
                           uint32_t*,
                           uint32_t,
                           uint32_t*> insert_tuple;
                std::get<0>(insert_tuple) = pain_intensity;
                std::get<1>(insert_tuple) = pain_condition;
                auto [num_frames, frames] = csv_frame_reader::readFramesFromCSV(entry.path().string());
                std::get<4>(insert_tuple) = num_frames;
                std::get<5>(insert_tuple) = frames;

                auto [num_choices, picked_choices] = frame_defs::generatePickedChoicesList(num_frames, frames);
                std::get<2>(insert_tuple) = num_choices;
                std::get<3>(insert_tuple) = picked_choices;

                auto it = frame_blocks.find(subject_name);
                if (it == frame_blocks.end())
                {
                    frame_blocks[subject_name] = new std::vector<std::tuple<float,
                                                                            std::string,
                                                                            uint32_t,
                                                                            uint32_t*,
                                                                            uint32_t,
                                                                            uint32_t*>>();
                }
                frame_blocks[subject_name]->push_back(insert_tuple);
            }
        }
    }
    catch (const std::exception& ex) 
    {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    std::unordered_map<std::string, float*> results;
    for (auto it = frame_blocks.begin(); it != frame_blocks.end(); ++it)
    {
        const auto& pair = *it;

        auto subject = pair.first;
        auto blocks = pair.second;

        std::cout<< subject << std::endl;
        std::cout<< blocks->size() << std::endl;

    #ifdef DEBUG_SINGLE_TEST
    //0, -0.9, 58.1, 2, -31, 0.51
        std::vector<int> slice_type = { PLAIN_TICKS, PLAIN_TICKS, PLAIN_TICKS, PLAIN_TICKS, PLAIN_TICKS, PLAIN_TICKS };
        std::vector<std::vector<float>> slices = {
            {0},
            {-0.9},
            {58.1},
            {2},
            {-31},
            {0.51}
        };    
    #else
        std::vector<int> slice_type = { PLAIN_TICKS, SLICED_TICKS, SLICED_TICKS, PLAIN_TICKS, SLICED_TICKS, SLICED_TICKS };
        std::vector<std::vector<float>> slices = {
            {0.125, 0.25, 0.5, 1, 2, 4, 8, 16},
            {-1, 1, 0.1},
            {1, 100, 1},
            {1},
            {-100, 100, 1},
            {0, 1, 0.001}
        };
    #endif
        float** parameter_ticks = nullptr;
        uint32_t* num_ticks = nullptr;
        uint64_t grid_size = generateGrid(num_ticks, parameter_ticks, slice_type, slices);

        // Define separate vectors for each element
        std::vector<float> pain_intensity;
        std::vector<uint32_t> num_choices;
        std::vector<uint32_t*> pickup_choices;
        std::vector<uint32_t> num_frames;
        std::vector<uint32_t*> frames;

        // Extract elements from each tuple into separate vectors
        for (size_t i = 0; i < blocks->size(); i++){
            const auto& tuple = blocks->at(i);
            pain_intensity.push_back(std::get<0>(tuple));
            num_choices.push_back(std::get<2>(tuple));
            pickup_choices.push_back(std::get<3>(tuple));
            num_frames.push_back(std::get<4>(tuple));
            frames.push_back(std::get<5>(tuple));
        }

        float* fitted_result = new float[model::NUM_FREE_PARAMETER + 1]; // last one is inaccuracy

        gridSearch(grid_size, blocks->size(), fitted_result, num_ticks, parameter_ticks, pain_intensity.data(), num_choices.data(),
                   pickup_choices.data(), num_frames.data(), frames.data());

        results[subject] = fitted_result;
    }

#ifndef DEBUG_SINGLE_TEST
    exportFittedResult(results, in_path.substr(in_path.length() - 9));
#endif
}