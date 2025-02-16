/*
 * Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
 * Licensed under the MIT License.
 */

#ifndef CSV_FRAME_READER_H
#define CSV_FRAME_READER_H

#include "frame_defs.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <tuple>

namespace csv_frame_reader
{
    std::tuple<uint32_t, uint32_t*> readFramesFromCSV(std::string csv_file_path)
    {
        std::ifstream file_in = std::ifstream(csv_file_path);
        std::string line_in;
        
        int num_frames = std::count(std::istreambuf_iterator<char>(file_in), 
                                    std::istreambuf_iterator<char>(), '\n');

        // Reset ifstream progress, go to beginning
        file_in.clear();
        file_in.seekg(0);
        // ignore first column name line
        num_frames -= 1;
        std::getline(file_in, line_in);
        
        uint32_t* frame_block = new uint32_t[frame_defs::FRAME_SIZE * num_frames];

        int cell_counter = 0;
        while (std::getline(file_in, line_in))
        {
            std::stringstream line_stream(line_in);
            std::string cell;
            int column_counter = 0;

            while (std::getline(line_stream, cell, ','))
            {
                frame_block[cell_counter] = frame_defs::OBJ_IDENTIFIER_CONVERTER(cell, column_counter);
                // std::cout << frame_defs::OBJ_IDENTIFIER_CONVERTER(cell) << std::endl;
                cell_counter++;
                column_counter++;
            }
            //std:: cout << cell_counter << std::endl;
            assert (cell_counter % frame_defs::FRAME_SIZE == 0);
        }

        return std::tuple<uint32_t, uint32_t*>(cell_counter / frame_defs::FRAME_SIZE, frame_block);
    }

    std::tuple<std::string, float, std::string> blockFileNameParser(std::string filename)
    {
        filename = filename.substr(0, filename.length() - 4);
        std::istringstream ss(filename);
        std::tuple<std::string, float, std::string> ret_tuple;

        std::getline(ss, std::get<0>(ret_tuple), '_');
        ss >> std::get<1>(ret_tuple);
        ss.ignore(); // Ignore the delimiter '_'
        std::getline(ss, std::get<2>(ret_tuple), '_');

        return ret_tuple;
    }
}

#endif // CSV_FRAME_READER_H