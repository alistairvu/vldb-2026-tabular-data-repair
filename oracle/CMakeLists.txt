cmake_minimum_required(VERSION 3.20)
project(sherma_dataset_oracle LANGUAGES CXX)

add_executable(dataset_oracle main.cpp)
target_compile_features(dataset_oracle PRIVATE cxx_std_20)
target_include_directories(dataset_oracle PRIVATE vendor "${CMAKE_CURRENT_BINARY_DIR}")
