cmake_minimum_required(VERSION 3.0.0)
project(LearnedIndex VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)

include(CTest)
enable_testing()

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRCS)


add_executable(LearnedIndex 
    ${SRCS}
    )




target_include_directories(LearnedIndex
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)


#set execute path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# #set library path
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

target_compile_options(LearnedIndex
    PRIVATE
    -ggdb
    -Wall
    -O0
)