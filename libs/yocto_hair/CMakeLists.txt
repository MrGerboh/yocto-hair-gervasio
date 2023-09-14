add_library(yocto_hair yocto_hair.h yocto_hair.cpp)

set_target_properties(yocto_hair PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories(yocto_hair PUBLIC ${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(yocto_hair yocto)