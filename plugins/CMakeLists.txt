cmake_minimum_required(VERSION 3.10)
project(CryptaVisionPlugins)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable the strict warning that is breaking the build
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-parameter -Wno-unused-parameter")

# Find GStreamer via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)

# Explicitly add Homebrew library paths for Apple Silicon Macs
link_directories(${GST_LIBRARY_DIRS} /opt/homebrew/lib /usr/local/lib)
include_directories(${GST_INCLUDE_DIRS})

# Build crypta_frame_filter
add_library(crypta_frame_filter SHARED crypta_frame_filter.cpp)
target_link_libraries(crypta_frame_filter ${GST_LIBRARIES})
# Remove 'lib' prefix so GStreamer recognizes it as a plugin module
set_target_properties(crypta_frame_filter PROPERTIES PREFIX "")

# Build crypta_network_sink
add_library(crypta_network_sink SHARED crypta_network_sink.cpp)
target_link_libraries(crypta_network_sink ${GST_LIBRARIES})
set_target_properties(crypta_network_sink PROPERTIES PREFIX "")
