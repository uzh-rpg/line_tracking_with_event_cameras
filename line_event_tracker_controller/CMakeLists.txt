cmake_minimum_required(VERSION 3.10)
project(line_event_tracker_controller)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(PACKAGE_DEPENDENCIES
    roscpp
    std_msgs
    geometry_msgs
    line_event_tracker_msgs
)


find_package(catkin REQUIRED 
    COMPONENTS
    ${PACKAGE_DEPENDENCIES}
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS ${PACKAGE_DEPENDS}
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
    src/controller.cpp
)

add_dependencies(${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node 
    src/node.cpp
)


add_dependencies(${PROJECT_NAME}_node
    ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
)





