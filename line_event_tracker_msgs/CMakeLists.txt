cmake_minimum_required(VERSION 3.10)
project(line_event_tracker_msgs)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    message_generation
    )

add_message_files(
    FILES
    Line.msg
    Lines.msg
)

generate_messages(
    DEPENDENCIES
    geometry_msgs
    std_msgs
)

catkin_package(
    CATKIN_DEPENDS message_runtime
)

include_directories(include
    ${catkin_INCLUDE_DIRS}
)
