cmake_minimum_required(VERSION 3.13)

project(nano_inf LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include(FindProtobuf)
# find_package(Protobuf REQUIRED)
# include_directories(${PROTOBUF_INCLUDE_DIR})

find_package(protobuf CONFIG REQUIRED)
add_subdirectory(src)
add_subdirectory(headers)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/headers)
# target_link_libraries(${PROJECT_NAME} PRIVATE protobuf)
# target_link_libraries(${PROJECT_NAME} PRIVATE ${PROTOBUF_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

include(GNUInstallDirs)
install(TARGETS nano_inf
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)