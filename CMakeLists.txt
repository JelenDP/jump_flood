cmake_minimum_required(
  VERSION 3.7
)

project(gpu_voronoi
  LANGUAGES CXX
)

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

foreach(Ver RANGE 0 2)

  set(Sources gpu_voronoi_v${Ver}.cpp)

  add_executable(${PROJECT_NAME}_v${Ver}
    ${Sources}
  )

  target_compile_features(${PROJECT_NAME}_v${Ver}
    PRIVATE
      cxx_std_17
  )

  set_target_properties(${PROJECT_NAME}_v${Ver}
    PROPERTIES
      CXX_EXTENSIONS OFF
  )

  target_link_libraries(${PROJECT_NAME}_v${Ver}
    PRIVATE
      OpenCL::OpenCL
      Threads::Threads
  )

  target_compile_definitions(${PROJECT_NAME}_v${Ver}
    PRIVATE
      CL_HPP_MINIMUM_OPENCL_VERSION=120
      CL_HPP_TARGET_OPENCL_VERSION=120
      CL_HPP_ENABLE_EXCEPTIONS
  )

  source_group("Sources" FILES ${Sources})

endforeach()