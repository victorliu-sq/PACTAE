file(GLOB CUDA_FILES "*.cu")

foreach (CU_FILE ${CUDA_FILES})
    # Extract the filename without the directory and extension
    get_filename_component(EXE_NAME ${CU_FILE} NAME_WE)

    # Add executable for each CUDA file
    add_executable(${EXE_NAME} ${CU_FILE})

    # Link libraries to the executable
    #    target_link_libraries(${EXE_NAME}
    #            bamboosmp_lib
    #            benchmark
    #    )

    target_link_libraries(${EXE_NAME}
            bamboosmp_lib
    )
endforeach ()
