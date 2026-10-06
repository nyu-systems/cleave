add_executable(test_fingerprint_ops
  kernel/test_fingerprint_ops.cc)
target_link_libraries(test_fingerprint_ops cleave_superoptimizer)
add_test(NAME test_fingerprint_ops COMMAND test_fingerprint_ops)

add_executable(test_split_k
  kernel/test_split_k.cc)
target_link_libraries(test_split_k cleave_superoptimizer)
add_test(NAME test_split_k COMMAND test_split_k)

add_executable(test_create_node
  kernel/test_create_node.cc)
target_link_libraries(test_create_node cleave_superoptimizer)
add_test(NAME test_create_node COMMAND test_create_node)

add_executable(test_reducibility_inference
  kernel/test_reducibility_inference.cc)
target_link_libraries(test_reducibility_inference cleave_superoptimizer)
add_test(NAME test_reducibility_inference COMMAND test_reducibility_inference)

function(add_cleave_superoptimizer_search_test CATEGORY TEST_NAME)
  set(TARGET_NAME test_search_${CATEGORY}_${TEST_NAME})
  add_executable(${TARGET_NAME}
    search/${CATEGORY}/test_${TEST_NAME}.cc)
  target_link_libraries(${TARGET_NAME} cleave_superoptimizer)
  target_include_directories(${TARGET_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/search)

  set(TEST_WORKING_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/search/${CATEGORY}/${TEST_NAME})
  file(MAKE_DIRECTORY ${TEST_WORKING_DIRECTORY})

  add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
  set_tests_properties(${TARGET_NAME}
    PROPERTIES WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})
endfunction()

add_cleave_superoptimizer_search_test(manual bmha_flash_decode)
add_cleave_superoptimizer_search_test(manual bmm_bcast)
add_cleave_superoptimizer_search_test(manual gated_mlp)
add_cleave_superoptimizer_search_test(manual gqa_flash_decode)
add_cleave_superoptimizer_search_test(manual lora)
add_cleave_superoptimizer_search_test(manual matmul_matmul_bhsd)
add_cleave_superoptimizer_search_test(manual mm_splitk)
add_cleave_superoptimizer_search_test(manual mm_splitk_2)
add_cleave_superoptimizer_search_test(manual sha_flash_decode)

add_cleave_superoptimizer_search_test(symbolic gated_mlp)
add_cleave_superoptimizer_search_test(symbolic gqa_flash_decode)
add_cleave_superoptimizer_search_test(symbolic matmul)
