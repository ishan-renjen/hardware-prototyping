cc_library(
    name = "fpga_client",
    srcs = ["src/pjrt_fpga_client.cc"],
    hdrs = ["pjrt_fpga_client.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@xla//xla:executable_run_options",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:async_work_runner",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_future",
        "@xla//xla/pjrt:semaphore",
        "@xla//xla/pjrt:transpose",
        "@xla//xla/service:buffer_assignment",
        "@xla//xla/service:computation_placer",
        "@xla//xla/service:executable",
        "@xla//xla/service:hlo_cost_analysis",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:logging",
    ],
)

cc_binary(
    name = "pjrt_c_api_prototype_plugin.so",
    srcs = ["src/pjrt_wrapper.cc"],
    additional_linker_inputs = [":pjrt_wrapper_version_script.lds"],
    linkopts = [
        "-Wl,--version-script,$(location :pjrt_wrapper_version_script.lds)",
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    deps = [
        ":fpga_client",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@xla//xla/pjrt/c:pjrt_c_api_status_utils",
        "@xla//xla/pjrt/c:pjrt_c_api_wrapper_impl",
    ],
)

cc_test(
    name = "fpga_client_test",
    srcs = ["test/pjrt_fpga_client_test.cc"],
    deps = [
        ":fpga_client",
        "@googletest//:gtest_main",
        "@xla//xla:shape_util",
    ],
)
