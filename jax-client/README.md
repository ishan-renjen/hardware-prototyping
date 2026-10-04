to build:
1. (cd jax-client && bazel build //main:pjrt_c_api_prototype_plugin.so)
2.   ln -sf "$(cd jax-client && bazel info bazel-bin)/main/pjrt_c_api_prototype_plugin.so" jax_plugins/fpga/