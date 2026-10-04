#pragma once

#include <vector>

#include "absl/types/span.h"
#include "absl/strings/string_view.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

// transport.h
struct TensorView { 
    std::string dtype;
    absl::Span<const int64_t> dims;
    absl::Span<const uint8_t> data;
};

struct Tensor {
    std::string dtype;
    std::vector<int64_t> dims;
    std::vector<uint8_t> data;
};

class Transport {
public:
    using DoneCallback = absl::AnyInvocable<void(absl::StatusOr<std::vector<Tensor>>) &&>;
    virtual ~Transport() = default;
    // Must copy `inputs` before returning; must call `done` exactly once, on any thread.
    virtual void Execute(absl::string_view module, absl::Span<const TensorView> inputs,
                         DoneCallback done) = 0;
};