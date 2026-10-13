#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

struct TensorView
{
    std::string dtype;
    absl::Span<const int64_t> dims;
    absl::Span<const uint8_t> data;
};

struct Tensor
{
    std::string dtype;
    std::vector<int64_t> dims;
    std::vector<uint8_t> data;
};

struct RemoteDevice
{
    int id;
    std::string kind;
    std::vector<std::string> modules;
};

class Transport
{
public:
    using DoneCallback = absl::AnyInvocable<void(absl::StatusOr<std::vector<Tensor>>) &&>;

    virtual ~Transport() = default;

    virtual void Execute(int device, uint64_t seq, absl::string_view module, absl::Span<const TensorView> inputs, DoneCallback done) = 0;
};