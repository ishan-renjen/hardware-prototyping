#pragma once

#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "main/inc/transport.h"
#include "main/proto/unary_sim.grpc.pb.h"


class GrpcTransport final : public Transport
{
public:
    explicit GrpcTransport(const std::string &target);

    void Execute(int device, uint64_t seq, absl::string_view module, absl::Span<const TensorView> inputs, DoneCallback done) override;

    absl::StatusOr<std::vector<RemoteDevice>> ListDevices();

private:
    std::unique_ptr<simulator::Simulator::Stub> stub_;
};