#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "main/inc/grpc_transport.h"

GrpcTransport::GrpcTransport(const std::string &target)
{
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(256 << 20); // gRPC's default is 4 MB
    args.SetMaxSendMessageSize(256 << 20);
    stub_ = simulator::Simulator::NewStub(
        grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args));
}

void GrpcTransport::Execute(absl::string_view module, absl::Span<const TensorView> inputs,
                            DoneCallback done)
{
    // Per-call state must outlive this function: the RPC completes on a gRPC thread.
    struct Call
    {
        grpc::ClientContext context;
        simulator::ExecuteRequest request;
        simulator::ExecuteResponse response;
        DoneCallback done;
    };
    auto call = std::make_shared<Call>();
    call->done = std::move(done);
    // TODO: fill call->request with `module` and `inputs`.

    stub_->async()->Execute(&call->context, &call->request, &call->response,
                            [call](grpc::Status status)
                            {
                                if (!status.ok())
                                {
                                    std::move(call->done)(absl::Status(
                                        static_cast<absl::StatusCode>(status.error_code()),
                                        status.error_message()));
                                    return;
                                }
                                // TODO: convert call->response.outputs() to std::vector<Tensor>.
                                std::move(call->done)(std::vector<Tensor>{});
                            });
}