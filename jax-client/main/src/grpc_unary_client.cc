#include "main/inc/grpc_transport.h"

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace
{
    absl::Status ToAbslStatus(const grpc::Status &status)
    {
        return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                            status.error_message());
    }
} //namespace

GrpcTransport::GrpcTransport(const std::string &target)
{
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(256 << 20);  //gRPC default is 4 MB
    args.SetMaxSendMessageSize(256 << 20);
    stub_ = simulator::Simulator::NewStub(
        grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args));
} //GrpcTransport::GrpcTransport()

void GrpcTransport::Execute(int device, uint64_t seq, absl::string_view module,
                            absl::Span<const TensorView> inputs, DoneCallback done)
{
    struct Call
    {
        grpc::ClientContext context;
        simulator::ExecuteRequest request;
        simulator::ExecuteResponse response;
        int device;
        uint64_t seq;
        DoneCallback done;
    };
    auto call = std::make_shared<Call>();
    call->device = device;
    call->seq = seq;
    call->done = std::move(done);

    call->request.set_module(std::string(module));
    call->request.set_device(device);
    call->request.set_seq(seq);
    for (const TensorView &view : inputs)
    {
        simulator::Tensor *t = call->request.add_inputs();
        t->set_dtype(view.dtype);
        for (int64_t d : view.dims)
        {
            t->add_dims(d);
        }
        t->mutable_data()->assign(reinterpret_cast<const char *>(view.data.data()),
                                  view.data.size());
    }

    stub_->async()->Execute(
        &call->context, &call->request, &call->response,
        [call](grpc::Status status)
        {
            if (!status.ok())
            {
                std::move(call->done)(ToAbslStatus(status));
                return;
            }
            if (call->response.device() != call->device || call->response.seq() != call->seq)
            {
                std::move(call->done)(absl::InternalError(absl::StrCat(
                    "gateway reply mismatch: sent device ", call->device, " seq ", call->seq,
                    ", got device ", call->response.device(), " seq ", call->response.seq())));
                return;
            }

            std::vector<Tensor> outputs;
            outputs.reserve(call->response.outputs_size());
            for (const simulator::Tensor &t : call->response.outputs())
            {
                Tensor out;
                out.dtype = t.dtype();
                out.dims.assign(t.dims().begin(), t.dims().end());
                const std::string &bytes = t.data();
                out.data.assign(bytes.begin(), bytes.end());
                outputs.push_back(std::move(out));
            }
            std::move(call->done)(std::move(outputs));
        });
}//GrpcTransport::Execute()

absl::StatusOr<std::vector<RemoteDevice>> GrpcTransport::ListDevices()
{
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    simulator::ListDevicesRequest request;
    simulator::ListDevicesResponse response;

    grpc::Status status = stub_->ListDevices(&context, request, &response);
    if (!status.ok())
    {
        return ToAbslStatus(status);
    }

    std::vector<RemoteDevice> devices;
    devices.reserve(response.devices_size());
    for (const simulator::DeviceInfo &info : response.devices())
    {
        devices.push_back(RemoteDevice{
            info.id(), info.kind(),
            std::vector<std::string>(info.modules().begin(), info.modules().end())});
    }
    return devices;
}//GrpcTransport::ListDevices()