// PJRT C-API entry point for the prototype backend
#include <memory>
#include <utility>
#include <vector>
#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_executable.h"

#include "main/inc/pjrt_fpga_client.h"
#include "main/inc/grpc_transport.h"

namespace pjrt
{
    namespace fpga_plugin
    {

        const PJRT_Api *GetPrototypePjrtApi();

        PJRT_Error *PJRT_Client_Create(PJRT_Client_Create_Args *args)
        {
            PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
                "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE, args->struct_size));

            const char *env = std::getenv("PROTOTYPE_SIM_TARGET");
            const std::string target = env != nullptr ? env : "localhost:50051";

            auto transport = std::make_shared<::GrpcTransport>(target);

            absl::StatusOr<std::vector<RemoteDevice>> remote = transport->ListDevices();
            if(!remote.ok())
            {
                return StatusToPjRtError(absl::UnavailableError(absl::StrCat("Simulation gateway not reachable @ ", target, ": ", remote.status().message())));
            }
            if(remote->empty()){
                return StatusToPjRtError(absl::FailedPreconditionError(absl::StrCat("Simulation gateway @ ", target, "reports no devices")));
            }

            std::vector<std::unique_ptr<xla::PrototypeDevice>> devices;
            for (RemoteDevice &rd : *remote)
            {
                devices.push_back(std::make_unique<xla::PrototypeDevice>(rd.id, std::move(rd.kind), std::move(rd.modules), transport));
            }
            auto client = std::make_unique<xla::PrototypeClient>(
                /*process_index=*/0, std::move(devices), /*num_threads=*/1);

            args->client = CreateWrapperClient(GetPrototypePjrtApi(), std::move(client));
            return nullptr;
        }

        // Generic; kept verbatim from the CPU plugin.
        PJRT_Error *PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args *args)
        {
            PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
                "PJRT_ExecuteContext_Create_Args",
                PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, args->struct_size));
            auto execute_context = std::make_unique<xla::ExecuteContext>();
            args->context = CreateWrapperExecuteContext(std::move(execute_context));
            return nullptr;
        }

        PJRT_Error *PJRT_PrototypeTopology_Create(PJRT_TopologyDescription_Create_Args *args)
        {
            return StatusToPjRtError(
                absl::UnimplementedError("Topology not supported for the prototype backend."));
        }

        const PJRT_Api *GetPrototypePjrtApi()
        {
            static const PJRT_Api pjrt_api = CreatePjrtApi(
                PJRT_Client_Create,
                PJRT_ExecuteContext_Create,
                PJRT_PrototypeTopology_Create,
                PJRT_Plugin_Initialize_NoOp,
                /*extension_start=*/nullptr,
                PJRT_Plugin_Attributes_Xla);
            return &pjrt_api;
        }

    } // namespace fpga_plugin
} // namespace pjrt

extern "C" const PJRT_Api *GetPjrtApi()
{
    return pjrt::fpga_plugin::GetPrototypePjrtApi();
}