#include "main/pjrt_fpga_client.h"
#include <vector>
#include "absl/strings/str_cat.h"
#include <memory>
#include <utility>

namespace xla {
//======================================PrototypeDeviceDescription=======================================
PrototypeDeviceDescription::PrototypeDeviceDescription(int id) : 
                                                    id_(id),
                                                    device_kind_("prototype"),
                                                    debug_string_(absl::StrCat("PrototypeDevice(id=", id, ")")),
                                                    to_string_(absl::StrCat("PrototypeDevice(", id, ")"))
{
}//PrototypeDeviceDescription::PrototypeDeviceDescription()
//=======================================================================================================
//============================================PrototypeDevice============================================
PrototypeDevice::PrototypeDevice(int id) : description_(id)
{
}//PrototypeDevice::PrototypeDevice()

void PrototypeDevice::AttachMemorySpace(PjRtMemorySpace* memory_space)
{
    CHECK(memory_space_ == nullptr);
    memory_space_ = memory_space;
}//void PrototypeDevice::AttachMemorySpace()
//=======================================================================================================
//============================================PrototypeClient============================================
PrototypeClient::PrototypeClient(int process_index, std::vector<std::unique_ptr<PrototypeDevice>> devices, size_t num_threads) : 
                                process_index_(process_index), owned_devices_(std::move(devices))
{
    for(const std::unique_ptr<PrototypeDevice>& device: owned_devices_)
    {
        device->SetClient(this);
        devices_.push_back(device.get());
        addressable_devices_.push_back(device.get());
        id_to_device_[device->global_device_id().value()] = device.get();

        auto memory_space = std::make_unique<PrototypeMemorySpace>(static_cast<int>(owned_memory_spaces_.size()), device.get());
        device->AttachMemorySpace(memory_space.get());
        memory_spaces_.push_back(memory_space.get());
        owned_memory_spaces_.push_back(std::move(memory_space));
    }
}//PrototypeClient::PrototypeClient()

PrototypeClient::~PrototypeClient() = default;

absl::StatusOr<PjRtDevice *> PrototypeClient::LookupDevice(GlobalDeviceId global_device_id) const {
    auto device = id_to_device_.find(global_device_id.value());
    if(device != id_to_device_.end()){
        return device->second;
    }
    else{
            return absl::NotFoundError(absl::StrCat("No device with id ", global_device_id.value()));
    }
}//absl::StatusOr<PjRtDevice *> LookupDevice()
//=======================================================================================================
//=========================================PrototypeMemorySpace==========================================
PrototypeMemorySpace::PrototypeMemorySpace(int id, PjRtDevice *device) : 
                                        id_(id),
                                        device_(device),
                                        debug_string_(absl::StrCat("PrototypeMemorySpace(id=", id,
                                                                    ", device=", device->global_device_id().value(), ")")),
                                        to_string_(absl::StrCat("PrototypeMemorySpace(", id, ")"))
{
}//PrototypeMemorySpace::PrototypeMemorySpace()
//=======================================================================================================
} //namespace xla