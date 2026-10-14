#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <array>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

#include "main/inc/transport.h"

namespace xla
{
    struct BufferState
    {
        std::vector<uint8_t> bytes;
        Future<> ready;
    };

    class PrototypeMemorySpace;

    class PrototypeDeviceDescription final : public PjRtDeviceDescription
    {
    public:
        explicit PrototypeDeviceDescription(int id, std::string kind);

        int id() const override { return id_; }
        int process_index() const override { return 0; }

        absl::string_view device_kind() const override { return device_kind_; }
        absl::string_view DebugString() const override { return debug_string_; }
        absl::string_view ToString() const override { return to_string_; }
        const absl::flat_hash_map<std::string, PjRtDeviceAttribute> &Attributes() const override { return attributes_; }

    private:
        int id_;
        std::string device_kind_;
        std::string debug_string_;
        std::string to_string_;
        absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_ = {};
    }; // class PrototypeDeviceDescription

    class PrototypeDevice final : public PjRtDevice
    {
    public:
        explicit PrototypeDevice(int id, std::string kind = "prototype",
                                 std::vector<std::string> modules = {},
                                 std::shared_ptr<Transport> transport = nullptr);
        Transport *transport() const { return transport_.get(); }
        const std::vector<std::string> &modules() const { return modules_; }
        // per-device call counter
        uint64_t NextSeq() { return next_seq_.fetch_add(1, std::memory_order_relaxed); }

        const PrototypeDeviceDescription &description() const override
        {
            return description_;
        }

        void SetClient(PjRtClient *client)
        {
            CHECK(client_ == nullptr);
            client_ = client;
        }

        PjRtClient *client() const override { return client_; }

        bool IsAddressable() const override { return process_index() == client()->process_index(); }
        LocalChipId local_hardware_id() const override { return LocalChipId(description_.id()); }

        absl::Status TransferToInfeed(const LiteralSlice &literal) override { return absl::UnimplementedError("TransferToInfeed not supported"); }
        absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override { return absl::UnimplementedError("TransferFromOutfeed not supported"); }

        absl::Span<PjRtMemorySpace *const> memory_spaces() const override { return absl::MakeConstSpan(&memory_space_, 1); }
        absl::StatusOr<PjRtMemorySpace *> default_memory_space() const override { return memory_space_; }
        void AttachMemorySpace(PjRtMemorySpace *memory_space);

        std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(absl::string_view description) const override { return nullptr; }

    private:
        PjRtClient *client_ = nullptr;
        PrototypeDeviceDescription description_;
        std::vector<std::string> modules_;
        PjRtMemorySpace *memory_space_ = nullptr;
        std::shared_ptr<Transport> transport_;
        std::atomic<uint64_t> next_seq_{0};
    }; // class PrototypeDevice

    class PrototypeClient final : public PjRtClient
    {
    public:
        PrototypeClient(int process_index, std::vector<std::unique_ptr<PrototypeDevice>> devices, size_t num_threads);
        ~PrototypeClient() override;

        int process_index() const override { return process_index_; }
        int device_count() const override { return devices_.size(); }

        absl::Span<PjRtDevice *const> devices() const override { return devices_; }
        int addressable_device_count() const override { return addressable_devices_.size(); }
        absl::Span<PjRtDevice *const> addressable_devices() const override { return addressable_devices_; }
        absl::StatusOr<PjRtDevice *> LookupDevice(GlobalDeviceId global_device_id) const override;

        absl::Span<PjRtMemorySpace *const> memory_spaces() const override { return memory_spaces_; }

        PjRtPlatformId platform_id() const override { return tsl::Fingerprint64(platform_name()); }
        absl::string_view platform_name() const override { return "prototype"; }
        absl::string_view platform_version() const override { return "<unknown>"; }

        absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(MaybeOwningMlirModule module, CompileOptions options) override;

        absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(const void *data, PrimitiveType type, absl::Span<int64_t const> dims,
                                                                         std::optional<absl::Span<int64_t const>> byte_strides, HostBufferSemantics host_buffer_semantics,
                                                                         absl::AnyInvocable<void() &&> on_done_with_host_buffer, PjRtMemorySpace *memory_space,
                                                                         const Layout *device_layout) override;

    private:
        int process_index_;
        std::vector<std::unique_ptr<PrototypeDevice>> owned_devices_;
        std::vector<PjRtDevice *> devices_;
        absl::flat_hash_map<int, PrototypeDevice *> id_to_device_;

        std::vector<std::unique_ptr<PrototypeMemorySpace>> owned_memory_spaces_;
        std::vector<PjRtMemorySpace *> memory_spaces_;

        std::vector<PjRtDevice *> addressable_devices_;
    }; // class PrototypeClient

    class PrototypeMemorySpace final : public PjRtMemorySpace
    {
    public:
        PrototypeMemorySpace(int id, PjRtDevice *device);

        PjRtClient *client() const override { return device_->client(); }

        absl::Span<PjRtDevice *const> devices() const override { return absl::MakeConstSpan(&device_, 1); }

        int id() const override { return id_; }

        absl::string_view kind() const override { return "device"; }

        int kind_id() const override { return 0; }

        absl::string_view DebugString() const override { return debug_string_; }

        absl::string_view ToString() const override { return to_string_; }

        PJRT_Memory *ToCApiPtr() override { return capi_delegator_.ToCApiPtr(); }

    private:
        int id_;
        PjRtDevice *device_;
        std::string debug_string_;
        std::string to_string_;

        PjRtMemorySpaceCApiDelegator capi_delegator_{this};
    }; // class PrototypeMemorySpace

    class PrototypeBuffer final : public PjRtBuffer
    {
    public:
        PrototypeBuffer(Shape shape, PrototypeMemorySpace *memory_space, std::shared_ptr<BufferState> state) : shape_(std::move(shape)),
                                                                                                               memory_space_(memory_space),
                                                                                                               buffer_state(std::move(state)) {} // TODO: also take the data

        std::shared_ptr<BufferState> state() const;
        const Shape &on_device_shape() const override { return shape_; }
        PjRtMemorySpace *memory_space() const override { return memory_space_; }

        PjRtDevice *device() const override { return memory_space_->devices()[0]; }
        PjRtClient *client() const override { return memory_space_->client(); }
        bool IsOnCpu() const override { return false; }

        Future<> ToLiteral(MutableLiteralBase *literal) override;
        Future<> LazyToLiteral(absl::AnyInvocable<Future<MutableLiteralBase *>() &&> generator) override;

        absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override { return ShapeUtil::ByteSizeOf(shape_); }

        Future<> GetReadyFuture() override;

        void Delete() override;

        bool IsDeleted() const override;

        absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference() override { return Unimplemented("AcquireExternalReference not implemented."); }
        absl::StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override { return Unimplemented("ReleaseDeviceMemoryOwnership not implemented."); }

        Future<> CopyRawToHost(void *dst, int64_t offset, int64_t transfer_size) override { return Future<>(Unimplemented("CopyRawToHost not implemented.")); }
        absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(PjRtMemorySpace *dst_memory_space) override { return Unimplemented("CopyToMemorySpace not implemented."); }
        absl::StatusOr<std::unique_ptr<PjRtBuffer>> Bitcast(PrimitiveType element_type, absl::Span<const int64_t> dims, const Layout *device_layout) override { return Unimplemented("Bitcast not implemented."); }
        void CopyToRemoteDevice(Future<std::string> serialized_descriptor, RemoteSendCallback on_done) override { on_done(Unimplemented("CopyToRemoteDevice not implemented."), /*sends_were_enqueued=*/false); }

    private:
        Shape shape_;
        std::vector<uint8_t> bytes_;
        PrototypeMemorySpace *memory_space_;

        mutable absl::Mutex mu_;
        std::shared_ptr<BufferState> buffer_state ABSL_GUARDED_BY(mu_);
    }; // class PrototypeBuffer

    class PrototypeLoadedExecutable final : public PjRtLoadedExecutable
    {
    public:
        PrototypeLoadedExecutable(std::shared_ptr<HloModule> hlo_module, PrototypeDevice *device, CompileOptions compile_options);

        // Keeps the base class's convenience overloads visible
        using PjRtLoadedExecutable::Execute;
        using PjRtLoadedExecutable::ExecutePortable;
        using PjRtLoadedExecutable::ExecuteSharded;

        absl::string_view name() const override { return hlo_module_->name(); }

        int num_replicas() const override { return 1; }

        int num_partitions() const override { return 1; }

        int64_t SizeOfGeneratedCodeInBytes() const override { return 0; }

        absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules() const override 
        { 
            return std::vector<std::shared_ptr<HloModule>>{hlo_module_};
        }

        absl::StatusOr<std::string> FingerprintExecutable() const override;
        absl::StatusOr<CompileOptions> GetCompileOptions() const override { return compile_options_; }
        absl::StatusOr<std::string> SerializeExecutable() const override
        {
            return absl::UnimplementedError("SerializeExecutable not supported");
        }
        absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override
        {
            return absl::UnimplementedError("GetCompiledMemoryStats not supported");
        }

        absl::StatusOr<std::vector<std::vector<absl::string_view>>>
        GetOutputMemoryKinds() const override
        {
            return std::vector<std::vector<absl::string_view>>{
                std::vector<absl::string_view>(output_shapes_.size(), "device")};
        }

        virtual absl::StatusOr<std::vector<std::vector<absl::string_view>>>
        GetParameterMemoryKinds() const override
        {
            return std::vector<std::vector<absl::string_view>>{
                std::vector<absl::string_view>(num_parameters_, "device")};
        }

        PjRtClient *client() const override { return device_->client(); }

        const DeviceAssignment &device_assignment() const override { return device_assignment_; }
        absl::Span<const LogicalDeviceIds> addressable_device_logical_ids() const override { return logical_ids_; }
        absl::Span<PjRtDevice *const> addressable_devices() const override { return addressable_devices_; }

        absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
        Execute(absl::Span<const std::vector<PjRtBuffer *>> argument_handles,
                const ExecuteOptions &options,
                std::optional<std::vector<Future<>>> &returned_futures) const override;

        absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
        ExecuteSharded(absl::Span<PjRtBuffer *const> argument_handles,
                       PjRtDevice *device, const ExecuteOptions &options,
                       std::optional<Future<>> &returned_future,
                       bool fill_future) const override;

        absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
        ExecutePortable(absl::Span<PjRtBuffer *const> argument_handles,
                        PjRtDevice *device, const ExecuteOptions &options,
                        std::optional<Future<>> &returned_future,
                        bool fill_future) const override;

        void Delete() override { deleted_.store(true); }

        bool IsDeleted() const override { return deleted_.load(); }

    private:
        absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteHelper(absl::Span<PjRtBuffer *const> args, Future<> &done_future) const;
        std::shared_ptr<HloModule> hlo_module_;
        PrototypeDevice *device_;
        std::vector<Shape> output_shapes_;
        int num_parameters_;
        DeviceAssignment device_assignment_;
        std::array<LogicalDeviceIds, 1> logical_ids_{{{0, 0}}};
        std::array<PjRtDevice *, 1> addressable_devices_;
        std::atomic<bool> deleted_{false};
        CompileOptions compile_options_;
    }; // class PrototypeLoadedExecutable
} // namespace xla