#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/transpose.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
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
        explicit PrototypeDeviceDescription(int id);

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
        explicit PrototypeDevice(int id, std::unique_ptr<Transport> transport = nullptr);
        Transport* transport() const { return transport_.get(); }

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
        PjRtMemorySpace *memory_space_ = nullptr;
        std::unique_ptr<Transport> transport_;
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

        absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(MaybeOwningMlirModule module, CompileOptions options) override { return absl::UnimplementedError("CompileAndLoad() not implemented"); }

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

    // =========================================================================
    // Stubs. Every body is a placeholder so the file compiles; replace each.
    //   [needed]   required for the round-trip proof of concept
    //   [optional] can stay Unimplemented for the single-device PoC
    // =========================================================================

    class PrototypeMemorySpace final : public PjRtMemorySpace
    {
    public:
        PrototypeMemorySpace(int id, PjRtDevice *device);

        // [needed] All accessors. string_view returns must point at stored members.
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
        // [needed] Accessors.
        const Shape &on_device_shape() const override { return shape_; }

        PjRtMemorySpace *memory_space() const override { return memory_space_; }

        PjRtDevice *device() const override { return memory_space_->devices()[0]; }
        PjRtClient *client() const override { return memory_space_->client(); }
        bool IsOnCpu() const override { return false; }

        // [needed] Device -> host readback; how JAX reads results.
        Future<> ToLiteral(MutableLiteralBase *literal) override;
        // [needed] Same as ToLiteral, but the literal comes from `generator`.
        Future<> LazyToLiteral(absl::AnyInvocable<Future<MutableLiteralBase *>() &&> generator) override;

        // [needed]
        absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override { return ShapeUtil::ByteSizeOf(shape_); }

        // [needed] Becomes ready when the data exists (e.g. an Execute output finished).
        Future<> GetReadyFuture() override;

        // [needed]
        void Delete() override;

        bool IsDeleted() const override;

        // [optional] DLPack / ownership handoff.
        absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference() override { return Unimplemented("AcquireExternalReference not implemented."); }
        absl::StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override { return Unimplemented("ReleaseDeviceMemoryOwnership not implemented."); }

        // [optional]
        Future<> CopyRawToHost(void *dst, int64_t offset, int64_t transfer_size) override { return Future<>(Unimplemented("CopyRawToHost not implemented.")); }
        // [optional] Needed later for multi-device.
        absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(PjRtMemorySpace *dst_memory_space) override { return Unimplemented("CopyToMemorySpace not implemented."); }
        // [optional] Reinterpret the buffer with a new type/shape/layout, no copy.
        absl::StatusOr<std::unique_ptr<PjRtBuffer>> Bitcast(PrimitiveType element_type, absl::Span<const int64_t> dims, const Layout *device_layout) override { return Unimplemented("Bitcast not implemented."); }
        // [optional] Cross-host transfers; round-trip never uses these.
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
        PrototypeLoadedExecutable() {} // TODO: constructor arguments

        // Keeps the base class's convenience overloads visible (name hiding).
        using PjRtLoadedExecutable::Execute;
        using PjRtLoadedExecutable::ExecutePortable;
        using PjRtLoadedExecutable::ExecuteSharded;

        // [needed] PjRtExecutable pure virtuals. The exact set varies by XLA
        // commit; the compiler will flag any that are missing or renamed.
        absl::string_view name() const override { LOG(FATAL) << "TODO"; }

        int num_replicas() const override { LOG(FATAL) << "TODO"; }

        int num_partitions() const override { LOG(FATAL) << "TODO"; }

        int64_t SizeOfGeneratedCodeInBytes() const override { LOG(FATAL) << "TODO"; }

        absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules() const override
        {
            return Unimplemented("TODO: GetHloModules");
        }

        absl::StatusOr<std::vector<std::vector<absl::string_view>>>
        GetOutputMemoryKinds() const override
        {
            return Unimplemented("TODO: GetOutputMemoryKinds");
        }

        virtual absl::StatusOr<std::vector<std::vector<absl::string_view>>>
        GetParameterMemoryKinds() const override
        {
            return Unimplemented("TODO: GetParameterMemoryKinds");
        }

        // [needed] PjRtLoadedExecutable accessors.
        PjRtClient *client() const override { LOG(FATAL) << "TODO"; }

        const DeviceAssignment &device_assignment() const override { LOG(FATAL) << "TODO"; }

        absl::Span<const LogicalDeviceIds> addressable_device_logical_ids() const override
        {
            LOG(FATAL) << "TODO";
        }

        absl::Span<PjRtDevice *const> addressable_devices() const override { LOG(FATAL) << "TODO"; }

        // [needed] The core: input buffers -> transport -> output buffers + future.
        absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
        Execute(absl::Span<const std::vector<PjRtBuffer *>> argument_handles,
                const ExecuteOptions &options,
                std::optional<std::vector<Future<>>> &returned_futures) const override
        {
            return Unimplemented("TODO: Execute");
        }

        // [needed] With 1 replica / 1 partition, these reduce to the same thing as Execute.
        absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
        ExecuteSharded(absl::Span<PjRtBuffer *const> argument_handles,
                       PjRtDevice *device, const ExecuteOptions &options,
                       std::optional<Future<>> &returned_future,
                       bool fill_future) const override
        {
            return Unimplemented("TODO: ExecuteSharded");
        }

        absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
        ExecutePortable(absl::Span<PjRtBuffer *const> argument_handles,
                        PjRtDevice *device, const ExecuteOptions &options,
                        std::optional<Future<>> &returned_future,
                        bool fill_future) const override
        {
            return Unimplemented("TODO: ExecutePortable");
        }

        // [needed]
        void Delete() override { LOG(FATAL) << "TODO"; }

        bool IsDeleted() const override { LOG(FATAL) << "TODO"; }

        // [optional] Override to return true once Execute fills returned_futures.
        // bool IsReturnedFutureSupported() const override;

    private:
        // TODO: members
    }; // class PrototypeLoadedExecutable

} // namespace xla