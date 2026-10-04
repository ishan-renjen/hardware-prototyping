#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <cstring>

#include "absl/strings/str_cat.h"
#include "xla/shape.h"
#include "xla/literal.h"
#include "xla/shape_util.h"

#include "main/inc/pjrt_fpga_client.h"

namespace xla
{
    namespace
    {
        // Shared by ToLiteral and LazyToLiteral. Equal() also compares layouts, and the memcpy depends on that.
        absl::Status CopyToLiteral(const Shape &shape, const std::vector<uint8_t> &bytes, MutableLiteralBase *literal)
        {
            if (!ShapeUtil::Equal(shape, literal->shape()))
            {
                return absl::InvalidArgumentError(absl::StrCat(
                    "ToLiteral shape mismatch: buffer ", ShapeUtil::HumanStringWithLayout(shape),
                    ", literal ", ShapeUtil::HumanStringWithLayout(literal->shape())));
            }
            if (!bytes.empty())
            {
                std::memcpy(literal->untyped_data(), bytes.data(), bytes.size());
            }
            return absl::OkStatus();
        }
    } // namespace
    //======================================PrototypeDeviceDescription=======================================
    PrototypeDeviceDescription::PrototypeDeviceDescription(int id) : id_(id),
                                                                     device_kind_("prototype"),
                                                                     debug_string_(absl::StrCat("PrototypeDevice(id=", id, ")")),
                                                                     to_string_(absl::StrCat("PrototypeDevice(", id, ")"))
    {
    } // PrototypeDeviceDescription::PrototypeDeviceDescription()
    //=======================================================================================================
    //============================================PrototypeDevice============================================
    PrototypeDevice::PrototypeDevice(int id, std::unique_ptr<Transport> transport)
    : description_(id), transport_(std::move(transport))
    {
    } // PrototypeDevice::PrototypeDevice()

    void PrototypeDevice::AttachMemorySpace(PjRtMemorySpace *memory_space)
    {
        CHECK(memory_space_ == nullptr);
        memory_space_ = memory_space;
    } // void PrototypeDevice::AttachMemorySpace()
    //=======================================================================================================
    //============================================PrototypeClient============================================
    PrototypeClient::PrototypeClient(int process_index, std::vector<std::unique_ptr<PrototypeDevice>> devices, size_t num_threads) : process_index_(process_index), owned_devices_(std::move(devices))
    {
        for (const std::unique_ptr<PrototypeDevice> &device : owned_devices_)
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
    } // PrototypeClient::PrototypeClient()

    PrototypeClient::~PrototypeClient() = default;

    absl::StatusOr<PjRtDevice *> PrototypeClient::LookupDevice(GlobalDeviceId global_device_id) const
    {
        auto device = id_to_device_.find(global_device_id.value());
        if (device != id_to_device_.end())
        {
            return device->second;
        }
        else
        {
            return absl::NotFoundError(absl::StrCat("No device with id ", global_device_id.value()));
        }
    } // PrototypeClient::LookupDevice()

    absl::StatusOr<std::unique_ptr<PjRtBuffer>> PrototypeClient::BufferFromHostBuffer(const void *data, PrimitiveType type, absl::Span<int64_t const> dims,
                                                                                      std::optional<absl::Span<int64_t const>> byte_strides, HostBufferSemantics host_buffer_semantics,
                                                                                      absl::AnyInvocable<void() &&> on_done_with_host_buffer, PjRtMemorySpace *memory_space,
                                                                                      const Layout *device_layout)
    {
        if (memory_space->client() != this)
        {
            return absl::InvalidArgumentError("Memory space belongs to a different client");
        }
        if (device_layout != nullptr)
        {
            return absl::InvalidArgumentError(absl::StrCat("Layout is null"));
        }

        Shape shape = ShapeUtil::MakeShape(type, dims);

        if (byte_strides.has_value())
        {
            if (byte_strides->size() != dims.size())
            {
                return absl::InvalidArgumentError("byte_strides rank does not match dims");
            }
            int64_t expected = ShapeUtil::ByteSizeOfPrimitiveType(type);
            for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i)
            {
                // Size-1 dims can carry any stride and still be dense
                if (dims[i] != 1 && (*byte_strides)[i] != expected)
                {
                    return absl::UnimplementedError("Only dense row-major host buffers are supported");
                }
                expected *= dims[i];
            }
        }

        auto state = std::make_shared<BufferState>();
        state->bytes.resize(ShapeUtil::ByteSizeOf(shape));
        if (!state->bytes.empty()) // data may be null for zero-size arrays
        {
            std::memcpy(state->bytes.data(), data, state->bytes.size());
        }
        std::move(on_done_with_host_buffer)();
        state->ready = Future<>(absl::OkStatus());

        return std::make_unique<PrototypeBuffer>(std::move(shape), static_cast<PrototypeMemorySpace *>(memory_space), std::move(state));
    } // PrototypeClient::BufferFromHostBuffer()
    //=======================================================================================================
    //=========================================PrototypeMemorySpace==========================================
    PrototypeMemorySpace::PrototypeMemorySpace(int id, PjRtDevice *device) : id_(id),
                                                                             device_(device),
                                                                             debug_string_(absl::StrCat("PrototypeMemorySpace(id=", id,
                                                                                                        ", device=", device->global_device_id().value(), ")")),
                                                                             to_string_(absl::StrCat("PrototypeMemorySpace(", id, ")"))
    {
    } // PrototypeMemorySpace::PrototypeMemorySpace()
    //=======================================================================================================
    //============================================PrototypeBuffer============================================
    void PrototypeBuffer::Delete()
    {
        absl::MutexLock lock(mu_);
        buffer_state.reset();
    }

    bool PrototypeBuffer::IsDeleted() const
    {
        absl::MutexLock lock(mu_);
        return buffer_state == nullptr;
    }

    std::shared_ptr<BufferState> PrototypeBuffer::state() const
    {
        absl::MutexLock lock(mu_);
        return buffer_state;
    }

    Future<> PrototypeBuffer::GetReadyFuture()
    {
        std::shared_ptr<BufferState> st = state();
        if (st == nullptr)
        {
            // Exact text matters: the base class's BlockHostUntilReady matches on it.
            return Future<>(absl::InvalidArgumentError("GetReadyFuture() called on deleted or donated buffer"));
        }
        return st->ready;
    }

    Future<> PrototypeBuffer::ToLiteral(MutableLiteralBase *literal)
    {
        std::shared_ptr<BufferState> st = state();
        if (st == nullptr)
        {
            return Future<>(absl::FailedPreconditionError("ToLiteral called on deleted buffer"));
        }
        auto [promise, done] = MakePromise<>();
        st->ready.OnReady(
            [st, shape = shape_, literal, promise = std::move(promise)](absl::Status s) mutable
            {
                if (s.ok())
                    s = CopyToLiteral(shape, st->bytes, literal);
                promise.Set(std::move(s));
            });
        return std::move(done);
    }

    Future<> PrototypeBuffer::LazyToLiteral(absl::AnyInvocable<Future<MutableLiteralBase *>() &&> generator)
    {
        std::shared_ptr<BufferState> st = state();
        if (st == nullptr)
        {
            return Future<>(absl::FailedPreconditionError("LazyToLiteral called on deleted buffer"));
        }
        auto [promise, done] = MakePromise<>();
        st->ready.OnReady(
            [st, shape = shape_, generator = std::move(generator),
             promise = std::move(promise)](absl::Status s) mutable
            {
                if (!s.ok())
                {
                    promise.Set(std::move(s));
                    return;
                }
                std::move(generator)().OnReady(
                    [st, shape, promise = std::move(promise)](absl::StatusOr<MutableLiteralBase *> literal) mutable
                    {
                        if (!literal.ok())
                        {
                            promise.Set(literal.status());
                            return;
                        }
                        promise.Set(CopyToLiteral(shape, st->bytes, *literal));
                    });
            });
        return std::move(done);
    }
    //=======================================================================================================
} // namespace xla