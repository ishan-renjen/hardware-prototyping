#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <cstring>

#include "absl/strings/str_cat.h"

#include "xla/shape.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/layout_util.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/hlo_module_config.h"

#include "main/inc/pjrt_fpga_client.h"

namespace xla
{
    namespace
    {
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
        } // CopyToLiteral()

        absl::StatusOr<std::string> DtypeName(PrimitiveType type)
        {
            switch (type)
            {
            case F32:
                return std::string("f32");
            case F16:
                return std::string("f16");
            case S32:
                return std::string("s32");
            default:
                return absl::UnimplementedError(absl::StrCat(
                    "Unsupported dtype ", PrimitiveType_Name(type)));
            }
        }

        absl::Status FillOutputs(absl::StatusOr<std::vector<Tensor>> result,
                                 const std::vector<Shape> &shapes,
                                 const std::vector<std::shared_ptr<BufferState>> &states)
        {
            if (!result.ok())
                return result.status();
            if (result->size() != states.size())
                return absl::InternalError(absl::StrCat("Hardware returned ", result->size(),
                                                        " outputs, expected ", states.size()));
            for (size_t i = 0; i < states.size(); ++i)
            {
                const int64_t expected = ShapeUtil::ByteSizeOf(shapes[i]);
                if (static_cast<int64_t>((*result)[i].data.size()) != expected)
                    return absl::InternalError(absl::StrCat("Output ", i, ": got ", (*result)[i].data.size(),
                                                            " bytes, expected ", expected));
            }
            for (size_t i = 0; i < states.size(); ++i)
                states[i]->bytes = std::move((*result)[i].data);
            return absl::OkStatus();
        }
    } // namespace
    //======================================PrototypeDeviceDescription=======================================
    PrototypeDeviceDescription::PrototypeDeviceDescription(int id, std::string kind)
        : id_(id),
          device_kind_(std::move(kind)),
          debug_string_(absl::StrCat("PrototypeDevice(id=", id, ", kind=", device_kind_, ")")),
          to_string_(absl::StrCat("PrototypeDevice(", id, ")"))
    {
    } // PrototypeDeviceDescription::PrototypeDeviceDescription()
    //=======================================================================================================
    //============================================PrototypeDevice============================================
    PrototypeDevice::PrototypeDevice(int id, std::string kind, std::vector<std::string> modules, std::shared_ptr<Transport> transport)
        : description_(id, std::move(kind)),
          modules_(std::move(modules)),
          transport_(std::move(transport))
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

    absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> PrototypeClient::CompileAndLoad(MaybeOwningMlirModule module, CompileOptions options)
    {
        XlaComputation computation;
        ABSL_RETURN_IF_ERROR(MlirToXlaComputation(module.mlir_module(), computation,
                                                  /*use_tuple_args=*/false, /*return_tuple=*/false,
                                                  /*exec_build_options=*/nullptr));
        ABSL_ASSIGN_OR_RETURN(ProgramShape program_shape, computation.GetProgramShape());
        ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo, HloModule::CreateFromProto(computation.proto(), HloModuleConfig(program_shape)));

        int device_id = 0;
        const ExecutableBuildOptions &build = options.executable_build_options;
        if (build.has_device_assignment())
        {
            device_id = build.device_assignment()(0, 0); // replica 0, partition 0
        }
        ABSL_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device, LookupDevice(GlobalDeviceId(device_id)));
        auto *device = static_cast<PrototypeDevice *>(pjrt_device);

        const std::string &name = hlo->name();
        if (std::find(device->modules().begin(), device->modules().end(), name) == device->modules().end())
        {
            return absl::UnimplementedError(absl::StrCat(
                "device ", device_id, " (", device->device_kind(), ") does not implement ", name));
        }

        return std::make_unique<PrototypeLoadedExecutable>(std::shared_ptr<HloModule>(std::move(hlo)), device, std::move(options));
    } // PrototypeClient::CompileAndLoad()
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
    //=======================================PrototypeLoadedExecutable=======================================
    PrototypeLoadedExecutable::PrototypeLoadedExecutable(std::shared_ptr<HloModule> hlo_module,
                                                     PrototypeDevice *device,
                                                     CompileOptions compile_options)
                                                    : hlo_module_(std::move(hlo_module)),
                                                    device_(device),
                                                    device_assignment_(1, 1),
                                                    addressable_devices_{device},
                                                    compile_options_(std::move(compile_options))
    {
        device_assignment_(0, 0) = device_->global_device_id().value();

        const HloComputation *entry = hlo_module_->entry_computation();
        num_parameters_ = entry->num_parameters();

        const Shape &root = entry->root_instruction()->shape();
        if (root.IsTuple())
            output_shapes_.assign(root.tuple_shapes().begin(), root.tuple_shapes().end());
        else
            output_shapes_.push_back(root);

        for (Shape &s : output_shapes_)
            LayoutUtil::SetToDefaultLayout(&s);
    } // PrototypeLoadedExecutable::PrototypeLoadedExecutable()

    absl::StatusOr<std::string> PrototypeLoadedExecutable::FingerprintExecutable() const
    {
        return absl::StrCat(tsl::Fingerprint64(hlo_module_->ToString()), "-",
                            device_->global_device_id().value());
    }

    absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
    PrototypeLoadedExecutable::ExecuteHelper(absl::Span<PjRtBuffer *const> args,
                                             Future<> &done_future) const
    {
        if (IsDeleted())
            return absl::FailedPreconditionError(absl::StrCat("Executable ", name(), " was deleted"));
        if (args.size() != static_cast<size_t>(num_parameters_))
            return absl::InvalidArgumentError(absl::StrCat(
                name(), " expects ", num_parameters_, " arguments, got ", args.size()));
        if (device_->transport() == nullptr)
            return absl::FailedPreconditionError("Device has no transport");

        std::vector<std::shared_ptr<BufferState>> input_states; // keeps the bytes alive during the call
        std::vector<TensorView> inputs;
        input_states.reserve(args.size());
        inputs.reserve(args.size());
        for (PjRtBuffer *arg : args)
        {
            if (arg->device() != device_)
                return absl::InvalidArgumentError(absl::StrCat("Argument is not on device ",
                                                               device_->global_device_id().value()));
            auto *buf = static_cast<PrototypeBuffer *>(arg);
            std::shared_ptr<BufferState> st = buf->state();
            if (st == nullptr)
                return absl::InvalidArgumentError("Execute called with a deleted buffer");
            ABSL_RETURN_IF_ERROR(st->ready.Await());
            ABSL_ASSIGN_OR_RETURN(std::string dtype, DtypeName(buf->element_type()));
            inputs.push_back(TensorView{std::move(dtype), buf->dimensions(), st->bytes});
            input_states.push_back(std::move(st));
        }

        auto *memory_space = static_cast<PrototypeMemorySpace *>(device_->memory_spaces()[0]);
        auto [promise, done] = MakePromise<>();
        std::vector<std::shared_ptr<BufferState>> output_states;
        std::vector<std::unique_ptr<PjRtBuffer>> outputs;
        for (const Shape &shape : output_shapes_)
        {
            auto st = std::make_shared<BufferState>();
            st->ready = done;
            outputs.push_back(std::make_unique<PrototypeBuffer>(shape, memory_space, st));
            output_states.push_back(std::move(st));
        }

        device_->transport()->Execute(
            device_->global_device_id().value(), device_->NextSeq(), name(), inputs,
            [output_states, output_shapes = output_shapes_, promise = std::move(promise)](
                absl::StatusOr<std::vector<Tensor>> result) mutable
            {
                promise.Set(FillOutputs(std::move(result), output_shapes, output_states));
            });

        done_future = done;
        return outputs;
    } // PrototypeLoadedExecutable::ExecuteHelper()

    absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
    PrototypeLoadedExecutable::Execute(absl::Span<const std::vector<PjRtBuffer *>> argument_handles,
                                       const ExecuteOptions &options,
                                       std::optional<std::vector<Future<>>> &returned_futures) const
    {
        if (argument_handles.size() != 1)
        {
            return absl::InvalidArgumentError(absl::StrCat(
                name(), " runs on 1 device, got arguments for ", argument_handles.size()));
        }

        Future<> done;
        ABSL_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<PjRtBuffer>> outputs,
                              ExecuteHelper(argument_handles[0], done));

        if (returned_futures.has_value())
        {
            returned_futures->assign(1, done);
        }

        std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result;
        result.push_back(std::move(outputs));
        return result;
    } // PrototypeLoadedExecutable::Execute()

    absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
    PrototypeLoadedExecutable::ExecuteSharded(absl::Span<PjRtBuffer *const> argument_handles,
                                              PjRtDevice *device, const ExecuteOptions &options,
                                              std::optional<Future<>> &returned_future,
                                              bool fill_future) const
    {
        if (device != device_)
        {
            return absl::InvalidArgumentError(absl::StrCat(
                name(), " was compiled for device ", device_->global_device_id().value(),
                ", not ", device->global_device_id().value()));
        }

        Future<> done;
        ABSL_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<PjRtBuffer>> outputs,
                              ExecuteHelper(argument_handles, done));
        if (fill_future)
        {
            returned_future = done;
        }
        return outputs;
    } // PrototypeLoadedExecutable::ExecuteSharded()

    absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
    PrototypeLoadedExecutable::ExecutePortable(absl::Span<PjRtBuffer *const> argument_handles,
                                               PjRtDevice *device, const ExecuteOptions &options,
                                               std::optional<Future<>> &returned_future,
                                               bool fill_future) const
    {
        return ExecuteSharded(argument_handles, device, options, returned_future, fill_future);
    } // ExecutePortable()
    //=======================================================================================================
} // namespace xla