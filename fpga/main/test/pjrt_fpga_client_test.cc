#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "main/pjrt_fpga_client.h" // adjust to your header's path
#include <xla/shape_util.h>
#include "xla/literal.h"

namespace xla {
namespace {

// Fresh client with one device for every test. Assumes the client constructor
// creates one PrototypeMemorySpace per device and calls SetClient on each.
class PrototypeClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<std::unique_ptr<PrototypeDevice>> devices;
    devices.push_back(std::make_unique<PrototypeDevice>(/*id=*/0));
    client_ = std::make_unique<PrototypeClient>(
        /*process_index=*/0, std::move(devices), /*num_threads=*/1);
  }

  PjRtDevice* device() const { return client_->devices()[0]; }

  PrototypeMemorySpace* memory_space() const {
    absl::StatusOr<PjRtMemorySpace*> ms = device()->default_memory_space();
    CHECK(ms.ok()) << ms.status();
    return static_cast<PrototypeMemorySpace*>(*ms);
  }

  // Creates an f32 buffer from `data` through the public PJRT entry point,
  // the same call JAX's device_put makes. Sets *done when the client says it
  // no longer needs the host data.
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> MakeF32Buffer(
      const std::vector<float>& data, absl::Span<const int64_t> dims,
      std::optional<absl::Span<const int64_t>> byte_strides = std::nullopt,
      bool* done = nullptr, PjRtMemorySpace* target = nullptr) {
    return client_->BufferFromHostBuffer(
        data.data(), F32, dims, byte_strides,
        PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
        [done]() {
          if (done != nullptr) *done = true;
        },
        target != nullptr ? target : memory_space(),
        /*device_layout=*/nullptr);
  }

  static std::vector<float> ReadBack(PjRtBuffer& buffer) {
    absl::StatusOr<std::shared_ptr<Literal>> literal = buffer.ToLiteral().Await();
    CHECK(literal.ok()) << literal.status();
    absl::Span<const float> values = (*literal)->data<float>();
    return std::vector<float>(values.begin(), values.end());
  }

  std::unique_ptr<PrototypeClient> client_;
};

// ============================== Step 1: wiring ==============================
TEST_F(PrototypeClientTest, ClientOwnsOneAddressableDevice) {
  EXPECT_EQ(client_->device_count(), 1);
  EXPECT_EQ(client_->addressable_device_count(), 1);
  ASSERT_EQ(client_->devices().size(), 1);
  EXPECT_EQ(client_->addressable_devices()[0], device());
  EXPECT_EQ(device()->client(), client_.get());
  EXPECT_TRUE(device()->IsAddressable());
}

TEST_F(PrototypeClientTest, DeviceAndMemorySpacePointAtEachOther) {
  PrototypeMemorySpace* ms = memory_space();
  ASSERT_EQ(ms->devices().size(), 1);
  EXPECT_EQ(ms->devices()[0], device());
  EXPECT_EQ(ms->client(), client_.get());

  ASSERT_EQ(device()->memory_spaces().size(), 1);
  EXPECT_EQ(device()->memory_spaces()[0], ms);

  ASSERT_EQ(client_->memory_spaces().size(), 1);
  EXPECT_EQ(client_->memory_spaces()[0], ms);
}

TEST_F(PrototypeClientTest, LookupDevice) {
  absl::StatusOr<PjRtDevice*> found = client_->LookupDevice(GlobalDeviceId(0));
  ASSERT_TRUE(found.ok()) << found.status();
  EXPECT_EQ(*found, device());

  EXPECT_FALSE(client_->LookupDevice(GlobalDeviceId(1)).ok());
}

TEST_F(PrototypeClientTest, PlatformIdentity) {
  EXPECT_EQ(client_->platform_name(), "prototype");  // whatever name you pick
}

TEST_F(PrototypeClientTest, StringsAreNonEmpty) {
  EXPECT_FALSE(device()->device_kind().empty());
  EXPECT_FALSE(device()->DebugString().empty());
  EXPECT_FALSE(device()->ToString().empty());
  EXPECT_FALSE(memory_space()->kind().empty());
  EXPECT_FALSE(memory_space()->DebugString().empty());
  EXPECT_FALSE(memory_space()->ToString().empty());
}

// Constructing each class proves no pure virtual is left unimplemented.
TEST_F(PrototypeClientTest, BufferAndExecutableConstruct) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto state = std::make_shared<BufferState>();
  state->ready = Future<>(absl::OkStatus());
  PrototypeBuffer buffer(shape, memory_space(), state);
  EXPECT_EQ(buffer.memory_space(), memory_space());
  EXPECT_EQ(buffer.device(), device());
  EXPECT_EQ(buffer.client(), client_.get());
  EXPECT_TRUE(ShapeUtil::Equal(buffer.on_device_shape(), shape));

  PrototypeLoadedExecutable executable;
  (void)executable;
}

// ========================= Step 2: buffers both ways =========================
TEST_F(PrototypeClientTest, HostBufferRoundTrip) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  bool done = false;
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer =
      MakeF32Buffer(data, {2, 2}, std::nullopt, &done);
  ASSERT_TRUE(buffer.ok()) << buffer.status();

  // We copy during the call, so the host data is released before it returns.
  EXPECT_TRUE(done);

  EXPECT_EQ((*buffer)->memory_space(), memory_space());
  EXPECT_EQ((*buffer)->device(), device());
  EXPECT_EQ((*buffer)->element_type(), F32);
  EXPECT_EQ((*buffer)->dimensions(), absl::Span<const int64_t>({2, 2}));

  EXPECT_TRUE((*buffer)->GetReadyFuture().Await().ok());
  EXPECT_EQ(ReadBack(**buffer), data);
}

TEST_F(PrototypeClientTest, HostDataIsCopied) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer = MakeF32Buffer(data, {2, 2});
  ASSERT_TRUE(buffer.ok()) << buffer.status();

  // kImmutableOnlyDuringCall lets the caller reuse its memory after the call.
  data = {9.f, 9.f, 9.f, 9.f};
  EXPECT_EQ(ReadBack(**buffer), std::vector<float>({1.f, 2.f, 3.f, 4.f}));
}

TEST_F(PrototypeClientTest, OnDeviceSize) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer = MakeF32Buffer(data, {2, 2});
  ASSERT_TRUE(buffer.ok()) << buffer.status();

  absl::StatusOr<size_t> size = (*buffer)->GetOnDeviceSizeInBytes();
  ASSERT_TRUE(size.ok()) << size.status();
  EXPECT_EQ(*size, 16u);
}

TEST_F(PrototypeClientTest, BufferIsNotOnCpu) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer = MakeF32Buffer(data, {2, 2});
  ASSERT_TRUE(buffer.ok()) << buffer.status();

  // false keeps JAX from aliasing our private bytes zero-copy.
  EXPECT_FALSE((*buffer)->IsOnCpu());
}

TEST_F(PrototypeClientTest, ToLiteralRejectsWrongShape) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer = MakeF32Buffer(data, {2, 2});
  ASSERT_TRUE(buffer.ok()) << buffer.status();

  Literal wrong(ShapeUtil::MakeShape(F32, {3}));
  EXPECT_FALSE((*buffer)->ToLiteral(&wrong).Await().ok());
}

TEST_F(PrototypeClientTest, LazyToLiteral) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer = MakeF32Buffer(data, {2, 2});
  ASSERT_TRUE(buffer.ok()) << buffer.status();

  Literal literal(ShapeUtil::MakeShape(F32, {2, 2}));
  Future<> copied = (*buffer)->LazyToLiteral(
      [&literal]() -> Future<MutableLiteralBase*> {
        MutableLiteralBase* target = &literal;
        return Future<MutableLiteralBase*>(target);
      });
  ASSERT_TRUE(copied.Await().ok());

  absl::Span<const float> values = literal.data<float>();
  EXPECT_EQ(std::vector<float>(values.begin(), values.end()), data);
}

TEST_F(PrototypeClientTest, DeleteInvalidatesBuffer) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer = MakeF32Buffer(data, {2, 2});
  ASSERT_TRUE(buffer.ok()) << buffer.status();

  EXPECT_FALSE((*buffer)->IsDeleted());
  (*buffer)->Delete();
  EXPECT_TRUE((*buffer)->IsDeleted());

  // Errors, not crashes.
  EXPECT_FALSE((*buffer)->GetReadyFuture().Await().ok());
  EXPECT_FALSE((*buffer)->ToLiteralSync().ok());
}

TEST_F(PrototypeClientTest, AcceptsExplicitDenseStrides) {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  const int64_t dense[] = {8, 4};  // row-major f32[2,2], in bytes
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer =
      MakeF32Buffer(data, {2, 2}, absl::Span<const int64_t>(dense));
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  EXPECT_EQ(ReadBack(**buffer), data);
}

TEST_F(PrototypeClientTest, RejectsNonDenseStrides) {
  std::vector<float> data(8, 0.f);  // large enough even if strides were honored
  const int64_t padded[] = {16, 4};  // each row padded to 16 bytes
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer =
      MakeF32Buffer(data, {2, 2}, absl::Span<const int64_t>(padded));
  EXPECT_FALSE(buffer.ok());
}

TEST_F(PrototypeClientTest, RejectsForeignMemorySpace) {
  std::vector<std::unique_ptr<PrototypeDevice>> devices;
  devices.push_back(std::make_unique<PrototypeDevice>(/*id=*/0));
  PrototypeClient other(/*process_index=*/0, std::move(devices), /*num_threads=*/1);

  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer =
      MakeF32Buffer(data, {2, 2}, std::nullopt, nullptr, other.memory_spaces()[0]);
  EXPECT_FALSE(buffer.ok());
}

}  // namespace
}  // namespace xla