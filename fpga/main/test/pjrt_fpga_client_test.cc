#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "main/pjrt_fpga_client.h" // adjust to your header's path
#include <xla/shape_util.h>

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

  std::unique_ptr<PrototypeClient> client_;
};

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
  PrototypeBuffer buffer(shape, memory_space());
  EXPECT_EQ(buffer.memory_space(), memory_space());
  EXPECT_EQ(buffer.device(), device());
  EXPECT_EQ(buffer.client(), client_.get());
  EXPECT_TRUE(ShapeUtil::Equal(buffer.on_device_shape(), shape));

  PrototypeLoadedExecutable executable;
  (void)executable;
}

}  // namespace
}  // namespace xla