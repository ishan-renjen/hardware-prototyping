// Requires the gateway running on localhost:50051 with the COMPONENTS in grpc-server/server.py:
//   device 0 "adder"  -> jit_add_one
//   device 1 "scaler" -> jit_times_two
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "main/inc/grpc_transport.h"

namespace
{
    constexpr char kGateway[] = "localhost:50051";

    absl::StatusOr<std::vector<Tensor>> RunSync(Transport &transport, int device, uint64_t seq,
                                                absl::string_view module,
                                                absl::Span<const TensorView> inputs)
    {
        absl::Notification done;
        absl::StatusOr<std::vector<Tensor>> result;
        transport.Execute(device, seq, module, inputs,
                          [&](absl::StatusOr<std::vector<Tensor>> r)
                          {
                              result = std::move(r);
                              done.Notify();
                          });
        done.WaitForNotification();
        return result;
    }

    TensorView F32View(const std::vector<float> &x, const std::vector<int64_t> &dims)
    {
        return TensorView{"f32", dims,
                          absl::MakeConstSpan(reinterpret_cast<const uint8_t *>(x.data()),
                                              x.size() * sizeof(float))};
    }

    std::vector<float> F32Values(const Tensor &t)
    {
        std::vector<float> v(t.data.size() / sizeof(float));
        std::memcpy(v.data(), t.data.data(), t.data.size());
        return v;
    }

    TEST(GrpcTransportTest, ListDevicesReportsComponents)
    {
        GrpcTransport transport(kGateway);
        auto devices = transport.ListDevices();
        ASSERT_TRUE(devices.ok()) << devices.status();
        ASSERT_EQ(devices->size(), 2u);
        EXPECT_EQ((*devices)[0].id, 0);
        EXPECT_EQ((*devices)[0].kind, "adder");
        EXPECT_EQ((*devices)[0].modules, std::vector<std::string>{"jit_add_one"});
        EXPECT_EQ((*devices)[1].id, 1);
        EXPECT_EQ((*devices)[1].kind, "scaler");
    }

    TEST(GrpcTransportTest, RoutesToEachDevice)
    {
        GrpcTransport transport(kGateway);
        std::vector<float> x = {0.f, 1.f, 2.f, 3.f};
        std::vector<int64_t> dims = {4};
        TensorView in = F32View(x, dims);

        auto add = RunSync(transport, /*device=*/0, /*seq=*/0, "jit_add_one", absl::MakeConstSpan(&in, 1));
        ASSERT_TRUE(add.ok()) << add.status();
        ASSERT_EQ(add->size(), 1u);
        EXPECT_EQ((*add)[0].dims, dims);
        EXPECT_EQ(F32Values((*add)[0]), (std::vector<float>{1.f, 2.f, 3.f, 4.f}));

        auto twice = RunSync(transport, /*device=*/1, /*seq=*/0, "jit_times_two", absl::MakeConstSpan(&in, 1));
        ASSERT_TRUE(twice.ok()) << twice.status();
        EXPECT_EQ(F32Values((*twice)[0]), (std::vector<float>{0.f, 2.f, 4.f, 6.f}));
    }

    TEST(GrpcTransportTest, ConcurrentCallsAcrossDevices)
    {
        GrpcTransport transport(kGateway);
        std::vector<float> x = {1.f};
        std::vector<int64_t> dims = {1};
        TensorView in = F32View(x, dims);

        constexpr int kCalls = 8;
        absl::Notification all_done;
        std::atomic<int> remaining{kCalls};
        std::atomic<int> failures{0};
        for (int i = 0; i < kCalls; ++i)
        {
            const int device = i % 2;
            transport.Execute(device, /*seq=*/i / 2, device == 0 ? "jit_add_one" : "jit_times_two",
                              absl::MakeConstSpan(&in, 1),
                              [&](absl::StatusOr<std::vector<Tensor>> r)
                              {
                                  if (!r.ok()) failures++;
                                  if (--remaining == 0) all_done.Notify();
                              });
        }
        all_done.WaitForNotification();
        EXPECT_EQ(failures.load(), 0);
    }

    TEST(GrpcTransportTest, UnknownDeviceIsNotFound)
    {
        GrpcTransport transport(kGateway);
        std::vector<float> x = {0.f};
        std::vector<int64_t> dims = {1};
        TensorView in = F32View(x, dims);

        auto result = RunSync(transport, /*device=*/7, /*seq=*/0, "jit_add_one", absl::MakeConstSpan(&in, 1));
        EXPECT_EQ(result.status().code(), absl::StatusCode::kNotFound);
    }

    TEST(GrpcTransportTest, ModuleNotOnDeviceIsUnimplemented)
    {
        GrpcTransport transport(kGateway);
        std::vector<float> x = {0.f};
        std::vector<int64_t> dims = {1};
        TensorView in = F32View(x, dims);

        // jit_add_one lives on device 0, not device 1.
        auto result = RunSync(transport, /*device=*/1, /*seq=*/0, "jit_add_one", absl::MakeConstSpan(&in, 1));
        EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
    }

    TEST(GrpcTransportTest, UnreachableServerIsAnError)
    {
        GrpcTransport transport("localhost:1"); // nothing listens on port 1
        EXPECT_FALSE(transport.ListDevices().ok());

        std::vector<float> x = {0.f};
        std::vector<int64_t> dims = {1};
        TensorView in = F32View(x, dims);
        EXPECT_FALSE(RunSync(transport, 0, 0, "jit_add_one", absl::MakeConstSpan(&in, 1)).ok());
    }
} //namespace