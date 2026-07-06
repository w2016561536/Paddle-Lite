// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
// Modified: calib + NCHW->NHWC(C16) on CPU, conv2d/hardswish/depthwise/hardswish on FPGA HLS IP.
// No C++ exceptions are used in this version.

#include "lite/kernels/intel_fpga/calib_conv2d.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define __ARM_NEON

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(CALIB_CONV2D_FORCE_NEON)
#include <arm_neon.h>
#define CALIB_CONV2D_HAS_NEON 1
#else
#define CALIB_CONV2D_HAS_NEON 0
#endif

#if CALIB_CONV2D_HAS_NEON && defined(__arm__) && !defined(__aarch64__) && defined(__GNUC__)
#define CALIB_CONV2D_HAS_ARMV7_ASM 1
#else
#define CALIB_CONV2D_HAS_ARMV7_ASM 0
#endif

#ifndef CALIB_CONV2D_USE_OPENMP
#if defined(_OPENMP)
#define CALIB_CONV2D_USE_OPENMP 1
#else
#define CALIB_CONV2D_USE_OPENMP 0
#endif
#endif

#define CALIB_CONV2D_PACK_INPUT_DIRECT_TO_CMA 1
#define CALIB_CONV2D_USE_CACHED_INPUT_STAGING 0
#define CALIB_CONV2D_USE_CMA_INPUT_MSYNC 0
#define CALIB_CONV2D_USE_CMA_OUTPUT_MSYNC 0

#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace intel_fpga {

// ============================================================================
// 1. Hardware/CMA configuration
// ============================================================================
#ifndef CALIB_CONV2D_IP_BASE_PHYS
#define CALIB_CONV2D_IP_BASE_PHYS 0xFF208000UL
#endif

#ifndef CALIB_CONV2D_LW_MAP_SIZE
#define CALIB_CONV2D_LW_MAP_SIZE 0x1000UL
#endif

#ifndef CALIB_CONV2D_CMA_DEV
#define CALIB_CONV2D_CMA_DEV "/dev/cmadrv0"
#endif

#ifndef CALIB_CONV2D_CLEAR_CMA_ON_ALLOC
#define CALIB_CONV2D_CLEAR_CMA_ON_ALLOC 0
#endif

#ifndef CALIB_CONV2D_CLEAR_OUTPUT_BEFORE_RUN
#define CALIB_CONV2D_CLEAR_OUTPUT_BEFORE_RUN 0
#endif

#ifndef CALIB_CONV2D_CLEAR_INPUT_PACK
#define CALIB_CONV2D_CLEAR_INPUT_PACK 0
#endif

#ifndef CALIB_CONV2D_USE_CACHED_IO_STAGING
#define CALIB_CONV2D_USE_CACHED_IO_STAGING 1
#endif

#ifndef CALIB_CONV2D_USE_CACHED_INPUT_STAGING
#define CALIB_CONV2D_USE_CACHED_INPUT_STAGING CALIB_CONV2D_USE_CACHED_IO_STAGING
#endif

#ifndef CALIB_CONV2D_USE_CACHED_OUTPUT_STAGING
#define CALIB_CONV2D_USE_CACHED_OUTPUT_STAGING 0
#endif

#ifndef CALIB_CONV2D_USE_CACHED_CONV2D_OUT_STAGING
#define CALIB_CONV2D_USE_CACHED_CONV2D_OUT_STAGING 0
#endif

#ifndef CALIB_CONV2D_ENABLE_VERBOSE_LOG
#define CALIB_CONV2D_ENABLE_VERBOSE_LOG 0
#endif

#ifndef CALIB_CONV2D_ENABLE_ERROR_LOG
#define CALIB_CONV2D_ENABLE_ERROR_LOG 1
#endif

#ifndef CALIB_CONV2D_ENABLE_TIMING_LOG
#define CALIB_CONV2D_ENABLE_TIMING_LOG 0
#endif

#ifndef CALIB_CONV2D_OMP_MIN_WORK
#define CALIB_CONV2D_OMP_MIN_WORK 4096
#endif

#ifndef CALIB_CONV2D_OMP_SPATIAL_TILE
#define CALIB_CONV2D_OMP_SPATIAL_TILE 512
#endif

#ifndef CALIB_CONV2D_CACHE_REG_WRITES
#define CALIB_CONV2D_CACHE_REG_WRITES 0
#endif

#ifndef CALIB_CONV2D_WAIT_CLOCK_CHECK_INTERVAL
#define CALIB_CONV2D_WAIT_CLOCK_CHECK_INTERVAL 1024
#endif

#ifndef CALIB_CONV2D_USE_CMA_MSYNC
#define CALIB_CONV2D_USE_CMA_MSYNC 1
#endif

#ifndef CALIB_CONV2D_ENABLE_MSYNC_LOG
#define CALIB_CONV2D_ENABLE_MSYNC_LOG 0
#endif

#ifndef CALIB_CONV2D_USE_CMA_INPUT_MSYNC
#define CALIB_CONV2D_USE_CMA_INPUT_MSYNC CALIB_CONV2D_USE_CMA_MSYNC
#endif

#ifndef CALIB_CONV2D_USE_CMA_OUTPUT_MSYNC
#define CALIB_CONV2D_USE_CMA_OUTPUT_MSYNC CALIB_CONV2D_USE_CMA_MSYNC
#endif

#ifndef CALIB_CONV2D_PACK_INPUT_DIRECT_TO_CMA
#define CALIB_CONV2D_PACK_INPUT_DIRECT_TO_CMA (!CALIB_CONV2D_USE_CACHED_INPUT_STAGING)
#endif

#if CALIB_CONV2D_ENABLE_ERROR_LOG
#define FPGA_CALIB_LOG_ERROR(msg) \
  do { std::cerr << "[IntelFPGA][calib_conv2d] ERROR: " << msg << std::endl; } while (0)
#else
#define FPGA_CALIB_LOG_ERROR(msg) do {} while (0)
#endif

// HLS CTRL_BUS register offsets.
static const uint32_t REG_CTRL = 0x00;
static const uint32_t REG_INPUT_L = 0x10;
static const uint32_t REG_INPUT_H = 0x14;
static const uint32_t REG_WEIGHT_CONV_L = 0x1c;
static const uint32_t REG_WEIGHT_CONV_H = 0x20;
static const uint32_t REG_WEIGHT_DW_L = 0x28;
static const uint32_t REG_WEIGHT_DW_H = 0x2c;
static const uint32_t REG_BIAS_CONV_L = 0x34;
static const uint32_t REG_BIAS_CONV_H = 0x38;
static const uint32_t REG_OUTPUT_FIXED_L = 0x40;
static const uint32_t REG_OUTPUT_FIXED_H = 0x44;
static const uint32_t REG_BIAS_DW_L = 0x4c;
static const uint32_t REG_BIAS_DW_H = 0x50;
static const uint32_t REG_OUTPUT_CONV2D_L = 0x58;
static const uint32_t REG_OUTPUT_CONV2D_H = 0x5c;
static const uint32_t REG_CONV_SCALE_L = 0x64;
static const uint32_t REG_CONV_SCALE_H = 0x68;
static const uint32_t REG_DW_SCALE_L = 0x70;
static const uint32_t REG_DW_SCALE_H = 0x74;
static const uint32_t REG_INV_DW_SCALE = 0x7c;
static const uint32_t REG_INPUT_HW = 0x84;
static const uint32_t REG_INPUT_CHANNELS = 0x8c;
static const uint32_t REG_OUT_CHANNELS = 0x94;
static const uint32_t REG_DW_KERNEL = 0x9c;
static const uint32_t REG_DW_STRIDE = 0xa4;
static const uint32_t REG_DW_PADDING = 0xac;
static const uint32_t REG_ENABLE_CONV_OUT = 0xb4;
static const uint32_t REG_LOCAL_GROUPS_PER_ROUND = 0xbc;
static const uint32_t REG_ROUNDS = 0xc4;

// HLS fixed_t = ap_fixed<32, 10, AP_RND, AP_SAT>, i.e. Q10.22.
static const int FIXED_TOTAL_BITS = 32;
static const int FIXED_INT_BITS = 10;
static const int FIXED_FRAC_BITS = FIXED_TOTAL_BITS - FIXED_INT_BITS;
static const int MAX_PREPACK_CONV_OC = 384;
static const int MAX_PREPACK_CONV_IC = 192;
static const int MAX_PREPACK_DW_OC = 384;
static const int MAX_PREPACK_DW_KH = 5;
static const int MAX_PREPACK_DW_KW = 5;
static const int HLS_MAX_INPUT_C_BLOCKS = 12;
static const int HLS_MAX_WEIGHT_ROWS = 192;
static const int HLS_MAX_LOCAL_GROUPS = 16;
static const int MAX_PERSIST_FEATURE_C = 192;
static const int MAX_PERSIST_FEATURE_HW = 208;
static const size_t MAX_PERSIST_FEATURE_HINT_ELEMS =
    static_cast<size_t>(MAX_PERSIST_FEATURE_C) * MAX_PERSIST_FEATURE_HW;
static const size_t MAX_PERSIST_FEATURE_ELEMS = 1000000;

// ============================================================================
// 2. CMA driver ABI copied from cma_example.c
// ============================================================================
struct cma_mblk_s {
  void* addr;
  void* virt;
  unsigned long phys;
  size_t size;
};

#define CMA_MAGIC_ID (('C' + 'M' + 'A' + 'B') / 4)
#define CMA_IOCTL_MAKE(cmd) (_IO(CMA_MAGIC_ID, cmd))
#define CMA_CMD_MGET 0x00
#define CMA_CMD_FREE 0x01

static inline size_t AlignUp(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

static inline int RoundUp16(int x) {
  return (x + 15) & ~15;
}

static inline int RoundUp4(int x) {
  return (x + 3) & ~3;
}

static inline void CalibFastCopy(void* dst, const void* src, size_t bytes) {
  if (dst == NULL || src == NULL || bytes == 0) return;
#if CALIB_CONV2D_HAS_NEON
  uint8_t* d = reinterpret_cast<uint8_t*>(dst);
  const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
  while (bytes >= 64) {
    __builtin_prefetch(s + 128);
    const uint8x16_t q0 = vld1q_u8(s + 0);
    const uint8x16_t q1 = vld1q_u8(s + 16);
    const uint8x16_t q2 = vld1q_u8(s + 32);
    const uint8x16_t q3 = vld1q_u8(s + 48);
    vst1q_u8(d + 0, q0);
    vst1q_u8(d + 16, q1);
    vst1q_u8(d + 32, q2);
    vst1q_u8(d + 48, q3);
    s += 64;
    d += 64;
    bytes -= 64;
  }
  while (bytes >= 16) {
    const uint8x16_t q = vld1q_u8(s);
    vst1q_u8(d, q);
    s += 16;
    d += 16;
    bytes -= 16;
  }
  if (bytes != 0) {
    std::memcpy(d, s, bytes);
  }
#else
  std::memcpy(dst, src, bytes);
#endif
}

static inline void CalibFastZero(void* dst, size_t bytes) {
  if (dst == NULL || bytes == 0) return;
#if CALIB_CONV2D_HAS_NEON
  uint8_t* d = reinterpret_cast<uint8_t*>(dst);
  const uint8x16_t z = vdupq_n_u8(0);
  while (bytes >= 64) {
    vst1q_u8(d + 0, z);
    vst1q_u8(d + 16, z);
    vst1q_u8(d + 32, z);
    vst1q_u8(d + 48, z);
    d += 64;
    bytes -= 64;
  }
  while (bytes >= 16) {
    vst1q_u8(d, z);
    d += 16;
    bytes -= 16;
  }
  if (bytes != 0) {
    std::memset(d, 0, bytes);
  }
#else
  std::memset(dst, 0, bytes);
#endif
}

static inline uint64_t CalibNowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

static inline double CalibNsToMs(uint64_t ns) {
  return static_cast<double>(ns) / 1000000.0;
}

static inline void CalibDmaWriteBarrier() {
#if defined(__arm__) || defined(__aarch64__)
  asm volatile("dsb sy" ::: "memory");
#else
  __sync_synchronize();
#endif
}

static inline void CalibDmaReadBarrier() {
#if defined(__arm__) || defined(__aarch64__)
  asm volatile("dsb sy" ::: "memory");
#else
  __sync_synchronize();
#endif
}

static inline uintptr_t CalibPageSize() {
  static uintptr_t page = 0;
  if (page == 0) {
    long page_long = sysconf(_SC_PAGESIZE);
    page = page_long > 0 ? static_cast<uintptr_t>(page_long) : 4096u;
  }
  return page;
}

static inline bool CalibMsyncRange(void* ptr, size_t bytes, int flags) {
  if (ptr == NULL || bytes == 0) return true;
  const uintptr_t page = CalibPageSize();
  const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t aligned_start = start & ~(page - 1u);
  const uintptr_t end = start + bytes;
  const uintptr_t aligned_end = (end + page - 1u) & ~(page - 1u);
  const size_t aligned_bytes = static_cast<size_t>(aligned_end - aligned_start);
  if (msync(reinterpret_cast<void*>(aligned_start), aligned_bytes, flags) != 0) {
#if CALIB_CONV2D_ENABLE_MSYNC_LOG
    FPGA_CALIB_LOG_ERROR("msync failed, errno=" << errno << ", " << strerror(errno));
#endif
    return false;
  }
  return true;
}

static inline void CalibSyncInputForFpga(void* ptr, size_t bytes) {
#if CALIB_CONV2D_USE_CMA_INPUT_MSYNC
  (void)CalibMsyncRange(ptr, bytes, MS_SYNC);
#else
  (void)ptr;
  (void)bytes;
#endif
  CalibDmaWriteBarrier();
}

static inline void CalibSyncOutputForCpu(void* ptr, size_t bytes) {
  CalibDmaReadBarrier();
#if CALIB_CONV2D_USE_CMA_OUTPUT_MSYNC
#if defined(MS_INVALIDATE)
  (void)CalibMsyncRange(ptr, bytes, MS_SYNC | MS_INVALIDATE);
#else
  (void)CalibMsyncRange(ptr, bytes, MS_SYNC);
#endif
#else
  (void)ptr;
  (void)bytes;
#endif
  CalibDmaReadBarrier();
}

struct ScopedCalibTimer {
  explicit ScopedCalibTimer(uint64_t* dst) {
#if CALIB_CONV2D_ENABLE_TIMING_LOG
    dst_ = dst;
    start_ns_ = CalibNowNs();
#else
    (void)dst;
#endif
  }

  ~ScopedCalibTimer() {
#if CALIB_CONV2D_ENABLE_TIMING_LOG
    if (dst_ != NULL) {
      *dst_ += CalibNowNs() - start_ns_;
    }
#endif
  }

 private:
  uint64_t* dst_{NULL};
  uint64_t start_ns_{0};
};

struct CalibConv2dTiming {
  uint64_t tensor_access_ns{0};
  uint64_t shape_check_ns{0};
  uint64_t cma_open_ns{0};
  uint64_t cma_alloc_ns{0};
  uint64_t arena_layout_ns{0};
  uint64_t static_pack_ns{0};
  uint64_t regs_open_ns{0};
  uint64_t regs_write_ns{0};
  uint64_t batch_clear_ns{0};
  uint64_t input_pack_ns{0};
  uint64_t input_sync_ns{0};
  uint64_t ip_start_cmd_ns{0};
  uint64_t ip_wait_done_ns{0};
  uint64_t output_sync_ns{0};
  uint64_t output_copy_ns{0};
  uint64_t conv2d_copy_ns{0};
  uint64_t cleanup_ns{0};

  uint64_t CpuInterfaceNs() const {
    return tensor_access_ns + shape_check_ns + cma_open_ns + cma_alloc_ns +
           arena_layout_ns + static_pack_ns + regs_open_ns + regs_write_ns +
           batch_clear_ns + input_pack_ns + input_sync_ns + ip_start_cmd_ns +
           output_sync_ns + output_copy_ns + conv2d_copy_ns + cleanup_ns;
  }

  void Print(int batch, uint64_t total_ns) const {
#if CALIB_CONV2D_ENABLE_TIMING_LOG
    const uint64_t cpu_ns = CpuInterfaceNs();
    const uint64_t tracked_ns = cpu_ns + ip_wait_done_ns;
    const uint64_t other_ns = total_ns > tracked_ns ? total_ns - tracked_ns : 0;
    const double total_ms = CalibNsToMs(total_ns);
    const double ip_ms = CalibNsToMs(ip_wait_done_ns);
    const double cpu_ms = CalibNsToMs(cpu_ns + other_ns);
    const double denom = total_ms > 0.0 ? total_ms : 1.0;

    std::cout << "[IntelFPGA][calib_conv2d][timing] total="
              << total_ms << " ms, fpga_ip_wait_done=" << ip_ms
              << " ms (" << (ip_ms * 100.0 / denom)
              << "%), cpu_interface=" << cpu_ms << " ms ("
              << (cpu_ms * 100.0 / denom) << "%), batch=" << batch
              << std::endl;
    std::cout << "[IntelFPGA][calib_conv2d][timing] setup: tensor_access="
              << CalibNsToMs(tensor_access_ns) << " ms, shape_check="
              << CalibNsToMs(shape_check_ns) << " ms, cma_open="
              << CalibNsToMs(cma_open_ns) << " ms, cma_alloc_mmap="
              << CalibNsToMs(cma_alloc_ns) << " ms, arena_layout="
              << CalibNsToMs(arena_layout_ns) << " ms, static_pack="
              << CalibNsToMs(static_pack_ns) << " ms, regs_open="
              << CalibNsToMs(regs_open_ns) << " ms, regs_write="
              << CalibNsToMs(regs_write_ns) << " ms" << std::endl;
    std::cout << "[IntelFPGA][calib_conv2d][timing] per_run: batch_clear="
              << CalibNsToMs(batch_clear_ns) << " ms, input_quant_pack="
              << CalibNsToMs(input_pack_ns) << " ms, input_sync="
              << CalibNsToMs(input_sync_ns) << " ms, ip_start_cmd="
              << CalibNsToMs(ip_start_cmd_ns) << " ms, fpga_ip_wait_done="
              << ip_ms << " ms, output_sync="
              << CalibNsToMs(output_sync_ns) << " ms, output_unpack_copy="
              << CalibNsToMs(output_copy_ns) << " ms, conv2d_out_copy="
              << CalibNsToMs(conv2d_copy_ns) << " ms, cleanup="
              << CalibNsToMs(cleanup_ns) << " ms, other="
              << CalibNsToMs(other_ns) << " ms" << std::endl;
    if (batch > 0) {
      std::cout << "[IntelFPGA][calib_conv2d][timing] avg_per_batch: input_quant_pack="
                << CalibNsToMs(input_pack_ns) / batch
                << " ms, input_sync=" << CalibNsToMs(input_sync_ns) / batch
                << " ms, fpga_ip_wait_done=" << ip_ms / batch
                << " ms, output_sync=" << CalibNsToMs(output_sync_ns) / batch
                << " ms, output_unpack_copy="
                << CalibNsToMs(output_copy_ns) / batch
                << " ms, conv2d_out_copy="
                << CalibNsToMs(conv2d_copy_ns) / batch << " ms"
                << std::endl;
    }
#else
    (void)batch;
    (void)total_ns;
#endif
  }
};

static inline uint32_t FixedRawFromFloat(float x) {
  double scaled = static_cast<double>(x) * static_cast<double>(1 << FIXED_FRAC_BITS);
  if (scaled > static_cast<double>(std::numeric_limits<int32_t>::max())) {
    return static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
  }
  if (scaled < static_cast<double>(std::numeric_limits<int32_t>::min())) {
    return static_cast<uint32_t>(std::numeric_limits<int32_t>::min());
  }
  int64_t q = llround(scaled);
  return static_cast<uint32_t>(static_cast<int32_t>(q));
}

static inline int8_t QuantizeToInt8SymInv(float x, float inv_scale) {
  const float scaled = x * inv_scale;
  if (scaled >= 127.f) return 127;
  if (scaled <= -127.f) return -127;
  const int32_t q = static_cast<int32_t>(scaled + (scaled >= 0.f ? 0.5f : -0.5f));
  return static_cast<int8_t>(q);
}

static inline int8_t QuantizeToInt8Sym(float x, float scale) {
  return QuantizeToInt8SymInv(x, 1.f / scale);
}

#if CALIB_CONV2D_HAS_NEON
#if CALIB_CONV2D_HAS_ARMV7_ASM
alignas(16) static const float kCalibArmv7QuantMin[4] = {-127.f, -127.f, -127.f, -127.f};

static inline int8x8_t QuantizeFloat32x8ToI8Armv7Asm(const float* src,
                                                     float inv_scale) {
  alignas(8) int8_t out[8];
  const float32x4_t vscale = vdupq_n_f32(inv_scale);
  const float32x4_t vzero = vdupq_n_f32(0.f);
  const float32x4_t vpoff = vdupq_n_f32(0.5f);
  const float32x4_t vnoff = vdupq_n_f32(-0.5f);
  int8_t* dst = out;
  asm volatile(
      "vld1.32    {d0-d3}, [%[src]]              \n"
      "vand.i32   q4, %q[vpoff], %q[vpoff]       \n"
      "vand.i32   q5, q4, q4                     \n"
      "vcgt.f32   q8, q0, %q[vzero]              \n"
      "vcgt.f32   q9, q1, %q[vzero]              \n"
      "vbif.f32   q4, %q[vnoff], q8              \n"
      "vbif.f32   q5, %q[vnoff], q9              \n"
      "vmla.f32   q4, q0, %q[vscale]             \n"
      "vmla.f32   q5, q1, %q[vscale]             \n"
      "vld1.32    {d0-d1}, [%[vmin]]             \n"
      "vcge.f32   q8, q4, q0                     \n"
      "vcge.f32   q9, q5, q0                     \n"
      "vbif       q4, q0, q8                     \n"
      "vbif       q5, q0, q9                     \n"
      "vcvt.s32.f32 q0, q4                       \n"
      "vcvt.s32.f32 q1, q5                       \n"
      "vqmovn.s32 d8, q0                         \n"
      "vqmovn.s32 d9, q1                         \n"
      "vqmovn.s16 d12, q4                        \n"
      "vst1.8     {d12}, [%[out]]                \n"
      : [out] "+r"(dst)
      : [src] "r"(src),
        [vscale] "w"(vscale),
        [vpoff] "w"(vpoff),
        [vnoff] "w"(vnoff),
        [vzero] "w"(vzero),
        [vmin] "r"(kCalibArmv7QuantMin)
      : "cc", "memory", "q0", "q1", "q4", "q5", "q6", "q8", "q9");
  return vld1_s8(out);
}
#endif

static inline int8x8_t QuantizeFloat32x8ToI8Neon(const float* src, float inv_scale) {
#if CALIB_CONV2D_HAS_ARMV7_ASM
  return QuantizeFloat32x8ToI8Armv7Asm(src, inv_scale);
#else
  const float32x4_t inv_v = vdupq_n_f32(inv_scale);
  const float32x4_t zero_v = vdupq_n_f32(0.f);
  const float32x4_t pos_half_v = vdupq_n_f32(0.5f);
  const float32x4_t neg_half_v = vdupq_n_f32(-0.5f);
  const int32x4_t min_v = vdupq_n_s32(-127);
  const int32x4_t max_v = vdupq_n_s32(127);

  float32x4_t v0 = vmulq_f32(vld1q_f32(src), inv_v);
  float32x4_t v1 = vmulq_f32(vld1q_f32(src + 4), inv_v);

  const uint32x4_t neg0 = vcltq_f32(v0, zero_v);
  const uint32x4_t neg1 = vcltq_f32(v1, zero_v);
  v0 = vaddq_f32(v0, vbslq_f32(neg0, neg_half_v, pos_half_v));
  v1 = vaddq_f32(v1, vbslq_f32(neg1, neg_half_v, pos_half_v));

  int32x4_t q0 = vcvtq_s32_f32(v0);
  int32x4_t q1 = vcvtq_s32_f32(v1);
  q0 = vmaxq_s32(min_v, vminq_s32(max_v, q0));
  q1 = vmaxq_s32(min_v, vminq_s32(max_v, q1));

  const int16x8_t q16 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
  return vmovn_s16(q16);
#endif
}

static inline uint8x8_t LoadQuantizedChannel8Neon(const float* src_nchw,
                                                  int spatial,
                                                  int channel,
                                                  int c,
                                                  int s,
                                                  float inv_scale) {
  if (channel >= c) {
    return vdup_n_u8(0);
  }
  const float* src = src_nchw + static_cast<size_t>(channel) * spatial + s;
  return vreinterpret_u8_s8(QuantizeFloat32x8ToI8Neon(src, inv_scale));
}

static inline uint8x8_t LoadQuantizedChannel8NoPadNeon(const float* src_nchw,
                                                       int spatial,
                                                       int channel,
                                                       int s,
                                                       float inv_scale) {
  const float* src = src_nchw + static_cast<size_t>(channel) * spatial + s;
  return vreinterpret_u8_s8(QuantizeFloat32x8ToI8Neon(src, inv_scale));
}

static inline void Transpose8x8U8Neon(uint8x8_t r0,
                                      uint8x8_t r1,
                                      uint8x8_t r2,
                                      uint8x8_t r3,
                                      uint8x8_t r4,
                                      uint8x8_t r5,
                                      uint8x8_t r6,
                                      uint8x8_t r7,
                                      uint8x8_t* c0,
                                      uint8x8_t* c1,
                                      uint8x8_t* c2,
                                      uint8x8_t* c3,
                                      uint8x8_t* c4,
                                      uint8x8_t* c5,
                                      uint8x8_t* c6,
                                      uint8x8_t* c7) {
  const uint8x8x2_t t01 = vtrn_u8(r0, r1);
  const uint8x8x2_t t23 = vtrn_u8(r2, r3);
  const uint8x8x2_t t45 = vtrn_u8(r4, r5);
  const uint8x8x2_t t67 = vtrn_u8(r6, r7);

  const uint16x4x2_t u02 = vtrn_u16(vreinterpret_u16_u8(t01.val[0]),
                                    vreinterpret_u16_u8(t23.val[0]));
  const uint16x4x2_t u13 = vtrn_u16(vreinterpret_u16_u8(t01.val[1]),
                                    vreinterpret_u16_u8(t23.val[1]));
  const uint16x4x2_t u46 = vtrn_u16(vreinterpret_u16_u8(t45.val[0]),
                                    vreinterpret_u16_u8(t67.val[0]));
  const uint16x4x2_t u57 = vtrn_u16(vreinterpret_u16_u8(t45.val[1]),
                                    vreinterpret_u16_u8(t67.val[1]));

  const uint32x2x2_t v04 = vtrn_u32(vreinterpret_u32_u16(u02.val[0]),
                                    vreinterpret_u32_u16(u46.val[0]));
  const uint32x2x2_t v26 = vtrn_u32(vreinterpret_u32_u16(u02.val[1]),
                                    vreinterpret_u32_u16(u46.val[1]));
  const uint32x2x2_t v15 = vtrn_u32(vreinterpret_u32_u16(u13.val[0]),
                                    vreinterpret_u32_u16(u57.val[0]));
  const uint32x2x2_t v37 = vtrn_u32(vreinterpret_u32_u16(u13.val[1]),
                                    vreinterpret_u32_u16(u57.val[1]));

  *c0 = vreinterpret_u8_u32(v04.val[0]);
  *c4 = vreinterpret_u8_u32(v04.val[1]);
  *c2 = vreinterpret_u8_u32(v26.val[0]);
  *c6 = vreinterpret_u8_u32(v26.val[1]);
  *c1 = vreinterpret_u8_u32(v15.val[0]);
  *c5 = vreinterpret_u8_u32(v15.val[1]);
  *c3 = vreinterpret_u8_u32(v37.val[0]);
  *c7 = vreinterpret_u8_u32(v37.val[1]);
}

static inline void StoreTransposed8x16Neon(uint8x8_t q0,
                                           uint8x8_t q1,
                                           uint8x8_t q2,
                                           uint8x8_t q3,
                                           uint8x8_t q4,
                                           uint8x8_t q5,
                                           uint8x8_t q6,
                                           uint8x8_t q7,
                                           uint8x8_t q8,
                                           uint8x8_t q9,
                                           uint8x8_t q10,
                                           uint8x8_t q11,
                                           uint8x8_t q12,
                                           uint8x8_t q13,
                                           uint8x8_t q14,
                                           uint8x8_t q15,
                                           uint8_t* dst,
                                           int dst_stride) {
  uint8x8_t lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7;
  uint8x8_t hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7;
  Transpose8x8U8Neon(q0, q1, q2, q3, q4, q5, q6, q7,
                     &lo0, &lo1, &lo2, &lo3, &lo4, &lo5, &lo6, &lo7);
  Transpose8x8U8Neon(q8, q9, q10, q11, q12, q13, q14, q15,
                     &hi0, &hi1, &hi2, &hi3, &hi4, &hi5, &hi6, &hi7);

  vst1q_u8(dst + static_cast<size_t>(0) * dst_stride, vcombine_u8(lo0, hi0));
  vst1q_u8(dst + static_cast<size_t>(1) * dst_stride, vcombine_u8(lo1, hi1));
  vst1q_u8(dst + static_cast<size_t>(2) * dst_stride, vcombine_u8(lo2, hi2));
  vst1q_u8(dst + static_cast<size_t>(3) * dst_stride, vcombine_u8(lo3, hi3));
  vst1q_u8(dst + static_cast<size_t>(4) * dst_stride, vcombine_u8(lo4, hi4));
  vst1q_u8(dst + static_cast<size_t>(5) * dst_stride, vcombine_u8(lo5, hi5));
  vst1q_u8(dst + static_cast<size_t>(6) * dst_stride, vcombine_u8(lo6, hi6));
  vst1q_u8(dst + static_cast<size_t>(7) * dst_stride, vcombine_u8(lo7, hi7));
}

template <bool Padded>
static inline void StoreQuantizedSpatial8x16Neon(const float* src_nchw,
                                                 int spatial,
                                                 int c,
                                                 int icb,
                                                 int s,
                                                 float inv_scale,
                                                 uint8_t* dst,
                                                 int dst_stride) {
  uint8x8_t q0, q1, q2, q3, q4, q5, q6, q7;
  uint8x8_t q8, q9, q10, q11, q12, q13, q14, q15;
  if (Padded) {
    q0 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 0, c, s, inv_scale);
    q1 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 1, c, s, inv_scale);
    q2 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 2, c, s, inv_scale);
    q3 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 3, c, s, inv_scale);
    q4 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 4, c, s, inv_scale);
    q5 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 5, c, s, inv_scale);
    q6 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 6, c, s, inv_scale);
    q7 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 7, c, s, inv_scale);
    q8 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 8, c, s, inv_scale);
    q9 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 9, c, s, inv_scale);
    q10 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 10, c, s, inv_scale);
    q11 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 11, c, s, inv_scale);
    q12 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 12, c, s, inv_scale);
    q13 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 13, c, s, inv_scale);
    q14 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 14, c, s, inv_scale);
    q15 = LoadQuantizedChannel8Neon(src_nchw, spatial, icb + 15, c, s, inv_scale);
  } else {
    q0 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 0, s, inv_scale);
    q1 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 1, s, inv_scale);
    q2 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 2, s, inv_scale);
    q3 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 3, s, inv_scale);
    q4 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 4, s, inv_scale);
    q5 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 5, s, inv_scale);
    q6 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 6, s, inv_scale);
    q7 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 7, s, inv_scale);
    q8 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 8, s, inv_scale);
    q9 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 9, s, inv_scale);
    q10 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 10, s, inv_scale);
    q11 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 11, s, inv_scale);
    q12 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 12, s, inv_scale);
    q13 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 13, s, inv_scale);
    q14 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 14, s, inv_scale);
    q15 = LoadQuantizedChannel8NoPadNeon(src_nchw, spatial, icb + 15, s, inv_scale);
  }
  StoreTransposed8x16Neon(q0, q1, q2, q3, q4, q5, q6, q7,
                          q8, q9, q10, q11, q12, q13, q14, q15,
                          dst, dst_stride);
}
#endif

template <bool Padded>
static inline void StoreQuantizedSpatial1x16Scalar(const float* src_nchw,
                                                   int spatial,
                                                   int c,
                                                   int icb,
                                                   int s,
                                                   float inv_scale,
                                                   uint8_t* dst) {
  const int valid_lanes = Padded ? std::min(16, c - icb) : 16;
  for (int lane = 0; lane < valid_lanes; ++lane) {
    const int channel = icb + lane;
    dst[lane] = static_cast<uint8_t>(QuantizeToInt8SymInv(
        src_nchw[static_cast<size_t>(channel) * spatial + s], inv_scale));
  }
  for (int lane = valid_lanes; lane < 16; ++lane) {
    dst[lane] = 0;
  }
}

static inline float FloatFromRawBits(uint32_t raw) {
  float value;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

#if CALIB_CONV2D_HAS_ARMV7_ASM
static inline void StoreFloatGHWC4x4ToNCHWArmv7Asm(const uint32_t* src,
                                                   float* dst0,
                                                   float* dst1,
                                                   float* dst2,
                                                   float* dst3) {
  const uint32_t* in = src;
  asm volatile(
      "vld1.32    {d0-d1}, [%[in]]!               \n"
      "vld1.32    {d2-d3}, [%[in]]!               \n"
      "vld1.32    {d4-d5}, [%[in]]!               \n"
      "vld1.32    {d6-d7}, [%[in]]                \n"
      "vtrn.32    q0, q1                          \n"
      "vtrn.32    q2, q3                          \n"
      "vswp       d1, d4                          \n"
      "vswp       d3, d6                          \n"
      "vst1.32    {d0-d1}, [%[dst0]]              \n"
      "vst1.32    {d2-d3}, [%[dst1]]              \n"
      "vst1.32    {d4-d5}, [%[dst2]]              \n"
      "vst1.32    {d6-d7}, [%[dst3]]              \n"
      : [in] "+r"(in)
      : [dst0] "r"(dst0),
        [dst1] "r"(dst1),
        [dst2] "r"(dst2),
        [dst3] "r"(dst3)
      : "memory", "q0", "q1", "q2", "q3");
}
#endif

static void ConvertFloatGHWC4RawToNCHWFloatArray(const uint32_t* raw,
                                                 float* dst,
                                                 int c,
                                                 int h,
                                                 int w) {
  if (raw == NULL || dst == NULL) return;

  const int groups = (c + 3) >> 2;
  const int spatial = h * w;
  const int spatial_tile = CALIB_CONV2D_OMP_SPATIAL_TILE > 0
                               ? CALIB_CONV2D_OMP_SPATIAL_TILE
                               : 512;
  const int spatial_tiles = (spatial + spatial_tile - 1) / spatial_tile;
#if CALIB_CONV2D_USE_OPENMP
#pragma omp parallel for collapse(2) schedule(static) if (groups * spatial >= CALIB_CONV2D_OMP_MIN_WORK)
#endif
  for (int g = 0; g < groups; ++g) {
    for (int tile = 0; tile < spatial_tiles; ++tile) {
      const uint32_t* src_group = raw + static_cast<size_t>(g) * spatial * 4;
      const int oc0 = (g << 2) + 0;
      const int oc1 = (g << 2) + 1;
      const int oc2 = (g << 2) + 2;
      const int oc3 = (g << 2) + 3;
      float* dst0 = dst + static_cast<size_t>(oc0) * spatial;
      float* dst1 = (oc1 < c) ? dst + static_cast<size_t>(oc1) * spatial : NULL;
      float* dst2 = (oc2 < c) ? dst + static_cast<size_t>(oc2) * spatial : NULL;
      float* dst3 = (oc3 < c) ? dst + static_cast<size_t>(oc3) * spatial : NULL;

      int s = tile * spatial_tile;
      const int s_end = std::min(spatial, s + spatial_tile);
#if CALIB_CONV2D_HAS_NEON
      for (; s + 4 <= s_end; s += 4) {
        const uint32_t* src_s = src_group + static_cast<size_t>(s) * 4;
#if CALIB_CONV2D_HAS_ARMV7_ASM
        if (dst3 != NULL) {
          StoreFloatGHWC4x4ToNCHWArmv7Asm(src_s, dst0 + s, dst1 + s, dst2 + s, dst3 + s);
          continue;
        }
#endif
        const uint32x4x4_t lanes = vld4q_u32(src_s);
        vst1q_f32(dst0 + s, vreinterpretq_f32_u32(lanes.val[0]));
        if (dst1 != NULL) {
          vst1q_f32(dst1 + s, vreinterpretq_f32_u32(lanes.val[1]));
        }
        if (dst2 != NULL) {
          vst1q_f32(dst2 + s, vreinterpretq_f32_u32(lanes.val[2]));
        }
        if (dst3 != NULL) {
          vst1q_f32(dst3 + s, vreinterpretq_f32_u32(lanes.val[3]));
        }
      }
#endif
      for (; s < s_end; ++s) {
        const uint32_t* src_s = src_group + static_cast<size_t>(s) * 4;
        dst0[s] = FloatFromRawBits(src_s[0]);
        if (dst1 != NULL) {
          dst1[s] = FloatFromRawBits(src_s[1]);
        }
        if (dst2 != NULL) {
          dst2[s] = FloatFromRawBits(src_s[2]);
        }
        if (dst3 != NULL) {
          dst3[s] = FloatFromRawBits(src_s[3]);
        }
      }
    }
  }
}

class CmaBlock {
 public:
  CmaBlock() {}
  CmaBlock(const CmaBlock&) = delete;
  CmaBlock& operator=(const CmaBlock&) = delete;

  ~CmaBlock() { Free(); }

  bool Allocate(int fd, size_t size) {
    Free();
    if (fd < 0 || size == 0) {
      FPGA_CALIB_LOG_ERROR("invalid CMA fd or size");
      return false;
    }

    fd_ = fd;
    std::memset(&blk_, 0, sizeof(blk_));
    blk_.size = size;

    if (ioctl(fd_, CMA_IOCTL_MAKE(CMA_CMD_MGET), &blk_) < 0) {
      FPGA_CALIB_LOG_ERROR("ioctl(CMA_CMD_MGET) failed, errno=" << errno << ", " << strerror(errno));
      std::memset(&blk_, 0, sizeof(blk_));
      fd_ = -1;
      return false;
    }

    map_ = mmap(NULL, blk_.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, blk_.phys);
    if (map_ == MAP_FAILED) {
      map_ = NULL;
      FPGA_CALIB_LOG_ERROR("mmap CMA failed, errno=" << errno << ", " << strerror(errno));
      Free();
      return false;
    }

#if CALIB_CONV2D_CLEAR_CMA_ON_ALLOC
    CalibFastZero(map_, blk_.size);
#endif
    return true;
  }

  void Free() {
    if (map_ != NULL) {
      munmap(map_, blk_.size);
      map_ = NULL;
    }
    if (fd_ >= 0 && blk_.size != 0) {
      cma_mblk_s tmp;
      std::memset(&tmp, 0, sizeof(tmp));
      tmp.addr = blk_.addr;
      tmp.virt = blk_.virt;
      tmp.phys = blk_.phys;
      tmp.size = blk_.size;
      if (ioctl(fd_, CMA_IOCTL_MAKE(CMA_CMD_FREE), &tmp) < 0) {
        FPGA_CALIB_LOG_ERROR("ioctl(CMA_CMD_FREE) failed, errno=" << errno << ", " << strerror(errno));
      }
    }
    std::memset(&blk_, 0, sizeof(blk_));
    fd_ = -1;
  }

  uint8_t* virt() { return reinterpret_cast<uint8_t*>(map_); }
  unsigned long phys() const { return blk_.phys; }
  size_t size() const { return blk_.size; }
  bool valid() const { return map_ != NULL && blk_.size != 0; }

 private:
  int fd_{-1};
  cma_mblk_s blk_{};
  void* map_{NULL};
};

class LwAxiRegs {
 public:
  LwAxiRegs() {}
  LwAxiRegs(const LwAxiRegs&) = delete;
  LwAxiRegs& operator=(const LwAxiRegs&) = delete;

  ~LwAxiRegs() { Close(); }

  bool Open() {
    Close();
    fd_ = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_ < 0) {
      FPGA_CALIB_LOG_ERROR("open(/dev/mem) failed, errno=" << errno << ", " << strerror(errno));
      return false;
    }

    void* p = mmap(NULL,
                   CALIB_CONV2D_LW_MAP_SIZE,
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED,
                   fd_,
                   CALIB_CONV2D_IP_BASE_PHYS);
    if (p == MAP_FAILED) {
      FPGA_CALIB_LOG_ERROR("mmap LW AXI failed, errno=" << errno << ", " << strerror(errno));
      close(fd_);
      fd_ = -1;
      return false;
    }

    regs_ = reinterpret_cast<volatile uint32_t*>(p);
    return true;
  }

  void Close() {
    if (regs_ != NULL) {
      munmap(const_cast<uint32_t*>(regs_), CALIB_CONV2D_LW_MAP_SIZE);
      regs_ = NULL;
    }
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

  bool valid() const { return regs_ != NULL; }

  bool Write32(uint32_t off, uint32_t v) {
    if (regs_ == NULL) return false;
    if (off + sizeof(uint32_t) > CALIB_CONV2D_LW_MAP_SIZE) {
      FPGA_CALIB_LOG_ERROR("register offset out of range: 0x" << std::hex << off << std::dec);
      return false;
    }
    regs_[off >> 2] = v;
    return true;
  }

  uint32_t Read32(uint32_t off) const {
    if (regs_ == NULL) return 0;
    if (off + sizeof(uint32_t) > CALIB_CONV2D_LW_MAP_SIZE) return 0;
    return regs_[off >> 2];
  }

  bool WriteAddr(uint32_t off_l, uint32_t off_h, unsigned long phys) {
    const uint64_t phys64 = static_cast<uint64_t>(phys);
    if (!Write32(off_l, static_cast<uint32_t>(phys64 & 0xffffffffULL))) return false;
    if (!Write32(off_h, static_cast<uint32_t>((phys64 >> 32) & 0xffffffffULL))) return false;
    return true;
  }

  bool StartAndWait(uint64_t* start_cmd_ns = NULL, uint64_t* wait_done_ns = NULL) {
    if (regs_ == NULL) return false;
    {
      ScopedCalibTimer timer(start_cmd_ns);
      (void)Read32(REG_CTRL);  // clear stale COR bits if any
      const uint64_t ready_begin_ns = CalibNowNs();
      const uint64_t timeout_ns = 10ULL * 1000ULL * 1000ULL * 1000ULL;
      uint32_t ctrl = Read32(REG_CTRL);
      while ((ctrl & 0x0cu) == 0u) {
        if (CalibNowNs() - ready_begin_ns > timeout_ns) {
          FPGA_CALIB_LOG_ERROR("HLS IP wait ready timeout, ctrl=0x"
                               << std::hex << ctrl << std::dec);
          return false;
        }
        ctrl = Read32(REG_CTRL);
      }
      if (!Write32(REG_CTRL, 0x01u)) return false;
    }
    {
      ScopedCalibTimer timer(wait_done_ns);
      const uint64_t wait_begin_ns = CalibNowNs();
      const uint64_t timeout_ns = 10ULL * 1000ULL * 1000ULL * 1000ULL;
      uint32_t polls_until_time_check = CALIB_CONV2D_WAIT_CLOCK_CHECK_INTERVAL;
      if (polls_until_time_check == 0u) {
        polls_until_time_check = 1u;
      }
      uint32_t polls = 0;
      while ((Read32(REG_CTRL) & 0x02u) == 0u) {
        ++polls;
        if (polls >= polls_until_time_check) {
          polls = 0;
          if (CalibNowNs() - wait_begin_ns > timeout_ns) {
            FPGA_CALIB_LOG_ERROR("HLS IP wait done timeout, ctrl=0x"
                                 << std::hex << Read32(REG_CTRL) << std::dec);
            return false;
          }
        }
      }
    }
    return true;
  }

 private:
  int fd_{-1};
  volatile uint32_t* regs_{NULL};
};

class Arena {
 public:
  Arena(uint8_t* base, unsigned long phys_base) : base_(base), phys_base_(phys_base) {}

  size_t Alloc(size_t bytes, size_t align = 64) {
    offset_ = AlignUp(offset_, align);
    size_t ret = offset_;
    offset_ += bytes;
    return ret;
  }

  uint8_t* Ptr(size_t off) { return base_ + off; }
  unsigned long Phys(size_t off) const { return phys_base_ + off; }
  size_t used() const { return offset_; }

 private:
  uint8_t* base_{NULL};
  unsigned long phys_base_{0};
  size_t offset_{0};
};

static void UnpackConv2dOutGHWC4ToNCHWInt8(const uint32_t* src,
                                           int8_t* dst,
                                           int c,
                                           int h,
                                           int w) {
  if (src == NULL || dst == NULL) return;

  const int groups = (c + 3) >> 2;
  const int spatial = h * w;
#if CALIB_CONV2D_USE_OPENMP
#pragma omp parallel for schedule(static) if (groups * spatial >= CALIB_CONV2D_OMP_MIN_WORK)
#endif
  for (int g = 0; g < groups; ++g) {
    const int oc0 = (g << 2) + 0;
    const int oc1 = (g << 2) + 1;
    const int oc2 = (g << 2) + 2;
    const int oc3 = (g << 2) + 3;
    int8_t* dst0 = dst + static_cast<size_t>(oc0) * spatial;
    int8_t* dst1 = (oc1 < c) ? dst + static_cast<size_t>(oc1) * spatial : NULL;
    int8_t* dst2 = (oc2 < c) ? dst + static_cast<size_t>(oc2) * spatial : NULL;
    int8_t* dst3 = (oc3 < c) ? dst + static_cast<size_t>(oc3) * spatial : NULL;

    int s = 0;
#if CALIB_CONV2D_HAS_NEON
    for (; s + 16 <= spatial; s += 16) {
      uint8x16x4_t lanes;
      if (groups == 1) {
        const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src + s);
        lanes = vld4q_u8(src_bytes);
      } else {
        alignas(16) uint32_t word_buf[16];
        for (int i = 0; i < 16; ++i) {
          word_buf[i] = src[static_cast<size_t>(s + i) * groups + g];
        }
        lanes = vld4q_u8(reinterpret_cast<const uint8_t*>(word_buf));
      }

      vst1q_u8(reinterpret_cast<uint8_t*>(dst0 + s), lanes.val[0]);
      if (dst1 != NULL) {
        vst1q_u8(reinterpret_cast<uint8_t*>(dst1 + s), lanes.val[1]);
      }
      if (dst2 != NULL) {
        vst1q_u8(reinterpret_cast<uint8_t*>(dst2 + s), lanes.val[2]);
      }
      if (dst3 != NULL) {
        vst1q_u8(reinterpret_cast<uint8_t*>(dst3 + s), lanes.val[3]);
      }
    }
#endif
    for (; s < spatial; ++s) {
      const uint32_t word = src[static_cast<size_t>(s) * groups + g];
      dst0[s] = static_cast<int8_t>(word & 0xffu);
      if (dst1 != NULL) dst1[s] = static_cast<int8_t>((word >> 8) & 0xffu);
      if (dst2 != NULL) dst2[s] = static_cast<int8_t>((word >> 16) & 0xffu);
      if (dst3 != NULL) dst3[s] = static_cast<int8_t>((word >> 24) & 0xffu);
    }
  }
}
static void UnpackConv2dOutHWG4ToNCHWInt8SourceMajor(const uint32_t* src,
                                                     int8_t* dst,
                                                     int c,
                                                     int h,
                                                     int w) {
  UnpackConv2dOutGHWC4ToNCHWInt8(src, dst, c, h, w);
}

template <typename T>
static void PackArrayQ10_22x4(const T* src,
                              uint32_t count,
                              uint32_t* dst_words,
                              float extra_mul = 1.f) {
  const uint32_t groups = (count + 3) >> 2;
  for (uint32_t g = 0; g < groups; ++g) {
    for (uint32_t lane = 0; lane < 4; ++lane) {
      const uint32_t idx = (g << 2) + lane;
      const float v = (idx < count && src != NULL) ? static_cast<float>(src[idx]) * extra_mul : 0.f;
      dst_words[g * 4 + lane] = FixedRawFromFloat(v);
    }
  }
}

static void PackScaleArrayQ10_22x4(const std::vector<float>& scale,
                                   uint32_t count,
                                   uint32_t* dst_words,
                                   float extra_mul) {
  const uint32_t groups = (count + 3) >> 2;
  for (uint32_t g = 0; g < groups; ++g) {
    for (uint32_t lane = 0; lane < 4; ++lane) {
      const uint32_t idx = (g << 2) + lane;
      const float s = idx < count ? scale[idx] * extra_mul : 0.f;
      dst_words[g * 4 + lane] = FixedRawFromFloat(s);
    }
  }
}

static void PackConv1x1WeightsOC_ICB16(const int8_t* src_oc_ic_1_1,
                                       int oc,
                                       int ic,
                                       int ic_aligned,
                                       uint8_t* dst) {
  const int ic_blocks = ic_aligned >> 4;
  CalibFastZero(dst, static_cast<size_t>(oc) * ic_blocks * 16);
  for (int o = 0; o < oc; ++o) {
    for (int i = 0; i < ic; ++i) {
      const int icb = i >> 4;
      const int lane = i & 15;
      dst[(o * ic_blocks + icb) * 16 + lane] =
          static_cast<uint8_t>(src_oc_ic_1_1[o * ic + i]);
    }
  }
}

static void PackDepthwiseWeights(const int8_t* src_oc_1_kh_kw,
                                 int oc,
                                 int kh,
                                 int kw,
                                 uint8_t* dst) {
  const int words_per_oc = (kh == 5 || kw == 5) ? 2 : 1;
  CalibFastZero(dst, static_cast<size_t>(oc) * words_per_oc * 16);
  for (int o = 0; o < oc; ++o) {
    for (int r = 0; r < kh; ++r) {
      for (int c = 0; c < kw; ++c) {
        const int kidx = r * kw + c;
        dst[(o * words_per_oc * 16) + kidx] =
            static_cast<uint8_t>(src_oc_1_kh_kw[(o * kh + r) * kw + c]);
      }
    }
  }
}

template <typename T>
static void QuantizeNCHWToNHWCC16(const T* src_nchw,
                                  int c,
                                  int h,
                                  int w,
                                  int c_aligned,
                                  float calib_scale,
                                  uint8_t* dst_nhwc) {
  const int spatial = h * w;
  const int c_blocks = c_aligned >> 4;
  const float inv_scale = 1.f / calib_scale;
#if CALIB_CONV2D_CLEAR_INPUT_PACK
  CalibFastZero(dst_nhwc, static_cast<size_t>(spatial) * c_aligned);
#endif
  const int spatial_tile = CALIB_CONV2D_OMP_SPATIAL_TILE > 0
                               ? CALIB_CONV2D_OMP_SPATIAL_TILE
                               : 512;
  const int spatial_tiles = (spatial + spatial_tile - 1) / spatial_tile;
#if CALIB_CONV2D_USE_OPENMP
#pragma omp parallel for collapse(2) schedule(static) if (c_blocks * spatial >= CALIB_CONV2D_OMP_MIN_WORK)
#endif
  for (int icb_idx = 0; icb_idx < c_blocks; ++icb_idx) {
    for (int tile = 0; tile < spatial_tiles; ++tile) {
      const int icb = icb_idx << 4;
      const bool padded = icb + 16 > c;
      const int valid_lanes = padded ? std::min(16, c - icb) : 16;
      int s = tile * spatial_tile;
      const int s_end = std::min(spatial, s + spatial_tile);
      for (; s < s_end; ++s) {
        uint8_t* dst = dst_nhwc + static_cast<size_t>(s) * c_aligned + icb;
        for (int lane = 0; lane < valid_lanes; ++lane) {
          const int channel = icb + lane;
          dst[lane] = static_cast<uint8_t>(QuantizeToInt8SymInv(
              static_cast<float>(src_nchw[static_cast<size_t>(channel) * spatial + s]),
              inv_scale));
        }
        for (int lane = valid_lanes; lane < 16; ++lane) {
          dst[lane] = 0;
        }
      }
    }
  }
}

static void QuantizeNCHWToNHWCC16Float(const float* src_nchw,
                                       int c,
                                       int h,
                                       int w,
                                       int c_aligned,
                                       float calib_scale,
                                       uint8_t* dst_nhwc) {
  if (src_nchw == NULL || dst_nhwc == NULL) return;
  const int spatial = h * w;
#if CALIB_CONV2D_CLEAR_INPUT_PACK
  CalibFastZero(dst_nhwc, static_cast<size_t>(spatial) * c_aligned);
#endif

#if CALIB_CONV2D_HAS_NEON
  const float inv_scale = 1.f / calib_scale;
  const int spatial_tile = CALIB_CONV2D_OMP_SPATIAL_TILE > 0
                               ? CALIB_CONV2D_OMP_SPATIAL_TILE
                               : 512;
  const int spatial_tiles = (spatial + spatial_tile - 1) / spatial_tile;

  const int c_blocks = c_aligned >> 4;
#if CALIB_CONV2D_USE_OPENMP
#pragma omp parallel for collapse(2) schedule(static) if (c_blocks * spatial >= CALIB_CONV2D_OMP_MIN_WORK)
#endif
  for (int icb_idx = 0; icb_idx < c_blocks; ++icb_idx) {
    for (int tile = 0; tile < spatial_tiles; ++tile) {
      const int icb = icb_idx << 4;
      const bool padded = icb + 16 > c;
      int s = tile * spatial_tile;
      const int s_end = std::min(spatial, s + spatial_tile);
      if (padded) {
        for (; s + 8 <= s_end; s += 8) {
          StoreQuantizedSpatial8x16Neon<true>(
              src_nchw,
              spatial,
              c,
              icb,
              s,
              inv_scale,
              dst_nhwc + static_cast<size_t>(s) * c_aligned + icb,
              c_aligned);
        }
        for (; s < s_end; ++s) {
          uint8_t* dst = dst_nhwc + static_cast<size_t>(s) * c_aligned + icb;
          StoreQuantizedSpatial1x16Scalar<true>(
              src_nchw, spatial, c, icb, s, inv_scale, dst);
        }
      } else {
        for (; s + 8 <= s_end; s += 8) {
          StoreQuantizedSpatial8x16Neon<false>(
              src_nchw,
              spatial,
              c,
              icb,
              s,
              inv_scale,
              dst_nhwc + static_cast<size_t>(s) * c_aligned + icb,
              c_aligned);
        }
        for (; s < s_end; ++s) {
          uint8_t* dst = dst_nhwc + static_cast<size_t>(s) * c_aligned + icb;
          StoreQuantizedSpatial1x16Scalar<false>(
              src_nchw, spatial, c, icb, s, inv_scale, dst);
        }
      }
    }
  }
#else
  QuantizeNCHWToNHWCC16<float>(src_nchw, c, h, w, c_aligned, calib_scale, dst_nhwc);
#endif
}

static bool AllEqualPadding(const std::vector<int>& paddings, int* pad_out) {
  if (pad_out == NULL) return false;
  if (paddings.empty()) {
    *pad_out = 0;
    return true;
  }
  const int p = paddings[0];
  for (size_t i = 0; i < paddings.size(); ++i) {
    if (paddings[i] != p) return false;
  }
  *pad_out = p;
  return true;
}

// ============================================================================
// 3. Paddle kernel
// ============================================================================
template <>
void CalibConv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::ReleasePersistentCma() {
  if (cma_runtime_.map != NULL) {
    munmap(cma_runtime_.map, cma_runtime_.size);
  }

  if (cma_runtime_.fd >= 0 && cma_runtime_.size != 0) {
    cma_mblk_s tmp;
    std::memset(&tmp, 0, sizeof(tmp));
    tmp.addr = cma_runtime_.addr;
    tmp.virt = cma_runtime_.virt;
    tmp.phys = cma_runtime_.phys;
    tmp.size = cma_runtime_.size;
    if (ioctl(cma_runtime_.fd, CMA_IOCTL_MAKE(CMA_CMD_FREE), &tmp) < 0) {
      FPGA_CALIB_LOG_ERROR("persistent CMA free failed, errno=" << errno << ", " << strerror(errno));
    }
  }

  if (cma_runtime_.fd >= 0) {
    close(cma_runtime_.fd);
  }

  cma_runtime_.ClearMeta();
  reg_config_.Clear();
}

template <>
CalibConv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::~CalibConv2dCompute() {
  delete regs_;
  regs_ = NULL;
  ReleasePersistentCma();
}

template <>
void CalibConv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  typedef float T;
#if CALIB_CONV2D_ENABLE_VERBOSE_LOG
  const uint64_t prepare_start_ns = CalibNowNs();
#endif
  packed_static_.Clear();
  reg_config_.Clear();

  auto& param = this->template Param<operators::ConvParam>();
  const int8_t* conv_w_q = param.filter ? param.filter->data<int8_t>() : NULL;
  const int8_t* dw_w_q = param.depthwise_filter ? param.depthwise_filter->data<int8_t>() : NULL;
  const T* conv_bias = param.bias ? param.bias->data<T>() : NULL;
  const T* dw_bias = param.depthwise_bias ? param.depthwise_bias->data<T>() : NULL;

  if (param.x == NULL || param.output == NULL ||
      param.filter == NULL || param.depthwise_filter == NULL ||
      conv_w_q == NULL || dw_w_q == NULL) {
    FPGA_CALIB_LOG_ERROR("PrepareForRun null tensor/filter data");
    return;
  }

  auto x_dims = param.x->dims();
  auto conv_w_dims = param.filter->dims();
  auto dw_w_dims = param.depthwise_filter->dims();
  auto out_dims = param.output->dims();

  const int batch = static_cast<int>(x_dims[0]);
  const int in_h = static_cast<int>(x_dims[2]);
  const int in_w = static_cast<int>(x_dims[3]);
  const int conv_oc = static_cast<int>(conv_w_dims[0]);
  const int conv_ic = static_cast<int>(conv_w_dims[1]);
  const int conv_kh = static_cast<int>(conv_w_dims[2]);
  const int conv_kw = static_cast<int>(conv_w_dims[3]);
  const int dw_oc = static_cast<int>(dw_w_dims[0]);
  const int dw_ic_per_group = static_cast<int>(dw_w_dims[1]);
  const int dw_kh = static_cast<int>(dw_w_dims[2]);
  const int dw_kw = static_cast<int>(dw_w_dims[3]);
  const int out_c = static_cast<int>(out_dims[1]);
  const int out_h = static_cast<int>(out_dims[2]);
  const int out_w = static_cast<int>(out_dims[3]);
  const int stride_h = param.strides.empty() ? 0 : param.strides[0];
  int prep_pad = 0;
  if (param.paddings != NULL) {
    (void)AllEqualPadding(*param.paddings, &prep_pad);
  }

  if (conv_kh != 1 || conv_kw != 1) {
    FPGA_CALIB_LOG_ERROR("PrepareForRun supports only 1x1 first conv");
    return;
  }
  if (dw_ic_per_group != 1 || !((dw_kh == 3 && dw_kw == 3) || (dw_kh == 5 && dw_kw == 5))) {
    FPGA_CALIB_LOG_ERROR("PrepareForRun supports only depthwise 3x3/5x5 with IC_PER_GROUP=1");
    return;
  }
  if (conv_oc > MAX_PREPACK_CONV_OC || conv_ic > MAX_PREPACK_CONV_IC ||
      dw_oc > MAX_PREPACK_DW_OC || dw_kh > MAX_PREPACK_DW_KH ||
      dw_kw > MAX_PREPACK_DW_KW) {
    FPGA_CALIB_LOG_ERROR("PrepareForRun filter exceeds prepack max size");
    return;
  }
  if (dw_oc != conv_oc) {
    FPGA_CALIB_LOG_ERROR("PrepareForRun requires DW_OC == Conv_OC");
    return;
  }

  const std::vector<float>& conv_filter_scale = param.Conv2d_Filter0_scale;
  const std::vector<float>& dw_filter_scale = param.Depthwise2d_Filter0_scale;
  if (static_cast<int>(conv_filter_scale.size()) < conv_oc ||
      static_cast<int>(dw_filter_scale.size()) < dw_oc) {
    FPGA_CALIB_LOG_ERROR("PrepareForRun scale array is shorter than output channels");
    return;
  }

  const int in_c_aligned = RoundUp16(conv_ic);
  const int in_c_blocks = in_c_aligned >> 4;
  const size_t channel_groups = (static_cast<size_t>(conv_oc) + 3) >> 2;
  const size_t conv_w_bytes = static_cast<size_t>(conv_oc) * in_c_blocks * 16;
  const size_t dw_words_per_oc = (dw_kh == 5) ? 2 : 1;
  const size_t dw_w_bytes = static_cast<size_t>(dw_oc) * dw_words_per_oc * 16;
  const size_t bias_conv_words = channel_groups * 4;
  const size_t bias_dw_words = channel_groups * 4;
  const size_t conv_scale_words = channel_groups * 4;
  const size_t dw_scale_words = channel_groups * 4;

  packed_static_.input_channels = conv_ic;
  packed_static_.input_channels_aligned = in_c_aligned;
  packed_static_.input_channel_blocks = in_c_blocks;
  packed_static_.conv_out_channels = conv_oc;
  packed_static_.depthwise_out_channels = dw_oc;
  packed_static_.depthwise_kernel_h = dw_kh;
  packed_static_.depthwise_kernel_w = dw_kw;
  packed_static_.calib_scale = param.calib_scale;
  packed_static_.depthwise_scale_value = param.depthwise_scale;
  packed_static_.conv_weight_bytes = conv_w_bytes;
  packed_static_.depthwise_weight_bytes = dw_w_bytes;
  packed_static_.bias_conv_bytes = bias_conv_words * sizeof(uint32_t);
  packed_static_.bias_depthwise_bytes = bias_dw_words * sizeof(uint32_t);
  packed_static_.conv_scale_bytes = conv_scale_words * sizeof(uint32_t);
  packed_static_.depthwise_scale_bytes = dw_scale_words * sizeof(uint32_t);

  packed_static_.conv_weight.assign(conv_w_bytes, 0);
  packed_static_.depthwise_weight.assign(dw_w_bytes, 0);
  packed_static_.bias_conv.assign(bias_conv_words, 0);
  packed_static_.bias_depthwise.assign(bias_dw_words, 0);
  packed_static_.conv_scale.assign(conv_scale_words, 0);
  packed_static_.depthwise_scale.assign(dw_scale_words, 0);

  PackConv1x1WeightsOC_ICB16(
      conv_w_q, conv_oc, conv_ic, in_c_aligned, packed_static_.conv_weight.data());
  PackDepthwiseWeights(dw_w_q, dw_oc, dw_kh, dw_kw, packed_static_.depthwise_weight.data());

  PackArrayQ10_22x4<T>(
      conv_bias, static_cast<uint32_t>(conv_oc), packed_static_.bias_conv.data(), 1.f);
  PackArrayQ10_22x4<T>(
      dw_bias, static_cast<uint32_t>(dw_oc), packed_static_.bias_depthwise.data(), 1.f);
  PackScaleArrayQ10_22x4(
      conv_filter_scale, static_cast<uint32_t>(conv_oc), packed_static_.conv_scale.data(), param.calib_scale);
  PackScaleArrayQ10_22x4(
      dw_filter_scale, static_cast<uint32_t>(dw_oc), packed_static_.depthwise_scale.data(), param.depthwise_scale);

  packed_static_.valid = true;

  const size_t input_bytes = static_cast<size_t>(in_h) * in_w * in_c_aligned;
  const size_t output_fixed_words =
      static_cast<size_t>((out_c + 3) >> 2) * out_h * out_w * 4;
  const size_t output_fixed_bytes = output_fixed_words * sizeof(uint32_t);
  const size_t conv2d_out_count = static_cast<size_t>(conv_oc) * in_h * in_w;
  const size_t conv2d_out_groups = static_cast<size_t>((conv_oc + 3) >> 2);
  const size_t conv2d_out_words = conv2d_out_groups * in_h * in_w;
  const size_t conv2d_out_bytes = conv2d_out_words * sizeof(uint32_t);
  const size_t persist_feature_elems =
      std::max(MAX_PERSIST_FEATURE_ELEMS, MAX_PERSIST_FEATURE_HINT_ELEMS);
  const size_t input_capacity_bytes = std::max(input_bytes, persist_feature_elems);
  const size_t output_fixed_capacity_bytes =
      std::max(output_fixed_bytes, persist_feature_elems * sizeof(uint32_t));
  const size_t conv2d_out_capacity_bytes = conv2d_out_bytes;

  size_t total_bytes = 0;
  total_bytes = AlignUp(total_bytes, 64) + AlignUp(input_capacity_bytes, 64);
  total_bytes = AlignUp(total_bytes, 64) + AlignUp(packed_static_.conv_weight_bytes, 64);
  total_bytes = AlignUp(total_bytes, 64) + AlignUp(packed_static_.depthwise_weight_bytes, 64);
  total_bytes = AlignUp(total_bytes, 64) + AlignUp(packed_static_.bias_conv_bytes, 64);
  total_bytes = AlignUp(total_bytes, 64) + AlignUp(output_fixed_capacity_bytes, 64);
  total_bytes = AlignUp(total_bytes, 64) + AlignUp(packed_static_.bias_depthwise_bytes, 64);
  total_bytes = AlignUp(total_bytes, 64) + AlignUp(conv2d_out_capacity_bytes, 64);
  total_bytes = AlignUp(total_bytes, 64) + AlignUp(packed_static_.conv_scale_bytes, 64);
  total_bytes = AlignUp(total_bytes, 64) + AlignUp(packed_static_.depthwise_scale_bytes, 64);

  const bool need_new_cma =
      !cma_runtime_.valid ||
      cma_runtime_.size < total_bytes + 4096 ||
      cma_runtime_.input_capacity_bytes < input_capacity_bytes ||
      cma_runtime_.output_fixed_capacity_bytes < output_fixed_capacity_bytes ||
      cma_runtime_.conv2d_out_capacity_bytes < conv2d_out_capacity_bytes;

  if (need_new_cma) {
    ReleasePersistentCma();

    cma_runtime_.fd = open(CALIB_CONV2D_CMA_DEV, O_RDWR);
    if (cma_runtime_.fd < 0) {
      FPGA_CALIB_LOG_ERROR("PrepareForRun open CMA device failed, errno=" << errno << ", " << strerror(errno));
      cma_runtime_.ClearMeta();
      return;
    }

    cma_mblk_s blk;
    std::memset(&blk, 0, sizeof(blk));
    blk.size = total_bytes + 4096;
    if (ioctl(cma_runtime_.fd, CMA_IOCTL_MAKE(CMA_CMD_MGET), &blk) < 0) {
      FPGA_CALIB_LOG_ERROR("PrepareForRun CMA alloc failed, errno=" << errno << ", " << strerror(errno));
      close(cma_runtime_.fd);
      cma_runtime_.ClearMeta();
      return;
    }

    void* map = mmap(NULL, blk.size, PROT_READ | PROT_WRITE, MAP_SHARED, cma_runtime_.fd, blk.phys);
    if (map == MAP_FAILED) {
      FPGA_CALIB_LOG_ERROR("PrepareForRun CMA mmap failed, errno=" << errno << ", " << strerror(errno));
      cma_mblk_s tmp;
      std::memset(&tmp, 0, sizeof(tmp));
      tmp.addr = blk.addr;
      tmp.virt = blk.virt;
      tmp.phys = blk.phys;
      tmp.size = blk.size;
      (void)ioctl(cma_runtime_.fd, CMA_IOCTL_MAKE(CMA_CMD_FREE), &tmp);
      close(cma_runtime_.fd);
      cma_runtime_.ClearMeta();
      return;
    }

    cma_runtime_.map = map;
    cma_runtime_.addr = blk.addr;
    cma_runtime_.virt = blk.virt;
    cma_runtime_.phys = blk.phys;
    cma_runtime_.size = blk.size;
  }

  Arena arena(reinterpret_cast<uint8_t*>(cma_runtime_.map), cma_runtime_.phys);
  cma_runtime_.off_input = arena.Alloc(input_capacity_bytes);
  cma_runtime_.off_conv_w = arena.Alloc(packed_static_.conv_weight_bytes);
  cma_runtime_.off_dw_w = arena.Alloc(packed_static_.depthwise_weight_bytes);
  cma_runtime_.off_bias_conv = arena.Alloc(packed_static_.bias_conv_bytes);
  cma_runtime_.off_output_fixed = arena.Alloc(output_fixed_capacity_bytes);
  cma_runtime_.off_bias_dw = arena.Alloc(packed_static_.bias_depthwise_bytes);
  cma_runtime_.off_output_conv2d = arena.Alloc(conv2d_out_capacity_bytes);
  cma_runtime_.off_conv_scale = arena.Alloc(packed_static_.conv_scale_bytes);
  cma_runtime_.off_dw_scale = arena.Alloc(packed_static_.depthwise_scale_bytes);

  if (arena.used() > cma_runtime_.size) {
    FPGA_CALIB_LOG_ERROR("PrepareForRun persistent CMA arena overflow");
    ReleasePersistentCma();
    return;
  }

  cma_runtime_.batch = batch;
  cma_runtime_.input_channels = conv_ic;
  cma_runtime_.input_h = in_h;
  cma_runtime_.input_w = in_w;
  cma_runtime_.input_channels_aligned = in_c_aligned;
  cma_runtime_.conv_out_channels = conv_oc;
  cma_runtime_.output_channels = out_c;
  cma_runtime_.output_h = out_h;
  cma_runtime_.output_w = out_w;
  cma_runtime_.enable_conv2d_out = false;
  cma_runtime_.input_bytes = input_bytes;
  cma_runtime_.input_capacity_bytes = input_capacity_bytes;
  cma_runtime_.conv_weight_bytes = packed_static_.conv_weight_bytes;
  cma_runtime_.depthwise_weight_bytes = packed_static_.depthwise_weight_bytes;
  cma_runtime_.bias_conv_bytes = packed_static_.bias_conv_bytes;
  cma_runtime_.bias_depthwise_bytes = packed_static_.bias_depthwise_bytes;
  cma_runtime_.conv_scale_bytes = packed_static_.conv_scale_bytes;
  cma_runtime_.depthwise_scale_bytes = packed_static_.depthwise_scale_bytes;
  cma_runtime_.output_fixed_bytes = output_fixed_bytes;
  cma_runtime_.output_fixed_capacity_bytes = output_fixed_capacity_bytes;
  cma_runtime_.conv2d_out_bytes = conv2d_out_bytes;
  cma_runtime_.conv2d_out_capacity_bytes = conv2d_out_capacity_bytes;
  cma_runtime_.total_bytes = total_bytes;
  cma_runtime_.used_bytes = arena.used();

#if CALIB_CONV2D_USE_CACHED_INPUT_STAGING && !CALIB_CONV2D_PACK_INPUT_DIRECT_TO_CMA
  if (input_pack_cache_bytes_ < input_capacity_bytes) {
    input_pack_cache_.resize(input_capacity_bytes);
    input_pack_cache_bytes_ = input_pack_cache_.size();
  }
#endif
#if CALIB_CONV2D_USE_CACHED_OUTPUT_STAGING
  const size_t output_fixed_capacity_words =
      output_fixed_capacity_bytes / sizeof(uint32_t);
  if (output_fixed_cache_words_ < output_fixed_capacity_words) {
    output_fixed_cache_.resize(output_fixed_capacity_words);
    output_fixed_cache_words_ = output_fixed_cache_.size();
  }
#endif
#if CALIB_CONV2D_USE_CACHED_CONV2D_OUT_STAGING
  const bool prepare_enable_conv2d_out = param.need_conv2d_output && param.conv2d_output;
  if (prepare_enable_conv2d_out) {
    const size_t conv2d_out_capacity_words =
        conv2d_out_capacity_bytes / sizeof(uint32_t);
    if (conv2d_out_cache_words_ < conv2d_out_capacity_words) {
      conv2d_out_cache_.resize(conv2d_out_capacity_words);
      conv2d_out_cache_words_ = conv2d_out_cache_.size();
    }
  }
#endif

  uint8_t* cma_base = reinterpret_cast<uint8_t*>(cma_runtime_.map);
  CalibFastCopy(cma_base + cma_runtime_.off_conv_w, packed_static_.conv_weight.data(), packed_static_.conv_weight_bytes);
  CalibFastCopy(cma_base + cma_runtime_.off_dw_w, packed_static_.depthwise_weight.data(), packed_static_.depthwise_weight_bytes);
  CalibFastCopy(cma_base + cma_runtime_.off_bias_conv, packed_static_.bias_conv.data(), packed_static_.bias_conv_bytes);
  CalibFastCopy(cma_base + cma_runtime_.off_bias_dw, packed_static_.bias_depthwise.data(), packed_static_.bias_depthwise_bytes);
  CalibFastCopy(cma_base + cma_runtime_.off_conv_scale, packed_static_.conv_scale.data(), packed_static_.conv_scale_bytes);
  CalibFastCopy(cma_base + cma_runtime_.off_dw_scale, packed_static_.depthwise_scale.data(), packed_static_.depthwise_scale_bytes);
  CalibSyncInputForFpga(cma_base + cma_runtime_.off_conv_w, packed_static_.conv_weight_bytes);
  CalibSyncInputForFpga(cma_base + cma_runtime_.off_dw_w, packed_static_.depthwise_weight_bytes);
  CalibSyncInputForFpga(cma_base + cma_runtime_.off_bias_conv, packed_static_.bias_conv_bytes);
  CalibSyncInputForFpga(cma_base + cma_runtime_.off_bias_dw, packed_static_.bias_depthwise_bytes);
  CalibSyncInputForFpga(cma_base + cma_runtime_.off_conv_scale, packed_static_.conv_scale_bytes);
  CalibSyncInputForFpga(cma_base + cma_runtime_.off_dw_scale, packed_static_.depthwise_scale_bytes);

  cma_runtime_.valid = true;

#if CALIB_CONV2D_ENABLE_VERBOSE_LOG
  std::cout << "[IntelFPGA] calib_conv2d prepacked static params: conv_w="
            << packed_static_.conv_weight_bytes << " bytes, dw_w="
            << packed_static_.depthwise_weight_bytes << " bytes, cma="
            << cma_runtime_.size << " bytes, shape=N" << batch
            << " IC" << conv_ic << " IH" << in_h << " IW" << in_w
            << " OC" << conv_oc << " OH" << out_h << " OW" << out_w
            << " DWK" << dw_kh << " S" << stride_h << " P" << prep_pad
            << ", time="
            << CalibNsToMs(CalibNowNs() - prepare_start_ns) << " ms" << std::endl;
#endif
}

template <>
void CalibConv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  typedef float T;
#if CALIB_CONV2D_ENABLE_TIMING_LOG
  const uint64_t run_start_ns = CalibNowNs();
#endif
  CalibConv2dTiming timing;

#if CALIB_CONV2D_ENABLE_TIMING_LOG
  uint64_t stage_start_ns = CalibNowNs();
#endif
  auto& param = this->template Param<operators::ConvParam>();

  const T* x_data = param.x->data<T>();
  T* out_data = param.output->mutable_data<T>();
  const int8_t* conv_w_q = param.filter ? param.filter->data<int8_t>() : NULL;
  const T* conv_bias = param.bias ? param.bias->data<T>() : NULL;

  auto x_dims = param.x->dims();
  auto conv_w_dims = param.filter->dims();
  auto dw_w_dims = param.depthwise_filter->dims();
  auto out_dims = param.output->dims();
#if CALIB_CONV2D_ENABLE_TIMING_LOG
  timing.tensor_access_ns += CalibNowNs() - stage_start_ns;

  stage_start_ns = CalibNowNs();
#endif
  const int batch = static_cast<int>(x_dims[0]);
  const int in_c = static_cast<int>(x_dims[1]);
  const int in_h = static_cast<int>(x_dims[2]);
  const int in_w = static_cast<int>(x_dims[3]);

  const int conv_oc = static_cast<int>(conv_w_dims[0]);
  const int conv_ic = static_cast<int>(conv_w_dims[1]);
  const int conv_kh = static_cast<int>(conv_w_dims[2]);
  const int conv_kw = static_cast<int>(conv_w_dims[3]);

  const int dw_oc = static_cast<int>(dw_w_dims[0]);
  const int dw_ic_per_group = static_cast<int>(dw_w_dims[1]);
  const int dw_kh = static_cast<int>(dw_w_dims[2]);
  const int dw_kw = static_cast<int>(dw_w_dims[3]);

  const int out_c = static_cast<int>(out_dims[1]);
  const int out_h = static_cast<int>(out_dims[2]);
  const int out_w = static_cast<int>(out_dims[3]);

  if (x_data == NULL || out_data == NULL || param.filter == NULL ||
      param.depthwise_filter == NULL || conv_w_q == NULL) {
    FPGA_CALIB_LOG_ERROR("null input/filter/output tensor data");
    return;
  }
  if (in_h != in_w) {
    FPGA_CALIB_LOG_ERROR("HLS calib_conv2d currently requires square input H == W");
    return;
  }
  if (conv_kh != 1 || conv_kw != 1) {
    FPGA_CALIB_LOG_ERROR("HLS calib_conv2d currently supports only 1x1 first conv");
    return;
  }
  if (conv_ic != in_c) {
    FPGA_CALIB_LOG_ERROR("conv filter IC does not match input C");
    return;
  }
  if (dw_oc != conv_oc || out_c != dw_oc || dw_ic_per_group != 1) {
    FPGA_CALIB_LOG_ERROR("HLS depthwise path requires DW_OC == Conv_OC == Output_C and IC_PER_GROUP == 1");
    return;
  }
  if (!((dw_kh == 3 && dw_kw == 3) || (dw_kh == 5 && dw_kw == 5))) {
    FPGA_CALIB_LOG_ERROR("HLS depthwise path supports only 3x3 or 5x5 kernels");
    return;
  }
  if (param.strides[0] != param.strides[1]) {
    FPGA_CALIB_LOG_ERROR("HLS depthwise path requires stride_h == stride_w");
    return;
  }
  if (param.paddings == NULL) {
    FPGA_CALIB_LOG_ERROR("paddings is null");
    return;
  }

  int pad = 0;
  if (!AllEqualPadding(*param.paddings, &pad)) {
    FPGA_CALIB_LOG_ERROR("HLS depthwise path requires symmetric equal padding");
    return;
  }

  const int stride = param.strides[0];
  const int expected_out = (in_h + 2 * pad - dw_kh) / stride + 1;
  if (expected_out != out_h || expected_out != out_w) {
    FPGA_CALIB_LOG_ERROR("HLS output shape mismatch with depthwise params");
    return;
  }

  const float calib_scale = param.calib_scale;
  const float depthwise_scale = param.depthwise_scale;
  if (calib_scale == 0.f || depthwise_scale == 0.f) {
    FPGA_CALIB_LOG_ERROR("calib_scale/depthwise_scale must be non-zero");
    return;
  }

  const std::vector<float>& conv_filter_scale = param.Conv2d_Filter0_scale;
  const std::vector<float>& dw_filter_scale = param.Depthwise2d_Filter0_scale;
  if (static_cast<int>(conv_filter_scale.size()) < conv_oc ||
      static_cast<int>(dw_filter_scale.size()) < dw_oc) {
    FPGA_CALIB_LOG_ERROR("per-channel scale array is shorter than output channels");
    return;
  }

  const int in_c_aligned = RoundUp16(in_c);
  const int in_c_blocks = in_c_aligned >> 4;
  if (in_c_blocks <= 0 || in_c_blocks > HLS_MAX_INPUT_C_BLOCKS) {
    FPGA_CALIB_LOG_ERROR("input channel blocks exceed HLS streaming conv capacity");
    return;
  }

  if (!packed_static_.valid ||
      packed_static_.input_channels != in_c ||
      packed_static_.input_channels_aligned != in_c_aligned ||
      packed_static_.input_channel_blocks != in_c_blocks ||
      packed_static_.conv_out_channels != conv_oc ||
      packed_static_.depthwise_out_channels != dw_oc ||
      packed_static_.depthwise_kernel_h != dw_kh ||
      packed_static_.depthwise_kernel_w != dw_kw ||
      packed_static_.calib_scale != calib_scale ||
      packed_static_.depthwise_scale_value != depthwise_scale) {
    PrepareForRun();
  }
  if (!packed_static_.valid) {
    FPGA_CALIB_LOG_ERROR("static packed params are not ready");
    return;
  }

  int local_groups_per_round = HLS_MAX_WEIGHT_ROWS / in_c_blocks;
  if (local_groups_per_round > HLS_MAX_LOCAL_GROUPS) {
    local_groups_per_round = HLS_MAX_LOCAL_GROUPS;
  }
  if (local_groups_per_round > 255) local_groups_per_round = 255;
  if (local_groups_per_round <= 0) {
    FPGA_CALIB_LOG_ERROR("local_groups_per_round is zero");
    return;
  }
  const int local_channels_per_round = local_groups_per_round * 4;
  int rounds = (conv_oc + local_channels_per_round - 1) / local_channels_per_round;
  if (rounds > 255) {
    FPGA_CALIB_LOG_ERROR("rounds exceeds uint8 register range");
    return;
  }

  const bool enable_conv2d_out = param.need_conv2d_output && param.conv2d_output;
  int8_t* conv2d_out_tensor = NULL;
  if (enable_conv2d_out) {
    conv2d_out_tensor = param.conv2d_output->mutable_data<int8_t>();
    if (conv2d_out_tensor == NULL) {
      FPGA_CALIB_LOG_ERROR("conv2d_output mutable data is null");
      return;
    }
  }

  const size_t input_bytes = static_cast<size_t>(in_h) * in_w * in_c_aligned;
  const size_t output_fixed_words =
      static_cast<size_t>((out_c + 3) >> 2) * out_h * out_w * 4;
  const size_t output_fixed_bytes = output_fixed_words * sizeof(uint32_t);
  const size_t conv2d_out_count = static_cast<size_t>(conv_oc) * in_h * in_w;
  const size_t conv2d_out_groups = static_cast<size_t>((conv_oc + 3) >> 2);
  const size_t conv2d_out_words = conv2d_out_groups * in_h * in_w;
  const size_t conv2d_out_bytes = conv2d_out_words * sizeof(uint32_t);

  if (!cma_runtime_.valid ||
      cma_runtime_.map == NULL ||
      cma_runtime_.input_capacity_bytes < input_bytes ||
      cma_runtime_.output_fixed_capacity_bytes < output_fixed_bytes ||
      cma_runtime_.conv2d_out_capacity_bytes < conv2d_out_bytes ||
      cma_runtime_.conv_weight_bytes != packed_static_.conv_weight_bytes ||
      cma_runtime_.depthwise_weight_bytes != packed_static_.depthwise_weight_bytes ||
      cma_runtime_.bias_conv_bytes != packed_static_.bias_conv_bytes ||
      cma_runtime_.bias_depthwise_bytes != packed_static_.bias_depthwise_bytes ||
      cma_runtime_.conv_scale_bytes != packed_static_.conv_scale_bytes ||
      cma_runtime_.depthwise_scale_bytes != packed_static_.depthwise_scale_bytes) {
    PrepareForRun();
  }
  if (!cma_runtime_.valid || cma_runtime_.map == NULL) {
    FPGA_CALIB_LOG_ERROR("persistent CMA is not ready");
    return;
  }
  cma_runtime_.batch = batch;
  cma_runtime_.input_channels = in_c;
  cma_runtime_.input_h = in_h;
  cma_runtime_.input_w = in_w;
  cma_runtime_.input_channels_aligned = in_c_aligned;
  cma_runtime_.conv_out_channels = conv_oc;
  cma_runtime_.output_channels = out_c;
  cma_runtime_.output_h = out_h;
  cma_runtime_.output_w = out_w;
  cma_runtime_.enable_conv2d_out = enable_conv2d_out;
  cma_runtime_.input_bytes = input_bytes;
  cma_runtime_.output_fixed_bytes = output_fixed_bytes;
  cma_runtime_.conv2d_out_bytes = conv2d_out_bytes;
  uint8_t* cma_base = reinterpret_cast<uint8_t*>(cma_runtime_.map);
#if CALIB_CONV2D_ENABLE_TIMING_LOG
  timing.shape_check_ns += CalibNowNs() - stage_start_ns;
#endif

  if (!regs_) {
    regs_ = new LwAxiRegs();
  }
  LwAxiRegs& regs = *regs_;
  if (!regs.valid()) {
    ScopedCalibTimer timer(&timing.regs_open_ns);
    if (!regs.Open()) {
      return;
    }
    reg_config_.Clear();
  }

  HlsRegConfig next_reg;
  next_reg.valid = true;
  next_reg.input_phys = cma_runtime_.phys + cma_runtime_.off_input;
  next_reg.weight_conv_phys = cma_runtime_.phys + cma_runtime_.off_conv_w;
  next_reg.weight_dw_phys = cma_runtime_.phys + cma_runtime_.off_dw_w;
  next_reg.bias_conv_phys = cma_runtime_.phys + cma_runtime_.off_bias_conv;
  next_reg.output_fixed_phys = cma_runtime_.phys + cma_runtime_.off_output_fixed;
  next_reg.bias_dw_phys = cma_runtime_.phys + cma_runtime_.off_bias_dw;
  next_reg.output_conv2d_phys = cma_runtime_.phys + cma_runtime_.off_output_conv2d;
  next_reg.conv_scale_phys = cma_runtime_.phys + cma_runtime_.off_conv_scale;
  next_reg.dw_scale_phys = cma_runtime_.phys + cma_runtime_.off_dw_scale;
  next_reg.inv_dw_scale = FixedRawFromFloat(1.f / depthwise_scale);
  next_reg.input_hw = static_cast<uint32_t>(in_h);
  next_reg.input_channels = static_cast<uint32_t>(in_c);
  next_reg.out_channels = static_cast<uint32_t>(conv_oc);
  next_reg.dw_kernel = static_cast<uint32_t>(dw_kh);
  next_reg.dw_stride = static_cast<uint32_t>(stride);
  next_reg.dw_padding = static_cast<uint32_t>(pad);
  next_reg.enable_conv_out = enable_conv2d_out ? 1u : 0u;
  next_reg.local_groups_per_round = static_cast<uint32_t>(local_groups_per_round);
  next_reg.rounds = static_cast<uint32_t>(rounds);

  bool reg_ok = true;
#if CALIB_CONV2D_CACHE_REG_WRITES
  const bool need_reg_write = !reg_config_.SameAs(next_reg);
#else
  const bool need_reg_write = true;
#endif
  if (need_reg_write) {
    ScopedCalibTimer timer(&timing.regs_write_ns);
    reg_ok = reg_ok && regs.WriteAddr(REG_WEIGHT_CONV_L, REG_WEIGHT_CONV_H, next_reg.weight_conv_phys);
    reg_ok = reg_ok && regs.WriteAddr(REG_WEIGHT_DW_L, REG_WEIGHT_DW_H, next_reg.weight_dw_phys);
    reg_ok = reg_ok && regs.WriteAddr(REG_BIAS_CONV_L, REG_BIAS_CONV_H, next_reg.bias_conv_phys);
    reg_ok = reg_ok && regs.WriteAddr(REG_BIAS_DW_L, REG_BIAS_DW_H, next_reg.bias_dw_phys);
    reg_ok = reg_ok && regs.WriteAddr(REG_CONV_SCALE_L, REG_CONV_SCALE_H, next_reg.conv_scale_phys);
    reg_ok = reg_ok && regs.WriteAddr(REG_DW_SCALE_L, REG_DW_SCALE_H, next_reg.dw_scale_phys);

    reg_ok = reg_ok && regs.Write32(REG_INV_DW_SCALE, next_reg.inv_dw_scale);
    reg_ok = reg_ok && regs.Write32(REG_INPUT_HW, next_reg.input_hw);
    reg_ok = reg_ok && regs.Write32(REG_INPUT_CHANNELS, next_reg.input_channels);
    reg_ok = reg_ok && regs.Write32(REG_OUT_CHANNELS, next_reg.out_channels);
    reg_ok = reg_ok && regs.Write32(REG_DW_KERNEL, next_reg.dw_kernel);
    reg_ok = reg_ok && regs.Write32(REG_DW_STRIDE, next_reg.dw_stride);
    reg_ok = reg_ok && regs.Write32(REG_DW_PADDING, next_reg.dw_padding);
    reg_ok = reg_ok && regs.Write32(REG_ENABLE_CONV_OUT, next_reg.enable_conv_out);
    reg_ok = reg_ok && regs.Write32(REG_LOCAL_GROUPS_PER_ROUND, next_reg.local_groups_per_round);
    reg_ok = reg_ok && regs.Write32(REG_ROUNDS, next_reg.rounds);

    reg_ok = reg_ok && regs.WriteAddr(REG_INPUT_L, REG_INPUT_H, next_reg.input_phys);
    reg_ok = reg_ok && regs.WriteAddr(REG_OUTPUT_FIXED_L, REG_OUTPUT_FIXED_H, next_reg.output_fixed_phys);
    reg_ok = reg_ok && regs.WriteAddr(REG_OUTPUT_CONV2D_L, REG_OUTPUT_CONV2D_H, next_reg.output_conv2d_phys);
    if (reg_ok) {
      reg_config_ = next_reg;
    }
  }
  if (!reg_ok) {
    FPGA_CALIB_LOG_ERROR("failed to write one or more HLS registers");
    return;
  }

#if CALIB_CONV2D_ENABLE_VERBOSE_LOG
  std::cout << "[IntelFPGA] calib_conv2d launch: N" << batch
            << " IC" << in_c << " ICB" << in_c_blocks
            << " IH" << in_h << " IW" << in_w
            << " OC" << conv_oc << " OH" << out_h << " OW" << out_w
            << " DWK" << dw_kh << " S" << stride << " P" << pad
            << " local_groups=" << local_groups_per_round
            << " rounds=" << rounds << std::endl;
#endif

  const size_t input_img_numel = static_cast<size_t>(in_c) * in_h * in_w;
  const size_t output_img_numel = static_cast<size_t>(out_c) * out_h * out_w;

#if CALIB_CONV2D_USE_CACHED_INPUT_STAGING && !CALIB_CONV2D_PACK_INPUT_DIRECT_TO_CMA
  if (input_pack_cache_bytes_ < input_bytes) {
    input_pack_cache_.resize(input_bytes);
    input_pack_cache_bytes_ = input_pack_cache_.size();
  }
#endif
#if CALIB_CONV2D_USE_CACHED_OUTPUT_STAGING
  if (output_fixed_cache_words_ < output_fixed_words) {
    output_fixed_cache_.resize(output_fixed_words);
    output_fixed_cache_words_ = output_fixed_cache_.size();
  }
#endif
#if CALIB_CONV2D_USE_CACHED_CONV2D_OUT_STAGING
  if (enable_conv2d_out && conv2d_out_cache_words_ < conv2d_out_words) {
    conv2d_out_cache_.resize(conv2d_out_words);
    conv2d_out_cache_words_ = conv2d_out_cache_.size();
  }
#endif

  for (int n = 0; n < batch; ++n) {
    const T* x_n = x_data + n * input_img_numel;
    T* out_n = out_data + n * output_img_numel;

#if CALIB_CONV2D_CLEAR_OUTPUT_BEFORE_RUN
    {
      ScopedCalibTimer timer(&timing.batch_clear_ns);
      CalibFastZero(cma_base + cma_runtime_.off_output_fixed, output_fixed_bytes);
      CalibFastZero(cma_base + cma_runtime_.off_output_conv2d, conv2d_out_bytes);
    }
#endif

    {
      ScopedCalibTimer timer(&timing.input_pack_ns);
#if CALIB_CONV2D_PACK_INPUT_DIRECT_TO_CMA
      uint8_t* input_pack_dst = cma_base + cma_runtime_.off_input;
#elif CALIB_CONV2D_USE_CACHED_INPUT_STAGING
      uint8_t* input_pack_dst = input_pack_cache_.data();
#else
      uint8_t* input_pack_dst = cma_base + cma_runtime_.off_input;
#endif
      QuantizeNCHWToNHWCC16Float(x_n,
                                 in_c,
                                 in_h,
                                 in_w,
                                 in_c_aligned,
                                 calib_scale,
                                 input_pack_dst);
#if CALIB_CONV2D_USE_CACHED_INPUT_STAGING && !CALIB_CONV2D_PACK_INPUT_DIRECT_TO_CMA
      CalibFastCopy(cma_base + cma_runtime_.off_input, input_pack_cache_.data(), input_bytes);
#endif
    }

    {
      ScopedCalibTimer timer(&timing.input_sync_ns);
      CalibSyncInputForFpga(cma_base + cma_runtime_.off_input, input_bytes);
    }
    if (!regs.StartAndWait(&timing.ip_start_cmd_ns, &timing.ip_wait_done_ns)) {
      FPGA_CALIB_LOG_ERROR("HLS IP execution failed");
      return;
    }
    {
      ScopedCalibTimer timer(&timing.output_sync_ns);
      CalibSyncOutputForCpu(cma_base + cma_runtime_.off_output_fixed,
                            output_fixed_bytes);
      if (enable_conv2d_out) {
        CalibSyncOutputForCpu(cma_base + cma_runtime_.off_output_conv2d,
                              conv2d_out_bytes);
      }
    }

    if (enable_conv2d_out) {
      ScopedCalibTimer timer(&timing.conv2d_copy_ns);
      int8_t* conv2d_out_n = conv2d_out_tensor + n * conv2d_out_count;
#if CALIB_CONV2D_USE_CACHED_CONV2D_OUT_STAGING
      CalibFastCopy(conv2d_out_cache_.data(),
                    cma_base + cma_runtime_.off_output_conv2d,
                    conv2d_out_bytes);
      const uint32_t* packed_conv2d_out = conv2d_out_cache_.data();
      UnpackConv2dOutGHWC4ToNCHWInt8(packed_conv2d_out, conv2d_out_n, conv_oc, in_h, in_w);
#else
      const uint32_t* packed_conv2d_out = reinterpret_cast<const uint32_t*>(
          cma_base + cma_runtime_.off_output_conv2d);
      UnpackConv2dOutHWG4ToNCHWInt8SourceMajor(
          packed_conv2d_out, conv2d_out_n, conv_oc, in_h, in_w);
#endif
    }

    {
      ScopedCalibTimer timer(&timing.output_copy_ns);
#if CALIB_CONV2D_USE_CACHED_OUTPUT_STAGING
      CalibFastCopy(output_fixed_cache_.data(), cma_base + cma_runtime_.off_output_fixed, output_fixed_bytes);
      const uint32_t* raw_out = output_fixed_cache_.data();
#else
      const uint32_t* raw_out = reinterpret_cast<const uint32_t*>(cma_base + cma_runtime_.off_output_fixed);
#endif
      ConvertFloatGHWC4RawToNCHWFloatArray(raw_out, out_n, out_c, out_h, out_w);
    }

  }

#if CALIB_CONV2D_ENABLE_TIMING_LOG
  timing.Print(batch, CalibNowNs() - run_start_ns);
#endif
}

}  // namespace intel_fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::intel_fpga::CalibConv2dCompute<PRECISION(kFloat), PRECISION(kFloat)> ConvFp32;

REGISTER_LITE_KERNEL(calib_conv2d, kIntelFPGA, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output_Conv2d", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("calib_conv2d", 1)
    .Finalize();
