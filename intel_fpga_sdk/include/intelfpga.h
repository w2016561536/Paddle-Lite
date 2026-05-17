#ifndef _INTELFPGA_H_
#define _INTELFPGA_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define INTELFPGA_SDK_VERSTR "1.0.0"
#define INTELFPGA_SDK_VERNUM 0x1000000

// Activation type
#define INTELFPGA_ACT_NONE 0
#define INTELFPGA_ACT_RELU 1
#define INTELFPGA_ACT_RELU6 2
#define INTELFPGA_ACT_LEAKYRELU 3

// Kernel parameters
struct intelfpga_kprm_s {
	uint32_t kw; // width
	uint32_t kh; // height
	uint32_t ws; // x stride(s)
	uint32_t hs; // y stride(s)
};

// Input parameters, nchw
struct intelfpga_iprm_s {
	uint32_t in; // nbr of batch {1}
	uint32_t ic; // nbr of channels {1}
	uint32_t iw; // width
	uint32_t ih; // height
	uint32_t pl; // padding x-left in bytes {0}
	uint32_t pr; // padding x-right in bytes {0}
	uint32_t pt; // padding y-top in bytes {0}
	uint32_t pb; // padding y-bottom in bytes {0}
	uint32_t dx; // dilation for x {1}
	uint32_t dy; // dilation for y {1}
};

// Output parameters, nchw
struct intelfpga_oprm_s {
	uint32_t on; // nbr of batch {1}
	uint32_t oc; // nbr of channels {1}
	uint32_t ow; // width
	uint32_t oh; // height
};

// Basic convolution
struct intelfpga_conv2d_s {
	uint32_t          at; // activation type {0}, None=0, RELU=1
	uint32_t          ng; // nbr of groups {1}
	float          alpha; // Leaky Relu alpha
	float*            ia; // input address, [N,Ci,Hi,Wi]
	float*            ka; // kernel address, [Co,Ci,Hk,Wk]
	float*            ba; // bias address, [Co,1]
	float*            oa; // output address, [N,Co,Ho,Wo]
	struct intelfpga_iprm_s ip; // input
	struct intelfpga_kprm_s kp; // kernel
	struct intelfpga_oprm_s op; // output
};

// Pooling convolution
struct intelfpga_pool2d_s {
	uint8_t           gp; // global pooling {0}
	uint8_t           pm; // pooling mode {0}, Max=0, AVG=1
	uint8_t           cm; // ceil mode {0}, ceil=0, floor=1
	uint8_t           ex; // exclusive {1}, if ignore padding in avg pooling
	float*            ia; // input address, [N,Ci,Hi,Wi]
	float*            oa; // output address, [N,Ci,Ho,Wo]
	struct intelfpga_iprm_s ip; // input
	struct intelfpga_kprm_s kp; // kernel
	struct intelfpga_oprm_s op; // output
};

// Full connection
struct intelfpga_fcon_s {
	uint32_t at; // activation type {0}, None=0, RELU=1
	float alpha; // Leaky Relu alpha
	float*   ia; // input address, [M,K]
	float*   ka; // kernel address, [K,N]
	float*   ba; // bias address, [M,N]
	float*   oa; // output address, [M,N] = ia[M,K] * wa[K,N] + ba[M,N]
	int m, n, k; // dims
};

/* function declarations */

void* intelfpga_malloc(size_t size);
void intelfpga_free(void *ptr);

int intelfpga_conv2d(struct intelfpga_conv2d_s* argp);
int intelfpga_pool2d(struct intelfpga_pool2d_s* argp);
int intelfpga_fullconnect(struct intelfpga_fcon_s* argp);

#ifdef __cplusplus
}
#endif

#endif
