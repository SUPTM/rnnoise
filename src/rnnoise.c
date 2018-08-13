#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "rnnoise.h"
#include "rnn_weight.h"
#include "hsfft.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

static float fast_sqrt(float x) {
    float s;
#if defined(__x86_64__)
    __asm__ __volatile__ ("sqrtss %1, %0" : "=x"(s) : "x"(x));
#elif defined(__i386__)
    s = x;
__asm__ __volatile__ ("fsqrt" : "+t"(s));
#elif defined(__arm__) && defined(__VFP_FP__)
__asm__ __volatile__ ("vsqrt.f32 %0, %1" : "=w"(s) : "w"(x));
#else
s = sqrtf(x);
#endif
    return s;
}

static const float tansig_table[201] = {
        0.000000f, 0.039979f, 0.079830f, 0.119427f, 0.158649f,
        0.197375f, 0.235496f, 0.272905f, 0.309507f, 0.345214f,
        0.379949f, 0.413644f, 0.446244f, 0.477700f, 0.507977f,
        0.537050f, 0.564900f, 0.591519f, 0.616909f, 0.641077f,
        0.664037f, 0.685809f, 0.706419f, 0.725897f, 0.744277f,
        0.761594f, 0.777888f, 0.793199f, 0.807569f, 0.821040f,
        0.833655f, 0.845456f, 0.856485f, 0.866784f, 0.876393f,
        0.885352f, 0.893698f, 0.901468f, 0.908698f, 0.915420f,
        0.921669f, 0.927473f, 0.932862f, 0.937863f, 0.942503f,
        0.946806f, 0.950795f, 0.954492f, 0.957917f, 0.961090f,
        0.964028f, 0.966747f, 0.969265f, 0.971594f, 0.973749f,
        0.975743f, 0.977587f, 0.979293f, 0.980869f, 0.982327f,
        0.983675f, 0.984921f, 0.986072f, 0.987136f, 0.988119f,
        0.989027f, 0.989867f, 0.990642f, 0.991359f, 0.992020f,
        0.992631f, 0.993196f, 0.993718f, 0.994199f, 0.994644f,
        0.995055f, 0.995434f, 0.995784f, 0.996108f, 0.996407f,
        0.996682f, 0.996937f, 0.997172f, 0.997389f, 0.997590f,
        0.997775f, 0.997946f, 0.998104f, 0.998249f, 0.998384f,
        0.998508f, 0.998623f, 0.998728f, 0.998826f, 0.998916f,
        0.999000f, 0.999076f, 0.999147f, 0.999213f, 0.999273f,
        0.999329f, 0.999381f, 0.999428f, 0.999472f, 0.999513f,
        0.999550f, 0.999585f, 0.999617f, 0.999646f, 0.999673f,
        0.999699f, 0.999722f, 0.999743f, 0.999763f, 0.999781f,
        0.999798f, 0.999813f, 0.999828f, 0.999841f, 0.999853f,
        0.999865f, 0.999875f, 0.999885f, 0.999893f, 0.999902f,
        0.999909f, 0.999916f, 0.999923f, 0.999929f, 0.999934f,
        0.999939f, 0.999944f, 0.999948f, 0.999952f, 0.999956f,
        0.999959f, 0.999962f, 0.999965f, 0.999968f, 0.999970f,
        0.999973f, 0.999975f, 0.999977f, 0.999978f, 0.999980f,
        0.999982f, 0.999983f, 0.999984f, 0.999986f, 0.999987f,
        0.999988f, 0.999989f, 0.999990f, 0.999990f, 0.999991f,
        0.999992f, 0.999992f, 0.999993f, 0.999994f, 0.999994f,
        0.999994f, 0.999995f, 0.999995f, 0.999996f, 0.999996f,
        0.999996f, 0.999997f, 0.999997f, 0.999997f, 0.999997f,
        0.999997f, 0.999998f, 0.999998f, 0.999998f, 0.999998f,
        0.999998f, 0.999998f, 0.999999f, 0.999999f, 0.999999f,
        0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f,
        0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f,
        1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
        1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
        1.000000f,
};
#define RNN_INLINE inline
#define celt_assert(cond)
#define MIN16(a, b) ((a) < (b) ? (a) : (b))   /**< Minimum 16-bit value.   */
#define MAX16(a, b) ((a) > (b) ? (a) : (b))   /**< Maximum 16-bit value.   */
#define MIN32(a, b) ((a) < (b) ? (a) : (b))   /**< Minimum 32-bit value.   */
#define MAX32(a, b) ((a) > (b) ? (a) : (b))   /**< Maximum 32-bit value.   */
#define   M_PI       3.14159265358979323846   // pi

#define Q15ONE 1.0f
#define QCONST16(x, bits) (x)
#define EXTRACT16(x) (x)
#define EXTEND32(x) (x)
#define SHR32(a, shift) (a)
#define SHL32(a, shift) (a)
#define VSHR32(a, shift) (a)
#define ROUND16(a, shift)  (a)
#define HALF16(x)       (.5*(x))
#define HALF32(x)       (.5*(x))
#define ADD32(a, b) ((a)+(b))
#define MULT16_16(a, b)     ((float)(a)*(float)(b))
#define MAC16_16(c, a, b)     ((c)+(float)(a)*(float)(b))
#define MULT16_32_Q15(a, b)     ((a)*(b))
#define MULT32_32_Q31(a, b)     ((a)*(b))
#define MULT16_16_Q15(a, b)     ((a)*(b))


static RNN_INLINE int celt_isnan(float x) {
    union {
        float f;
        uint32_t i;
    } in;
    in.f = x;
    return ((in.i >> 23) & 0xFF) == 0xFF && (in.i & 0x007FFFFF) != 0;
}

void pitch_downsample(float *x[], float *x_lp, int len, int C);

void pitch_search(const float *x_lp, float *y, int len, int max_pitch, int *pitch);

float remove_doubling(float *x, int maxperiod, int minperiod,
                      int N, int *T0, int prev_period, float prev_gain);


void celt_pitch_xcorr(const float *_x, const float *_y,
                      float *xcorr, int len, int max_pitch);

/* OPT: This is the kernel you really want to optimize. It gets used a lot
   by the prefilter and by the PLC. */
static RNN_INLINE void xcorr_kernel(const float *x, const float *y, float sum[4], int len) {
    int j;
    float y_0, y_1, y_2, y_3;
    celt_assert(len >= 3);
    y_3 = 0; /* gcc doesn't realize that y_3 can't be used uninitialized */
    y_0 = *y++;
    y_1 = *y++;
    y_2 = *y++;
    for (j = 0; j < len - 3; j += 4) {
        float tmp;
        tmp = *x++;
        y_3 = *y++;
        sum[0] = MAC16_16(sum[0], tmp, y_0);
        sum[1] = MAC16_16(sum[1], tmp, y_1);
        sum[2] = MAC16_16(sum[2], tmp, y_2);
        sum[3] = MAC16_16(sum[3], tmp, y_3);
        tmp = *x++;
        y_0 = *y++;
        sum[0] = MAC16_16(sum[0], tmp, y_1);
        sum[1] = MAC16_16(sum[1], tmp, y_2);
        sum[2] = MAC16_16(sum[2], tmp, y_3);
        sum[3] = MAC16_16(sum[3], tmp, y_0);
        tmp = *x++;
        y_1 = *y++;
        sum[0] = MAC16_16(sum[0], tmp, y_2);
        sum[1] = MAC16_16(sum[1], tmp, y_3);
        sum[2] = MAC16_16(sum[2], tmp, y_0);
        sum[3] = MAC16_16(sum[3], tmp, y_1);
        tmp = *x++;
        y_2 = *y++;
        sum[0] = MAC16_16(sum[0], tmp, y_3);
        sum[1] = MAC16_16(sum[1], tmp, y_0);
        sum[2] = MAC16_16(sum[2], tmp, y_1);
        sum[3] = MAC16_16(sum[3], tmp, y_2);
    }
    if (j++ < len) {
        float tmp = *x++;
        y_3 = *y++;
        sum[0] = MAC16_16(sum[0], tmp, y_0);
        sum[1] = MAC16_16(sum[1], tmp, y_1);
        sum[2] = MAC16_16(sum[2], tmp, y_2);
        sum[3] = MAC16_16(sum[3], tmp, y_3);
    }
    if (j++ < len) {
        float tmp = *x++;
        y_0 = *y++;
        sum[0] = MAC16_16(sum[0], tmp, y_1);
        sum[1] = MAC16_16(sum[1], tmp, y_2);
        sum[2] = MAC16_16(sum[2], tmp, y_3);
        sum[3] = MAC16_16(sum[3], tmp, y_0);
    }
    if (j < len) {
        float tmp = *x++;
        y_1 = *y++;
        sum[0] = MAC16_16(sum[0], tmp, y_2);
        sum[1] = MAC16_16(sum[1], tmp, y_3);
        sum[2] = MAC16_16(sum[2], tmp, y_0);
        sum[3] = MAC16_16(sum[3], tmp, y_1);
    }
}

static RNN_INLINE void dual_inner_prod(const float *x, const float *y01, const float *y02,
                                       int N, float *xy1, float *xy2) {
    int i;
    float xy01 = 0;
    float xy02 = 0;
    for (i = 0; i < N; i++) {
        xy01 = MAC16_16(xy01, x[i], y01[i]);
        xy02 = MAC16_16(xy02, x[i], y02[i]);
    }
    *xy1 = xy01;
    *xy2 = xy02;
}

/*We make sure a C version is always available for cases where the overhead of
  vectorization and passing around an arch flag aren't worth it.*/
static RNN_INLINE float celt_inner_prod(const float *x, const float *y, int N) {
    int i;
    float xy = 0;
    for (i = 0; i < N; i++)
        xy = MAC16_16(xy, x[i], y[i]);
    return xy;
}


#define WEIGHTS_SCALE (1.f/256)
#define MAX_NEURONS 128

#define ACTIVATION_TANH    0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_RELU    2

typedef signed char rnn_weight;

typedef struct {
    const rnn_weight *bias;
    const rnn_weight *input_weights;
    int nb_inputs;
    int nb_neurons;
    int activation;
} DenseLayer;

typedef struct {
    const rnn_weight *bias;
    const rnn_weight *input_weights;
    const rnn_weight *recurrent_weights;
    int nb_inputs;
    int nb_neurons;
    int activation;
} GRULayer;

typedef struct RNNState RNNState;

void compute_dense(const DenseLayer *layer, float *output, const float *input);

void compute_gru(const GRULayer *gru, float *state, const float *input);

void compute_rnn(RNNState *rnn, float *gains, float *vad, const float *input);


/*This file is automatically generated from a Keras model*/

#define INPUT_SIZE 42

#define INPUT_DENSE_SIZE 24
extern const DenseLayer input_dense;

#define VAD_GRU_SIZE 24
extern const GRULayer vad_gru;

#define NOISE_GRU_SIZE 48
extern const GRULayer noise_gru;

#define DENOISE_GRU_SIZE 96
extern const GRULayer denoise_gru;

extern const DenseLayer denoise_output;

extern const DenseLayer vad_output;

typedef struct RNNState {
    float vad_gru_state[VAD_GRU_SIZE];
    float noise_gru_state[NOISE_GRU_SIZE];
    float denoise_gru_state[DENOISE_GRU_SIZE];
} RNNState;


#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)
#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)
#define SQUARE(x) ((x)*(x))
#define NB_BANDS 22
#define CEPS_MEM 8
#define NB_DELTA_CEPS 6
#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)

#ifndef TRAINING
#define TRAINING 0
#endif
static const int16_t eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};

typedef struct {
    int init;
    float half_window[FRAME_SIZE];
    float dct_table[NB_BANDS * NB_BANDS];
} CommonState;

struct DenoiseState {
    float analysis_mem[FRAME_SIZE];
    float cepstral_mem[CEPS_MEM][NB_BANDS];
    int memid;
    float synthesis_mem[FRAME_SIZE];
    float pitch_buf[PITCH_BUF_SIZE];
    float last_gain;
    int last_period;
    float mem_hp_x[2];
    float lastg[NB_BANDS];
    RNNState rnn;
};


void compute_band_energy(float *bandE, const fft_t *X) {
    int i;
    float sum[NB_BANDS] = {0};
    for (i = 0; i < NB_BANDS - 1; i++) {
        int j;
        int band_size;
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
        for (j = 0; j < band_size; j++) {
            float tmp;
            float frac = (float) j / band_size;
            tmp = SQUARE(X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].re);
            tmp += SQUARE(X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].im);
            sum[i] += (1 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2;
    sum[NB_BANDS - 1] *= 2;
    for (i = 0; i < NB_BANDS; i++) {
        bandE[i] = sum[i];
    }
}

void compute_band_corr(float *bandE, const fft_t *X, const fft_t *P) {
    int i;
    float sum[NB_BANDS] = {0};
    for (i = 0; i < NB_BANDS - 1; i++) {
        int j;
        int band_size;
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
        for (j = 0; j < band_size; j++) {
            float tmp;
            float frac = (float) j / band_size;
            tmp = X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].re * P[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].re;
            tmp += X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].im * P[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].im;
            sum[i] += (1 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2;
    sum[NB_BANDS - 1] *= 2;
    for (i = 0; i < NB_BANDS; i++) {
        bandE[i] = sum[i];
    }
}

void interp_band_gain(float *g, const float *bandE) {
    int i;
    memset(g, 0, FREQ_SIZE);
    for (i = 0; i < NB_BANDS - 1; i++) {
        int j;
        int band_size;
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
        for (j = 0; j < band_size; j++) {
            float frac = (float) j / band_size;
            g[(eband5ms[i] << FRAME_SIZE_SHIFT) + j] = (1 - frac) * bandE[i] + frac * bandE[i + 1];
        }
    }
}


CommonState common;

static void check_init() {
    int i;
    if (common.init) return;
    for (i = 0; i < FRAME_SIZE; i++)
        common.half_window[i] = sinf(
                .5f * M_PI * sinf(.5f * M_PI * (i + .5f) / FRAME_SIZE) * sinf(.5f * M_PI * (i + .5f) / FRAME_SIZE));
    for (i = 0; i < NB_BANDS; i++) {
        int j;
        for (j = 0; j < NB_BANDS; j++) {
            common.dct_table[i * NB_BANDS + j] = cosf((i + .5f) * j * M_PI / NB_BANDS);
            if (j == 0) common.dct_table[i * NB_BANDS + j] *= fast_sqrt(.5f);
        }
    }
    common.init = 1;
}

static void dct(float *out, const float *in) {
    int i;
    check_init();
    for (i = 0; i < NB_BANDS; i++) {
        int j;
        float sum = 0;
        for (j = 0; j < NB_BANDS; j++) {
            sum += in[j] * common.dct_table[j * NB_BANDS + i];
        }
        out[i] = sum * fast_sqrt(2. / 22);
    }
}


static void forward_transform(fft_t *out, float *in) {
    fft_t y[WINDOW_SIZE];
    fft_real_object fftPlan = fft_real_init(WINDOW_SIZE, 1);
    fft_r2c_exec(fftPlan, in, y);
    free_real_fft(fftPlan);
    memcpy(out, y, sizeof(fft_t) * FREQ_SIZE);
}


static void apply_window(float *x) {
    int i;
    check_init();
    for (i = 0; i < FRAME_SIZE; i++) {
        x[i] *= common.half_window[i];
        x[WINDOW_SIZE - 1 - i] *= common.half_window[i];
    }
}

int rnnoise_get_size() {
    return sizeof(DenoiseState);
}

int rnnoise_init(DenoiseState *st) {
    memset(st, 0, sizeof(*st));
    return 0;
}

DenoiseState *rnnoise_create() {
    DenoiseState *st;
    st = (DenoiseState *) malloc(rnnoise_get_size());
    rnnoise_init(st);
    return st;
}

void rnnoise_destroy(DenoiseState *st) {
    free(st);
}

#if TRAINING
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;
#endif

static void frame_analysis(DenoiseState *st, fft_t *X, float *Ex, const float *in) {
    int i;
    float x[WINDOW_SIZE];
    memcpy(x, st->analysis_mem, FRAME_SIZE * sizeof(*x) + 0 * (x - st->analysis_mem));

    for (i = 0; i < FRAME_SIZE; i++) x[FRAME_SIZE + i] = in[i];

    memcpy(st->analysis_mem, in, FRAME_SIZE * sizeof(*st->analysis_mem) + 0 * (st->analysis_mem - in));
    apply_window(x);
    forward_transform(X, x);

#if TRAINING
    for (i=lowpass;i<FREQ_SIZE;i++)
        X[i].r = X[i].i = 0;
#endif
    compute_band_energy(Ex, X);
}

static int compute_frame_features(DenoiseState *st, fft_t *X, fft_t *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
    int i;
    float E = 0;
    float *ceps_0, *ceps_1, *ceps_2;
    float spec_variability = 0;
    float Ly[NB_BANDS];
    float p[WINDOW_SIZE];
    float pitch_buf[PITCH_BUF_SIZE >> 1];
    int pitch_index;
    float gain;
    float *(pre[1]);
    float tmp[NB_BANDS];
    float follow, logMax;
    frame_analysis(st, X, Ex, in);
    memmove(st->pitch_buf, &st->pitch_buf[FRAME_SIZE],
            (PITCH_BUF_SIZE - FRAME_SIZE) * sizeof(*st->pitch_buf) + 0 * (st->pitch_buf - &st->pitch_buf[FRAME_SIZE]));

    memcpy(&st->pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE], in,
           FRAME_SIZE * sizeof(*&st->pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE]) +
           0 * (&st->pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE] - in));

    pre[0] = &st->pitch_buf[0];
    pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
    pitch_search(pitch_buf + (PITCH_MAX_PERIOD >> 1), pitch_buf, PITCH_FRAME_SIZE,
                 PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD, &pitch_index);
    pitch_index = PITCH_MAX_PERIOD - pitch_index;

    gain = remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
                           PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
    st->last_period = pitch_index;
    st->last_gain = gain;
    for (i = 0; i < WINDOW_SIZE; i++)
        p[i] = st->pitch_buf[PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index + i];
    apply_window(p);
    forward_transform(P, p);
    compute_band_energy(Ep, P);
    compute_band_corr(Exp, X, P);
    for (i = 0; i < NB_BANDS; i++) Exp[i] = Exp[i] / fast_sqrt(.001f + Ex[i] * Ep[i]);
    dct(tmp, Exp);
    for (i = 0; i < NB_DELTA_CEPS; i++) features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = tmp[i];
    features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3f;
    features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9f;
    features[NB_BANDS + 3 * NB_DELTA_CEPS] = .01f * (pitch_index - 300);
    logMax = -2;
    follow = -2;
    for (i = 0; i < NB_BANDS; i++) {
        Ly[i] = log10f(1e-2f + Ex[i]);
        Ly[i] = MAX16(logMax - 7, MAX16(follow - 1.5f, Ly[i]));
        logMax = MAX16(logMax, Ly[i]);
        follow = MAX16(follow - 1.5f, Ly[i]);
        E += Ex[i];
    }
    if (!TRAINING && E < 0.04f) {
        /* If there's no audio, avoid messing up the state. */
        memset(features, 0, (NB_FEATURES) * sizeof(*features));
        return 1;
    }
    dct(features, Ly);
    features[0] -= 12;
    features[1] -= 4;
    ceps_0 = st->cepstral_mem[st->memid];
    ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM + st->memid - 1] : st->cepstral_mem[st->memid - 1];
    ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM + st->memid - 2] : st->cepstral_mem[st->memid - 2];
    for (i = 0; i < NB_BANDS; i++) ceps_0[i] = features[i];
    st->memid++;
    for (i = 0; i < NB_DELTA_CEPS; i++) {
        features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
        features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
        features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2 * ceps_1[i] + ceps_2[i];
    }
    /* Spectral variability features. */
    if (st->memid == CEPS_MEM) st->memid = 0;
    for (i = 0; i < CEPS_MEM; i++) {
        int j;
        float mindist = 1e15f;
        for (j = 0; j < CEPS_MEM; j++) {
            int k;
            float dist = 0;
            for (k = 0; k < NB_BANDS; k++) {
                float tmp;
                tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
                dist += tmp * tmp;
            }
            if (j != i)
                mindist = MIN32(mindist, dist);
        }
        spec_variability += mindist;
    }
    features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM - 2.1f;
    return TRAINING && E < 0.1f;
}

static void frame_synthesis(DenoiseState *st, short *out, fft_t *input) {

    float output[WINDOW_SIZE];
    int i;
    fft_real_object ifftPlan = fft_real_init(WINDOW_SIZE, -1);
    fft_c2r_exec(ifftPlan, input, output);
    free_real_fft(ifftPlan);
    float norm = 1.f / WINDOW_SIZE;
    for (int i = 0; i < WINDOW_SIZE; ++i) {
        output[i] = (output[i] * norm);
    }
    apply_window(output);
    for (i = 0; i < FRAME_SIZE; i++)
        out[i] = (short) (output[i] + st->synthesis_mem[i]);
    memcpy(st->synthesis_mem, &output[FRAME_SIZE],
           FRAME_SIZE * sizeof(*st->synthesis_mem) + 0 * (st->synthesis_mem - &output[FRAME_SIZE]));
}

static void biquad(float *y, float mem[2], const short *x, const float *b, const float *a, int N) {
    int i;
    for (i = 0; i < N; i++) {
        float xi, yi;
        xi = x[i];
        yi = x[i] + mem[0];
        mem[0] = mem[1] + (b[0] * xi - a[0] * yi);
        mem[1] = (b[1] * xi - a[1] * yi);
        y[i] = yi;
    }
}

void pitch_filter(fft_t *X, const fft_t *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
    int i;
    float r[NB_BANDS];
    float rf[FREQ_SIZE] = {0};
    for (i = 0; i < NB_BANDS; i++) {

        if (Exp[i] > g[i]) r[i] = 1;
        else r[i] = SQUARE(Exp[i]) * (1 - SQUARE(g[i])) / (.001f + SQUARE(g[i]) * (1 - SQUARE(Exp[i])));
        r[i] = fast_sqrt(MIN16(1, MAX16(0, r[i])));
        r[i] *= fast_sqrt(Ex[i] / (1e-8f + Ep[i]));
    }
    interp_band_gain(rf, r);
    for (i = 0; i < FREQ_SIZE; i++) {
        X[i].re += rf[i] * P[i].re;
        X[i].im += rf[i] * P[i].im;
    }
    float newE[NB_BANDS];
    compute_band_energy(newE, X);
    float norm[NB_BANDS];
    float normf[FREQ_SIZE] = {0};
    for (i = 0; i < NB_BANDS; i++) {
        norm[i] = fast_sqrt(Ex[i] / (1e-8 + newE[i]));
    }
    interp_band_gain(normf, norm);
    for (i = 0; i < FREQ_SIZE; i++) {
        X[i].re *= normf[i];
        X[i].im *= normf[i];
    }
}

float rnnoise_process_frame(DenoiseState *st, short *out, const short *in) {
    int i;
    fft_t X[FREQ_SIZE];
    fft_t P[WINDOW_SIZE];
    float x[FRAME_SIZE];
    float Ex[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    float gf[FREQ_SIZE] = {1};
    float vad_prob = 0;
    int silence;
    static const float a_hp[2] = {-1.99599f, 0.99600f};
    static const float b_hp[2] = {-2, 1};
    biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
    silence = compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);

    if (!silence) {
        compute_rnn(&st->rnn, g, &vad_prob, features);
        pitch_filter(X, P, Ex, Ep, Exp, g);
        for (i = 0; i < NB_BANDS; i++) {
            float alpha = .6f;
            g[i] = MAX16(g[i], alpha * st->lastg[i]);
            st->lastg[i] = g[i];
        }
        interp_band_gain(gf, g);
#if 1
        for (i = 0; i < FREQ_SIZE; i++) {
            X[i].re *= gf[i];
            X[i].im *= gf[i];
        }
#endif
    }

    frame_synthesis(st, out, X);
    return vad_prob;
}

#if TRAINING

static float uni_rand() {
return rand()/(double)RAND_MAX-.5f;
}

static void rand_resp(float *a, float *b) {
a[0] = .75f*uni_rand();
a[1] = .75f*uni_rand();
b[0] = .75f*uni_rand();
b[1] = .75f*uni_rand();
}

int main(int argc, char **argv) {
int i;
int count=0;
static const float a_hp[2] = {-1.99599f, 0.99600f};
static const float b_hp[2] = {-2, 1};
float a_noise[2] = {0};
float b_noise[2] = {0};
float a_sig[2] = {0};
float b_sig[2] = {0};
float mem_hp_x[2]={0};
float mem_hp_n[2]={0};
float mem_resp_x[2]={0};
float mem_resp_n[2]={0};
float x[FRAME_SIZE];
float n[FRAME_SIZE];
float xn[FRAME_SIZE];
int vad_cnt=0;
int gain_change_count=0;
float speech_gain = 1, noise_gain = 1;
FILE *f1, *f2, *fout;
DenoiseState *st;
DenoiseState *noise_state;
DenoiseState *noisy;
st = rnnoise_create();
noise_state = rnnoise_create();
noisy = rnnoise_create();
if (argc!=4) {
  fprintf(stderr, "usage: %s <speech> <noise> <output denoised>\n", argv[0]);
  return 1;
}
f1 = fopen(argv[1], "r");
f2 = fopen(argv[2], "r");
fout = fopen(argv[3], "w");
for(i=0;i<150;i++) {
  short tmp[FRAME_SIZE];
  fread(tmp, sizeof(short), FRAME_SIZE, f2);
}
while (1) {
  kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE], P[WINDOW_SIZE];
  float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float Ln[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float gf[FREQ_SIZE]={1};
  short tmp[FRAME_SIZE];
  float vad=0;
  float vad_prob;
  float E=0;
  if (count==50000000) break;
  if (++gain_change_count > 2821) {
    speech_gain = powf(10.f, (-40+(rand()%60))/20.f);
    noise_gain = powf(10.f, (-30+(rand()%50))/20.f);
    if (rand()%10==0) noise_gain = 0;
    noise_gain *= speech_gain;
    if (rand()%10==0) speech_gain = 0;
    gain_change_count = 0;
    rand_resp(a_noise, b_noise);
    rand_resp(a_sig, b_sig);
    lowpass = FREQ_SIZE * 3000.f/24000.f * powf(50.f, rand()/(float)RAND_MAX);
    for (i=0;i<NB_BANDS;i++) {
      if (eband5ms[i]<<FRAME_SIZE_SHIFT > lowpass) {
        band_lp = i;
        break;
      }
    }
  }
  if (speech_gain != 0) {
    fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1)) {
      rewind(f1);
      fread(tmp, sizeof(short), FRAME_SIZE, f1);
    }
    for (i=0;i<FRAME_SIZE;i++) x[i] = speech_gain*tmp[i];
    for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];
  } else {
    for (i=0;i<FRAME_SIZE;i++) x[i] = 0;
    E = 0;
  }
  if (noise_gain!=0) {
    fread(tmp, sizeof(short), FRAME_SIZE, f2);
    if (feof(f2)) {
      rewind(f2);
      fread(tmp, sizeof(short), FRAME_SIZE, f2);
    }
    for (i=0;i<FRAME_SIZE;i++) n[i] = noise_gain*tmp[i];
  } else {
    for (i=0;i<FRAME_SIZE;i++) n[i] = 0;
  }
  biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
  biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
  biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
  biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) xn[i] = x[i] + n[i];
  if (E > 1e9f) {
    vad_cnt=0;
  } else if (E > 1e8f) {
    vad_cnt -= 5;
  } else if (E > 1e7f) {
    vad_cnt++;
  } else {
    vad_cnt+=2;
  }
  if (vad_cnt < 0) vad_cnt = 0;
  if (vad_cnt > 15) vad_cnt = 15;

  if (vad_cnt >= 10) vad = 0;
  else if (vad_cnt > 0) vad = 0.5f;
  else vad = 1.f;

  frame_analysis(st, Y, Ey, x);
  frame_analysis(noise_state, N, En, n);
  for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-2+En[i]);
  int silence = compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, xn);
  pitch_filter(X, P, Ex, Ep, Exp, g);
  //printf("%f %d\n", noisy->last_gain, noisy->last_period);
  for (i=0;i<NB_BANDS;i++) {
    g[i] = fast_sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
    if (g[i] > 1) g[i] = 1;
    if (silence || i > band_lp) g[i] = -1;
    if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
    if (vad==0 && noise_gain==0) g[i] = -1;
  }
  count++;
#if 0
  for (i=0;i<NB_FEATURES;i++) printf("%f ", features[i]);
  for (i=0;i<NB_BANDS;i++) printf("%f ", g[i]);
  for (i=0;i<NB_BANDS;i++) printf("%f ", Ln[i]);
  printf("%f\n", vad);
#endif
#if 1
  fwrite(features, sizeof(float), NB_FEATURES, stdout);
  fwrite(g, sizeof(float), NB_BANDS, stdout);
  fwrite(Ln, sizeof(float), NB_BANDS, stdout);
  fwrite(&vad, sizeof(float), 1, stdout);
#endif
#if 0
  compute_rnn(&noisy->rnn, g, &vad_prob, features);
  interp_band_gain(gf, g);
#if 1
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= gf[i];
    X[i].i *= gf[i];
  }
#endif
  frame_synthesis(noisy, xn, X);

  for (i=0;i<FRAME_SIZE;i++) tmp[i] = xn[i];
  fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
#endif
}
fprintf(stderr, "matrix size: %d x %d\n", count, NB_FEATURES + 2*NB_BANDS + 1);
fclose(f1);
fclose(f2);
fclose(fout);
return 0;
}

#endif

void _celt_lpc(
        float *_lpc, /* out: [0...p-1] LPC coefficients      */
        const float *ac,  /* in:  [0...p] autocorrelation values  */
        int p
) {
    int i, j;
    float r;
    float error = ac[0];
    float *lpc = _lpc;
    memset(lpc, 0, p * sizeof(*lpc));
    if (ac[0] != 0) {
        for (i = 0; i < p; i++) {
            /* Sum up this iteration's reflection coefficient */
            float rr = 0;
            for (j = 0; j < i; j++)
                rr += MULT32_32_Q31(lpc[j], ac[i - j]);
            rr += SHR32(ac[i + 1], 3);
            r = -SHL32(rr, 3) / error;
            /*  Update LPC coefficients and total error */
            lpc[i] = SHR32(r, 3);
            for (j = 0; j < (i + 1) >> 1; j++) {
                float tmp1, tmp2;
                tmp1 = lpc[j];
                tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + MULT32_32_Q31(r, tmp2);
                lpc[i - 1 - j] = tmp2 + MULT32_32_Q31(r, tmp1);
            }
            error = error - MULT32_32_Q31(MULT32_32_Q31(r, r), error);
            /* Bail out once we get 30 dB gain */
            if (error < .001f * ac[0])
                break;
        }
    }
}


int _celt_autocorr(const float *x,   /*  in: [0...n-1] samples x   */
                   float *ac,  /* out: [0...lag-1] ac values */  const float *window, int overlap, int lag, int n) {
    float d;
    int i, k;
    int fastN = n - lag;
    int shift;
    const float *xptr;
    float *xx = (float *) malloc(n * sizeof(float));
    if (xx == NULL) return -1;
    celt_assert(n > 0);
    celt_assert(overlap >= 0);
    if (overlap == 0) {
        xptr = x;
    } else {
        for (i = 0; i < n; i++)
            xx[i] = x[i];
        for (i = 0; i < overlap; i++) {
            xx[i] = MULT16_16_Q15(x[i], window[i]);
            xx[n - i - 1] = MULT16_16_Q15(x[n - i - 1], window[i]);
        }
        xptr = xx;
    }
    shift = 0;
    celt_pitch_xcorr(xptr, xptr, ac, fastN, lag + 1);
    for (k = 0; k <= lag; k++) {
        for (i = k + fastN, d = 0; i < n; i++)
            d = MAC16_16(d, xptr[i], xptr[i - k]);
        ac[k] += d;
    }
    free(xx);
    return shift;
}


static void find_best_pitch(float *xcorr, float *y, int len, int max_pitch, int *best_pitch) {
    int i, j;
    float Syy = 1;
    float best_num[2];
    float best_den[2];
    best_num[0] = -1;
    best_num[1] = -1;
    best_den[0] = 0;
    best_den[1] = 0;
    best_pitch[0] = 0;
    best_pitch[1] = 1;
    for (j = 0; j < len; j++)
        Syy = ADD32(Syy, SHR32(MULT16_16(y[j], y[j]), yshift));
    for (i = 0; i < max_pitch; i++) {
        if (xcorr[i] > 0) {
            float num;
            float xcorr16;
            xcorr16 = EXTRACT16(VSHR32(xcorr[i], xshift));
            num = MULT16_16_Q15(xcorr16, xcorr16);
            if (MULT16_32_Q15(num, best_den[1]) > MULT16_32_Q15(best_num[1], Syy)) {
                if (MULT16_32_Q15(num, best_den[0]) > MULT16_32_Q15(best_num[0], Syy)) {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = Syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = Syy;
                    best_pitch[1] = i;
                }
            }
        }
        Syy += SHR32(MULT16_16(y[i + len], y[i + len]), yshift) - SHR32(MULT16_16(y[i], y[i]), yshift);
        Syy = MAX32(1, Syy);
    }
}

static void celt_fir5(const float *x,
                      const float *num,
                      float *y,
                      int N,
                      float *mem) {
    int i;
    float num0, num1, num2, num3, num4;
    float mem0, mem1, mem2, mem3, mem4;
    num0 = num[0];
    num1 = num[1];
    num2 = num[2];
    num3 = num[3];
    num4 = num[4];
    mem0 = mem[0];
    mem1 = mem[1];
    mem2 = mem[2];
    mem3 = mem[3];
    mem4 = mem[4];
    for (i = 0; i < N; i++) {
        float sum = SHL32(EXTEND32(x[i]), SIG_SHIFT);
        sum = MAC16_16(sum, num0, mem0);
        sum = MAC16_16(sum, num1, mem1);
        sum = MAC16_16(sum, num2, mem2);
        sum = MAC16_16(sum, num3, mem3);
        sum = MAC16_16(sum, num4, mem4);
        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = x[i];
        y[i] = ROUND16(sum, SIG_SHIFT);
    }
    mem[0] = mem0;
    mem[1] = mem1;
    mem[2] = mem2;
    mem[3] = mem3;
    mem[4] = mem4;
}


void pitch_downsample(float *x[], float *x_lp,
                      int len, int C) {
    int i;
    float ac[5];
    float tmp = Q15ONE;
    float lpc[4], mem[5] = {0, 0, 0, 0, 0};
    float lpc2[5];
    float c1 = QCONST16(.8f, 15);
    for (i = 1; i < len >> 1; i++)
        x_lp[i] = SHR32(HALF32(HALF32(x[0][(2 * i - 1)] + x[0][(2 * i + 1)]) + x[0][2 * i]), shift);
    x_lp[0] = SHR32(HALF32(HALF32(x[0][1]) + x[0][0]), shift);
    if (C == 2) {
        for (i = 1; i < len >> 1; i++)
            x_lp[i] += SHR32(HALF32(HALF32(x[1][(2 * i - 1)] + x[1][(2 * i + 1)]) + x[1][2 * i]), shift);
        x_lp[0] += SHR32(HALF32(HALF32(x[1][1]) + x[1][0]), shift);
    }

    _celt_autocorr(x_lp, ac, NULL, 0,
                   4, len >> 1);

    /* Noise floor -40 dB */
    ac[0] *= 1.0001f;
    /* Lag windowing */
    for (i = 1; i <= 4; i++) {
        /*ac[i] *= exp(-.5*(2*M_PI*.002*i)*(2*M_PI*.002*i));*/
        ac[i] -= ac[i] * (.008f * i) * (.008f * i);
    }
    _celt_lpc(lpc, ac, 4);
    for (i = 0; i < 4; i++) {
        tmp = MULT16_16_Q15(QCONST16(.9f, 15), tmp);
        lpc[i] = MULT16_16_Q15(lpc[i], tmp);
    }
    /* Add a zero */
    lpc2[0] = lpc[0] + QCONST16(.8f, SIG_SHIFT);
    lpc2[1] = lpc[1] + MULT16_16_Q15(c1, lpc[0]);
    lpc2[2] = lpc[2] + MULT16_16_Q15(c1, lpc[1]);
    lpc2[3] = lpc[3] + MULT16_16_Q15(c1, lpc[2]);
    lpc2[4] = MULT16_16_Q15(c1, lpc[3]);
    celt_fir5(x_lp, lpc2, x_lp, len >> 1, mem);
}

void celt_pitch_xcorr(const float *_x, const float *_y, float *xcorr, int len, int max_pitch) {
    int i;
    /*The EDSP version requires that max_pitch is at least 1, and that _x is
     32-bit aligned.
    Since it's hard to put asserts in assembly, put them here.*/
    celt_assert(max_pitch > 0);
    celt_assert((((unsigned char *) _x - (unsigned char *) NULL) & 3) == 0);
    for (i = 0; i < max_pitch - 3; i += 4) {
        float sum[4] = {0, 0, 0, 0};
        xcorr_kernel(_x, _y + i, sum, len);
        xcorr[i] = sum[0];
        xcorr[i + 1] = sum[1];
        xcorr[i + 2] = sum[2];
        xcorr[i + 3] = sum[3];

    }
    /* In case max_pitch isn't a multiple of 4, do non-unrolled version. */
    for (; i < max_pitch; i++) {
        float sum;
        sum = celt_inner_prod(_x, _y + i, len);
        xcorr[i] = sum;
    }
}

void pitch_search(const float *x_lp, float *y,
                  int len, int max_pitch, int *pitch) {
    int i, j;
    int lag;
    int best_pitch[2] = {0, 0};
    int offset;
    celt_assert(len > 0);
    celt_assert(max_pitch > 0);
    lag = len + max_pitch;
    float *cache_mem = calloc((len >> 2) + (lag >> 2) + (max_pitch >> 1), sizeof(float));
    if (cache_mem == NULL) return;
    float *x_lp4 = cache_mem;
    float *y_lp4 = x_lp4 + (len >> 2);
    float *xcorr = y_lp4 + (lag >> 2);
    /* Downsample by 2 again */
    for (j = 0; j < len >> 2; j++)
        x_lp4[j] = x_lp[2 * j];
    for (j = 0; j < lag >> 2; j++)
        y_lp4[j] = y[2 * j];
    /* Coarse search with 4x decimation */
    celt_pitch_xcorr(x_lp4, y_lp4, xcorr, len >> 2, max_pitch >> 2);
    find_best_pitch(xcorr, y_lp4, len >> 2, max_pitch >> 2, best_pitch);
    /* Finer search with 2x decimation */
    for (i = 0; i < max_pitch >> 1; i++) {
        float sum;
        xcorr[i] = 0;
        if (abs(i - 2 * best_pitch[0]) > 2 && abs(i - 2 * best_pitch[1]) > 2)
            continue;
        sum = celt_inner_prod(x_lp, y + i, len >> 1);
        xcorr[i] = MAX32(-1, sum);
    }
    find_best_pitch(xcorr, y, len >> 1, max_pitch >> 1, best_pitch);
    /* Refine by pseudo-interpolation */
    if (best_pitch[0] > 0 && best_pitch[0] < (max_pitch >> 1) - 1) {
        float a, b, c;
        a = xcorr[best_pitch[0] - 1];
        b = xcorr[best_pitch[0]];
        c = xcorr[best_pitch[0] + 1];
        if ((c - a) > MULT16_32_Q15(QCONST16(.7f, 15), b - a))
            offset = 1;
        else if ((a - c) > MULT16_32_Q15(QCONST16(.7f, 15), b - c))
            offset = -1;
        else
            offset = 0;
    } else {
        offset = 0;
    }
    *pitch = 2 * best_pitch[0] - offset;
    free(cache_mem);
}

static float compute_pitch_gain(float xy, float xx, float yy) {
    return xy / fast_sqrt(1.0 + xx * yy);
}


static const int second_check[16] = {0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2};

float remove_doubling(float *x, int maxperiod, int minperiod,
                      int N, int *T0_, int prev_period, float prev_gain) {
    int k, i, T, T0;
    float g, g0;
    float pg;
    float xy, xx, yy, xy2;
    float xcorr[3];
    float best_xy, best_yy;
    int offset;
    int minperiod0;

    minperiod0 = minperiod;
    maxperiod /= 2;
    minperiod /= 2;
    *T0_ /= 2;
    prev_period /= 2;
    N /= 2;
    x += maxperiod;
    if (*T0_ >= maxperiod)
        *T0_ = maxperiod - 1;

    T = T0 = *T0_;
    float *yy_lookup = (float *) calloc(maxperiod + 1, sizeof(float));
    if (yy_lookup == NULL) return 0;
    dual_inner_prod(x, x, x - T0, N, &xx, &xy);
    yy_lookup[0] = xx;
    yy = xx;
    for (i = 1; i <= maxperiod; i++) {
        yy = yy + MULT16_16(x[-i], x[-i]) - MULT16_16(x[N - i], x[N - i]);
        yy_lookup[i] = MAX32(0, yy);
    }
    yy = yy_lookup[T0];
    best_xy = xy;
    best_yy = yy;
    g = g0 = compute_pitch_gain(xy, xx, yy);
    /* Look for any pitch at T/k */
    for (k = 2; k <= 15; k++) {
        int T1, T1b;
        float g1;
        float cont = 0;
        float thresh;
        T1 = (2 * T0 + k) / (2 * k);
        if (T1 < minperiod)
            break;
        /* Look for another strong correlation at T1b */
        if (k == 2) {
            if (T1 + T0 > maxperiod)
                T1b = T0;
            else
                T1b = T0 + T1;
        } else {
            T1b = (2 * second_check[k] * T0 + k) / (2 * k);
        }
        dual_inner_prod(x, &x[-T1], &x[-T1b], N, &xy, &xy2);
        xy = HALF32(xy + xy2);
        yy = HALF32(yy_lookup[T1] + yy_lookup[T1b]);
        g1 = compute_pitch_gain(xy, xx, yy);
        if (abs(T1 - prev_period) <= 1)
            cont = prev_gain;
        else if (abs(T1 - prev_period) <= 2 && 5 * k * k < T0)
            cont = HALF16(prev_gain);
        else
            cont = 0;
        thresh = MAX16(QCONST16(.3f, 15), MULT16_16_Q15(QCONST16(.7f, 15), g0) - cont);
        /* Bias against very high pitch (very short period) to avoid false-positives
       due to short-term correlation */
        if (T1 < 3 * minperiod)
            thresh = MAX16(QCONST16(.4f, 15), MULT16_16_Q15(QCONST16(.85f, 15), g0) - cont);
        else if (T1 < 2 * minperiod)
            thresh = MAX16(QCONST16(.5f, 15), MULT16_16_Q15(QCONST16(.9f, 15), g0) - cont);
        if (g1 > thresh) {
            best_xy = xy;
            best_yy = yy;
            T = T1;
            g = g1;
        }
    }
    best_xy = MAX32(0, best_xy);
    if (best_yy <= best_xy)
        pg = Q15ONE;
    else
        pg = best_xy / (best_yy + 1);

    for (k = 0; k < 3; k++)
        xcorr[k] = celt_inner_prod(x, x - (T + k - 1), N);
    if ((xcorr[2] - xcorr[0]) > MULT16_32_Q15(QCONST16(.7f, 15), xcorr[1] - xcorr[0]))
        offset = 1;
    else if ((xcorr[0] - xcorr[2]) > MULT16_32_Q15(QCONST16(.7f, 15), xcorr[1] - xcorr[2]))
        offset = -1;
    else
        offset = 0;
    if (pg > g)
        pg = g;
    *T0_ = 2 * T + offset;

    if (*T0_ < minperiod0)
        *T0_ = minperiod0;
    free(yy_lookup);
    return pg;
}

static RNN_INLINE float tansig_approx(float x) {
    int i;
    float y, dy;
    float sign = 1;
    /* Tests are reversed to catch NaNs */
    if (x >= 8)
        return 1;
    if (x <= -8)
        return -1;
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
        return 0;
    if (x < 0) {
        x = -x;
        sign = -1;
    }
    i = (int) floorf(.5f + 25 * x);
    x -= .04f * i;
    y = tansig_table[i];
    dy = 1 - y * y;
    y = y + x * dy * (1 - y * x);
    return sign * y;
}

static RNN_INLINE float sigmoid_approx(float x) {
    return .5f + .5f * tansig_approx(.5f * x);
}

static RNN_INLINE float relu(float x) {
    return x < 0 ? 0 : x;
}

void compute_dense(const DenseLayer *layer, float *output, const float *input) {
    int i, j;
    int N, M;
    int stride;
    M = layer->nb_inputs;
    N = layer->nb_neurons;
    stride = N;
    for (i = 0; i < N; i++) {
        /* Compute update gate. */
        float sum = layer->bias[i];
        for (j = 0; j < M; j++)
            sum += layer->input_weights[j * stride + i] * input[j];
        output[i] = WEIGHTS_SCALE * sum;
    }
    if (layer->activation == ACTIVATION_SIGMOID) {
        for (i = 0; i < N; i++)
            output[i] = sigmoid_approx(output[i]);
    } else if (layer->activation == ACTIVATION_TANH) {
        for (i = 0; i < N; i++)
            output[i] = tansig_approx(output[i]);
    } else if (layer->activation == ACTIVATION_RELU) {
        for (i = 0; i < N; i++)
            output[i] = relu(output[i]);
    } else {
        *(int *) 0 = 0;
    }
}

void compute_gru(const GRULayer *gru, float *state, const float *input) {
    int i, j;
    int N, M;
    int stride;
    float z[MAX_NEURONS];
    float r[MAX_NEURONS];
    float h[MAX_NEURONS];
    M = gru->nb_inputs;
    N = gru->nb_neurons;
    stride = 3 * N;
    for (i = 0; i < N; i++) {
        /* Compute update gate. */
        float sum = gru->bias[i];
        for (j = 0; j < M; j++)
            sum += gru->input_weights[j * stride + i] * input[j];
        for (j = 0; j < N; j++)
            sum += gru->recurrent_weights[j * stride + i] * state[j];
        z[i] = sigmoid_approx(WEIGHTS_SCALE * sum);
    }
    for (i = 0; i < N; i++) {
        /* Compute reset gate. */
        float sum = gru->bias[N + i];
        for (j = 0; j < M; j++)
            sum += gru->input_weights[N + j * stride + i] * input[j];
        for (j = 0; j < N; j++)
            sum += gru->recurrent_weights[N + j * stride + i] * state[j];
        r[i] = sigmoid_approx(WEIGHTS_SCALE * sum);
    }
    for (i = 0; i < N; i++) {
        /* Compute output. */
        float sum = gru->bias[2 * N + i];
        for (j = 0; j < M; j++)
            sum += gru->input_weights[2 * N + j * stride + i] * input[j];
        for (j = 0; j < N; j++)
            sum += gru->recurrent_weights[2 * N + j * stride + i] * state[j] * r[j];
        if (gru->activation == ACTIVATION_SIGMOID) sum = sigmoid_approx(WEIGHTS_SCALE * sum);
        else if (gru->activation == ACTIVATION_TANH) sum = tansig_approx(WEIGHTS_SCALE * sum);
        else if (gru->activation == ACTIVATION_RELU) sum = relu(WEIGHTS_SCALE * sum);
        else *(int *) 0 = 0;
        h[i] = z[i] * state[i] + (1 - z[i]) * sum;
    }
    for (i = 0; i < N; i++)
        state[i] = h[i];
}


void compute_rnn(RNNState *rnn, float *gains, float *vad, const float *input) {
    int i;
    float dense_out[MAX_NEURONS];
    float noise_input[MAX_NEURONS * 3];
    float denoise_input[MAX_NEURONS * 3];
    compute_dense(&input_dense, dense_out, input);
    compute_gru(&vad_gru, rnn->vad_gru_state, dense_out);
    compute_dense(&vad_output, vad, rnn->vad_gru_state);
    for (i = 0; i < INPUT_DENSE_SIZE; i++) noise_input[i] = dense_out[i];
    for (i = 0; i < VAD_GRU_SIZE; i++) noise_input[i + INPUT_DENSE_SIZE] = rnn->vad_gru_state[i];
    for (i = 0; i < INPUT_SIZE; i++) noise_input[i + INPUT_DENSE_SIZE + VAD_GRU_SIZE] = input[i];
    compute_gru(&noise_gru, rnn->noise_gru_state, noise_input);

    for (i = 0; i < VAD_GRU_SIZE; i++) denoise_input[i] = rnn->vad_gru_state[i];
    for (i = 0; i < NOISE_GRU_SIZE; i++) denoise_input[i + VAD_GRU_SIZE] = rnn->noise_gru_state[i];
    for (i = 0; i < INPUT_SIZE; i++) denoise_input[i + VAD_GRU_SIZE + NOISE_GRU_SIZE] = input[i];
    compute_gru(&denoise_gru, rnn->denoise_gru_state, denoise_input);
    compute_dense(&denoise_output, gains, rnn->denoise_gru_state);
}

const GRULayer denoise_gru = {
        denoise_gru_bias,
        denoise_gru_weights,
        denoise_gru_recurrent_weights,
        114, 96, ACTIVATION_RELU
};


const DenseLayer denoise_output = {
        denoise_output_bias,
        denoise_output_weights,
        96, 22, ACTIVATION_SIGMOID
};


const DenseLayer input_dense = {
        input_dense_bias,
        input_dense_weights,
        42, 24, ACTIVATION_TANH
};
const GRULayer vad_gru = {
        vad_gru_bias,
        vad_gru_weights,
        vad_gru_recurrent_weights,
        24, 24, ACTIVATION_RELU
};

const GRULayer noise_gru = {
        noise_gru_bias,
        noise_gru_weights,
        noise_gru_recurrent_weights,
        90, 48, ACTIVATION_RELU
};
const DenseLayer vad_output = {
        vad_output_bias,
        vad_output_weights,
        24, 1, ACTIVATION_SIGMOID
};


void denoise_frames(DenoiseState *st, short *data_in, int in_size, int frameSize) {
    size_t samples = in_size / frameSize;
    int patch = in_size % frameSize;
    if (patch != 0)
        return;
    if (data_in != 0 && st != 0) {
        int16_t *data_in_ptr = data_in;
        for (int i = 0; i < samples; ++i) {
            rnnoise_process_frame(st, data_in_ptr, data_in_ptr);
            data_in_ptr += frameSize;
        }
    }
}