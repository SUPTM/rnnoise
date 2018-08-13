
#include <stdio.h>
#include "rnnoise.h"
#include <stdlib.h>
#include <stdint.h>

#define DR_WAV_IMPLEMENTATION

#include "dr_wav.h"

#if   defined(__APPLE__)
# include <mach/mach_time.h>
#elif defined(_WIN32)
# define WIN32_LEAN_AND_MEAN

# include <windows.h>

#else // __linux

# include <time.h>

# ifndef  CLOCK_MONOTONIC //_RAW
#  define CLOCK_MONOTONIC CLOCK_REALTIME
# endif
#endif

static
uint64_t nanotimer() {
    static int ever = 0;
#if defined(__APPLE__)
    static mach_timebase_info_data_t frequency;
    if (!ever) {
        if (mach_timebase_info(&frequency) != KERN_SUCCESS) {
            return 0;
        }
        ever = 1;
    }
    return  (mach_absolute_time() * frequency.numer / frequency.denom);
#elif defined(_WIN32)
    static LARGE_INTEGER frequency;
    if (!ever) {
        QueryPerformanceFrequency(&frequency);
        ever = 1;
    }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (t.QuadPart * (uint64_t) 1e9) / frequency.QuadPart;
#else // __linux
    struct timespec t = {0};
    if (!ever) {
        if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
            return 0;
        }
        ever = 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * (uint64_t) 1e9) + t.tv_nsec;
#endif
}

static double now() {
    static uint64_t epoch = 0;
    if (!epoch) {
        epoch = nanotimer();
    }
    return (nanotimer() - epoch) / 1e9;
};

static double calcElapsed(double start, double end) {
    double took = -start;
    return took + end;
}


void wavWrite_int16(char *filename, int16_t *buffer, int sampleRate, uint32_t totalSampleCount) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = 1;
    format.sampleRate = (drwav_uint32) sampleRate;
    format.bitsPerSample = 16;
    drwav *pWav = drwav_open_file_write(filename, &format);
    if (pWav) {
        drwav_uint64 samplesWritten = drwav_write(pWav, totalSampleCount, buffer);
        drwav_uninit(pWav);
        if (samplesWritten != totalSampleCount) {
            fprintf(stderr, "ERROR\n");
            exit(1);
        }
    }
}

int16_t *wavRead_int16(char *filename, uint32_t *sampleRate, uint64_t *totalSampleCount) {
    unsigned int channels;
    int16_t *buffer = drwav_open_and_read_file_s16(filename, &channels, sampleRate, totalSampleCount);
    if (buffer == NULL) {
        fprintf(stderr, "ERROR\n");
        exit(1);
    }
    if (channels != 1) {
        drwav_free(buffer);
        buffer = NULL;
        *sampleRate = 0;
        *totalSampleCount = 0;
    }
    return buffer;
}

void splitpath(const char *path, char *drv, char *dir, char *name, char *ext) {
    const char *end;
    const char *p;
    const char *s;
    if (path[0] && path[1] == ':') {
        if (drv) {
            *drv++ = *path++;
            *drv++ = *path++;
            *drv = '\0';
        }
    } else if (drv)
        *drv = '\0';
    for (end = path; *end && *end != ':';)
        end++;
    for (p = end; p > path && *--p != '\\' && *p != '/';)
        if (*p == '.') {
            end = p;
            break;
        }
    if (ext)
        for (s = end; (*ext = *s++);)
            ext++;
    for (p = end; p > path;)
        if (*--p == '\\' || *p == '/') {
            p++;
            break;
        }
    if (name) {
        for (s = p; s < end;)
            *name++ = *s++;
        *name = '\0';
    }
    if (dir) {
        for (s = path; s < p;)
            *dir++ = *s++;
        *dir = '\0';
    }
}

void resampleData(const int16_t *sourceData, int32_t sampleRate, uint32_t srcSize, int16_t *destinationData,
                  int32_t newSampleRate) {
    if (sampleRate == newSampleRate) {
        memcpy(destinationData, sourceData, srcSize * sizeof(int16_t));
        return;
    }
    uint32_t last_pos = srcSize - 1;
    uint32_t dstSize = (uint32_t) (srcSize * ((float) newSampleRate / sampleRate));
    for (uint32_t idx = 0; idx < dstSize; idx++) {
        float index = ((float) idx * sampleRate) / (newSampleRate);
        uint32_t p1 = (uint32_t) index;
        float coef = index - p1;
        uint32_t p2 = (p1 == last_pos) ? last_pos : p1 + 1;
        destinationData[idx] = (int16_t) ((1.0f - coef) * sourceData[p1] + coef * sourceData[p2]);
    }
}

void denoise_proc(int16_t *buffer, uint32_t buffen_len) {
	#define  FRAME_SIZE   480 
    DenoiseState *st;
    st = rnnoise_create();
    int16_t patch_buffer[FRAME_SIZE];
    if (st != NULL) {
        uint32_t frames = buffen_len / FRAME_SIZE;
        uint32_t lastFrame = buffen_len % FRAME_SIZE;
        for (int i = 0; i < frames; ++i) {
            rnnoise_process_frame(st, buffer, buffer);
            buffer += FRAME_SIZE;
        }
        if (lastFrame != 0) {
            memset(patch_buffer, 0, FRAME_SIZE * sizeof(int16_t));
            memcpy(patch_buffer, buffer, lastFrame * sizeof(int16_t));
            rnnoise_process_frame(st, patch_buffer, patch_buffer);
            memcpy(buffer, patch_buffer, lastFrame * sizeof(int16_t));
        }
    }
    rnnoise_destroy(st);
}

void rnnDeNoise(char *in_file, char *out_file) {
    uint32_t in_sampleRate = 0;
    uint64_t in_size = 0;
    int16_t *data_in = wavRead_int16(in_file, &in_sampleRate, &in_size);
    uint32_t out_sampleRate = 48000;
    uint32_t out_size = (uint32_t) (in_size * ((float) out_sampleRate / in_sampleRate));
    int16_t *data_out = (int16_t *) malloc(out_size * sizeof(int16_t));
    if (data_in != NULL && data_out != NULL) {
        resampleData(data_in, in_sampleRate, (uint32_t) in_size, data_out, out_sampleRate);
        double startTime = now();
        denoise_proc(data_out, out_size);
        double time_interval = calcElapsed(startTime, now());
        printf("time interval: %d ms\n ", (int) (time_interval * 1000));
        resampleData(data_out, out_sampleRate, (uint32_t) out_size, data_in, in_sampleRate);
        wavWrite_int16(out_file, data_in, in_sampleRate, (uint32_t) in_size);
        free(data_in);
        free(data_out);
    } else {
        if (data_in) free(data_in);
        if (data_out) free(data_out);
    }
}


int main(int argc, char **argv) {
    printf("Audio Noise Reduction\n");
    printf("blog:http://cpuimage.cnblogs.com/\n");
    printf("e-mail:gaozhihan@vip.qq.com\n");
    if (argc < 2)
        return -1;

    char *in_file = argv[1];
    char drive[3];
    char dir[256];
    char fname[256];
    char ext[256];
    char out_file[1024];
    splitpath(in_file, drive, dir, fname, ext);
    sprintf(out_file, "%s%s%s_out%s", drive, dir, fname, ext);
    rnnDeNoise(in_file, out_file);
    printf("press any key to exit.\n");
    getchar();
    return 0;
}
