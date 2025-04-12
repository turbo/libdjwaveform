#ifndef DJWAVEFORM_H
#define DJWAVEFORM_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>  // for uint8_t
#include <sndfile.h> // SF_INFO

/**
 * Return codes
 */
#define DJW_OK 0
#define DJW_ERR_MEMORY 1
#define DJW_ERR_FILE_OPEN 2
#define DJW_ERR_SF_READ 3
#define DJW_ERR_INVALID_PARAM 4
#define DJW_ERR_FFTW_INIT 5
#define DJW_ERR_RANGE_INVALID 6
#define DJW_ERR_UNSUPPORTED_FMT 7
#define DJW_ERR_UNKNOWN 99

  /**
   * Frequency→Color control point
   */
  typedef struct
  {
    int freq;              // frequency threshold (Hz)
    unsigned char r, g, b; // color at that threshold
  } DJW_FreqColorPoint;

  /**
   * Color gradient: an array of freq→color control points, sorted asc by freq
   */
  typedef struct
  {
    const DJW_FreqColorPoint *points; // array pointer
    int numPoints;                    // count
  } DJW_ColorGradient;

  /**
   * Window function enum
   */
  typedef enum
  {
    DJW_WINDOW_NONE = 0,
    DJW_WINDOW_HANN,
    DJW_WINDOW_HAMMING,
    DJW_WINDOW_BLACKMAN
  } DJW_WindowType;

  /**
   * Multi-channel rendering mode
   */
  typedef enum
  {
    DJW_MULTI_MIXDOWN = 0, // average all channels into one
    DJW_MULTI_STACK        // each channel gets its own vertical band
  } DJW_MultiChannelMode;

  /**
   * Range mode: process entire audio, or partial by sample index or ms
   */
  typedef enum
  {
    DJW_RANGE_FULL = 0,
    DJW_RANGE_SAMPLES,
    DJW_RANGE_MS
  } DJW_RangeMode;

  /**
   * Configuration struct
   */
  typedef struct
  {
    int width;  // final image width
    int height; // final image height

    int fftSize;       // e.g. 2048
    int overlapFactor; // e.g. 2 => 50%, 4 => 75%

    DJW_WindowType windowType;
    DJW_ColorGradient gradient;

    int useLogFreq; // 0 => linear, 1 => log freq mapping
    DJW_MultiChannelMode channelMode;

    float sampleRate; // if 0 => use file's rate; else override
    int numThreads;   // if >1 => parallel columns

    DJW_RangeMode rangeMode;
    long long rangeStart; // start sample or ms
    long long rangeEnd;   // end sample or ms
  } DJW_Config;

  /**
   * Opaque handle for library usage
   */
  typedef struct DJW_Handle_ DJW_Handle;

  /**
   * Create/destroy handle
   */
  DJW_Handle *djw_create_handle(const DJW_Config *cfg);
  void djw_destroy_handle(DJW_Handle *handle);

  /**
   * Non-streaming: read entire audio with libsndfile, write final PPM
   */
  int djw_generate_waveform_file(const char *inFilename,
                                 const char *outFilename,
                                 const DJW_Config *cfg);

  /**
   * Non-streaming: memory in (float samples), memory out (RGB)
   */
  int djw_generate_waveform_memory(const float *interleavedSamples,
                                   int numChannels,
                                   long long numFrames,
                                   DJW_Handle *handle,
                                   uint8_t *outRGB);

  /**
   * ---------------- STREAMING API ----------------
   *
   * Allows incrementally feeding audio. Accumulate columns in memory, then at 
   * the end resample those columns to produce exactly cfg->width columns in 
   * the final image.
   */

  /**
   * Begin streaming
   *
   * @param handle        a valid DJW_Handle
   * @param numChannels   must be 1 or 2
   * @param sampleRate    must be in {11025,22050,44100,48000,96000,192000} ± small tolerance
   * @param totalFrames   if known, >=0. If unknown, pass -1
   *
   * The library will prepare for incremental push. If the user wants partial
   * range, we apply it once we know totalFrames or in stream_end if totalFrames<0.
   */
  int djw_stream_begin(DJW_Handle *handle,
                       int numChannels,
                       float sampleRate,
                       long long totalFrames);

  /**
   * Push frames (float interleaved). The library does ring-buffer accumulation.
   * Each time we have enough samples for a “hop”, we generate one “column” internally.
   *
   * @param handle     streaming must be in progress
   * @param frames     pointer to nFrames * numChannels floats
   * @param nFrames    how many frames in this chunk
   */
  int djw_stream_push_frames(DJW_Handle *handle, const float *frames, int nFrames);

  /**
   * End streaming and produce final image data (width*height*3).
   *
   * The library finalizes partial windows, has an array of columns,
   * and resamples them to exactly `cfg->width` columns. Each column
   * is `cfg->height` tall. The result is placed in outRGB.
   */
  int djw_stream_end(DJW_Handle *handle, uint8_t *outRGB);

#ifdef __cplusplus
}
#endif

#endif /* DJWAVEFORM_H */
