#include "djwaveform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <sndfile.h>
#include <omp.h>

/** Check if sampleRate is in {11025,22050,44100,48000,96000,192000} ± ~1Hz */
static int checkSampleRateAcceptable(float sr)
{
  static const float rates[] = {
      11025.f, 22050.f, 44100.f, 48000.f, 96000.f, 192000.f};
  for (int i = 0; i < (int)(sizeof(rates) / sizeof(rates[0])); i++)
  {
    float diff = fabsf(sr - rates[i]);
    if (diff < 1.f) // e
      return 1;
  }
  return 0;
}

/** Weighted color accumulation helper */
static inline void addWeightedColorF(float *rAcc, float *gAcc, float *bAcc,
                                     float weight,
                                     unsigned char R, unsigned char G, unsigned char B)
{
  *rAcc += weight * (float)R;
  *gAcc += weight * (float)G;
  *bAcc += weight * (float)B;
}

/** Apply a chosen window function in-place */
static void applyWindow(float *buf, int size, DJW_WindowType wt)
{
  if (wt == DJW_WINDOW_NONE)
    return;
  for (int i = 0; i < size; i++)
  {
    float alpha = 1.f;
    float ratio = (float)i / (float)(size - 1);
    switch (wt)
    {
    case DJW_WINDOW_HANN:
      alpha = 0.5f * (1.f - cosf(2.f * (float)M_PI * ratio));
      break;
    case DJW_WINDOW_HAMMING:
      alpha = 0.54f - 0.46f * cosf(2.f * (float)M_PI * ratio);
      break;
    case DJW_WINDOW_BLACKMAN:
      alpha = 0.42659f - 0.49656f * cosf(2.f * (float)M_PI * ratio) + 0.076849f * cosf(4.f * (float)M_PI * ratio);
      break;
    default:
      break;
    }
    buf[i] *= alpha;
  }
}

/** Frequency→color, either linear or log scale */
static void getColorFromFreq(float freq,
                             const DJW_ColorGradient *grad,
                             int useLog,
                             unsigned char *r, unsigned char *g, unsigned char *b)
{
  float key = freq;

  if (useLog)
  {
    // this is a bit arbitrary, but it works well
    key = 200.f * logf(1.f + freq / 50.f);
  }

  if (key <= grad->points[0].freq)
  {
    *r = grad->points[0].r;
    *g = grad->points[0].g;
    *b = grad->points[0].b;
    return;
  }

  if (key >= grad->points[grad->numPoints - 1].freq)
  {
    *r = grad->points[grad->numPoints - 1].r;
    *g = grad->points[grad->numPoints - 1].g;
    *b = grad->points[grad->numPoints - 1].b;
    return;
  }

  // linear interpolation
  for (int i = 0; i < grad->numPoints - 1; i++)
  {
    float f1 = (float)grad->points[i].freq;
    float f2 = (float)grad->points[i + 1].freq;

    if (key >= f1 && key < f2)
    {
      float t = (key - f1) / (f2 - f1);
      float rr = grad->points[i].r + (grad->points[i + 1].r - grad->points[i].r) * t;
      float gg = grad->points[i].g + (grad->points[i + 1].g - grad->points[i].g) * t;
      float bb = grad->points[i].b + (grad->points[i + 1].b - grad->points[i].b) * t;

      *r = (unsigned char)(rr + 0.5f);
      *g = (unsigned char)(gg + 0.5f);
      *b = (unsigned char)(bb + 0.5f);

      return;
    }
  }

  // fallback
  *r = grad->points[grad->numPoints - 1].r;
  *g = grad->points[grad->numPoints - 1].g;
  *b = grad->points[grad->numPoints - 1].b;
}

/** Compute final [start..end] sample range from config */
static int computeRange(const DJW_Config *cfg,
                        long long totalFrames,
                        float sr,
                        long long *startOut,
                        long long *endOut)
{
  if (cfg->rangeMode == DJW_RANGE_FULL)
  {
    *startOut = 0;
    *endOut = totalFrames;
    return DJW_OK;
  }

  long long st = cfg->rangeStart;
  long long en = cfg->rangeEnd;

  if (st < 0)
    st = 0;
  if (en < 0)
    en = 0;

  if (cfg->rangeMode == DJW_RANGE_MS)
  {
    double s1 = (double)st * sr / 1000.0;
    double s2 = (double)en * sr / 1000.0;
    st = (long long)(s1 + 0.5);
    en = (long long)(s2 + 0.5);
  }

  if (st > en)
    return DJW_ERR_RANGE_INVALID;

  if (st >= totalFrames)
    return DJW_ERR_RANGE_INVALID;

  if (en > totalFrames)
    en = totalFrames;

  *startOut = st;
  *endOut = en;
  return DJW_OK;
}

/** Per-thread FFT context to avoid re-initialization */
typedef struct
{
  fftwf_plan plan;
  float *timeBuf;
  fftwf_complex *freqBuf;
} DJW_ThreadCtx;

/**
 * For streaming columns, each column is a block of 3*height bytes
 */
typedef struct
{
  unsigned char *colData; // length = 3 * config.height
} DJW_Column;

struct DJW_Handle_
{
  DJW_Config config;
  int nThreads;

  // Per-thread
  DJW_ThreadCtx *tctx;

  // --------------- streaming state ---------------
  int streamingActive;
  int streamNumChannels;
  float streamSampleRate;
  long long streamTotalFrames; // if known, else -1
  long long streamFramesPushed;

  // We handle partial range in streaming as well
  long long streamStartSample;
  long long streamEndSample;

  // ring buffer for single-channel mix
  float *ringBuf;
  int ringFilled;
  int ringWritePos;

  // A dynamic array of columns, each column is 3*h->config.height bytes
  DJW_Column *columns;
  int columnsCount;
  int columnsAlloc; // how many allocated
};

/** Destroy handle => free everything */
void djw_destroy_handle(DJW_Handle *handle)
{
  if (!handle)
    return;

  // free per-thread
  if (handle->tctx)
  {
    for (int i = 0; i < handle->nThreads; i++)
    {
      if (handle->tctx[i].plan)
      {
        fftwf_destroy_plan(handle->tctx[i].plan);
      }
      if (handle->tctx[i].freqBuf)
      {
        fftwf_free(handle->tctx[i].freqBuf);
      }
      if (handle->tctx[i].timeBuf)
      {
        fftwf_free(handle->tctx[i].timeBuf);
      }
    }
    free(handle->tctx);
  }

  // free ring
  if (handle->ringBuf)
    free(handle->ringBuf);

  // free columns
  if (handle->columns)
  {
    for (int i = 0; i < handle->columnsCount; i++)
    {
      if (handle->columns[i].colData)
        free(handle->columns[i].colData);
    }
    free(handle->columns);
  }

  free(handle);
}

/** Create handle => prepare per-thread FFT plan, etc. */
DJW_Handle *djw_create_handle(const DJW_Config *cfg)
{
  if (!cfg)
    return NULL;

  if (cfg->width < 1 || cfg->height < 1 ||
      cfg->fftSize < 2 || cfg->overlapFactor < 1)
  {
    return NULL; // invalid
  }

  DJW_Handle *h = (DJW_Handle *)calloc(1, sizeof(DJW_Handle));
  if (!h)
    return NULL;
  h->config = *cfg;

  if (cfg->numThreads > 1)
    h->nThreads = cfg->numThreads;
  else
    h->nThreads = 1;

  // allocate thread contexts
  h->tctx = (DJW_ThreadCtx *)calloc(h->nThreads, sizeof(DJW_ThreadCtx));
  if (!h->tctx)
  {
    free(h);
    return NULL;
  }

  int N = cfg->fftSize;
  for (int i = 0; i < h->nThreads; i++)
  {
    h->tctx[i].timeBuf = (float *)fftwf_malloc(sizeof(float) * N);
    h->tctx[i].freqBuf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N);

    if (!h->tctx[i].timeBuf || !h->tctx[i].freqBuf)
    {
      djw_destroy_handle(h);
      return NULL;
    }

    // see https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html
    h->tctx[i].plan = fftwf_plan_dft_r2c_1d(N,
                                            h->tctx[i].timeBuf,
                                            h->tctx[i].freqBuf,
                                            FFTW_ESTIMATE);
    if (!h->tctx[i].plan)
    {
      djw_destroy_handle(h);
      return NULL;
    }
  }

  return h;
}

/** File→File (non-streaming) */
int djw_generate_waveform_file(const char *inFilename,
                               const char *outFilename,
                               const DJW_Config *cfg)
{
  // open with libsndfile
  SF_INFO sfinfo;
  memset(&sfinfo, 0, sizeof(sfinfo));
  SNDFILE *sf = sf_open(inFilename, SFM_READ, &sfinfo);

  if (!sf)
  {
    fprintf(stderr, "[djwaveform] cannot open '%s'\n", inFilename);
    return DJW_ERR_FILE_OPEN;
  }

  // check channel <=2, sampleRate acceptable
  if (sfinfo.channels < 1 || sfinfo.channels > 2)
  {
    sf_close(sf);
    return DJW_ERR_UNSUPPORTED_FMT;
  }

  if (!checkSampleRateAcceptable((float)sfinfo.samplerate))
  {
    sf_close(sf);
    return DJW_ERR_UNSUPPORTED_FMT;
  }

  long long totalFrames = sfinfo.frames;
  float *samples = (float *)malloc(sfinfo.channels * sizeof(float) * totalFrames);

  if (!samples)
  {
    sf_close(sf);
    return DJW_ERR_MEMORY;
  }

  sf_count_t got = sf_readf_float(sf, samples, totalFrames);
  sf_close(sf);
  if (got < totalFrames)
    totalFrames = got;

  // create handle
  DJW_Handle *h = djw_create_handle(cfg);
  if (!h)
  {
    free(samples);
    return DJW_ERR_MEMORY;
  }

  // allocate output
  uint8_t *rgb = (uint8_t *)calloc(3, cfg->width * cfg->height);
  if (!rgb)
  {
    free(samples);
    djw_destroy_handle(h);
    return DJW_ERR_MEMORY;
  }

  // do memory-based
  int ret = djw_generate_waveform_memory(samples,
                                         sfinfo.channels,
                                         totalFrames,
                                         h, rgb);

  if (ret == DJW_OK)
  {
    // write PPM
    FILE *fout = fopen(outFilename, "wb");
    if (!fout)
    {
      free(rgb);
      free(samples);
      djw_destroy_handle(h);
      return DJW_ERR_FILE_OPEN;
    }

    // PPM P6
    fprintf(fout, "P6\n%d %d\n255\n", cfg->width, cfg->height);
    fwrite(rgb, 3, cfg->width * cfg->height, fout);
    fclose(fout);
  }

  free(rgb);
  free(samples);
  djw_destroy_handle(h);
  return ret;
}

/** Non-streaming, memory→memory */
int djw_generate_waveform_memory(const float *interleavedSamples,
                                 int numChannels,
                                 long long numFrames,
                                 DJW_Handle *handle,
                                 uint8_t *outRGB)
{
  if (!handle || !interleavedSamples || !outRGB)
    return DJW_ERR_INVALID_PARAM;

  const DJW_Config *cfg = &handle->config;

  if (numFrames < cfg->fftSize)
  {
    // trivial => no visible waveform
    memset(outRGB, 0, 3 * cfg->width * cfg->height);
    return DJW_OK;
  }

  if (numChannels < 1 || numChannels > 2)
  {
    memset(outRGB, 0, 3 * cfg->width * cfg->height);
    return DJW_ERR_UNSUPPORTED_FMT;
  }

  // figure sample rate
  float sr = (cfg->sampleRate > 0.f) ? cfg->sampleRate : 44100.f;
  if (!checkSampleRateAcceptable(sr))
  {
    memset(outRGB, 0, 3 * cfg->width * cfg->height);
    return DJW_ERR_UNSUPPORTED_FMT;
  }

  // compute range
  long long stS, enS;
  int rc = computeRange(cfg, numFrames, sr, &stS, &enS);
  if (rc != DJW_OK)
  {
    memset(outRGB, 0, 3 * cfg->width * cfg->height);
    return rc;
  }

  long long length = enS - stS;
  if (length < cfg->fftSize)
  {
    // empty
    memset(outRGB, 0, 3 * cfg->width * cfg->height);
    return DJW_OK;
  }

  // prepare channel data
  int outCh = (cfg->channelMode == DJW_MULTI_STACK ? numChannels : 1);
  float **chanData = (float **)calloc(outCh, sizeof(float *));
  if (!chanData)
  {
    memset(outRGB, 0, 3 * cfg->width * cfg->height);
    return DJW_ERR_MEMORY;
  }

  for (int c = 0; c < outCh; c++)
  {
    chanData[c] = (float *)calloc(length, sizeof(float));
    if (!chanData[c])
    {
      for (int x = 0; x < c; x++)
      {
        free(chanData[x]);
      }
      free(chanData);
      memset(outRGB, 0, 3 * cfg->width * cfg->height);
      return DJW_ERR_MEMORY;
    }
  }

  // fill
  if (cfg->channelMode == DJW_MULTI_STACK)
  {
    // one channel on top of the other (squeezed into the same height)
    for (long long f = 0; f < length; f++)
    {
      long long inf = stS + f;
      for (int c = 0; c < numChannels; c++)
      {
        if (c < outCh)
        {
          chanData[c][f] = interleavedSamples[inf * numChannels + c];
        }
      }
    }
  }
  else
  {
    // mixdown
    for (long long f = 0; f < length; f++)
    {
      long long inf = stS + f;
      double sum = 0.0;
      for (int c = 0; c < numChannels; c++)
      {
        sum += interleavedSamples[inf * numChannels + c];
      }
      chanData[0][f] = (float)(sum / (double)numChannels);
    }
  }

  // build columns => total = cfg->width
  // each column => fraction col/(width-1)
  // wStart = fraction*(length-fftSize)
  double lastW = (double)(length - cfg->fftSize);
  if (lastW < 0)
    lastW = 0;

  typedef struct
  {
    unsigned char *colData; // 3 * cfg->height
  } ColTmp;

  ColTmp *cols = (ColTmp *)calloc(cfg->width, sizeof(ColTmp));

  if (!cols)
  {
    for (int c = 0; c < outCh; c++)
      free(chanData[c]);
    free(chanData);
    memset(outRGB, 0, 3 * cfg->width * cfg->height);
    return DJW_ERR_MEMORY;
  }

  for (int i = 0; i < cfg->width; i++)
  {
    cols[i].colData = (unsigned char *)calloc(3, cfg->height);
  }

  // If multi-stack, each channel has a sub-height
  int channelHeight = cfg->height / outCh;

#pragma omp parallel for num_threads(handle->nThreads)
  for (int col = 0; col < cfg->width; col++)
  {
    int tid = omp_get_thread_num();
    DJW_ThreadCtx *tc = &handle->tctx[tid];

    double frac = (cfg->width > 1) ? (double)col / (cfg->width - 1) : 0.0;
    double ws = frac * lastW;
    long long wStart = (long long)(ws + 0.5);

    if (wStart + cfg->fftSize > length)
    {
      if (length > cfg->fftSize)
        wStart = length - cfg->fftSize;
      else
        wStart = 0;
    }

    for (int chIdx = 0; chIdx < outCh; chIdx++)
    {
      // copy & find peak
      float peakAmp = 0.f;

      for (int i = 0; i < cfg->fftSize; i++)
      {
        float s = chanData[chIdx][wStart + i];
        if (fabsf(s) > peakAmp)
          peakAmp = fabsf(s);
        tc->timeBuf[i] = s;
      }

      applyWindow(tc->timeBuf, cfg->fftSize, cfg->windowType);
      fftwf_execute(tc->plan);

      float totalP = 0.f, accR = 0.f, accG = 0.f, accB = 0.f;
      int N2 = cfg->fftSize / 2;

      for (int k = 1; k <= N2; k++)
      {
        float re = tc->freqBuf[k][0];
        float im = tc->freqBuf[k][1];
        float pwr = re * re + im * im;

        if (pwr > 0.f)
        {
          float freqHz = (float)k * (sr / (float)cfg->fftSize);
          unsigned char R, G, B;
          getColorFromFreq(freqHz, &cfg->gradient, cfg->useLogFreq, &R, &G, &B);
          addWeightedColorF(&accR, &accG, &accB, pwr, R, G, B);
        }

        totalP += pwr;
      }

      unsigned char cR = 0, cG = 0, cB = 0;

      if (totalP > 1e-12f)
      {
        float inv = 1.f / totalP;
        float rr = accR * inv, gg = accG * inv, bb = accB * inv;

        if (rr > 255.f)
          rr = 255.f;
        if (gg > 255.f)
          gg = 255.f;
        if (bb > 255.f)
          bb = 255.f;

        cR = (unsigned char)(rr + 0.5f);
        cG = (unsigned char)(gg + 0.5f);
        cB = (unsigned char)(bb + 0.5f);
      }

      if (peakAmp > 1.f)
        peakAmp = 1.f;

      int yStart = chIdx * channelHeight;
      int yEnd = yStart + channelHeight;
      int center = yStart + channelHeight / 2;
      float amp = peakAmp * (float)(channelHeight / 2);

      int yMin = (int)(center - amp);
      if (yMin < yStart)
        yMin = yStart;

      int yMax = (int)(center + amp);
      if (yMax >= yEnd)
        yMax = yEnd - 1;

      for (int row = yMin; row <= yMax; row++)
      {
        int idx = row * 3;
        cols[col].colData[idx + 0] = cR;
        cols[col].colData[idx + 1] = cG;
        cols[col].colData[idx + 2] = cB;
      }
    }
  }

  // fill outRGB
  for (int col = 0; col < cfg->width; col++)
  {
    for (int row = 0; row < cfg->height; row++)
    {
      int outI = (row * cfg->width + col) * 3;
      int inI = row * 3;
      outRGB[outI + 0] = cols[col].colData[inI + 0];
      outRGB[outI + 1] = cols[col].colData[inI + 1];
      outRGB[outI + 2] = cols[col].colData[inI + 2];
    }
  }

  // cleanup
  for (int i = 0; i < cfg->width; i++)
  {
    free(cols[i].colData);
  }

  free(cols);

  for (int c = 0; c < outCh; c++)
  {
    free(chanData[c]);
  }

  free(chanData);
  return DJW_OK;
}

/* ------------------------------------------------------------------------
   STREAMING IMPLEMENTATION
   We produce *one column per hop* and store them in handle->columns[].
   Later, at djw_stream_end, we "resample" these columns to exactly cfg->width.
   This approach handles unknown total frames, partial ranges, etc.
 ------------------------------------------------------------------------*/

/** Realloc column array if needed */
static int ensureColumnCapacity(DJW_Handle *h)
{
  if (h->columnsCount >= h->columnsAlloc)
  {
    int newCap = (h->columnsAlloc < 8) ? 8 : h->columnsAlloc * 2;

    DJW_Column *p = (DJW_Column *)realloc(h->columns, newCap * sizeof(DJW_Column));

    if (!p)
      return DJW_ERR_MEMORY;

    h->columns = p;

    // set newly allocated portion to zero
    for (int i = h->columnsAlloc; i < newCap; i++)
    {
      h->columns[i].colData = NULL;
    }

    h->columnsAlloc = newCap;
  }

  return DJW_OK;
}

/** Actually compute 1 column from ringBuf => store in new columns[] entry */
static int streamingComputeOneColumn(DJW_Handle *h)
{
  // single-thread for demo; if multi-thread needed, store ring in local copy
  // for now, straightforward approach:
  int idx = h->columnsCount;
  int err = ensureColumnCapacity(h);
  if (err != DJW_OK)
    return err;

  // allocate colData
  h->columns[idx].colData = (unsigned char *)calloc(3, h->config.height);
  if (!h->columns[idx].colData)
    return DJW_ERR_MEMORY;

  // gather from ringBuf
  float peakAmp = 0.f;
  DJW_ThreadCtx *tc = &(h->tctx[0]); // just use thread 0 for streaming
  // copy ring => timeBuf
  for (int i = 0; i < h->config.fftSize; i++)
  {
    float s = h->ringBuf[i];
    if (fabsf(s) > peakAmp)
      peakAmp = fabsf(s);
    tc->timeBuf[i] = s;
  }
  applyWindow(tc->timeBuf, h->config.fftSize, h->config.windowType);
  fftwf_execute(tc->plan);

  // accumulate color
  float totalP = 0.f, accR = 0.f, accG = 0.f, accB = 0.f;
  int N2 = h->config.fftSize / 2;
  float sr = (h->config.sampleRate > 0.f) ? h->config.sampleRate : h->streamSampleRate;
  for (int k = 1; k <= N2; k++)
  {
    float re = tc->freqBuf[k][0];
    float im = tc->freqBuf[k][1];
    float pwr = re * re + im * im;
    if (pwr > 0.f)
    {
      float freqHz = (float)k * (sr / (float)h->config.fftSize);
      unsigned char R, G, B;
      getColorFromFreq(freqHz, &h->config.gradient, h->config.useLogFreq, &R, &G, &B);
      addWeightedColorF(&accR, &accG, &accB, pwr, R, G, B);
    }
    totalP += pwr;
  }
  unsigned char cR = 0, cG = 0, cB = 0;
  if (totalP > 1e-12f)
  {
    float inv = 1.f / totalP;
    float rr = accR * inv, gg = accG * inv, bb = accB * inv;
    if (rr > 255.f)
      rr = 255.f;
    if (gg > 255.f)
      gg = 255.f;
    if (bb > 255.f)
      bb = 255.f;
    cR = (unsigned char)(rr + 0.5f);
    cG = (unsigned char)(gg + 0.5f);
    cB = (unsigned char)(bb + 0.5f);
  }

  // amplitude => fill center
  if (peakAmp > 1.f)
    peakAmp = 1.f;
  int center = h->config.height / 2;
  float amp = peakAmp * (float)(h->config.height / 2);
  int yMin = (int)(center - amp);
  if (yMin < 0)
    yMin = 0;
  int yMax = (int)(center + amp);
  if (yMax >= h->config.height)
    yMax = h->config.height - 1;
  for (int row = yMin; row <= yMax; row++)
  {
    int px = row * 3;
    h->columns[idx].colData[px + 0] = cR;
    h->columns[idx].colData[px + 1] = cG;
    h->columns[idx].colData[px + 2] = cB;
  }

  h->columnsCount++;
  return DJW_OK;
}

int djw_stream_begin(DJW_Handle *handle,
                     int numChannels,
                     float sampleRate,
                     long long totalFrames)
{
  if (!handle)
    return DJW_ERR_INVALID_PARAM;
  if (numChannels < 1 || numChannels > 2)
    return DJW_ERR_UNSUPPORTED_FMT;
  if (!checkSampleRateAcceptable(sampleRate))
    return DJW_ERR_UNSUPPORTED_FMT;

  handle->streamingActive = 1;
  handle->streamNumChannels = numChannels;
  handle->streamSampleRate = sampleRate;
  handle->streamTotalFrames = totalFrames;
  handle->streamFramesPushed = 0;

  // ring
  if (!handle->ringBuf)
  {
    handle->ringBuf = (float *)calloc(handle->config.fftSize, sizeof(float));
    if (!handle->ringBuf)
      return DJW_ERR_MEMORY;
  }
  handle->ringFilled = 0;
  handle->ringWritePos = 0;

  // columns array
  if (!handle->columns)
  {
    handle->columnsAlloc = 0;
    handle->columnsCount = 0;
    handle->columns = NULL;
  }

  // compute range if totalFrames>=0
  // If user wants partial range, we store it in streamStartSample..streamEndSample
  // If totalFrames<0 => we do not fully confirm end; we'll do it in stream_end
  if (handle->config.rangeMode == DJW_RANGE_FULL)
  {
    handle->streamStartSample = 0;
    handle->streamEndSample = (totalFrames < 0) ? (1LL << 50) : totalFrames; // large if unknown
  }
  else
  {
    // we do a best guess. If totalFrames<0 => we wait until stream_end to clamp
    // or we compute partial if totalFrames>0
    if (totalFrames >= 0)
    {
      long long st = 0, en = 0;
      int rc = computeRange(&handle->config, totalFrames, sampleRate, &st, &en);
      if (rc != DJW_OK)
      {
        // let's store a dummy (the user might fix it in end)
        handle->streamStartSample = 0;
        handle->streamEndSample = 0;
        return rc;
      }
      handle->streamStartSample = st;
      handle->streamEndSample = en;
    }
    else
    {
      // unknown total => just store partial for now
      // We'll finalize in stream_end
      handle->streamStartSample = handle->config.rangeStart; // if ms => we convert in end
      handle->streamEndSample = (1LL << 50);                 // big
    }
  }
  return DJW_OK;
}

/** shift ring left by 'hop' */
static void ringShiftLeft(DJW_Handle *h, int hop)
{
  int remain = h->config.fftSize - hop;
  for (int i = 0; i < remain; i++)
  {
    h->ringBuf[i] = h->ringBuf[i + hop];
  }
  h->ringWritePos -= hop;
  if (h->ringWritePos < 0)
    h->ringWritePos = 0;
  h->ringFilled -= hop;
}

int djw_stream_push_frames(DJW_Handle *handle, const float *frames, int nFrames)
{
  if (!handle || !handle->streamingActive || !frames)
    return DJW_ERR_INVALID_PARAM;
  int numCh = handle->streamNumChannels;
  int hop = handle->config.fftSize / handle->config.overlapFactor;

  for (int i = 0; i < nFrames; i++)
  {
    long long globalSamp = handle->streamFramesPushed + i;
    // If we haven't fully resolved endSample (when totalFrames<0 + partial range), we do a partial approach
    // Let's skip samples < streamStartSample, or beyond streamEndSample
    if (globalSamp < handle->streamStartSample)
    {
      continue; // skip
    }
    if (globalSamp >= handle->streamEndSample)
    {
      // ignore => done reading the range
      break;
    }
    // mixdown
    double sum = 0.0;
    for (int c = 0; c < numCh; c++)
    {
      sum += frames[i * numCh + c];
    }
    float mono = (float)(sum / (double)numCh);

    // push to ring
    handle->ringBuf[handle->ringWritePos] = mono;
    handle->ringWritePos++;
    handle->ringFilled++;
    if (handle->ringWritePos >= handle->config.fftSize)
    {
      handle->ringWritePos = 0;
    }

    // produce columns if ringFilled >= fftSize
    while (handle->ringFilled >= handle->config.fftSize)
    {
      // compute one column => append to handle->columns
      int e = streamingComputeOneColumn(handle);
      if (e != DJW_OK)
        return e;
      ringShiftLeft(handle, hop);
    }
  }
  handle->streamFramesPushed += nFrames;
  return DJW_OK;
}

/** Resample columns[] => exactly config.width columns, store in outRGB */
static void finalizeStreamingImage(DJW_Handle *h, uint8_t *outRGB)
{
  // If we have columnsCount == 0 => fill black
  if (h->columnsCount < 1)
  {
    memset(outRGB, 0, 3 * h->config.width * h->config.height);
    return;
  }
  // If columnsCount == width => direct copy
  if (h->columnsCount == h->config.width)
  {
    for (int col = 0; col < h->config.width; col++)
    {
      for (int row = 0; row < h->config.height; row++)
      {
        int outI = (row * h->config.width + col) * 3;
        int inI = (row * 3);
        outRGB[outI + 0] = h->columns[col].colData[inI + 0];
        outRGB[outI + 1] = h->columns[col].colData[inI + 1];
        outRGB[outI + 2] = h->columns[col].colData[inI + 2];
      }
    }
    return;
  }
  // If columnsCount < width => upsample
  // If columnsCount > width => downsample
  // We do a simple linear approach
  for (int x = 0; x < h->config.width; x++)
  {
    float frac = (h->config.width > 1) ? (float)x / (float)(h->config.width - 1) : 0.f;
    float src = frac * (float)(h->columnsCount - 1);
    int i0 = (int)floorf(src);
    int i1 = i0 + 1;
    float alpha = src - (float)i0;
    if (i0 < 0)
      i0 = 0;
    if (i1 >= h->columnsCount)
      i1 = h->columnsCount - 1;

    for (int row = 0; row < h->config.height; row++)
    {
      int outI = (row * h->config.width + x) * 3;
      int inRow = row * 3;
      // bilinear in the horizontal dimension
      // column i0 => c0, column i1 => c1
      unsigned char c0r = h->columns[i0].colData[inRow + 0];
      unsigned char c0g = h->columns[i0].colData[inRow + 1];
      unsigned char c0b = h->columns[i0].colData[inRow + 2];
      unsigned char c1r = h->columns[i1].colData[inRow + 0];
      unsigned char c1g = h->columns[i1].colData[inRow + 1];
      unsigned char c1b = h->columns[i1].colData[inRow + 2];
      float rr = ((float)c0r + alpha * ((float)c1r - (float)c0r));
      float gg = ((float)c0g + alpha * ((float)c1g - (float)c0g));
      float bb = ((float)c0b + alpha * ((float)c1b - (float)c0b));
      // clamp
      if (rr < 0.f)
        rr = 0.f;
      if (rr > 255.f)
        rr = 255.f;
      if (gg < 0.f)
        gg = 0.f;
      if (gg > 255.f)
        gg = 255.f;
      if (bb < 0.f)
        bb = 0.f;
      if (bb > 255.f)
        bb = 255.f;
      outRGB[outI + 0] = (unsigned char)(rr + 0.5f);
      outRGB[outI + 1] = (unsigned char)(gg + 0.5f);
      outRGB[outI + 2] = (unsigned char)(bb + 0.5f);
    }
  }
}

int djw_stream_end(DJW_Handle *handle, uint8_t *outRGB)
{
  if (!handle || !handle->streamingActive || !outRGB)
    return DJW_ERR_INVALID_PARAM;

  // If range was unknown and we have partial user config => finalize range now
  // But we already skipped frames beyond end in push.
  // We might do a final flush if we want the last partial window.
  // Typically, wave displays only columns for full windows. We'll skip partial leftover.
  // If you wanted to do a partial final column, you'd do it here.

  handle->streamingActive = 0;
  // finalize the image => resample handle->columns => outRGB
  finalizeStreamingImage(handle, outRGB);
  return DJW_OK;
}
