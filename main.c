#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "djwaveform.h"
#include <sndfile.h>

static DJW_FreqColorPoint sPoints[] = {
    {10, 0x83, 0x1E, 0x1E},
    {50, 0x94, 0x1E, 0x1E},
    {100, 0xA1, 0x1E, 0x1E},
    {150, 0xBF, 0x1E, 0x1E},
    {250, 0xBF, 0x1E, 0x1E},
    {300, 0xBD, 0x2C, 0x1E},
    {350, 0x8A, 0x48, 0x1F},
    {400, 0x91, 0x59, 0x1E},
    {450, 0x73, 0x60, 0x1E},
    {500, 0x49, 0x63, 0x1E},
    {550, 0x42, 0x71, 0x1F},
    {600, 0x30, 0x61, 0x1E},
    {650, 0x33, 0xA6, 0x1D},
    {700, 0x27, 0xBF, 0x1E},
    {800, 0x27, 0xBF, 0x1E},
    {850, 0x1D, 0x9D, 0x1F},
    {900, 0x1D, 0x9D, 0x1F},
    {950, 0x1E, 0x8D, 0x1F},
    {1300, 0x1E, 0x8D, 0x1F},
    {1350, 0x1D, 0xBF, 0x25},
    {1450, 0x1D, 0xBF, 0x25},
    {1500, 0x1E, 0xA8, 0x2D},
    {1550, 0x1E, 0x5E, 0x2D},
    {1600, 0x1E, 0x73, 0x2D},
    {1650, 0x1E, 0x5E, 0x2D},
    {2600, 0x1E, 0x5E, 0x2D},
    {2650, 0x1E, 0x5E, 0x3B},
    {2700, 0x1E, 0x5E, 0x4F},
    {2750, 0x1E, 0x5C, 0x6D},
    {2800, 0x1D, 0x54, 0x79},
    {2850, 0x1E, 0x42, 0x79},
    {2900, 0x1E, 0x22, 0x6A},
    {2950, 0x1E, 0x1E, 0x61},
    {3000, 0x1E, 0x1E, 0x5C},
    {5400, 0x1E, 0x1E, 0x5C},
    {5450, 0x1E, 0x1E, 0x71},
    {5500, 0x1E, 0x1E, 0x85},
    {5550, 0x1E, 0x1E, 0xB5},
    {5600, 0x1E, 0x1E, 0xBF},
    {5650, 0x1E, 0x1E, 0xBF},
    {5700, 0x1E, 0x1E, 0xAD},
    {5750, 0x1E, 0x1E, 0x94},
    {5850, 0x1E, 0x1E, 0x94},
    {5900, 0x1E, 0x1E, 0x87},
    {10700, 0x1E, 0x1E, 0x87},
    {10750, 0x28, 0x28, 0xD6},
    {10800, 0x28, 0x28, 0xB9},
    {10850, 0x28, 0x28, 0xF0}};


int main(int argc, char **argv)
{
  if (argc < 5)
  {
    fprintf(stderr, "Usage: %s <input.wav> <output.ppm> <width> <height> [--stream]\n", argv[0]);
    fprintf(stderr, "  e.g. %s song.wav out.ppm 1200 200\n", argv[0]);
    return 1;
  }
  const char *inFile = argv[1];
  const char *outFile = argv[2];
  int width = atoi(argv[3]);
  int height = atoi(argv[4]);

  int useStreaming = 0;
  for (int i = 5; i < argc; i++)
  {
    if (!strcmp(argv[i], "--stream"))
    {
      useStreaming = 1;
    }
  }

  // Build a color gradient
  DJW_ColorGradient grad;
  grad.points = sPoints;
  grad.numPoints = (int)(sizeof(sPoints) / sizeof(sPoints[0]));

  // Build config
  DJW_Config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.width = width;
  cfg.height = height;
  cfg.fftSize = 2048;
  cfg.overlapFactor = 2; // 50%
  cfg.windowType = DJW_WINDOW_HANN;
  cfg.gradient = grad;
  cfg.useLogFreq = 0;
  cfg.channelMode = DJW_MULTI_MIXDOWN;
  cfg.sampleRate = 0.f;           // use actual
  cfg.numThreads = 4;             // parallel for non-streaming
  cfg.rangeMode = DJW_RANGE_FULL; // entire audio
  cfg.rangeStart = 0;
  cfg.rangeEnd = 0; // ignored if FULL

  if (!useStreaming)
  {
    // Non-streaming approach
    int ret = djw_generate_waveform_file(inFile, outFile, &cfg);
    if (ret != DJW_OK)
    {
      fprintf(stderr, "Error code=%d from djw_generate_waveform_file\n", ret);
      return ret;
    }
    printf("Wrote %s (non-streaming) OK\n", outFile);
  }
  else
  {
    // Streaming approach
    // 1) open with libsndfile
    SF_INFO sfinfo;
    memset(&sfinfo, 0, sizeof(sfinfo));
    SNDFILE *sf = sf_open(inFile, SFM_READ, &sfinfo);
    if (!sf)
    {
      fprintf(stderr, "Cannot open '%s'\n", inFile);
      return DJW_ERR_FILE_OPEN;
    }
    if (sfinfo.channels < 1 || sfinfo.channels > 2)
    {
      sf_close(sf);
      fprintf(stderr, "Only up to 2 channels supported.\n");
      return DJW_ERR_UNSUPPORTED_FMT;
    }
    // create handle
    DJW_Handle *h = djw_create_handle(&cfg);
    if (!h)
    {
      sf_close(sf);
      fprintf(stderr, "djw_create_handle failed.\n");
      return DJW_ERR_MEMORY;
    }
    // totalFrames known
    long long totalFrames = sfinfo.frames;
    float sr = (float)sfinfo.samplerate;
    int ret = djw_stream_begin(h, sfinfo.channels, sr, totalFrames);
    if (ret != DJW_OK)
    {
      sf_close(sf);
      djw_destroy_handle(h);
      fprintf(stderr, "djw_stream_begin error=%d\n", ret);
      return ret;
    }
    // push frames in small blocks
    const int BLOCK = 1024;
    float *buf = (float *)malloc(sizeof(float) * BLOCK * sfinfo.channels);
    if (!buf)
    {
      sf_close(sf);
      djw_destroy_handle(h);
      return DJW_ERR_MEMORY;
    }
    while (1)
    {
      sf_count_t got = sf_readf_float(sf, buf, BLOCK);
      if (got <= 0)
        break;
      int r2 = djw_stream_push_frames(h, buf, (int)got);
      if (r2 != DJW_OK)
      {
        fprintf(stderr, "djw_stream_push_frames error=%d\n", r2);
        free(buf);
        sf_close(sf);
        djw_destroy_handle(h);
        return r2;
      }
    }
    free(buf);
    sf_close(sf);

    // finalize
    uint8_t *rgb = (uint8_t *)calloc(3, cfg.width * cfg.height);
    if (!rgb)
    {
      djw_destroy_handle(h);
      return DJW_ERR_MEMORY;
    }
    int r3 = djw_stream_end(h, rgb);
    if (r3 == DJW_OK)
    {
      FILE *fpp = fopen(outFile, "wb");
      if (fpp)
      {
        fprintf(fpp, "P6\n%d %d\n255\n", cfg.width, cfg.height);
        fwrite(rgb, 3, cfg.width * cfg.height, fpp);
        fclose(fpp);
        printf("Wrote %s (streaming) OK\n", outFile);
      }
      else
      {
        fprintf(stderr, "Cannot write %s\n", outFile);
      }
    }
    else
    {
      fprintf(stderr, "djw_stream_end error=%d\n", r3);
    }
    free(rgb);
    djw_destroy_handle(h);
  }

  return 0;
}
