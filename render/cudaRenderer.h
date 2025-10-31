#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#include "circleRenderer.h"

class CudaRenderer : public CircleRenderer {
 private:
  Image* image;
  SceneName sceneName;

  int numCircles;
  float* position;
  float* velocity;
  float* color;
  float* radius;

  float* cudaDevicePosition;
  float* cudaDeviceVelocity;
  float* cudaDeviceColor;
  float* cudaDeviceRadius;
  float* cudaDeviceImageData;

  int tile_size;
  int num_tiles;
  int total_pixels;

 public:
  CudaRenderer();
  virtual ~CudaRenderer();

  const Image* getImage();

  void setup();

  void loadScene(SceneName name, int seed = 0);

  void allocOutputImage(int width, int height);

  void clearImage();

  void advanceAnimation();

  void render();

  int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
  }

  void shadePixel(int circleIndex, float pixelCenterX, float pixelCenterY,
                  float px, float py, float pz, float* pixelData);
};

#endif