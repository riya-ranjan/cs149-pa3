#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include "circleBoxTest.cu_inl"

#define SCAN_BLOCK_DIM   256
#include "exclusiveScan.cu_inl"

#include <thrust/device_ptr.h>    
#include <thrust/device_vector.h> 
#include <thrust/sequence.h>       
#include <thrust/transform.h>  
#include <thrust/scan.h>          
#include <thrust/functional.h>     

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

struct RowID {
    int numCols;
    __host__ __device__
    int operator()(int i) const {
        return i / numCols;
    }
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ float4
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr, float4 existingColor) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return existingColor;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    // if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

    //     const float kCircleMaxAlpha = .5f;
    //     const float falloffScale = 4.f;

    //     float normPixelDist = sqrt(pixelDist) / rad;
    //     rgb = lookupColor(normPixelDist);

    //     float maxAlpha = .6f + .4f * (1.f-p.z);
    //     maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
    //     alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    // } else {
    //     // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    // }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    //float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    return newColor;

    // global memory write
    //*imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// shadePixelSnowflake -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ float4
shadePixelSnowflake(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr, float4 existingColor) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return existingColor;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    float normPixelDist = sqrt(pixelDist) / rad;
    rgb = lookupColor(normPixelDist);

    float maxAlpha = .6f + .4f * (1.f-p.z);
    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
    alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    //float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    return newColor;

    // global memory write
    //*imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

struct ShadePixel {
    __device__ float4 operator()(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr, float4 existingColor) const {
        return shadePixel(circleIndex, pixelCenter, p, imagePtr, existingColor);
    }
};

struct ShadePixelSnowflake {
    __device__ float4 operator()(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr, float4 existingColor) const {
        return shadePixelSnowflake(circleIndex, pixelCenter, p, imagePtr, existingColor);
    }
};

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
template <typename ShadeFunc>
__global__ void kernelRenderCircles(ShadeFunc shade, int *circle_indices, int numTiles, int tile_size) {

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;
    int pixels = height * width;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixels) return;

    // at this point, index = pixel number. need to convert into its x, y coords
    // so we can match it to the other circles
    int x = index % width;
    int y = index / width;
    // printf("this has coordinates (%d, %d)\n", x, y);

    float invWidth = 1.f / width;
    float invHeight = 1.f / height;
    float pixelCenterNormX = invWidth  * (static_cast<float>(x) + 0.5f);
    float pixelCenterNormY = invHeight * (static_cast<float>(y) + 0.5f);

    // pointer to this pixels RGBA in global image
    float4* imgPtr = reinterpret_cast<float4*>(&cuConstRendererParams.imageData[4 * (index)]);

    // figure out what tile we are in
    int tile_index = ((x / tile_size) + (y / tile_size) * (width / tile_size));
    //printf("Current tile is %d\t", tile_index);
    // if (tile_index == 100) 
    //     printf("TRUE\n");
    // get the index of the beginning of circle indices
    int start_circle_index = tile_index * (cuConstRendererParams.numCircles);
    
    float4 existingColor = *imgPtr;
    
    // iterate through all the circles that are relevant
    for (int arr_ind=start_circle_index; arr_ind<start_circle_index+cuConstRendererParams.numCircles; arr_ind++) {

        int circleIndex = circle_indices[arr_ind];
        // if (circleIndex > 0 && tile_index > 0)
        //     printf("Now drawing circle %d for pixel in tile %d", circleIndex, tile_index);
        if (arr_ind % cuConstRendererParams.numCircles != 0 && circleIndex == 0) break;
        // params of the circle we're currently looking at
        int index3 = 3 * circleIndex;
        float px = cuConstRendererParams.position[index3];
        float py = cuConstRendererParams.position[index3+1];
        float pz = cuConstRendererParams.position[index3+2];
        float rad = cuConstRendererParams.radius[circleIndex];

        // compute the bounding box of the circle.  This bounding box
        // is in normalized coordinates
        float minX = px - rad;
        float maxX = px + rad;
        float minY = py - rad;
        float maxY = py + rad;

        // do nothing if the current pixel is not within the bounding box
        if (pixelCenterNormX < minX || pixelCenterNormX > maxX || minY > pixelCenterNormY || maxY <  pixelCenterNormY)  continue;
        
        //printf("the circle index %d has been detected\n", circleIndex);
        float2 pc = make_float2(pixelCenterNormX, pixelCenterNormY);
        float3 pos = make_float3(px, py, pz);

        existingColor = shade(circleIndex, pc, pos, imgPtr, existingColor);
        // imgPtr++;
    }
    *imgPtr = existingColor;

}

// kerneltSetTilesToCircles -- (CUDA device code)
//
// Sets the boolean index of tiles_to_circles to true if this particular
// tile and circle intersect.
// tile index: (overall index / numCircles)
// circle index: (overall index % numCircles)
__global__ void kernelSetTilesToCircles(int *tiles_and_circles, int length, int tile_size) {
    int overall_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (overall_index >= length) return;
    int tile_index = overall_index / cuConstRendererParams.numCircles;
    int circle_index = overall_index % cuConstRendererParams.numCircles;

    int index3 = 3 * circle_index;
    float circleX = cuConstRendererParams.position[index3];
    float circleY = cuConstRendererParams.position[index3+1];
    float circleRadius = cuConstRendererParams.radius[circle_index];

    // each tile is 32 x 32, but we need the relative width, so use the inverse
    const float invWidth  = 1.f / cuConstRendererParams.imageWidth;
    const float invHeight = 1.f / cuConstRendererParams.imageHeight;

    float boxL = static_cast<float>((tile_index % (cuConstRendererParams.imageWidth / tile_size)) * tile_size);
    float boxR = static_cast<float>(boxL + tile_size);
    boxL *= invWidth;
    boxR *= invWidth;
    float boxT = static_cast<float>((tile_index / (cuConstRendererParams.imageWidth / tile_size)) * tile_size);
    float boxB = static_cast<float>(boxT + tile_size);
    boxT *= invHeight;
    boxB *= invHeight;

    int circle_in_box = circleInBoxConservative(circleX, circleY, circleRadius, boxL, boxR, boxB, boxT);
    
    tiles_and_circles[overall_index] = circle_in_box;
    /** if (circle_in_box == 1) {
        printf("The circle %d is in tile %d, setting %d to %d\n", circle_index, tile_index, overall_index, tiles_and_circles[overall_index]);
    } */
}

//kernelGetCircleIndices -- (CUDA device code)
//
// returns output to circle_indices that tell us which circles we care about for each tile
// performs essentially a scatter
__global__ void kernelGetCircleIndices(int *circle_indices, int *flags, int *scanned_tiles, int total_length) {
    int overall_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (overall_index >= total_length) return;
    int tile_index = overall_index / cuConstRendererParams.numCircles;

    if (flags[overall_index]) {
        circle_indices[scanned_tiles[overall_index] + tile_index * cuConstRendererParams.numCircles] = overall_index % cuConstRendererParams.numCircles;
    }
}

// returns whether or not the circle intersects with the
// tile at the specified tile index
__device__ __inline__ int
boxCircleIntersect(int tile_index, int circle_index, int tile_size) {
    int index3 = 3 * circle_index;
    float circleX = cuConstRendererParams.position[index3];
    float circleY = cuConstRendererParams.position[index3+1];
    float circleRadius = cuConstRendererParams.radius[circle_index];

    const float invWidth  = 1.f / cuConstRendererParams.imageWidth;
    const float invHeight = 1.f / cuConstRendererParams.imageHeight;

    float boxL = static_cast<float>((tile_index % (cuConstRendererParams.imageWidth / tile_size)) * tile_size);
    float boxR = static_cast<float>(boxL + tile_size);
    boxL *= invWidth;
    boxR *= invWidth;
    float boxT = static_cast<float>((tile_index / (cuConstRendererParams.imageWidth / tile_size)) * tile_size);
    float boxB = static_cast<float>(boxT + tile_size);
    boxT *= invHeight;
    boxB *= invHeight;

    return circleInBoxConservative(circleX, circleY, circleRadius, boxL, boxR, boxB, boxT);
}

//kernelMapCircleIds -- (CUDA device code)
//
// This will be a single kernel where we:
// 1) flag which circles belong to each tile
// 2) perform an exclusive scan 
// 3) map the circle indices we care about to global memory
__global__ void kernelMapCircleIds(int *circle_indices, int total_length, int circles_pow_2, int tile_size) {
    const uint BLOCKSIZE = 256;
    int tile_index = blockIdx.x;
    int in_block_tid = threadIdx.x;

    __shared__ uint flags[BLOCKSIZE];
    __shared__ uint prefix_sum_output[BLOCKSIZE];
    __shared__ uint prefix_sum_scratch[2 * BLOCKSIZE];
    int prev_prefix_sum = 0;
    for (int i = in_block_tid; i < circles_pow_2; i += blockDim.x) {
        // set flags appropriately
        if (i >= cuConstRendererParams.numCircles) {
            flags[i] = 0;
        } else {
            flags[i] = boxCircleIntersect(tile_index, i, tile_size);
        }

        __syncthreads();
        // perform prefix scans
        sharedMemExclusiveScan(in_block_tid, flags, prefix_sum_output, prefix_sum_scratch, BLOCKSIZE);
        __syncthreads();

        // write to global memory
        if (flags[i]) {
            int current_row_index = tile_index * cuConstRendererParams.numCircles;
            circle_indices[prefix_sum_output[in_block_tid] + prev_prefix_sum + current_row_index] = i;
        }
        if (in_block_tid == 0) prev_prefix_sum += prefix_sum_output[BLOCKSIZE - 1] + flags[BLOCKSIZE - 1];
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene, int seed) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius, seed);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    int totalPixels = image->height * image->width;

    // we want each tile to be 32 x 32, unless the image is too big
    int numTiles = totalPixels / 1024; 
    int tile_size = 32;
    if (numCircles >= 2000000) {
        numTiles = totalPixels / 16384; // now each tile is 64 x 64
        tile_size = 128;
    }

    // create array of tiles to circles 
    // tile index: block index
    // circle index: thread index 
    int *tiles_and_circles;
    int total_length = numTiles * numCircles;
    cudaCheckError(cudaMalloc(&tiles_and_circles, total_length * sizeof(int)));
    
    // number of blocks should be = number of tiles
    // number of threads should be 256
    dim3 blockDim(256, 1);
    dim3 gridDim((total_length + blockDim.x - 1) / blockDim.x);

    // find all tiles/circles that intersect
    kernelSetTilesToCircles<<<gridDim, blockDim>>>(tiles_and_circles, total_length, tile_size);
    cudaDeviceSynchronize();

    // exclusive scan for each tile
    int *scanned_tiles;
    cudaCheckError(cudaMalloc(&scanned_tiles, total_length * sizeof(int)));
    perform_exclusive_scans(tiles_and_circles, scanned_tiles, total_length, numTiles);

    // get all the correct circle indices using the flags, and our exclusive scan
    int *circle_indices;
    cudaCheckError(cudaMalloc(&circle_indices, total_length * sizeof(int)));
    cudaMemset(circle_indices, 0, total_length * sizeof(int));
    kernelGetCircleIndices<<<gridDim, blockDim>>>(circle_indices, tiles_and_circles, scanned_tiles, total_length);
    cudaDeviceSynchronize();
    cudaFree(tiles_and_circles);
    cudaFree(scanned_tiles);

    // for each pixel, whatever tile the pixel is in, check for every overlapping circle how to 
    // color the pixel
    dim3 gridDim_2((totalPixels + blockDim.x - 1) / blockDim.x);
    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelRenderCircles<<<gridDim_2, blockDim>>>(ShadePixelSnowflake(), circle_indices, numTiles, tile_size);
    }
    else {
        kernelRenderCircles<<<gridDim_2, blockDim>>>(ShadePixel(), circle_indices, numTiles, tile_size);
    }
    cudaDeviceSynchronize();
    cudaFree(circle_indices);
}

void
CudaRenderer::perform_exclusive_scans(int *flags, int *output, int length, int numTiles) {
    thrust::device_vector<int> keys(length);
    thrust::sequence(keys.begin(), keys.end());
    thrust::transform(keys.begin(), keys.end(), keys.begin(), RowID{numCircles});

    thrust::device_ptr<int> in_ptr(flags);
    thrust::device_ptr<int>     out_ptr(output);
    thrust::exclusive_scan_by_key(keys.begin(), keys.end(), in_ptr, out_ptr);
    cudaDeviceSynchronize();
}


