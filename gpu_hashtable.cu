#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

// 4-byte integer hash, full avalanche
// source: https://burtleburtle.net/bob/hash/integer.html
inline __device__ std::size_t kernel_hash(int key) {
	key = (key+0x7ed55d16) + (key<<12);
    key = (key^0xc761c23c) ^ (key>>19);
    key = (key+0x165667b1) + (key<<5);
    key = (key+0xd3a2646c) ^ (key<<9);
    key = (key+0xfd7046c5) + (key<<3);
    key = (key^0xb55a4f09) ^ (key>>16);

	return key;
}

// increments the position and loops around the container size
inline __device__ void inc(size_t &pos, size_t container_size) {
	if (++pos == container_size) {
		pos = 0;
	}
}

// gets a pointer to the position of a given key, incrementing the `uniques`
// counter if the key isn't in the map.
inline __device__ hashentry_t *getToPosition(hashentry_t *container, int key,
								std::size_t container_size, int *uniques) {
	size_t crawler;
	int old_key;

	// loop until empty position or key found
	crawler = kernel_hash(key) % container_size;
	do {
		old_key = atomicCAS(&(container[crawler].first), GpuHashTable::NO_ENTRY, key);

		if (container[crawler].first == key) {
			break;
		}

		inc(crawler, container_size);
	} while (true);

	if (uniques != nullptr && old_key == GpuHashTable::NO_ENTRY) {
		atomicInc((unsigned int *)uniques, container_size);
	}

	return &container[crawler];
}

// inserts a <key, val> pair into container, returning how many
// unique keys were inserted.
__global__ void kernel_insert(int *keys, int *value, int numKeys,
				hashentry_t *container, size_t container_size, int *uniques) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numKeys) {
		return;
	}

	auto *location = getToPosition(container, keys[index], container_size, uniques);

	location->second = value[index];
}

// gets <key, val> pairs
__global__ void kernel_get(int *keys, int numKeys, int *retvals,
						hashentry_t *container, std::size_t container_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numKeys) {
		return;
	}

	auto *location = getToPosition(container, keys[index], container_size, nullptr);

	if (location->first != GpuHashTable::NO_ENTRY) {
		retvals[index] = location->second;
	}
}

// for all entries in `source`, if not null, rehash them into `destination`.
__global__ void kernel_rehash_keys(hashentry_t *destination, hashentry_t *source,
						std::size_t destination_size, std::size_t source_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= source_size) {
		return;
	}

	if (source[index].first == GpuHashTable::NO_ENTRY) {
		return;
	}

	auto *dst_location = getToPosition(destination, source[index].first, destination_size, nullptr);

	dst_location->second = source[index].second;
}

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators __cudaMalloc and __cudaFree
 */
GpuHashTable::GpuHashTable(int size) : container_size(size) {
	glbGpuAllocator->_cudaMalloc((void **) &container, container_size * sizeof(hashentry_t));

	cudaDeviceProp dev;
	cudaGetDeviceProperties(&dev, 0);
	cudaThreads = dev.maxThreadsPerBlock;
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(container);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	if (1.0 * (registeredKeys + numBucketsReshape) / container_size > GpuHashTable::thresho_load_factor) {
		reallocAndRehash((registeredKeys + numBucketsReshape) / GpuHashTable::desired_load_factor);
	}
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *devKeys;
	int *devValues;

	glbGpuAllocator->_cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void **) &devValues, numKeys * sizeof(int));

	cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	reshape(numKeys);

	int *uniques;
	glbGpuAllocator->_cudaMalloc((void **) &uniques, sizeof(int));

	kernel_insert<<<cudaBlocks(numKeys), cudaThreads>>>(devKeys, devValues, numKeys, container, container_size, uniques);

	int uniques_host;
	cudaMemcpy(&uniques_host, uniques, sizeof(int), cudaMemcpyDeviceToHost);

	registeredKeys += uniques_host;

	cudaDeviceSynchronize();
	glbGpuAllocator->_cudaFree(devKeys);
	glbGpuAllocator->_cudaFree(devValues);
	glbGpuAllocator->_cudaFree(uniques);

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *devKeys;
	int *devValues;
	int *values;

	glbGpuAllocator->_cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void **) &devValues, numKeys * sizeof(int));

	cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	kernel_get<<<cudaBlocks(numKeys), cudaThreads>>>(devKeys, numKeys, devValues, container, container_size);

	cudaDeviceSynchronize();

	values = (int *)malloc(numKeys * sizeof(int));

	cudaMemcpy(values, devValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	glbGpuAllocator->_cudaFree(devKeys);
	glbGpuAllocator->_cudaFree(devValues);

	return values;
}

// reallocs container and rehashes all entries
void GpuHashTable::reallocAndRehash(size_t newSize) {
	hashentry_t *aux;

	// if no keys to rehash, just realloc the container
	if (!registeredKeys) {
		glbGpuAllocator->_cudaFree(container);
		glbGpuAllocator->_cudaMalloc((void **) &container, newSize * sizeof(hashentry_t));
		container_size = newSize;

		return;
	}

	glbGpuAllocator->_cudaMalloc((void **) &aux, newSize * sizeof(hashentry_t));

	kernel_rehash_keys<<<cudaBlocks(container_size), cudaThreads>>>(aux, container, newSize, container_size);

	cudaDeviceSynchronize();
	glbGpuAllocator->_cudaFree(container);

	container_size = newSize;
	container = aux;
}

// computes how many blocks are needed for a given query
inline int GpuHashTable::cudaBlocks(int queries) {
	return queries / cudaThreads + (queries % cudaThreads != 0);
}
