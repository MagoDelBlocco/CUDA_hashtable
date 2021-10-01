#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>
#include <unordered_set>

using hashentry_t = std::pair<int, int>;

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ", " << __LINE__ << ", " << cudaGetErrorString(e); \
		exit(0); \
	 }\
}

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		static constexpr int NO_ENTRY = 0;

		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		~GpuHashTable();
	private:
		static constexpr double desired_load_factor = .55;
		static constexpr double thresho_load_factor = .75;

		hashentry_t *container;
		std::size_t container_size;
		// number of unique keys registered
		int registeredKeys = 0;
		int cudaThreads;

		void reallocAndRehash(std::size_t newSize);

		int cudaBlocks(int queries);
};

#endif

