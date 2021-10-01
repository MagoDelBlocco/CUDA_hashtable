#!/bin/sh
echo "Test 1"
./gpu_hashtable 1000000 1 10

echo "Test 2"
./gpu_hashtable 1000000 2 20

echo "Test 3"
./gpu_hashtable 1000000 8 40

echo "Test 4"
./gpu_hashtable 10000000 4 50

echo "Test 5"
./gpu_hashtable 40000000 2 50