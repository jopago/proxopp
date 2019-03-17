CC=g++
CFLAGS=-W -msse2 -mavx -O3

EIGEN=~/Documents/C++/Include/ # put the path to Eigen include dir here
OPENCV = `pkg-config opencv --cflags --libs`

all:
	$(CC) sparseImageExample.cpp -o example -I $(EIGEN) -Iinclude/ $(CFLAGS) $(OPENCV)
clea:
	rm example