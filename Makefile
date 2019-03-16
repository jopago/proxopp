CC=g++
EIGEN=~/Documents/C++/Include/ # put the path to Eigen include dir here
CFLAGS=-Wall -W -msse2 -mavx

all:
	$(CC) example.cpp -o example -I $(EIGEN) -Iinclude/ $(CFLAGS) 
clea:
	rm example