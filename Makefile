CC=g++
CFLAGS=-W -msse2 -mavx -O3 -std=c++11

EIGEN=~/Documents/C++/Include/ # put the path to Eigen include dir here
OPENCV = `pkg-config opencv --cflags --libs`

basisPursuit:
	$(CC) basisPursuit.cpp -o basisPursuit -I $(EIGEN) -Iinclude/ $(CFLAGS) $(OPENCV)
ptfOptimization:
	$(CC) regularizedPortfolio.cpp -o regularizedPortfolio -I $(EIGEN) -Iinclude/ $(CFLAGS)

clean:
	rm basisPursuit regularizedPortfolio
