CC=g++
CFLAGS=-W -msse2 -mavx -O3 -std=c++14

EIGEN=/Users/joan/Downloads/Eigen/ # put the path to Eigen include dir here
OPENCV = `pkg-config /usr/local/Cellar/opencv/4.1.0_2/lib/pkgconfig/opencv4.pc --cflags --libs`

basisPursuit:
	$(CC) basisPursuit.cpp -o basisPursuit -I $(EIGEN) -Iinclude/ $(CFLAGS) $(OPENCV)
ptfOptimization:
	$(CC) regularizedPortfolio.cpp -o regularizedPortfolio -I $(EIGEN) -Iinclude/ $(CFLAGS)

clean:
	rm basisPursuit regularizedPortfolio
