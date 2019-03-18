#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

#include "proximalGradient.hpp"
#include "douglasRachford.hpp"
#include "admmSolver.hpp"
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char **argv)
{
	srand(time(0));
	const int M = 1200;
	const int w = 64;
	int N = w*w;

	const int Sparsity = 200;

	VectorXf b = VectorXf::Zero(N); // ground truth sparse in the canonical basis

	for(int i = 0; i < Sparsity; i++)
	{
		b[rand() % N] = (float)(rand()) / std::numeric_limits<int>::max();
	}

	MatrixXf A = MatrixXf::Random(M,N); // Measurement matrix M by N
	VectorXf y = A*b; // Measurements without noise

	basisPursuitSolver basisPursuit(N);
	basisPursuit.swapProx();
	basisPursuit.setMaxSteps(200);

	VectorXf x 	= basisPursuit.solve(A,y);

	std::cout << "BP SNR: " << -20.0f*log10((b-x).norm()/b.norm()) << std::endl;

	cv::Size size(w,w);
	cv::Mat original(size, CV_32FC1), reconstruction(size,CV_32FC1);

	for(int i = 0; i < w; i++)
	{
		for(int j = 0; j < w; j++)
		{
			original.at<float>(j,i) 		= b[i*w+j];
			reconstruction.at<float>(j,i) 	= x[i*w+j];
		}
	}

	cv::Size display_size(512,512);
	cv::Mat frame(cv::Size(1024,512), CV_32FC1);
	cv::resize(original,original,display_size);
	cv::resize(reconstruction,reconstruction,display_size);

	original.copyTo(frame.rowRange(0, 512).colRange(0, 512));
	reconstruction.copyTo(frame.rowRange(0, 512).colRange(512, 1024));

	cv::cvtColor(frame,frame,CV_GRAY2RGB);
	cv::line(frame, cv::Point(512,0), cv::Point(512,512),CV_RGB(0,255,0));

	cv::putText(frame,"Original",
		cv::Point(200,30),cv::FONT_HERSHEY_DUPLEX,1.0,
		CV_RGB(100,150,10), 2);

	cv::putText(frame,"Reconstruction",
		cv::Point(720,30),cv::FONT_HERSHEY_DUPLEX,1.0,
		CV_RGB(100,150,10), 2);

	cv::namedWindow("Result", 1);
	cv::imshow("Result", frame);
	cv::waitKey(0);

	return 0;
}
