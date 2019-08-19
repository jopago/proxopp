#include <iostream>
#include <random>

#include "proximalGradient.hpp"
#include "douglasRachford.hpp"
#include "admmSolver.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

float snr(const Eigen::VectorXf& x_true, const Eigen::VectorXf& x_rec)
{
	return -20.0f*log10((x_true-x_rec).norm()/x_true.norm());
}

int main(int argc, char **argv)
{
	const int M = 800;
	const int w = 32; // w x w pixels image 
	const int Sparsity = 100;
	int N = w*w;
	std::default_random_engine generator;
  	std::uniform_real_distribution<float> random_float(0.0,1.0);
  	std::uniform_int_distribution<int> random_int(0, N-1);
	
	Eigen::VectorXf b = Eigen::VectorXf::Zero(N); // ground truth sparse in the canonical basis

	for(int i = 0; i < Sparsity; i++)
	{
		b[random_int(generator)] = random_float(generator); 
	}

	Eigen::MatrixXf A = Eigen::MatrixXf::Random(M,N); // Measurement matrix M by N
	Eigen::VectorXf y = A*b; // Measurements without noise

	proxopp::basisPursuitSolver basisPursuit(N);
	basisPursuit.setMaxSteps(200);

	Eigen::VectorXf x 	= basisPursuit.solve(A,y);

	std::cout << "BP SNR: " << snr(b,x) << std::endl;

	cv::Size size(w,w);
	cv::Mat original(size, CV_32FC1), reconstruction(size, CV_32FC1);

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

	cv::cvtColor(frame,frame,cv::COLOR_GRAY2RGB);
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
