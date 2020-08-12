#pragma once

#include <iostream>
#include <string>
#include <chrono>

class Timer
{
public:
	Timer() : beg_(clock_::now())
	{}
	void reset()
	{
		beg_ = clock_::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
	}

	void out(std::string message = "")
	{
		double t = elapsed();
		std::cout << message << " elasped time:" << t << "ms" << std::endl;
		reset();
	}

	double get_duration()const
	{
		return elapsed();
	}
private:
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::milli> second_;
	std::chrono::time_point<clock_> beg_;
};
