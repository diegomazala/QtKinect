#ifndef __TIMER_H__
#define __TIMER_H__

#include <iostream>
#include <chrono>
#include <thread>

class Timer
{
public:
	Timer(){}
	~Timer(){}

	void start()
	{
		startTime = std::chrono::steady_clock::now();
	}

	double diff_sec() const
	{
		return std::chrono::duration<double>(interval_ns()).count();
	}

	double diff_msec() const
	{
		return std::chrono::duration<double, std::milli>(interval_ns()).count();
	}

	void print_interval(const std::string& msg) const
	{
		std::cout << msg << diff_sec() << " seconds " << std::endl;
	}

	void print_interval_msec(const std::string& msg) const
	{
		std::cout << msg << diff_msec() << " msec " << std::endl;
	}

	std::chrono::seconds interval_sec() const
	{
		return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime);
	}

	std::chrono::milliseconds interval_ms() const
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime);
	}

	std::chrono::microseconds interval_us() const
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTime);
	}

	std::chrono::nanoseconds interval_ns() const
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTime);
	}

	static std::chrono::seconds interval_sec(const std::chrono::steady_clock::time_point& t1, const std::chrono::steady_clock::time_point& t0)
	{
		return std::chrono::duration_cast<std::chrono::seconds>(t1 - t0);
	}

	static std::chrono::milliseconds interval_ms(const std::chrono::steady_clock::time_point& t1, const std::chrono::steady_clock::time_point& t0)
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
	}

	static std::chrono::microseconds interval_us(const std::chrono::steady_clock::time_point& t1, const std::chrono::steady_clock::time_point& t0)
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
	}

	static std::chrono::nanoseconds interval_ns(const std::chrono::steady_clock::time_point& t1, const std::chrono::steady_clock::time_point& t0)
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
	}

	static void sleep_ms(unsigned int ms)
	{
		auto t0 = std::chrono::steady_clock::now();
		std::this_thread::sleep_for(std::chrono::milliseconds(ms));
		auto t1 = std::chrono::steady_clock::now();
		std::chrono::milliseconds total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

		std::cout << "this_thread_sleep (" << ms << ") milliseconds: " << total_ms.count() << "ms\n";
	}


	static void test()
	{
		std::cout << std::endl << "[================ Timer Test [begin] ================]" << std::endl << std::endl;

		Timer t;
		t.start();

		Timer::sleep_ms(1234);

		std::cout
			<< std::fixed << std::endl
			<< "Interval NanoSeconds           : " << t.interval_ns().count() << std::endl
			<< "Interval MicroSeconds          : " << t.interval_us().count() << std::endl
			<< "Interval MilliSeconds          : " << t.interval_ms().count() << std::endl
			<< "Interval Seconds               : " << t.interval_sec().count() << std::endl
			<< "Interval Seconds <double>      : " << t.diff_sec() << std::endl
			<< std::endl;


		Timer::sleep_ms(765);

		std::cout
			<< std::fixed << std::endl
			<< "Interval MilliSeconds          : " << t.interval_ms().count() << std::endl
			<< "Interval Seconds               : " << t.interval_sec().count() << std::endl
			<< "Interval Seconds <double>      : " << t.diff_sec() << std::endl;

		std::cout << std::endl << "[================ Timer Test [end] ==================]" << std::endl << std::endl;
	}

private:

	std::chrono::steady_clock::time_point startTime;

};

#endif // __TIMER_H__