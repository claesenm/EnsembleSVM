/**
 *  Copyright (C) 2013 KU Leuven
 *
 *  This file is part of EnsembleSVM.
 *
 *  EnsembleSVM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  EnsembleSVM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with EnsembleSVM.  If not, see <http://www.gnu.org/licenses/>.
 *
 * ThreadPool.hpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#include "config.h"
#ifdef HAVE_PTHREAD

/*************************************************************************************************/

#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <deque>
#include <functional>

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

extern const unsigned NUM_HARDWARE_THREADS{std::thread::hardware_concurrency()};

template <typename T>
class ThreadPool;

template<typename Ret, typename... Args>
class ThreadPool<Ret(Args...)>{
public:
	typedef typename std::deque<std::future<Ret>>::iterator iterator;
	typedef typename std::deque<std::future<Ret>>::const_iterator const_iterator;
	typedef std::function<Ret(Args...)> Fun;

private:
	typedef std::packaged_task<Ret()> job;

	// the job queue
	std::queue<job> jobs;
	std::mutex jobs_mutex;
	std::condition_variable jobs_cv;

	const Fun fun;

	std::deque<std::future<Ret>> futures;
	bool stop_{false}; // When set, this flag tells the threads that they should exit

	// maximum amount of jobs in queue or 0 if unbounded
	const unsigned maxjobs=0; // const to avoid the need for an extra mutex

	// used to signal the threads that add jobs
	std::condition_variable maxjobs_cv;

public:
	/**
	 * The actual execution of jobs is done by workers.
	 */
	class Worker{
	private:
		typedef ThreadPool<Ret(Args...)>::job job;
		ThreadPool<Ret(Args...)> &mgr;
		std::thread t;

		static void thread_func_static(Worker *t){
			std::unique_lock<std::mutex> l(t->mgr.jobs_mutex, std::defer_lock);
			while (true)
			{
				l.lock();

				// Wait until the queue won't be empty or stop is signaled
				t->mgr.jobs_cv.wait(l, [t](){
					return (t->mgr.stop_ || !t->mgr.jobs.empty());
				});

				// Stop was signaled, let's exit the thread
				if (t->mgr.stop_) { return; }

				// Pop one task from the queue...
				job j = std::move(t->mgr.jobs.front());
				t->mgr.jobs.pop();

				// notify potential job adding
				if(t->mgr.maxjobs > 0 && t->mgr.jobs.size() < t->mgr.maxjobs) t->mgr.maxjobs_cv.notify_one();
				l.unlock();

				// Execute the task!
				j();
			}
		};

	public:
		Worker(ThreadPool<Ret(Args...)> &mgr):mgr(mgr){
			t = std::thread(thread_func_static,this);
		};

		friend class ThreadPool<Ret(Args...)>;
	};

private:
	friend class Worker;

	// thread pool
	std::vector<Worker> threads;

public:
	/**
	 * Creates a ThreadPool to execute fun.
	 *
	 * The default number of threads is std::thread::hardware_concurrency().
	 * Using this constructor allows an infinite job queue.
	 */
	ThreadPool(Fun&& fun):ThreadPool{std::move(fun),std::thread::hardware_concurrency(),0}{};
	/**
	 * Creates a ThreadPool to execute fun with specified number of threads.
	 *
	 * When maxjobs=0 (default), an infinite job queue is permitted.
	 * When maxjobs>0, ThreadPool::addjob() will block until the queue
	 * is small enough.
	 */
	ThreadPool(Fun&& fun, unsigned numthreads, unsigned maxjobs=0)
	:fun{std::move(fun)},
	 stop_{false},
	 maxjobs{maxjobs}
	{
		// create worker threads
		if(!numthreads) numthreads=1;
		threads.reserve(numthreads);
		for(unsigned i=0; i<numthreads; ++i){
			threads.emplace_back(*this);
		}
	};

	ThreadPool(const ThreadPool<Ret(Args...)> &t) = delete;
	ThreadPool<Ret(Args...)>& operator=(const ThreadPool<Ret(Args...)> &t) = delete;
	ThreadPool<Ret(Args...)>& operator=(ThreadPool<Ret(Args...)>&& t) = default;
	~ThreadPool(){
		join();
	}

	/**
	 * Adds a new job to this ThreadPool's job queue.
	 */
	void addjob(Args... arguments){
		std::unique_lock<std::mutex> lck(jobs_mutex);

		// if there is a max job queue and we reached it, wait for jobs to disappear
		if(maxjobs > 0 && jobs.size() >= maxjobs) maxjobs_cv.wait(lck);
		job newjob(std::bind(fun,arguments...));

		// add the job
		futures.push_back(std::move(newjob.get_future()));
		jobs.push(std::move(newjob));

		// notify a worker thread
		jobs_cv.notify_one();
	}


	void join(){
		stop();
		for (auto &td: threads){ td.t.join(); }
	}

	void stop(){
		stop_=true;
		jobs_cv.notify_all();
	}

	/**
	 * Wait for all workers to finish.
	 */
	void wait(){
	    for (auto& f : futures) { f.wait(); }
	}

	void clear_futures(){ futures.clear(); }

	/**
	 * Returns the number of threads managed by this ThreadPool.
	 */
	unsigned num_threads() const{ return threads.size(); }

	/**
	 * Iterate over the futures of all posted jobs.
	 * Iteration occurs in the order the jobs were added.
	 */
	iterator begin(){ return futures.begin(); }
	iterator end(){ return futures.end(); }
	/**
	 * Iterate over the futures of all posted jobs.
	 * Iteration occurs in the order the jobs were added.
	 */
	const_iterator cbegin() const{ return futures.cbegin(); }
	const_iterator cend() const{ return futures.cend(); }
};

/*************************************************************************************************/

}  // ensemble namespace

/*************************************************************************************************/

#endif
