#ifndef MULTITHREAD_HPP
#define MULTITHREAD_HPP

#ifndef _WIN32
#include <numa.h>
#include <queue>
#include <thread>
#include <functional>
#include <vector>
#include <mutex>                // std::mutex, std::unique_lock
#include <condition_variable>    // std::condition_variable

struct multiThreads {
    int totalThreads;
    int totalThreadsCPP;
    int totalSockets;
    int threadsPerSocket;
    volatile bool ok;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::pair<v_size, v_size>> stVAndEdV; 

    // std::queue<std::function<void()>> tasks;
    // std::vector<std::thread> pools;
    // std::mutex mtx;
    // std::condition_variable condition;
    // bool flag;

    multiThreads() {
        totalThreads = numa_num_configured_cpus();
        totalThreadsCPP = std::thread::hardware_concurrency();
        totalSockets = numa_num_configured_nodes();
        threadsPerSocket = totalThreads / totalSockets;
        stVAndEdV.resize(totalThreads);

        cpu_set_t cpu_mask;                    //Allocate mask
        CPU_ZERO(     &cpu_mask);           //Set mask to zero
        for(int i = 0; i < totalThreads; i++) {
            CPU_SET(i, &cpu_mask);           //Set mask with thread #
        }
        int err = sched_setaffinity( (pid_t)0,  //Set the affinity
                                    sizeof(cpu_mask),
                                    &cpu_mask );
        assert(err == 0);
        // int cnt = 0;
        // #pragma omp parallel shared(cpu_mask) num_threads(totalThreads)
        // {
        //     int inum = omp_get_thread_num();
        //     int cpu = sched_getcpu();               //Confirm affinity
        //     // printf("thread %d on cpu %d\n", inum, cpu);
        //     // cnt++;
        // }
        // printf("cnt %d\n", cnt);

        CPU_ZERO(&cpu_mask); //置空
        int cnt = 0;
        sched_getaffinity(0, sizeof(cpu_mask), &cpu_mask); 
        for(int i=0;i<totalThreads;i++){
            if(CPU_ISSET(i,&cpu_mask)){
                cnt++;
            }
        }
        assert(cnt == totalThreads);

        ok = false;
    }

    void init(v_size * partitionOffset) {
        for(int i = 0; i < totalThreads; i++) {
            stVAndEdV[i].first = partitionOffset[i];
            stVAndEdV[i].second = partitionOffset[i + 1];
        }
      //  ok = true;
      //  cv.notify_all();
    }
    // template<class F, class... Args>
    // auto push(F && f, Args && ...args)->std::future<typename std::result_of<F(Args...)>::type>;
};


#else

#include <queue>
#include <thread>
#include <functional>
#include <vector>
#include <mutex>                // std::mutex, std::unique_lock
#include <condition_variable>    // std::condition_variable
#include <omp.h>
#include <string>
#include<cstdlib>
#include<cstdio>

struct multiThreads {
    int totalThreads;
    int totalThreadsCPP;
    int totalSockets;
    int threadsPerSocket;
    volatile bool ok;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::pair<v_size, v_size>> stVAndEdV; 

    // std::queue<std::function<void()>> tasks;
    // std::vector<std::thread> pools;
    // std::mutex mtx;
    // std::condition_variable condition;
    // bool flag;

    multiThreads() {
        totalThreads = totalThreadsCPP = omp_get_num_procs();
        // totalSockets = omp_get_num_places();
        // threadsPerSocket = totalThreads / totalSockets;
        stVAndEdV.resize(totalThreads);

        std::string s = "0";
        char tmp[10];
        for(int i = 1; i < totalThreads; i++) {
            itoa(i, tmp, 10);
            s += " ";
            s += tmp;
        }

        // setenv("GOMP_CPU_AFFINITY", s.c_str(), 0);

        _putenv_s("GOMP_CPU_AFFINITY", s.c_str());

        bool vis[500];
        memset(vis, false, sizeof(s));
        #pragma omp parallel for
        for(int i = 0; i < totalThreads; i++) {
            int id = omp_get_thread_num();
            vis[id] = true;
        }
        
        for(int i = 0; i < totalThreads; i++) {
            assert(vis[i]);
        }
    }

    void init(v_size * partitionOffset) {
        for(int i = 0; i < totalThreads; i++) {
            stVAndEdV[i].first = partitionOffset[i];
            stVAndEdV[i].second = partitionOffset[i + 1];
        }
      //  ok = true;
      //  cv.notify_all();
    }
    // template<class F, class... Args>
    // auto push(F && f, Args && ...args)->std::future<typename std::result_of<F(Args...)>::type>;
};

#endif
#endif