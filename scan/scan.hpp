#ifndef SCAN_H
#define SCAN_H

#include "../graph/graph.hpp"
#include "../tools/multiThreads.hpp"
#include "../tools/fastIO.hpp"
#include "../tools/parallelSort.hpp"
#include <unordered_map>
#include <algorithm>
// #include <x86intrin.h>
#include <immintrin.h>
#include <fstream>
using namespace std;

class SCAN {
    Graph * g;
    // v_size *clusterId = nullptr;
    int * sim = nullptr;
    vector< pair<double, v_size> > NO; 
    pair<double, v_size> ** CO;
    v_size *idx;
    // int * epsNei = nullptr;
    double eps;
    int mu;

    int interNum(v_size u, v_size v);
public:
    SCAN(Graph * g_, int mu_, double eps_);
    ~SCAN();
    void computeSim();
    void saveSim(const string & path);
    void sortNO();
    void sortCO();
};

SCAN::SCAN(Graph * g_, int mu_, double eps_) {
    g = g_;
    sim = new int[g->eCnt];
    CO = new pair<double, v_size> *[g->maxDeg + 1];

    // printf("%u\n", g->maxDeg);
    for(v_size i = 2; i <= g->maxDeg; i++) {
        // printf("%u ", g->degNum[i]);
        CO[i] = new pair<double, v_size>[g->degNum[i]];
    }
    eps = eps_;
    mu = mu_;
}

void SCAN::computeSim() {
    NO.resize(g->eCnt);
    multiThreads *threadController = new multiThreads();

    #pragma omp parallel num_threads(threadController->totalThreads)
    {
        int threadId = omp_get_thread_num();
        v_size st = g->partitionOffset[threadId];
        v_size ed = g->partitionOffset[threadId + 1];
        // printf("%d %u %u\n", threadId, st, ed);

        for(v_size u = st; u < ed; u++) {
            double uDeg = g->pIdx[u + 1] - g->pIdx[u];
            for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
                v_size v = g->pEdge[i];
                sim[i] = interNum(u, v);
                
                double vDeg = g->pIdx[v + 1] - g->pIdx[v];
                NO[i].first = (1.0*sim[i]*sim[i]) / (uDeg * vDeg);
                NO[i].second = v;
            }
            // printf("%u\n", u);
        }
    }
    delete threadController;
}

int SCAN::interNum(v_size u, v_size v) {
    auto pIdx = g->pIdx;
    auto pEdge = g->pEdge;
    // v_size uDeg = pIdx[u + 1] - pIdx[u];
    // v_size vDeg = pIdx[v + 1] - pIdx[v];

    v_size pu = pIdx[u], pv = pIdx[v];

    v_size ret = 0;

    while(true) {
        while(pu < pIdx[u + 1] && pEdge[pu] < pEdge[pv]) pu++;
        if(pu == pIdx[u + 1]) break;
        while(pv < pIdx[v + 1] && pEdge[pv] < pEdge[pu]) pv++;
        if(pv == pIdx[v + 1]) break;

        if(pEdge[pu] == pEdge[pv]) {
            ret++; pu++; pv++;
        }
    }

    return ret;
}

void SCAN::sortNO() {  
    multiThreads *threadController = new multiThreads();
    #pragma omp parallel num_threads(threadController->totalThreads)
    {
        int threadId = omp_get_thread_num();
        v_size st = g->partitionOffset[threadId];
        v_size ed = g->partitionOffset[threadId + 1];
        // printf("%d %u %u\n", threadId, st, ed);
        for(v_size u = st; u < ed; u++) {
            std::sort(NO.begin() + g->pIdx[u], NO.begin() + g->pIdx[u+1]);
        }
    }
    delete threadController;
}

void SCAN::sortCO() {
    //1. 归并 xxx
    //2. 先放在一起，再排序
    idx = new v_size[g->maxDeg + 1]();

    // for(v_size u = 0; u < g->vCnt; u++) {
    //     v_size uDeg = g->pIdx[u+1] - g->pIdx[u];
    //     for(v_size i = 2; i <= uDeg; i++) {
    //         CO[i][idx[i]].first = NO[g->pIdx[u] + i - 1].first;
    //         CO[i][idx[i]++].second = u;
    //     }
    // }

    #pragma omp parallel for
    for(v_size i = 2; i <= g->maxDeg; i++) {
        for(v_size u = 0; u < g->vCnt; u++) {
            v_size uDeg = g->pIdx[u+1] - g->pIdx[u];
            if(uDeg >= i) {
                CO[i][idx[i]].first = NO[g->pIdx[u] + i - 1].first;
                CO[i][idx[i]++].second = u;
            }
        }
    }

    // #pragma omp parallel for
    // for(v_size i = 2; i <= g->maxDeg; i++) {
    //     sort(CO[i], CO[i] + idx[i]);
    // }
    for(v_size i = 2; i <= g->maxDeg; i++) {
        if(idx[i] > 10000) {
            parallelSort(CO[i], CO[i] + idx[i]);
        }
        else {
            #pragma omp parallel for
            for(v_size j = i; j <= g->maxDeg; j++) {
                sort(CO[j], CO[j] + idx[j]);
            }

            break;
        }
    }
}

void SCAN::saveSim(const std::string & path) {
    // fastIO writer(simPath, "wb");

    // writer.writeArray<int>(sim, g->vCnt);
    std::ofstream outfile(path.c_str(), std::ios::out);
    if(!outfile.is_open()) {
        printf("create file error!!!!!!!!!!!!!!!\n");
        return;
    }
    uint32_t u = 0;
    for(uint32_t i = 0; i < g->eCnt; i++) {
        if(i == g->pIdx[u + 1]) {
            u++;
        }
        outfile << u << ' ' <<  g->pEdge[i] << ' ' << sim[i] << '\n';
    }
    outfile.close();

    // u = 1;
    // uint32_t v = 92;
    // for(uint32_t i = g->pIdx[u]; i < g->pIdx[u+1]; i++) {
    //     printf("%u ", g->pEdge[i]);
    // }
    // printf("\n");
    // for(uint32_t i = g->pIdx[v]; i < g->pIdx[v+1]; i++) {
    //     printf("%u ", g->pEdge[i]);
    // }
    // printf("\n");
}

SCAN::~SCAN() { 
    delete [] sim;
    for(v_size i = 2; i <= g->maxDeg; i++) {
        delete [] CO[i];
    }
    delete [] CO;
    delete [] idx;
}

#endif