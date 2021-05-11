#ifndef DISTRIBUTEDSCANRAW_HPP
#define DISTRIBUTEDSCANRAW_HPP

#include "../graph/distributedGraph.hpp"
#include "../tools/multiThreads.hpp"
#include "../tools/fastIO.hpp"
#include "../tools/ufs.hpp"
#include "../tools/bitmap.hpp"

#include <unordered_map>
#include <algorithm>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <cmath>
#include <sched.h>             //C API parameters and prototypes
#include <cstdio>
#include <bitset>

#include <x86intrin.h>
#include <immintrin.h>
#include <fstream>

using namespace std;
#define map unordered_map
constexpr v_size basicChunk = 64;
constexpr v_size bsEps = 1000;
constexpr v_size bsEps2 = 1000000;
constexpr double smallFloat = 1e-8;

class dSCAN {
private:
    dGraph * g;

    v_size *clusterId = nullptr; //[vCnt], 最终每个点的clusterId
    int * sim = nullptr; //[lECnt]
    int * epsNei = nullptr; //[lVCnt]
    
    v_size ** sendVertex = nullptr; //[pattitions][] 
    v_size * sendVerticesIndex  = nullptr; //[g->partitions]
    // map<v_size, std::pair<int, int>> hash; //<v_other_partition, <hashId, cnt>>
    v_size * hashVId = nullptr;
    int * vOsum = nullptr; 

    v_size vOCnt = 0;  //num-v_other_partition
    e_size eOCnt = 0;  //num-v_other_partition
    pair<v_size, int> **sourceVertex = nullptr; //[vOtherCnt][]
    int *sourceVertexIndex = nullptr;   //[vOtherCnt]
    v_size * reverseHash = nullptr; //[vOtherCnt]

    v_size * pOEdge = nullptr;  //[eOtherCnt]
    v_size * pOIdx = nullptr;   //[vOtherCnt]

    v_size * oPartitionOffset = nullptr; //[threads]

    int * totalEps = nullptr;   //[vCnt]
    map<v_size, v_size> vertexMap;  //<v, clusterId-v>
    parallelUfs2 myufs;

    multiThreads * threadController;

    string pEdgePath;

    double eps = 0.2;
    int e1, e2, e102, e10x;
    int mu = 5;

    //void prune();
    void prepareGetAdjList1(); //
    void prepareGetAdjList2();
    void createDoubleLayerGraph1();  void createDoubleLayerGraph12();
    void createDoubleLayerGraph2();
    void createDoubleLayerLocal();
    void computeDoubleLayer();
    void computeLoacl_naive(); void computeLoacl_without_lexi();
    void computeLoacl();
    void computeTotalEps();
    void uniteCoreLoacl();
    void uniteCoreInter();
    void uniteCoreInter2();
    void initClusterId();
    void clusterNoneCore();
    int getPartitionId(v_size v, bool f, int id);
    bool isInSendArea(int vPid);
    void mapFtoV(std::function<void(v_size)>, v_size *, bool );
    
public:
    dSCAN(dGraph * g, int mu, double eps, multiThreads * threadController_, const string & path);
    ~dSCAN();
    void run();
    void saveFile(string path);
};

void dSCAN::run() {
    #ifdef PRINT_DEBUG_MESSAGES
    double time1 = MPI_Wtime(), tmp;
    double timeS = time1;
    #endif

    // prune();
    // #ifdef PRINT_DEBUG_MESSAGES
    // tmp = MPI_Wtime();
    // printf("part %d, prune %f s\n", g->mpiController->partitionId, tmp - time1);
    // time1 = tmp;
    // #endif

    prepareGetAdjList1();
    #ifdef PRINT_DEBUG_MESSAGES
    tmp = MPI_Wtime();
    printf("part %d, prepareGetAdjList1 %f s\n", g->mpiController->partitionId, tmp - time1);
    time1 = tmp;
    #endif

    prepareGetAdjList2();
    #ifdef PRINT_DEBUG_MESSAGES
    tmp = MPI_Wtime();
    printf("part %d, prepareAdj2 %f s\n", g->mpiController->partitionId, tmp - time1);
    time1 = tmp;
    #endif

    createDoubleLayerGraph1();
    createDoubleLayerGraph2();
    // createDoubleLayerLocal();
    #ifdef PRINT_DEBUG_MESSAGES
    tmp = MPI_Wtime();
    printf("part %d, createDoubleLayer %f s\n", g->mpiController->partitionId, tmp - time1);
    time1 = tmp;
    #endif

    computeDoubleLayer();
    #ifdef PRINT_DEBUG_MESSAGES
    tmp = MPI_Wtime();
    printf("part %d, compute double local %f s\n", g->mpiController->partitionId, tmp - time1);
    time1 = tmp;
    #endif

    // computeLoacl_naive();
    // computeLoacl();
    computeLoacl_without_lexi();
    #ifdef PRINT_DEBUG_MESSAGES
    tmp = MPI_Wtime();
    printf("part %d, compute local cost %f s\n", g->mpiController->partitionId, tmp - time1);
    time1 = tmp;
    #endif

    computeTotalEps();
    #ifdef PRINT_DEBUG_MESSAGES
    tmp = MPI_Wtime();
    printf("part %d, compute totEps cost %f s\n", g->mpiController->partitionId, tmp - time1);
    time1 = tmp;
    #endif

    uniteCoreLoacl();
    #ifdef PRINT_DEBUG_MESSAGES
    tmp = MPI_Wtime();
    printf("part %d, uniteCoreLocal %f s\n", g->mpiController->partitionId, tmp - time1);
    time1 = tmp;
    #endif
    uniteCoreInter2();
    #ifdef PRINT_DEBUG_MESSAGES
    tmp = MPI_Wtime();
    printf("part %d, uniteCoreInter %f s\n", g->mpiController->partitionId, tmp - time1);
    time1 = tmp;
    #endif

    initClusterId();
    clusterNoneCore();
    #ifdef PRINT_DEBUG_MESSAGES
    tmp = MPI_Wtime();
    printf("part %d, cluster NoneCore %f s\n", g->mpiController->partitionId, tmp - time1);
    time1 = tmp;
    #endif

    #ifdef PRINT_DEBUG_MESSAGES
    printf("part %d, total cost %f s\n", g->mpiController->partitionId,  MPI_Wtime() - timeS);
    #endif
}

dSCAN::dSCAN(dGraph * g_, int mu_, double eps_, multiThreads * threadController_, const string & path) {
    g = g_;
    eps = eps_;
    e10x = eps * bsEps + smallFloat;
    mu = mu_;
    pEdgePath = path;

    threadController = threadController_;

    sim = new int[g->lECnt]();
    epsNei = new int[g->lVCnt]();
    
    totalEps = new int[g->vCnt]();
    hashVId = new v_size[g->vCnt]();
    memset(hashVId, 0xfffffffe, g->vCnt);

    myufs.init(g->vCnt);

    for(int i = 0; i < g->mpiController->partitions; i++) {
        if(isInSendArea(i)) {
            printf("send Area:%d %d\n", g->mpiController->partitionId, i);
        }
    }
    // bitMapPrune = new bitmap<g->lECnt>();
}

dSCAN::~dSCAN() {
    delete [] clusterId; //[vCnt], 最终每个点的clusterId
    delete [] sim; //[lECnt]
    delete [] epsNei; //[lVCnt]
    
    for(int i = 0; i < g->mpiController->partitions; i++) {
        delete [] sendVertex[i];
    }
    delete [] sendVertex ; //[pattitions][] 
    delete [] sendVerticesIndex  ; //[g->partitions]
    for(v_size i = 0; i < vOCnt; i++) {
        delete [] sourceVertex[i];
    }
    delete [] sourceVertex ; //[vOtherCnt][]
    delete [] sourceVertexIndex ;   //[vOtherCnt]
    delete [] reverseHash ; //[vOtherCnt]
    delete [] pOEdge ;  //[eOtherCnt]
    delete [] pOIdx ;   //[vOtherCnt]

    delete [] oPartitionOffset ; //[threads]
    delete [] totalEps ;   //[vCnt]
}

int dSCAN::getPartitionId(v_size v, bool f, int id = -1) {
    // int ret = g->mpiController->partitionId;
    // if(id != -1) ret = id;
    // if(f) {
    //     while(v >= g->disPartitionOffset[ret + 1]) ret++;
    // }
    // else {
    //     do {
    //         ret--;
    //     }while(v < g->disPartitionOffset[ret]);
    // }
    int ret = 0;
    while(v >= g->disPartitionOffset[ret + 1]) {
        ret++;
        // if(ret + 1 > g->mpiController->partitions) {
        //     assert(false);
        // }
    }

    return ret;
}

bool dSCAN::isInSendArea(int vPartitionId) {
    int ed1 = g->mpiController->partitionId + g->mpiController->partitions / 2;
    
    if(g->mpiController->partitions & 1) {
        if(ed1 < g->mpiController->partitions) 
            return g->mpiController->partitionId < vPartitionId && vPartitionId <= ed1;
        return (vPartitionId > g->mpiController->partitionId) || (vPartitionId <= ed1 - g->mpiController->partitions);
    }
    else {
        if(ed1 < g->mpiController->partitions)
            return  g->mpiController->partitionId < vPartitionId && vPartitionId <= ed1;
        return (vPartitionId >g->mpiController-> partitionId) || (vPartitionId < ed1 - g->mpiController->partitions);
    }

    return true;
}

void dSCAN::mapFtoV(std::function<void(v_size)> f, v_size * threadOffset, bool change = false) {
    threadController->init(threadOffset);
    if(change) {
        threadController->ok = true;
        threadController->cv.notify_all();
    }

    #pragma omp parallel num_threads(threadController->totalThreads)
    {
        int threadId = omp_get_thread_num();
        v_size st;
        auto &vec = threadController->stVAndEdV[threadId];
        v_size ed = vec.second;
// printf("part %d thread %d, st %u. ed %u\n", g->mpiController->partitionId, threadId, st, ed);
        while((st = __sync_fetch_and_add(&vec.first, basicChunk)) < ed) {
            v_size ed2 = min(st + basicChunk, ed);
            for(v_size u = st; u < ed2; u++) f(u);
        }
        // for(int i = 1; i < threadController->totalThreads; i++) {
        //     int tId = (threadId + i) % threadController->totalThreads;
        for(int tId = 0; tId < threadController->totalThreads; tId++) {
            while((st = __sync_fetch_and_add(&threadController->stVAndEdV[tId].first, basicChunk)) 
                < threadController->stVAndEdV[tId].second) {
                v_size ed2 = min(st + basicChunk, threadController->stVAndEdV[tId].second);
                for(v_size u = st; u < ed2; u++) f(u);
            }
        }
    }
}

// void dSCAN::prune()
// {
//     auto f = [&](v_size u) {
//         v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];
//         v_size uOff = u - g->stV;
//         for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
//             v_size edgeOffset = i - g->pIdxV0;
//             v_size v = g->pEdge[edgeOffset];
//             int vPartitionId = getPartitionId(v, v >= g->stV);
//             if(vPartitionId == g->mpiController->partitionId) continue;
//             v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
//             //d[u]+2 < ⌈ϵ d[u] + 1)(d[v] + 1)⌉
//             if((double)(uDeg + 2)*(uDeg + 2) < eps*eps*(uDeg+1)*(vDeg+1) || 
//                 ((double)vDeg + 2)*(vDeg + 2) < eps*eps*(uDeg+1)*(vDeg+1)) {
//                 sim[edgeOffset] = 2;
//                 continue;
//             }
//             if(4 >= eps*eps*(uDeg+1)*(vDeg+1)) {
//                 epsNei[uOff]++;
//                 sim[edgeOffset] = 2;
//                 continue;
//             }
//             if(!isInSendArea(vPartitionId)) continue;
//             __sync_fetch_and_add(&sendVerticesIndex[vPartitionId], 1);
//             bitMapPrune->set(edgeOffset);
//         }
//     };
//     mapFtoV(f, g->partitionOffset);
// }

void dSCAN::prepareGetAdjList1() {
    //alloc memory 1
    sendVerticesIndex = new v_size[g->mpiController->partitions]();
    
    mutex Mutex;

    //bitset<g->vCnt> *bitmap = new bitset<g->vCnt>();
    Bitmap * bitmap = new Bitmap(g->vCnt);
    // printf("count %u\n", bitmap->count());
    // v_size tmp3 = 0;
    auto f = [&](v_size u) {
        v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];
        v_size uOff = u - g->stV;//partition offset

        for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
            v_size edgeOffset = i - g->pIdxV0;

            v_size v = g->pEdge[edgeOffset];
            int vPartitionId = getPartitionId(v, v >= g->stV);
           
            v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];

            //d[u]+2 < ⌈ϵ d[u] + 1)(d[v] + 1)⌉
            double bound = eps*eps*(uDeg+1)*(vDeg+1);
            if((double)(uDeg + 2)*(uDeg + 2) + smallFloat < bound || 
                ((double)vDeg + 2)*(vDeg + 2) + smallFloat < bound) {
                sim[edgeOffset] = 2;
                continue;
            }

            if(4 + smallFloat >= bound) {
                epsNei[uOff]++;
                sim[edgeOffset] = 2;
                continue;
            }
            // long long bound = ((long long)e10x*e10x*(uDeg+1)*(vDeg+1) + bsEps2 - 1) / bsEps2;
            // if((long long)(uDeg + 2)*(uDeg + 2) < bound || 
            //     (long long)(vDeg + 2)*(vDeg + 2) < bound) {
            //     sim[edgeOffset] = 2;
            //     continue;
            // }

            // if(4 >= bound) {
            //     epsNei[uOff]++;
            //     sim[edgeOffset] = 2;
            //     continue;
            // }

            if(vPartitionId == g->mpiController->partitionId) {
                continue;
            }
            // else {
            //      __sync_fetch_and_add(&tmp3, 1);
            // }
            if(!isInSendArea(vPartitionId)) continue;
            
            __sync_fetch_and_add(&sendVerticesIndex[vPartitionId], 1);
            bitmap->SetBit(v);
        }
    };
    mapFtoV(f, g->partitionOffset);
    // v_size tmp = 0, tmp2 = 0;
    // for(v_size u = g->stV; u < g->edV; u++) {
    //     for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
    //         v_size v = g->pEdge[i - g->pIdxV0];
    //         int vPartitionId = getPartitionId(v, v >= g->stV);
    //         if(vPartitionId == g->mpiController->partitionId) tmp++;
    //         else tmp2++;
    //     }
    // }
    // printf("part%d, %.2f %.2f %.2f\n", g->mpiController->partitionId,
    //      (double)tmp/g->lECnt, (double)tmp2/g->lECnt, (double)tmp3/tmp2);

    vOCnt = bitmap->count();

    printf("part %d hash size %u\n", g->mpiController->partitionId, vOCnt);
    reverseHash = new v_size[vOCnt];

    v_size pr = 0;
    #pragma omp parallel for
    for(v_size i = 0; i < g->vCnt; i++) {
        if(bitmap->GetBit(i)) hashVId[i] = __sync_fetch_and_add(&pr, 1);
    }
    assert(pr == vOCnt);
    delete bitmap;

    vOsum = new int[vOCnt]();
    auto f3 = [&](v_size u) {
        for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
            v_size edgeOffset = i - g->pIdxV0;

            v_size v = g->pEdge[edgeOffset];
            int vPartitionId = getPartitionId(v, v >= g->stV);

            if(vPartitionId == g->mpiController->partitionId) continue;

            //d[u]+2 < ⌈ϵ d[u] + 1)(d[v] + 1)⌉
            if(sim[edgeOffset] != 0) continue;
            if(!isInSendArea(vPartitionId)) continue;
            
            __sync_fetch_and_add(&vOsum[hashVId[v]], 1);
            reverseHash[hashVId[v]] = v;
        }
    };
    mapFtoV(f3, g->partitionOffset);
    //alloc memory 2
    sendVertex = new v_size*[g->mpiController->partitions];
    for(int i = 0; i < g->mpiController->partitions; i++) {
        sendVertex[i] = new v_size[sendVerticesIndex[i]];
        sendVerticesIndex[i] = 0;
    }

    sourceVertex = new pair<v_size, int>*[vOCnt];
    for(v_size vId = 0; vId < vOCnt; vId++) {
        v_size v = reverseHash[vId];
        int uCnt = vOsum[vId];
        int partitionId = getPartitionId(v, v >= g->stV);

        sendVertex[partitionId][sendVerticesIndex[partitionId]++] = v;
        sourceVertex[vId] = new pair<v_size, int>[uCnt];
    }

    sourceVertexIndex = new int[vOCnt]();
}

void dSCAN::prepareGetAdjList2() {
    mutex Mutex;

    auto f = [&](v_size u) {
        v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];

        for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
            int edgeOffset = i - g->pIdxV0;
            //assert(edgeOffset < g->lECnt);
            v_size v = g->pEdge[edgeOffset];
            int vPartitionId = getPartitionId(v, v >= g->stV);

            if(vPartitionId == g->mpiController->partitionId) continue;
            if(!isInSendArea(vPartitionId)) continue;

            v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];

            //d[u]+2 < ⌈ϵ d[u] + 1)(d[v] + 1)⌉
            // if((double)(uDeg + 2)*(uDeg + 2) < eps*eps*(uDeg+1)*(vDeg+1) || 
            //     ((double)vDeg + 2)*(vDeg + 2) < eps*eps*(uDeg+1)*(vDeg+1)) {
            //     continue;
            // }

            // if(4 >= eps*eps*(uDeg+1)*(vDeg+1)) continue;
            if(sim[edgeOffset] != 0) continue;

            v_size vId = hashVId[v];
            
            // assert(vId < vOCnt);
            
            int pos = __sync_fetch_and_add(&sourceVertexIndex[vId], 1);

            // assert(pos < hash[v].second);
            sourceVertex[vId][pos].first = u;
            sourceVertex[vId][pos].second = edgeOffset;
        }
    };

    mapFtoV(f, g->partitionOffset);
   // printf("%d, %d %d %d %d\n", g->mpiController->partitionId , getPartitionId(0, 0>=g->stV),
     //  getPartitionId(1746, 1746>=g->stV),getPartitionId(4000, 4000>=g->stV),getPartitionId(2400, 2400>=g->stV) );
}

void dSCAN::createDoubleLayerGraph1() {
    oPartitionOffset = new v_size[threadController->totalThreads + 1];
    
    e_size *pSumDeg = new e_size[vOCnt]();
    pOIdx = new v_size[vOCnt + 1]();
    e_size sumDeg = 0, vSumDeg = 0;

    #pragma omp parallel for reduction(+:sumDeg)
    for(v_size vId = 0; vId < vOCnt; vId++) {
        e_size cnt = 0;

        for(int i = 0; i < sourceVertexIndex[vId]; i++) {
            v_size u = sourceVertex[vId][i].first;
            cnt += g->pIdx[u + 1] - g->pIdx[u];
        }

        pSumDeg[vId] = cnt;
        sumDeg += cnt;
    }

    #pragma omp parallel for reduction(+:vSumDeg)
    for(v_size vId = 0; vId < vOCnt; vId++) {
        v_size v = reverseHash[vId];
        v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];

        vSumDeg += vDeg;
        pOIdx[vId + 1] = vDeg;
    }
    eOCnt = vSumDeg;

    printf("part %d, eOcnt %lu\n", g->mpiController->partitionId, eOCnt);
    pOEdge = new v_size[eOCnt]();

    for(v_size vId = 1; vId <= vOCnt; vId++) {
        pOIdx[vId] += pOIdx[vId - 1];
    }
    assert(pOIdx[vOCnt] == eOCnt);

    oPartitionOffset[0] = 0;
    v_size u = 0;
    e_size partedEdges = 0;
    for(int i = 1; i < threadController->totalThreads; i++) {
        e_size expected = partedEdges + 
            (sumDeg - partedEdges) / (threadController->totalThreads - i + 1);
        if(expected > sumDeg) expected = sumDeg;

        while(partedEdges < expected) {
            partedEdges += pSumDeg[u++];
        }
        oPartitionOffset[i] = u;
    }
    oPartitionOffset[threadController->totalThreads] = vOCnt;

    delete [] pSumDeg;
}

void dSCAN::createDoubleLayerGraph12() {
    oPartitionOffset = new v_size[threadController->totalThreads + 1]; 
    e_size *pSumDeg = new e_size[vOCnt]();
    pOIdx = new v_size[vOCnt + 1]();
    e_size sumDeg = 0, vSumDeg = 0;

    #pragma omp parallel for reduction(+:vSumDeg)
    for(v_size vId = 0; vId < vOCnt; vId++) {
        v_size v = reverseHash[vId];
        v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];

        vSumDeg += vDeg;
        pOIdx[vId + 1] = vDeg;
        pSumDeg[vId] = vDeg;
    }
    sumDeg = eOCnt = vSumDeg;

    printf("part %d, eOcnt %lu\n", g->mpiController->partitionId, eOCnt);
    pOEdge = new v_size[eOCnt]();

    for(v_size vId = 1; vId <= vOCnt; vId++) {
        pOIdx[vId] += pOIdx[vId - 1];
    }
    assert(pOIdx[vOCnt] == eOCnt);

    oPartitionOffset[0] = 0;
    v_size u = 0;
    e_size partedEdges = 0;
    for(int i = 1; i < threadController->totalThreads; i++) {
        e_size expected = partedEdges + 
            (sumDeg - partedEdges) / (threadController->totalThreads - i + 1);
        if(expected > sumDeg) expected = sumDeg;

        while(partedEdges < expected) {
            partedEdges += pSumDeg[u++];
        }
        oPartitionOffset[i] = u;
    }
    oPartitionOffset[threadController->totalThreads] = vOCnt;

    delete [] pSumDeg;
}


enum messageTag
{
    RequestAdj,
    RequestAdjNum,
    SendAdj,
    UnionVertices,
    UniteFather,
    SendSim,
    UniteFather2,
    UniteNonCore,
    SendEps,
    Ready,
    SendReady1,
    SendReady2,
    SendSteal,
    SendStealAns
};

void dSCAN::createDoubleLayerGraph2() {
    int partitions = g->mpiController->partitions;
    int partitionId = g->mpiController->partitionId;

    if(g->mpiController->partitions == 1) return;

    v_size ** adjRequest = new v_size*[partitions];
    v_size * adjRequestIndex = new v_size[partitions]();

    std::thread recvAdjRequestNumThread([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        int recvPartitionsNum = partitions - 1;
        while (finished_count < recvPartitionsNum) {
            MPI_Probe(MPI_ANY_SOURCE, messageTag::RequestAdjNum, MPI_COMM_WORLD, &recv_status);
            int sourcePartitionId = recv_status.MPI_SOURCE;
            int recv_bytes;
            MPI_Get_count(&recv_status, MPI_BYTE, &recv_bytes);
            if (recv_bytes==sizeof(v_size)) {
                finished_count += 1;
                MPI_Recv(&adjRequestIndex[sourcePartitionId], sizeof(v_size), MPI_BYTE, sourcePartitionId, 
                    messageTag::RequestAdjNum, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                continue;
            }
            else printf("error part %d, from %d, bytes %d\n", partitionId, sourcePartitionId, recv_bytes);
        }
    });
    std::thread sendAdjRequestNumThread([&]() {
        for(int i = 1; i < partitions; i++) {
            int id = (partitionId + i) % partitions;
            MPI_Send(&sendVerticesIndex[id], sizeof(v_size), MPI_BYTE, id, 
                messageTag::RequestAdjNum, MPI_COMM_WORLD);
        }
    });
    
    sendAdjRequestNumThread.join();
    recvAdjRequestNumThread.join();

    for(int i = 0; i < partitions; i++) {
        adjRequest[i] = new v_size[adjRequestIndex[i]]();
        adjRequestIndex[i] = 0;
    }

    //对于发送区，按chunk发送请求
    std::thread recvAdjRequest([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        int recvPartitionsNum = partitions-1;

        while (finished_count < recvPartitionsNum) {
            MPI_Probe(MPI_ANY_SOURCE, messageTag::RequestAdj, MPI_COMM_WORLD, &recv_status);
            int sourcePartitionId = recv_status.MPI_SOURCE;
            // assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
            int recv_bytes;
            MPI_Get_count(&recv_status, MPI_BYTE, &recv_bytes);
            if (recv_bytes==1) {
                finished_count += 1;
                uint8_t c;
                /*MPI_Rev(buf,count,datatype,source,tag,comm,status)*/
                MPI_Recv(&c, 1, MPI_BYTE, sourcePartitionId, 
                    messageTag::RequestAdj, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                continue;
            }
            assert(recv_bytes % sizeof(v_size) == 0);

            v_size recvVertices = recv_bytes / sizeof(v_size);
            MPI_Recv(adjRequest[sourcePartitionId] + adjRequestIndex[sourcePartitionId], 
                recv_bytes, MPI_BYTE, sourcePartitionId, messageTag::RequestAdj, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            adjRequestIndex[sourcePartitionId] += recvVertices;
            // printf("recvAdjRequedt part %d source %d bytes %d\n", partitionId, sourcePartitionId, recv_bytes);
        }
    });//thread 接收请求
    std::thread sendAdjRequest([&]() {
        for(int i = 1; i < partitions; i++) {
            int id = (partitionId + i) % partitions;
            // if(i == partitions/2 && !(partitions & 1) && id < partitionId) continue;
            v_size idx = 0;
            for(; idx + CHUNKSIZE <= sendVerticesIndex[id]; idx += CHUNKSIZE) {
                MPI_Send(sendVertex[id] + idx, CHUNKSIZE*sizeof(v_size), MPI_BYTE, id, 
                    messageTag::RequestAdj, MPI_COMM_WORLD);
                // printf("sendrequest from %d to %d\n", partitionId, id);
            }
            if(idx < sendVerticesIndex[id]) {
                MPI_Send(sendVertex[id] + idx, (sendVerticesIndex[id]-idx)*sizeof(v_size), 
                    MPI_BYTE, id, messageTag::RequestAdj, MPI_COMM_WORLD);
                //printf("sendrequest from %d to %d\n", partitionId, id);
            }
            uint8_t tmp = 0;
            MPI_Send(&tmp, 1, MPI_BYTE, id, messageTag::RequestAdj, MPI_COMM_WORLD);
            //printf("sendrequest byte from %d to %d\n", partitionId, id);
        }
    });
    sendAdjRequest.join();
    recvAdjRequest.join();

    constexpr v_size adjBufferSize = CHUNKSIZE;
    v_size * recvAdjListBuffer = new v_size[adjBufferSize];

    constexpr v_size sendAdjBufferSize = CHUNKSIZE;
    v_size ** sendAdjListBuffer = new v_size*[partitions];
    for(int i = 0; i < partitions; i++) {
        if(i == g->mpiController->partitionId) continue;
        sendAdjListBuffer[i] = new v_size[sendAdjBufferSize];
    }

    std::thread recvAdjList([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        int sendPartitionsNum = partitions - 1;

        while (finished_count < sendPartitionsNum) {
            MPI_Probe(MPI_ANY_SOURCE, messageTag::SendAdj, MPI_COMM_WORLD, &recv_status);
            
            int sourcePartitionId = recv_status.MPI_SOURCE;

            int recv_bytes;
            MPI_Get_count(&recv_status, MPI_BYTE, &recv_bytes);
            if (recv_bytes==1) {
                finished_count += 1;
                uint8_t c;
                /*MPI_Rev(buf,count,datatype,source,tag,comm,status)*/
                MPI_Recv(&c, 1, MPI_BYTE, sourcePartitionId, 
                    messageTag::SendAdj, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("partition %d and %d over\n", sourcePartitionId, partition_id);  
                continue;
            }
            assert(recv_bytes % sizeof(v_size) == 0);

            int recvVertices = recv_bytes / sizeof(v_size);
            MPI_Recv(recvAdjListBuffer, recv_bytes, MPI_BYTE, sourcePartitionId, 
                messageTag::SendAdj, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("recvAdjList part %d source %d, %d u\n", partition_id, sourcePartitionId, recvVertices);
            
            int i = 0;
            while(i < recvVertices) {
                v_size v = recvAdjListBuffer[i++];
                v_size recvNum = recvAdjListBuffer[i++];

                int vId = hashVId[v];

                memcpy(pOEdge + pOIdx[vId],
                    recvAdjListBuffer + i, sizeof(v_size) * recvNum);
                i += recvNum;

                pOIdx[vId] += recvNum;
            }

            if(i != recvVertices) {
                printf("recvV %d %d\n", i, recvVertices);
            }
            assert(i == recvVertices);
        }
    });//thread 接收边

    v_size sendBufferIndex[partitions];
    for(int i = 0; i < partitions; i++)
        sendBufferIndex[i] = 0;
    
    v_size requestIndex[partitions];
    for(int i = 0; i < partitions; i++)
        requestIndex[i] = 0;

    int finished = 0, i = partitionId;
    // if(!(partitions & 1) && partition_id >= finished) finished--;
    for(int i = 0; i < partitions; i++) {
        if(adjRequestIndex[i] == 0) finished++;
    }
    while(finished < partitions) {
        i = (i + 1) % partitions;
        if(requestIndex[i] == adjRequestIndex[i]) continue;

        v_size v = adjRequest[i][requestIndex[i]++];

        if(requestIndex[i] == adjRequestIndex[i]) finished++;
        int sourcePartitionId = i;

        int eOff = g->pIdx[v] - g->pIdxV0;
        v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
    
        if(sendBufferIndex[i] + vDeg + 2 <= sendAdjBufferSize) {
            sendAdjListBuffer[i][sendBufferIndex[i]++] = v;
            sendAdjListBuffer[i][sendBufferIndex[i]++] = vDeg;

            memcpy(sendAdjListBuffer[i] + sendBufferIndex[i], 
              g->pEdge + eOff, sizeof(v_size)*vDeg);
            sendBufferIndex[i] += vDeg;

            if(sendBufferIndex[i] + 2 >= sendAdjBufferSize) {
                MPI_Send(sendAdjListBuffer[i], sendBufferIndex[i] * sizeof(v_size), MPI_BYTE,
                  sourcePartitionId, messageTag::SendAdj, MPI_COMM_WORLD);
                sendBufferIndex[i] = 0;
            }
        }
        else {
            if(sendBufferIndex[i] + 2 >= sendAdjBufferSize) {
                MPI_Send(sendAdjListBuffer[i], sendBufferIndex[i]*sizeof(v_size), MPI_BYTE,
                    sourcePartitionId, messageTag::SendAdj, MPI_COMM_WORLD);
                sendBufferIndex[i] = 0;
            }

            auto ptr = g->pEdge + eOff;
            auto end = g->pEdge + eOff + vDeg;
            v_size tmp;
            while(ptr < end) {
                sendAdjListBuffer[i][sendBufferIndex[i]++] = v;
                tmp = sendAdjListBuffer[i][sendBufferIndex[i]] = 
                    std::min(sendAdjBufferSize - sendBufferIndex[i] - 1, static_cast<v_size>(end - ptr));
                sendBufferIndex[i]++;
                memcpy(sendAdjListBuffer[i] + sendBufferIndex[i], ptr, tmp * sizeof(v_size));

                sendBufferIndex[i] += tmp;
                ptr += tmp;

                if(sendBufferIndex[i] + 2 >= sendAdjBufferSize) {
                    MPI_Send(sendAdjListBuffer[i], sendBufferIndex[i]*sizeof(v_size), MPI_BYTE,
                        sourcePartitionId, messageTag::SendAdj, MPI_COMM_WORLD);
                    sendBufferIndex[i] = 0;
                }
            }
            assert(ptr == end);
        }
    }

    for(int i = 0; i < partitions; i++) {
        if(i == g->mpiController->partitionId) continue;
        if(sendBufferIndex[i] > 0) {
            MPI_Send(sendAdjListBuffer[i], sendBufferIndex[i]*sizeof(v_size), MPI_BYTE, 
                i, messageTag::SendAdj, MPI_COMM_WORLD);
            sendBufferIndex[i] = 0;
        }
        uint8_t tmp = 0;
        MPI_Send(&tmp, 1, MPI_BYTE, i, messageTag::SendAdj, MPI_COMM_WORLD);
    }
  
   // sendAdjList.join();
    recvAdjList.join();

    #pragma omp parallel for
    for(v_size vId = 1; vId < vOCnt; vId++) {
        v_size v = reverseHash[vId];
        v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];

        assert(pOIdx[vId] - pOIdx[vId - 1] == vDeg);
    }
    assert(pOIdx[0] == g->pIdx[reverseHash[0] + 1] - g->pIdx[reverseHash[0]]);

    pOIdx[0] = 0;
    #pragma omp parallel for
    for(v_size vId = 0; vId < vOCnt; vId++) {
        v_size v = reverseHash[vId];
        v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];

        pOIdx[vId + 1] = vDeg;
    }

    for(v_size vId = 1; vId <= vOCnt; vId++) {
        pOIdx[vId] += pOIdx[vId - 1];
    }

    for(int i = 0; i < partitions; i++) {
        delete [] adjRequest[i];
    }
    delete [] adjRequest;
    delete [] adjRequestIndex;
    delete [] recvAdjListBuffer;
    for(int i = 0; i < partitions; i++) {
        if(i == g->mpiController->partitionId) continue;
        delete [] sendAdjListBuffer[i];
    }
    delete [] sendAdjListBuffer;

    // for(v_size vId = 0; vId < vOCnt; vId++) {
    //     for(v_size j = pOIdx[vId] + 1; j < pOIdx[vId + 1]; j++) {
    //         if(pOEdge[j] <= pOEdge[j - 1]) {
    //             printf("%u %u %u\n", vId, pOEdge[j], pOEdge[j - 1]);
    //         }
    //         assert(pOEdge[j] > pOEdge[j - 1]);
    //     }
    // }
}

void dSCAN::createDoubleLayerLocal() {
    FILE* f = fopen(pEdgePath.c_str(), "rb");
    v_size * tmp = new v_size[vOCnt];
    memcpy(tmp, reverseHash, vOCnt * sizeof(v_size));
    sort(tmp, tmp + vOCnt);
    // for(v_size i = 1; i < vOCnt; i++) assert(tmp[i] > tmp[i - 1]);
// v_size *buffer = new v_size[g->vCnt];
    v_size offset = 0;
    for(v_size i = 0; i < vOCnt; i++) {
        v_size v = tmp[i];
        v_size vId = hashVId[v];

        // assert(fseek(f, g->pIdx[v] * sizeof(v_size), SEEK_SET) == 0);
        assert(fseek(f, (g->pIdx[v] - offset) * sizeof(v_size), SEEK_CUR) == 0);
        assert(fread(pOEdge + pOIdx[vId], 4, pOIdx[vId + 1] - pOIdx[vId], f) 
            == pOIdx[vId + 1] - pOIdx[vId]);
        offset = g->pIdx[v + 1];
    }
    fclose(f);

    delete [] tmp;
// delete [] buffer;
    for(v_size vId = 0; vId < vOCnt; vId++) {
        for(v_size j = pOIdx[vId] + 1; j < pOIdx[vId + 1]; j++) {
            if(pOEdge[j] <= pOEdge[j - 1]) {
                printf("%u %u %u\n", vId, pOEdge[j], pOEdge[j - 1]);
            }
            assert(pOEdge[j] > pOEdge[j - 1]);
        }
    }
}

inline void computeSimAndEps(v_size * pu, v_size * pv, v_size * endPu, v_size * endPv, 
    int & sim, double bound)
{
    sim = 2;
    int sqb = ceil(sqrt(bound));

    while(true) {
        while(sim + endPu - pu >= sqb && *pu < *pv) pu++;
        if(sim + endPu - pu < sqb) break; 
        while(sim + endPv - pv >= sqb && *pv < *pu) pv++;
        if(sim + endPv - pv < sqb) break; 

        if(*pu == *pv) {
            ++pu, ++pv, ++sim;
            if(sim >= sqb) break;
        }
    }
}
/*
    // inline void simAndEpsSIMD(v_size * pu, v_size * pv, v_size * endPu, v_size * endPv, 
    //     int & sim, double bound) 
    // {
    //     sim = 2;
    //     int sqb = ceil(sqrt(bound));

    //     constexpr int parallelism = 4;

    //     while(true) {

    //         {
    //             __m128i pivotV = _mm_set1_epi32(*pv);
    //             while(pu + parallelism < endPu) {
    //                 __m128i fourU = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pu));
    //                 __m128i cmp = _mm_cmpgt_epi32(pivotV, fourU);

    //                 auto mask = _mm_movemask_epi8(cmp);
    //                 auto cnt = mask == 0xffff ? parallelism : _popcnt32(mask) >> 2;

    //                 pu += cnt;

    //                 if(sim + endPu - pu < sqb) break;
    //                 if(cnt < parallelism) break;
    //             }
    //         }
    //         if(sim + endPu - pu < sqb) break;
    //         if(pu + parallelism >= endPu) break;

    //         {
    //             __m128i pivotU = _mm_set1_epi32(*pu);
    //             while(pv + parallelism < endPv) {
    //                 __m128i fourV = _mm_load_si128(reinterpret_cast<const __m128i *>(pv));
    //                 __m128i cmp = _mm_cmpgt_epi32(pivotU, fourV);

    //                 auto mask = _mm_movemask_epi8(cmp);
    //                 auto cnt = mask == 0xffff ? parallelism : _popcnt32(mask) >> 2;

    //                 pv += cnt;

    //                 if(sim + endPv - pv < sqb) break;
    //                 if(cnt < parallelism) break;
    //             }
    //         }
    //         if(sim + endPv - pv < sqb) break;
    //         if(pv + parallelism >= endPv) break;

    //         if(*pu == *pv) {
    //             ++pu, ++pv, ++sim;
    //             if(sim >= sqb) break;
    //         }
    //     }


    //     if(sim + endPu - pu < sqb) return;
    //     if(sim + endPv - pv < sqb) return;

    //     while(true) {
    //         while(sim + endPu - pu >= sqb && *pu < *pv) pu++;
    //         if(sim + endPu - pu < sqb) break; 
    //         while(sim + endPv - pv >= sqb && *pv < *pu) pv++;
    //         if(sim + endPv - pv < sqb) break; 

    //         if(*pu == *pv) {
    //             ++pu, ++pv, ++sim;
    //             if(sim >= sqb) break;
    //         }
    //     }
    // }
*/
inline void simAndEpsSIMD(v_size * pu, v_size * pv, v_size * endPu, v_size * endPv, 
    int & sim, double bound) 
{
    sim = 2;
    int sqb = ceil(sqrt(bound));
    // assert(sqb * sqb >= bound);
    // if((double)sqb * sqb + smallFloat < bound) {
    //     printf("smallFloat error: %d %f %f\n", sqb, bound, (double)sqb * sqb + smallFloat);
    //     sqb++;
    // }

    constexpr int parallelism = 8;

    while(true) {
        {
            __m256i pivotV = _mm256_set1_epi32(*pv);
            while(pu + parallelism < endPu) {
                __m256i fourU = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pu));
                __m256i cmp = _mm256_cmpgt_epi32(pivotV, fourU);

                auto mask = _mm256_movemask_epi8(cmp);
                auto cnt = mask == 0xffffffff ? parallelism : _popcnt32(mask) >> 2;

                pu += cnt;

                if(sim + endPu - pu < sqb) break;
                if(cnt < parallelism) break;
            }
        }
        if(sim + endPu - pu < sqb) break;
        if(pu + parallelism >= endPu) break;

        {
            __m256i pivotU = _mm256_set1_epi32(*pu);
            while(pv + parallelism < endPv) {
                __m256i fourV = _mm256_load_si256(reinterpret_cast<const __m256i *>(pv));
                __m256i cmp = _mm256_cmpgt_epi32(pivotU, fourV);

                auto mask = _mm256_movemask_epi8(cmp);
                auto cnt = mask == 0xffffffff ? parallelism : _popcnt32(mask) >> 2;

                pv += cnt;

                if(sim + endPv - pv < sqb) break;
                if(cnt < parallelism) break;
            }
        }
        if(sim + endPv - pv < sqb) break;
        if(pv + parallelism >= endPv) break;

        if(*pu == *pv) {
            ++pu, ++pv, ++sim;
            if(sim >= sqb) return;
        }
    }

    if(sim + endPu - pu < sqb) return;
    if(sim + endPv - pv < sqb) return;

    while(true) {
        while(sim + endPu - pu >= sqb && *pu < *pv) pu++;
        if(sim + endPu - pu < sqb) break; 
        while(sim + endPv - pv >= sqb && *pv < *pu) pv++;
        if(sim + endPv - pv < sqb) break; 

        if(*pu == *pv) {
            ++pu, ++pv, ++sim;
            if(sim >= sqb) break;
        }
    }
}

constexpr int tt = 16896;
void dSCAN::computeDoubleLayer() {
    auto f = [&](v_size vId) {
        v_size v = reverseHash[vId];
        v_size vDeg = pOIdx[vId + 1] - pOIdx[vId];
        auto endPv = pOEdge + pOIdx[vId + 1];

        for(int j = 0; j < sourceVertexIndex[vId]; j++) {
            v_size u = sourceVertex[vId][j].first;
            int edgeOffset = sourceVertex[vId][j].second;

            int uDeg = g->pIdx[u + 1] - g->pIdx[u];
            v_size uOff = u - g->stV;

            auto pv = pOEdge + pOIdx[vId];
            auto pu = g->pEdge + g->pIdx[u] - g->pIdxV0;
            auto endPu = pu + uDeg;
            double bound = eps*eps*(uDeg+1)*(vDeg+1);
            // int bound = (e10x*e10x*(uDeg+1)*(vDeg+1) + bsEps2 - 1) / bsEps2;
            // int bound = eps*sqrt((uDeg+1)*(vDeg+1));
            // if(bound * bound < eps*eps*(uDeg+1)*(vDeg+1)) bound++;

            //computeSimAndEps(pu, pv, endPu, endPv, sim[edgeOffset], bound);
            simAndEpsSIMD(pu, pv, endPu, endPv, sim[edgeOffset], bound);
// if(v == tt) {
//     printf("%d, sim %d, bd %f\n", u, sim[edgeOffset], sqrt(eps*eps*(uDeg+1)*(vDeg + 1)));
// }
            if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=bound) {
                __sync_fetch_and_add(&epsNei[uOff], 1);
                totalEps[v]++;
            }
        }
    };

    mapFtoV(f, oPartitionOffset);
}

void dSCAN::computeLoacl_naive() {
    auto f = [&](v_size u) {
        v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];
        v_size uOff = u - g->stV;

        auto endPu = g->pEdge + g->pIdx[u] - g->pIdxV0 + uDeg;
       int up = uDeg;

        for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
            int edgeOffset = i - g->pIdxV0;
            v_size v = g->pEdge[edgeOffset];
            int vPartitionId = getPartitionId(v, v >= g->stV);

            if(g->mpiController->partitionId != vPartitionId) continue;

            v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
            
            double bound = eps*eps*(uDeg+1)*(vDeg+1);

            // if((double)(uDeg + 2)*(uDeg + 2) < bound || 
            //     ((double)vDeg + 2)*(vDeg + 2) < bound) {
            //     up--;
            //     if(up < mu) break;
            //     continue;
            // }

            // if(4 >= bound) {
            //     continue;
            // }
            if(sim[edgeOffset] != 0) continue;

         //  if(u > v) continue;

            auto pv = g->pEdge + g->pIdx[v] - g->pIdxV0;
            auto pu = g->pEdge + g->pIdx[u] - g->pIdxV0;
            
            // computeSimAndEps(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
            simAndEpsSIMD(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
// assert(sim[edgeOffset] != 0);
            if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=bound) {
                // __sync_fetch_and_add(&epsNei[uOff], 1);
                // __sync_fetch_and_add(&epsNei[v - g->stV], 1);
               epsNei[uOff]++;
                
            }
            // int vEOff = lower_bound(pv, pv + vDeg, u) - pv;
            // assert(g->pEdge[vEOff + g->pIdx[v] - g->pIdxV0] == u);
            // sim[vEOff + g->pIdx[v] - g->pIdxV0] = sim[edgeOffset];
            // if((double)sim[edgeOffset]*sim[edgeOffset]>=bound) {
            //     __sync_fetch_and_add(&epsNei[uOff], 1);
            //     __sync_fetch_and_add(&epsNei[v - g->stV], 1);
            //     // epsNei[uOff]++;
            //     int vEOff = lower_bound(pv, pv + vDeg, u) - pv;
            //     assert(g->pEdge[vEOff + g->pIdx[v] - g->pIdxV0] == u);
            //     sim[vEOff + g->pIdx[v] - g->pIdxV0] = sim[edgeOffset];
            //     // dn++;
            //     // if(dn >= mu) break;
            // }
            // else {
            //     up--;
            //     if(up < mu) break;
            // }
        }
    };
    mapFtoV(f, g->partitionOffset);
}

void dSCAN::computeLoacl_without_lexi() {
    vector<thread> manager;
    if(g->mpiController->partitionId == 0) {
        manager.reserve(1);
        manager.emplace_back([&](){
            int readyNum = 0;
            MPI_Status recv_status;
            int help[g->mpiController->partitions];
            for(int i = 0; i < g->mpiController->partitions; i++) {
                help[i] = i;
            }

            while(true) {
                MPI_Probe(MPI_ANY_SOURCE, messageTag::Ready, MPI_COMM_WORLD, &recv_status);
                int sourcePartitionId = recv_status.MPI_SOURCE;
                uint8_t b;
                MPI_Recv(&b, 1, MPI_BYTE, sourcePartitionId, 
                    messageTag::Ready, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // printf("part %d over\n", sourcePartitionId);
                readyNum++;
                if(readyNum >= g->mpiController->partitions) break;

                for(int j = 0; j < g->mpiController->partitions; j++) {
                    if(help[j] == sourcePartitionId) {
                        for(int i = 1; i < g->mpiController->partitions; i++) {
                            int id = (sourcePartitionId - i + g->mpiController->partitions) % g->mpiController->partitions;
                            if(help[id] != id) continue;
                            MPI_Send(&id, 1, MPI_INT, j, messageTag::SendReady1, MPI_COMM_WORLD);
                            MPI_Send(&j, 1, MPI_INT, id, messageTag::SendReady2, MPI_COMM_WORLD);
                            help[j] = id;
                            break;
                        }
                    }
                }
            }
            
            int temp = -1;
            // #pragma omp parallel for
            for(int i = 0; i < g->mpiController->partitions; i++) {
                MPI_Send(&temp, 1, MPI_INT, i, messageTag::SendReady1, MPI_COMM_WORLD);
                MPI_Send(&temp, 1, MPI_INT, i, messageTag::SendReady2, MPI_COMM_WORLD);
            }
        });
    }

    vector<thread> threadPool;
    threadPool.reserve(g->mpiController->partitions);

    auto stealF = [&](int id) {
        while(!threadController->ok) {
            printf("%d wait for %d\n", g->mpiController->partitionId, id);
            unique_lock <std::mutex> lck(threadController->mtx);
            threadController->cv.wait(lck);
        }
// printf("client %d is being helped by %d\n", g->mpiController->partitionId, id);
        v_size stealChunk = basicChunk*threadController->totalThreads*5;
        for(int tId = 0; tId < threadController->totalThreads; tId++) {
            v_size st;
            while((st = __sync_fetch_and_add(&threadController->stVAndEdV[tId].first, 
                                                    stealChunk)) 
                < threadController->stVAndEdV[tId].second) {
                v_size ed2 = min(st + stealChunk, 
                                    threadController->stVAndEdV[tId].second);
                //send to ready part;
                v_size buffer[] = {st, ed2};
                MPI_Send(buffer, 2, MPI_UINT32_T, id, messageTag::SendSteal, MPI_COMM_WORLD);

                e_size recvNum = g->pIdx[ed2+1] - g->pIdx[st];
                // double tmp = -MPI_Wtime();
                MPI_Recv(sim + g->pIdx[st]-g->pIdxV0, recvNum, MPI_INT, id, 
                    messageTag::SendStealAns, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// printf("\n%d help %d, %u-%u %lu\n", 
//     id, g->mpiController->partitionId, st, ed2, recvNum);
//                 for(v_size u = st; u < ed2; u++) {
//                     int uDeg = g->pIdx[u + 1] - g->pIdx[u];
//                     v_size uOff = u - g->stV;
//                     for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
//                         int edgeOffset = i - g->pIdxV0;
//                         v_size v = g->pEdge[edgeOffset];
//                         int vPartitionId = getPartitionId(v, v >= g->stV);
//                         if(vPartitionId != g->mpiController->partitionId) continue;
//                         assert(sim[edgeOffset] != 0);
//                         v_size vDeg = g->pIdx[v + 1] - g->pIdx[v]; 
//                         double bound = eps*eps*(uDeg+1)*(vDeg+1);
//                         if((double)(uDeg + 2)*(uDeg + 2) < bound || 
//                             ((double)vDeg + 2)*(vDeg + 2) < bound) {
//                             assert(sim[edgeOffset] == 2);
//                             continue;
//                         }
//                         if(4 >= bound) {
//                             assert(sim[edgeOffset] == 2);
//                             continue;
//                         }
//                         auto pv = g->pEdge + g->pIdx[v] - g->pIdxV0;
//                         auto pu = g->pEdge + g->pIdx[u] - g->pIdxV0; 
//                         int tmp;
//                         int sqb = ceil(sqrt(bound));
//                         // computeSimAndEps(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
// //                         simAndEpsSIMD(pu, pv, pu + uDeg, pv + vDeg, tmp, bound);
// // if(tmp >= mu && sim[edgeOffset] < mu) printf("error sim: %u %u %d %d %d\n", u, v, tmp, sim[edgeOffset], sqb);
// // if(sim[edgeOffset] >= mu && tmp < mu) printf("error sim: %u %u %d %d %d\n", u, v, tmp, sim[edgeOffset], sqb);
//                         // assert(tmp == sim[edgeOffset]);
//                         if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=bound) {
//                             epsNei[uOff]++;
//                         }
//                     }
//                 }
//                 double tmp2 = MPI_Wtime();
//                 tmp += tmp2;
// printf("%d help %d, %u-%u, %.2fs %.2fs\n", 
//     id, g->mpiController->partitionId, st, ed2, tmp, MPI_Wtime()-tmp2);
            }
        }

        v_size buffer[] = {0, 0};
        MPI_Send(buffer, 2, MPI_UINT32_T, id, messageTag::SendSteal, MPI_COMM_WORLD);
        // printf("%d help %d over\n", id, g->mpiController->partitionId);
    };
    auto recvHelpId = std::thread([&](){
        while(true) {
            int id;
            MPI_Recv(&id, 1, MPI_INT, 0, messageTag::SendReady2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(id == -1) break;
// printf("%d receive help %d\n", g->mpiController->partitionId, id);
            threadPool.emplace_back(std::bind(stealF, id));
        }
    });

    auto f = [&](v_size u) {
        v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];
        v_size uOff = u - g->stV;

        auto endPu = g->pEdge + g->pIdx[u] - g->pIdxV0 + uDeg;
        int up = uDeg;

        for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
            int edgeOffset = i - g->pIdxV0;
            v_size v = g->pEdge[edgeOffset];
            int vPartitionId = getPartitionId(v, v >= g->stV);

            if(g->mpiController->partitionId != vPartitionId) continue;

            v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
            
            double bound = eps*eps*(uDeg+1)*(vDeg+1);
            
            if(sim[edgeOffset] != 0) continue;
            // if(u > v) continue;

            auto pv = g->pEdge + g->pIdx[v] - g->pIdxV0;
            auto pu = g->pEdge + g->pIdx[u] - g->pIdxV0;
            
            // computeSimAndEps(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
            simAndEpsSIMD(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);

            if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=bound) {
                epsNei[uOff]++;
            }
        }
    };
    mapFtoV(f, g->partitionOffset, true);

    //send to part 0 ready
    //thread receive <unready part> from part 0, load graph
    //send to unready part ready.
    //recv <st> and parallel compute and send sim. until receive down , jump to the second line

    uint8_t b = 0;
    MPI_Send(&b, 1, MPI_BYTE, 0, messageTag::Ready, MPI_COMM_WORLD);

    v_size bufferSize = 0;
    for(int i = 0; i < g->mpiController->partitions; i++) {
        v_size st = g->disPartitionOffset[i], ed = g->disPartitionOffset[i + 1];
        bufferSize = max(bufferSize, g->pIdx[ed] - g->pIdx[st]);
    }
    int * simBuffer = new int[bufferSize];
    v_size * partitionOffset = new v_size[threadController->totalThreads + 1];
    
    while(true) {
        int id;
        MPI_Recv(&id, 1, MPI_INT, 0, messageTag::SendReady1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// printf("%d strat help %d\n", g->mpiController->partitionId, id);
        if(id == -1) break;

        v_size lVCnt = g->disPartitionOffset[id + 1] - g->disPartitionOffset[id];
        v_size stV = g->disPartitionOffset[id];
        v_size edV = g->disPartitionOffset[id + 1];
        e_size lECnt = g->pIdx[edV] - g->pIdx[stV];
        v_size pIdxV0 = g->pIdx[stV];
        v_size *pEdge = new v_size[lECnt];
        fastIO<v_size> * readEdge = new fastIO<v_size>(pEdgePath, "rb");
        readEdge->seek(pIdxV0 * sizeof(v_size));
        for(v_size i = 0; i < lECnt; i++) {
            pEdge[i] = readEdge->getFromBin();
        }
        
        delete readEdge;

        v_size buffer[2];

        while(true) {
// double t1 = MPI_Wtime();
            MPI_Recv(buffer, 2, MPI_UINT32_T, id, messageTag::SendSteal, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(buffer[0] == 0 && buffer[1] == 0) break;
// double t2 = MPI_Wtime();
            e_size recvNum = 0;
            e_size sumDeg = 0;
            #pragma omp parallel for reduction(+:recvNum, sumDeg)
            for(v_size u = buffer[0]; u < buffer[1]; u++) {
                recvNum += g->pIdx[u+1] - g->pIdx[u];
                sumDeg += g->pSumDeg[u];
            }

            partitionOffset[0] = buffer[0];
            v_size u = buffer[0];
            e_size partedEdges = 0;
            for(int i = 1; i < threadController->totalThreads; i++) {
                e_size expected = partedEdges + 
                    (sumDeg - partedEdges) / (threadController->totalThreads - i + 1);
                if(expected > sumDeg) expected = sumDeg;

                while(partedEdges < expected) {
                    partedEdges += g->pSumDeg[u++];
                }
                partitionOffset[i] = u;
            }
            partitionOffset[threadController->totalThreads] = buffer[1];

            auto f2 = [&](v_size u) {
                v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];

// bool flag = false;
                auto endPu = pEdge + g->pIdx[u + 1] - pIdxV0;
                int up = uDeg;

                for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
                    int edgeOffset = i - pIdxV0;
                    v_size v = pEdge[edgeOffset];
                    int vPartitionId = getPartitionId(v, v >= stV, id);

                    if(id != vPartitionId) continue;

                    v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
                    double bound = eps*eps*(uDeg+1)*(vDeg+1);
                    int simOffset = i - g->pIdx[buffer[0]];

                    if((double)(uDeg + 2)*(uDeg + 2) + smallFloat < bound || 
                        ((double)vDeg + 2)*(vDeg + 2) + smallFloat < bound) {
                    // sim[edgeOffset] = 2;
                        // up--;
                        // if(up < mu) break;
                        simBuffer[simOffset] = 2;
                        continue;
                    }

                    if(4.0 + smallFloat >= bound) {
                        simBuffer[simOffset] = 2;
                        continue;
                    }

                    //if(u > v) continue;

                    auto pv = pEdge + g->pIdx[v] - pIdxV0;
                    auto pu = pEdge + g->pIdx[u] - pIdxV0;

                    // computeSimAndEps(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
                    //assert(i - g->pIdx[buffer[0]] < recvNum);
                    simAndEpsSIMD(pu, pv, endPu, pv + vDeg, simBuffer[simOffset], bound);
                    if((double)simBuffer[simOffset]*simBuffer[simOffset]+smallFloat >= bound) {
                        totalEps[u]++;
                        // flag = true;
                    }
                }
// if(!flag) printf("%u ", u);
            };
            mapFtoV(f2, partitionOffset);
// double t3 = MPI_Wtime();
            MPI_Send(simBuffer, recvNum, MPI_INT, id, messageTag::SendStealAns, MPI_COMM_WORLD);
// printf("%d help %d, %u-%u, recv %.2fs compute %.2fs send %.2fs\n", 
    // g->mpiController->partitionId, id, buffer[0], buffer[1], t2 - t1, t3 - t2, MPI_Wtime()-t3);
// printf("%d help %d, %u-%u -1\n", 
//     g->mpiController->partitionId, id, buffer[0], buffer[1]);
        }

// printf("%d help %d over\n", g->mpiController->partitionId, id);
        delete [] pEdge;
    }

    if(simBuffer != nullptr) delete [] simBuffer;
    delete [] partitionOffset;

    if(g->mpiController->partitionId == 0) manager[0].join();
    recvHelpId.join();
    for(auto & t:threadPool) t.join();

    // auto ff = [&](v_size u) {
    //     v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];
    //     v_size uOff = u - g->stV;
    //     auto endPu = g->pEdge + g->pIdx[u] - g->pIdxV0 + uDeg;
    //     for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
    //         int edgeOffset = i - g->pIdxV0;
    //         v_size v = g->pEdge[edgeOffset];
    //         int vPartitionId = getPartitionId(v, v >= g->stV);
    //         if(g->mpiController->partitionId != vPartitionId) continue;
    //         v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];    
    //         double bound = eps*eps*(uDeg+1)*(vDeg+1);
    //         if((double)(uDeg + 2)*(uDeg + 2) < bound || 
    //             ((double)vDeg + 2)*(vDeg + 2) < bound) {
    //             assert(sim[edgeOffset] == 2);
    //             continue;
    //         }
    //         if(4.0 + smallFloat >= bound) {
    //             assert(sim[edgeOffset] == 2);
    //             continue;
    //         }
    //      //  if(u > v) continue;
    //         auto pv = g->pEdge + g->pIdx[v] - g->pIdxV0;
    //         auto pu = g->pEdge + g->pIdx[u] - g->pIdxV0;
    //         int tmp = 0;
    //         // computeSimAndEps(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
    //         simAndEpsSIMD(pu, pv, endPu, pv + vDeg, tmp, bound);
    //         assert(tmp == sim[edgeOffset]);
    //         if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=bound) {
    //            epsNei[uOff]++;
    //         }
    //     }
    // };
    // mapFtoV(ff, g->partitionOffset);
}

void dSCAN::computeLoacl() {
    vector<thread> manager;
    if(g->mpiController->partitionId == 0) {
        // int ready[partitons]
        // thread receive ready
        // thread send to unready part <a ready part id>
        manager.reserve(1);
        manager.emplace_back([&](){
            int readyNum = 0;
            MPI_Status recv_status;
            int help[g->mpiController->partitions];
            memset(help, -1, sizeof(help));

            while(true) {
                MPI_Probe(MPI_ANY_SOURCE, messageTag::Ready, MPI_COMM_WORLD, &recv_status);
                int sourcePartitionId = recv_status.MPI_SOURCE;
                uint8_t b;
                MPI_Recv(&b, 1, MPI_BYTE, sourcePartitionId, 
                    messageTag::Ready, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // printf("part %d over\n", sourcePartitionId);
                readyNum++;
                if(readyNum >= g->mpiController->partitions) break;

                help[sourcePartitionId] = -2;
                for(int j = 0; j < g->mpiController->partitions; j++) {
                    if(help[j] == sourcePartitionId) {
                        for(int i = 1; i < g->mpiController->partitions; i++) {
                            int id = (sourcePartitionId - i + g->mpiController->partitions) % g->mpiController->partitions;
                            if(help[id] != -1) continue;
                            MPI_Send(&id, 1, MPI_INT, j, messageTag::SendReady1, MPI_COMM_WORLD);
                            MPI_Send(&j, 1, MPI_INT, id, messageTag::SendReady2, MPI_COMM_WORLD);
                            help[j] = id;
                            break;
                        }
                    }
                }
                
                for(int i = 1; i < g->mpiController->partitions; i++) {
                    int id = (sourcePartitionId - i + g->mpiController->partitions) % g->mpiController->partitions;
                    if(help[id] != -1) continue;
                    MPI_Send(&id, 1, MPI_INT, sourcePartitionId, messageTag::SendReady1, MPI_COMM_WORLD);
                    help[sourcePartitionId] = id;
                    MPI_Send(&sourcePartitionId, 1, MPI_INT, id, messageTag::SendReady2, MPI_COMM_WORLD);
                    break;
                }
            }
            
            int temp = -1;
            // #pragma omp parallel for
            for(int i = 0; i < g->mpiController->partitions; i++) {
                MPI_Send(&temp, 1, MPI_INT, i, messageTag::SendReady1, MPI_COMM_WORLD);
                MPI_Send(&temp, 1, MPI_INT, i, messageTag::SendReady2, MPI_COMM_WORLD);
            }
        });
    }

    vector<thread> threadPool;
    threadPool.reserve(g->mpiController->partitions);

    auto stealF = [&](int id) {
        while(!threadController->ok) {
            printf("%d wait for %d\n", g->mpiController->partitionId, id);
            unique_lock <std::mutex> lck(threadController->mtx);
            threadController->cv.wait(lck);
        }
// printf("client %d was helped by %d\n", g->mpiController->partitionId, id);

        for(int tId = 0; tId < threadController->totalThreads; tId++) {
            v_size st;
            while((st = __sync_fetch_and_add(&threadController->stVAndEdV[tId].first, 
                                                    basicChunk*threadController->totalThreads)) 
                < threadController->stVAndEdV[tId].second) {
                v_size ed2 = min(st + basicChunk*threadController->totalThreads , 
                                    threadController->stVAndEdV[tId].second);
                //send to ready part;
                v_size buffer[] = {st, ed2};
                MPI_Send(buffer, 2, MPI_UINT32_T, id, messageTag::SendSteal, MPI_COMM_WORLD);

                e_size recvNum = 0;
                for(v_size u = st; u < ed2; u++) {
                    recvNum += g->pIdx[u+1] - g->pIdx[u];
                }
                // double tmp = -MPI_Wtime();
                MPI_Recv(sim + g->pIdx[st]-g->pIdxV0, recvNum, MPI_INT, id, 
                    messageTag::SendStealAns, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // double tmp2 = MPI_Wtime();
                // tmp += tmp2;

                for(v_size u = st; u < ed2; u++) {
                    v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];
                    v_size uOff = u - g->stV;
                    int up = uDeg;

                    for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
                        int edgeOffset = i - g->pIdxV0;
                        v_size v = g->pEdge[edgeOffset];
                        int vPartitionId = getPartitionId(v, v >= g->stV);

                        if(g->mpiController->partitionId != vPartitionId) continue;
                        if(u > v) continue;

                        v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
                        
                        double bound = eps*eps*(uDeg+1)*(vDeg+1);

                        if((double)(uDeg + 2)*(uDeg + 2) + smallFloat < bound || 
                            ((double)vDeg + 2)*(vDeg + 2) + smallFloat < bound) {
                            continue;
                        }

                        if(4.0 + smallFloat >= bound) continue;
assert(sim[edgeOffset] != 0);
                        if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=bound) {
                            __sync_fetch_and_add(&epsNei[uOff], 1);
                            __sync_fetch_and_add(&epsNei[v - g->stV], 1);
                            // epsNei[uOff]++;
                        }
                        if(st <= v && v < ed2) continue;
                        auto pv = g->pEdge + g->pIdx[v] - g->pIdxV0;
                        int vEOff = lower_bound(pv, pv + vDeg, u) - pv;
assert(g->pEdge[vEOff + g->pIdx[v] - g->pIdxV0] == u);
                        sim[vEOff + g->pIdx[v] - g->pIdxV0] = sim[edgeOffset];
                    }
                }
// printf("%d help %d, %u-%u, %.2fs %.2fs\n", 
//     id, g->mpiController->partitionId, st, ed2, tmp, MPI_Wtime()-tmp2);
            }
        }

        v_size buffer[] = {0, 0};
        MPI_Send(buffer, 2, MPI_UINT32_T, id, messageTag::SendSteal, MPI_COMM_WORLD);
        // printf("%d help %d over\n", id, g->mpiController->partitionId);
    };
    auto recvHelpId = std::thread([&](){
        while(true) {
            int id;
            MPI_Recv(&id, 1, MPI_INT, 0, messageTag::SendReady2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(id == -1) break;
// printf("%d receive help %d\n", g->mpiController->partitionId, id);
            threadPool.emplace_back(std::bind(stealF, id));
        }
    });

    auto f = [&](v_size u) {
        v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];
        v_size uOff = u - g->stV;

        auto endPu = g->pEdge + g->pIdx[u] - g->pIdxV0 + uDeg;
        int up = uDeg;

        for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
            int edgeOffset = i - g->pIdxV0;
            v_size v = g->pEdge[edgeOffset];
            int vPartitionId = getPartitionId(v, v >= g->stV);

            if(g->mpiController->partitionId != vPartitionId) continue;

            v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
            
            double bound = eps*eps*(uDeg+1)*(vDeg+1);
            
            if(sim[edgeOffset] != 0) continue;

            if(u > v) continue;

            auto pv = g->pEdge + g->pIdx[v] - g->pIdxV0;
            auto pu = g->pEdge + g->pIdx[u] - g->pIdxV0;
            
            // computeSimAndEps(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
            simAndEpsSIMD(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);

            if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=bound) {
                __sync_fetch_and_add(&epsNei[uOff], 1);
                __sync_fetch_and_add(&epsNei[v - g->stV], 1);
                // epsNei[uOff]++;
                
// assert(g->pEdge[vEOff + g->pIdx[v] - g->pIdxV0] == u);
                // dn++;
                // if(dn >= mu) break;
            }
            int vEOff = lower_bound(pv, pv + vDeg, u) - pv;
assert(g->pEdge[vEOff + g->pIdx[v] - g->pIdxV0] == u);
            sim[vEOff + g->pIdx[v] - g->pIdxV0] = sim[edgeOffset];
            // else {
            //     up--;
            //     if(up < mu) break;
            // }
        }
    };
    mapFtoV(f, g->partitionOffset, true);

    //send to part 0 ready
    //thread receive <unready part> from part 0, load graph
    //send to unready part ready.
    //recv <st> and parallel compute and send sim. until receive down , jump to the second line
    // for(auto &t : threadPool) t.join();

    uint8_t b = 0;
    MPI_Send(&b, 1, MPI_BYTE, 0, messageTag::Ready, MPI_COMM_WORLD);

    v_size bufferSize = 0;
    for(int i = 0; i < g->mpiController->partitions; i++) {
        v_size st = g->disPartitionOffset[i], ed = g->disPartitionOffset[i + 1];
        bufferSize = max(bufferSize, g->pIdx[ed] - g->pIdx[st]);
    }
    int * simBuffer = new int[bufferSize];
    v_size * partitionOffset = new v_size[threadController->totalThreads + 1];
    
    while(true) {
        int id;
        MPI_Recv(&id, 1, MPI_INT, 0, messageTag::SendReady1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// printf("%d strat help %d\n", g->mpiController->partitionId, id);
        if(id == -1) break;

        v_size lVCnt = g->disPartitionOffset[id + 1] - g->disPartitionOffset[id];
        v_size stV = g->disPartitionOffset[id];
        v_size edV = g->disPartitionOffset[id + 1];
        e_size lECnt = g->pIdx[edV] - g->pIdx[stV];
        v_size pIdxV0 = g->pIdx[stV];
        v_size *pEdge = new v_size[lECnt];
        fastIO<v_size> * readEdge = new fastIO<v_size>(pEdgePath, "rb");
        readEdge->seek(pIdxV0 * sizeof(v_size));
        for(v_size i = 0; i < lECnt; i++) {
            pEdge[i] = readEdge->getFromBin();
        }
        delete readEdge;

        v_size buffer[2];

        while(true) {
// double t1 = MPI_Wtime();
            MPI_Recv(buffer, 2, MPI_UINT32_T, id, messageTag::SendSteal, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(buffer[0] == 0 && buffer[1] == 0) break;
// double t2 = MPI_Wtime();
            e_size recvNum = 0;
            e_size sumDeg = 0;
            #pragma omp parallel for reduction(+:recvNum, sumDeg)
            for(v_size u = buffer[0]; u < buffer[1]; u++) {
                recvNum += g->pIdx[u+1] - g->pIdx[u];
                sumDeg += g->pSumDeg[u];
            }

            partitionOffset[0] = buffer[0];
            v_size u = buffer[0];
            e_size partedEdges = 0;
            for(int i = 1; i < threadController->totalThreads; i++) {
                e_size expected = partedEdges + 
                    (sumDeg - partedEdges) / (threadController->totalThreads - i + 1);
                if(expected > sumDeg) expected = sumDeg;

                while(partedEdges < expected) {
                    partedEdges += g->pSumDeg[u++];
                }
                partitionOffset[i] = u;
            }
            partitionOffset[threadController->totalThreads] = buffer[1];

            auto f2 = [&](v_size u) {
                v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];

                auto endPu = pEdge + g->pIdx[u] - pIdxV0 + uDeg;
                int up = uDeg;

                for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
                    int edgeOffset = i - pIdxV0;
                    v_size v = pEdge[edgeOffset];
                    int vPartitionId = getPartitionId(v, v >= stV, id);

                    if(id != vPartitionId) continue;

                    v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
                    double bound = eps*eps*(uDeg+1)*(vDeg+1);

                    if((double)(uDeg + 2)*(uDeg + 2)+ smallFloat  < bound || 
                        ((double)vDeg + 2)*(vDeg + 2)+ smallFloat  < bound) {
                    // sim[edgeOffset] = 2;
                        // up--;
                        // if(up < mu) break;
                        simBuffer[i - g->pIdx[buffer[0]]] = 2;
                        continue;
                    }

                    if(4.0 + smallFloat  >= bound) {
                        simBuffer[i - g->pIdx[buffer[0]]] = 2;
                        continue;
                    }

                    //if(u > v) continue;

                    auto pv = pEdge + g->pIdx[v] - pIdxV0;
                    auto pu = pEdge + g->pIdx[u] - pIdxV0;

                    // computeSimAndEps(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
                    //assert(i - g->pIdx[buffer[0]] < recvNum);
                    simAndEpsSIMD(pu, pv, endPu, pv + vDeg, simBuffer[i - g->pIdx[buffer[0]]], bound);
                }
            };
            mapFtoV(f2, partitionOffset);
// double t3 = MPI_Wtime();
            MPI_Send(simBuffer, recvNum, MPI_INT, id, messageTag::SendStealAns, MPI_COMM_WORLD);
// printf("%d help %d, %u-%u, recv %.2fs compute %.2fs send %.2fs\n", 
    // g->mpiController->partitionId, id, buffer[0], buffer[1], t2 - t1, t3 - t2, MPI_Wtime()-t3);
        }

// printf("%d help %d over\n", g->mpiController->partitionId, id);
        delete [] pEdge;
    }

    if(simBuffer != nullptr) delete [] simBuffer;
    delete [] partitionOffset;

    if(g->mpiController->partitionId == 0) manager[0].join();
    recvHelpId.join();
    for(auto & t:threadPool) t.join();
}

void dSCAN::computeTotalEps() {
    // printf("part %d, there\n", g->mpiController->partitionId);

    auto f = [&](v_size u) {
        totalEps[u] += epsNei[u - g->stV];
    };
    mapFtoV(f, g->partitionOffset);

    MPI_Allreduce(MPI_IN_PLACE, totalEps, g->vCnt, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // std::ofstream outfile("./log/totalEps", std::ios::out);
    // if(!outfile.is_open()) {
    //     printf("create file error!!!!!!!!!!!!!!!\n");
    //     return;
    // }
    // for(v_size i = 0; i < g->vCnt; i++) {
    //     outfile << totalEps[i] << '\n';
    // }
    // outfile.close();
}

void dSCAN::uniteCoreLoacl() {
    auto f = [&](v_size u) {
        if(totalEps[u] < mu) return;

        v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];
        auto endPu = g->pEdge + g->pIdx[u] - g->pIdxV0 + uDeg;

        for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
            int edgeOffset = i - g->pIdxV0;
            v_size v = g->pEdge[edgeOffset];
// if(u == tt) {
//     printf("%d, sim %d, bd %f, veps %d\n", v, sim[edgeOffset], sqrt(eps*eps*(uDeg+1)*(g->pIdx[v + 1] - g->pIdx[v] + 1)), totalEps[v]);
//     if(totalEps[u] < mu) continue;
// }
            if(totalEps[v] < mu) continue;

            int vPartitionId = getPartitionId(v, v >= g->stV);
            v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
            double bound = eps*eps*(uDeg+1)*(vDeg+1);

            if(vPartitionId == g->mpiController->partitionId && sim[edgeOffset] == 0) {
printf("%u %u\n", u, v);
assert(false);
                auto pv = g->pEdge + g->pIdx[v] - g->pIdxV0;
                auto pu = g->pEdge + g->pIdx[u] - g->pIdxV0;
                double bound = eps*eps*(uDeg+1)*(vDeg+1);
                
                // computeSimAndEps(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
                simAndEpsSIMD(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
            }
            
            if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=bound) {
                myufs.unite(u, v, true);
            }
        }
    };
    mapFtoV(f, g->partitionOffset);
    // #pragma omp parallel num_threads(threadController->totalThreads)
    // {
    //     int threadId = omp_get_thread_num();
    //     v_size st = g->partitionOffset[threadId];
    //     v_size ed = g->partitionOffset[threadId + 1];

    //     for(v_size u = st; u < ed; u++) {
            
    //     }
    // }
}

void dSCAN::uniteCoreInter() {
    if(g->mpiController->partitionId == 0) {
        MPI_Status recv_status;
        int bufferSize = CHUNKSIZE;
        v_size * recvBuffer = new v_size[bufferSize];
        
        int count = 0;
        while(count < g->mpiController->partitions - 1) {
            MPI_Probe(MPI_ANY_SOURCE, messageTag::UniteFather, MPI_COMM_WORLD, &recv_status);
            int sourcePartitionId = recv_status.MPI_SOURCE;
            int recv_bytes;
            MPI_Get_count(&recv_status, MPI_BYTE, &recv_bytes);

            if(recv_bytes == 1) {
                uint8_t tmp;
                MPI_Recv(&tmp, 1, MPI_BYTE, sourcePartitionId, messageTag::UniteFather, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                count++;
                continue;
            }
            assert(recv_bytes % sizeof(VertexId) == 0);
            int recvNum = recv_bytes / sizeof(VertexId);

            MPI_Recv(recvBuffer, recv_bytes, MPI_BYTE, sourcePartitionId, 
                messageTag::UniteFather, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("p %d send %d B\n", sourcePartitionId, recv_bytes);

            VertexId st = recvBuffer[0];
            for (int i = 1; i < recvNum; ++i) {
                myufs.unite(i-1+st, recvBuffer[i], true);
            }
        }
        delete [] recvBuffer;
    }
    else {
        int bufferSize = CHUNKSIZE;
        v_size * sendBuffer = new v_size[bufferSize];
        
        v_size fatherIndex = 0;

        while(fatherIndex < g->vCnt) {
            sendBuffer[0] = fatherIndex;
            if(fatherIndex + bufferSize - 1 < g->vCnt) {
                memcpy(sendBuffer+1, myufs.father+fatherIndex, sizeof(VertexId)*(bufferSize-1));
                MPI_Send(sendBuffer, bufferSize*sizeof(VertexId), MPI_BYTE, 0, 
                    messageTag::UniteFather, MPI_COMM_WORLD);
                fatherIndex += bufferSize - 1;
            }
            else {
                memcpy(sendBuffer+1, myufs.father+fatherIndex, sizeof(VertexId)*(g->vCnt-fatherIndex));
                MPI_Send(sendBuffer, (g->vCnt-fatherIndex+1)*sizeof(VertexId), 
                    MPI_BYTE, 0, messageTag::UniteFather, MPI_COMM_WORLD);
                fatherIndex = g->vCnt;
            }
        }
        uint8_t tmp = 0;
        MPI_Send(&tmp, 1, MPI_BYTE, 0, messageTag::UniteFather, MPI_COMM_WORLD);
        
        delete [] sendBuffer;
    }

    // std::ofstream outfile("./log/fa", std::ios::out);
    // if(!outfile.is_open()) {
    //     printf("create file error!!!!!!!!!!!!!!!\n");
    //     return;
    // }
    // for(v_size i = 0; i < g->vCnt; i++) {
    //     outfile << myufs.findFather(i, true) << '\n';
    // }
    // outfile.close();
}

void dSCAN::uniteCoreInter2() {
    if(g->mpiController->partitionId == 0) {
        MPI_Status recv_status;
        int bufferSize = CHUNKSIZE;
        v_size * recvBuffer = new v_size[bufferSize];
        
        int count = 0;
        while(count < g->mpiController->partitions - 1) {
            MPI_Probe(MPI_ANY_SOURCE, messageTag::UniteFather, MPI_COMM_WORLD, &recv_status);
            int sourcePartitionId = recv_status.MPI_SOURCE;
            int recv_bytes;
            MPI_Get_count(&recv_status, MPI_BYTE, &recv_bytes);

            if(recv_bytes == 1) {
                uint8_t tmp;
                MPI_Recv(&tmp, 1, MPI_BYTE, sourcePartitionId, messageTag::UniteFather, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                count++;
                continue;
            }
            assert(recv_bytes % sizeof(e_size) == 0);
            int recvNum = recv_bytes / sizeof(e_size);

            MPI_Recv(recvBuffer, recv_bytes, MPI_BYTE, sourcePartitionId, 
                messageTag::UniteFather, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("p %d send %d B\n", sourcePartitionId, recv_bytes);

            v_size st = recvBuffer[0];
            for (int i = 1; i < recvNum; ++i) {
                myufs.unite(i-1+st, recvBuffer[i], true);
            }
        }
        delete [] recvBuffer;
    }
    else {
        int bufferSize = CHUNKSIZE;
        v_size * sendBuffer = new v_size[bufferSize];
        
        v_size fatherIndex = 0;

        while(fatherIndex < g->vCnt) {
            sendBuffer[0] = fatherIndex;
            if(fatherIndex + bufferSize - 1 < g->vCnt) {
                // memcpy(sendBuffer+1, myufs.father+fatherIndex, sizeof(VertexId)*(bufferSize-1));
                for(int i = 1; i < bufferSize; i++) {
                    sendBuffer[i] = myufs.father[fatherIndex + i - 1];
                }
                MPI_Send(sendBuffer, bufferSize*sizeof(v_size), MPI_BYTE, 0, 
                    messageTag::UniteFather, MPI_COMM_WORLD);
                fatherIndex += bufferSize - 1;
            }
            else {
                // memcpy(sendBuffer+1, myufs.father+fatherIndex, sizeof(VertexId)*(g->vCnt-fatherIndex));
                for(int i = 1; i <= g->vCnt-fatherIndex; i++) {
                    sendBuffer[i] = myufs.father[fatherIndex + i - 1];
                }
                MPI_Send(sendBuffer, (g->vCnt-fatherIndex+1)*sizeof(VertexId), 
                    MPI_BYTE, 0, messageTag::UniteFather, MPI_COMM_WORLD);
                fatherIndex = g->vCnt;
            }
        }
        uint8_t tmp = 0;
        MPI_Send(&tmp, 1, MPI_BYTE, 0, messageTag::UniteFather, MPI_COMM_WORLD);
        
        delete [] sendBuffer;
    }

    // std::ofstream outfile("./log/fa", std::ios::out);
    // if(!outfile.is_open()) {
    //     printf("create file error!!!!!!!!!!!!!!!\n");
    //     return;
    // }
    // for(v_size i = 0; i < g->vCnt; i++) {
    //     outfile << myufs.findFather(i, true) << '\n';
    // }
    // outfile.close();
}

void dSCAN::initClusterId() {
    mutex Mutex;
    int partitionId = g->mpiController->partitionId;

    clusterId = new v_size[g->vCnt];
    if(partitionId == 0) {
        #pragma omp parallel for
        for(v_size i = 0; i < g->vCnt; i++) {
            if(totalEps[i] >= mu) {
                clusterId[i] = myufs.findFather(i, true);
            }
            else clusterId[i] = g->vCnt;
        }
    }

    MPI_Bcast(clusterId, sizeof(v_size)*g->vCnt, MPI_BYTE, 0, MPI_COMM_WORLD);

    //nonecore
    auto f = [&](v_size u) {
        if(totalEps[u] < mu) return;

        v_size uDeg = g->pIdx[u + 1] - g->pIdx[u];
        auto endPu = g->pEdge + g->pIdx[u] - g->pIdxV0 + uDeg;

        for(v_size i = g->pIdx[u]; i < g->pIdx[u + 1]; i++) {
            int edgeOffset = i - g->pIdxV0;
            v_size v = g->pEdge[edgeOffset];
            if(totalEps[v] >= mu) continue;

            int vPartitionId = getPartitionId(v, v >= g->stV);

            v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];
            
            double bound = eps*eps*(uDeg+1)*(vDeg+1);
            if(vPartitionId == partitionId && sim[edgeOffset] == 0) {
assert(false);
                auto pv = g->pEdge + g->pIdx[v] - g->pIdxV0;
                auto pu = g->pEdge + g->pIdx[u] - g->pIdxV0;
                
                // computeSimAndEps(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
                simAndEpsSIMD(pu, pv, endPu, pv + vDeg, sim[edgeOffset], bound);
            }

            if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=bound) {
                if(partitionId==0 && clusterId[v] > clusterId[u]) {
                    write_min(&clusterId[v], clusterId[u]);
                }
                else {
                    Mutex.lock();
                    if(vertexMap.find(v) == vertexMap.end()) 
                        vertexMap[v] = clusterId[u];
                    else
                        vertexMap[v] = std::min(vertexMap[v], clusterId[u]);
                    Mutex.unlock();
                    // if(v == zz) printf("!!noneCore 2, %u\n", u);
                }
            }
        }
    };
    mapFtoV(f, g->partitionOffset);

    if(g->mpiController->partitions == 1) return;

    auto f2 = [&](v_size vId) {
        v_size v = reverseHash[vId];
        if(totalEps[v] < mu) return ;
        v_size vDeg = g->pIdx[v + 1] - g->pIdx[v];

        for(int j = 0; j < sourceVertexIndex[vId]; j++) {
            v_size u = sourceVertex[vId][j].first;
            int edgeOffset = sourceVertex[vId][j].second;
            int uDeg = g->pIdx[u + 1] - g->pIdx[u];

            if(totalEps[u] >= mu) continue;

            if((double)sim[edgeOffset]*sim[edgeOffset]+smallFloat>=eps*eps*(uDeg+1)*(vDeg+1)) {
                if(g->mpiController->partitionId== 0) 
                    write_min(&clusterId[u], clusterId[v]);
                else {
                    Mutex.lock();
                    if(vertexMap.find(u) == vertexMap.end()) 
                        vertexMap[u] = clusterId[v];
                    else
                        vertexMap[u] = std::min(vertexMap[u], clusterId[v]);
                    Mutex.unlock();
                }
            }
        }
    };
    mapFtoV(f2, oPartitionOffset);
}

void dSCAN::clusterNoneCore() {
    auto partitions = g->mpiController->partitions;
    auto partitonId = g->mpiController->partitionId;

    int nonCoreBufferSize = CHUNKSIZE;
    v_size *nonCoreBuffer = new v_size[nonCoreBufferSize];

    if(partitonId == 0) {
        MPI_Status recv_status;
        int count = 0;
        while(count < partitions - 1) {
            MPI_Probe(MPI_ANY_SOURCE, messageTag::UniteNonCore, MPI_COMM_WORLD, &recv_status);
            int sourcePartitionId = recv_status.MPI_SOURCE;
            int recv_bytes;
            MPI_Get_count(&recv_status, MPI_BYTE, &recv_bytes);
            if(recv_bytes == 1) {
                uint8_t tmp = 0;
                MPI_Recv(&tmp, recv_bytes, MPI_BYTE, sourcePartitionId, 
                    messageTag::UniteNonCore, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                count++;
                continue;
            }
            assert(recv_bytes % (sizeof(VertexId)*2)==0);
            int recvNum = recv_bytes / sizeof(VertexId);

            MPI_Recv(nonCoreBuffer, recv_bytes, MPI_BYTE, sourcePartitionId, 
            messageTag::UniteNonCore, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // tot += recv_bytes;
            // printf("%d %d %dv:\n", partition_id, sourcePartitionId, recvNum);
            #pragma omp parallel for
            for(int i = 0; i < recvNum; i+=2) {
                clusterId[nonCoreBuffer[i]] 
                    = std::min(clusterId[nonCoreBuffer[i]], clusterId[nonCoreBuffer[i+1]]);
            }
        }
    }
    else {
        int nonCoreIdx = 0;
        for(auto it = vertexMap.begin(); it != vertexMap.end(); ++it) {
            if(nonCoreIdx + 2 > nonCoreBufferSize) {
                MPI_Send(nonCoreBuffer, nonCoreIdx*sizeof(v_size), MPI_BYTE, 0, 
                    messageTag::UniteNonCore, MPI_COMM_WORLD);
                nonCoreIdx = 0;
            }
            nonCoreBuffer[nonCoreIdx++] = it->first;
            nonCoreBuffer[nonCoreIdx++] = it->second;
        }
        if(nonCoreIdx > 0) {
            MPI_Send(nonCoreBuffer, nonCoreIdx*sizeof(v_size), MPI_BYTE, 0, 
                messageTag::UniteNonCore, MPI_COMM_WORLD);
            nonCoreIdx = 0;
        }
        uint8_t tmp = 0;
        MPI_Send(&tmp, 1, MPI_BYTE, 0, messageTag::UniteNonCore, MPI_COMM_WORLD);
    }
}

void dSCAN::saveFile(string path) {
    std::ofstream outfile(path.c_str(), std::ios::out);
    if(!outfile.is_open()) {
        printf("create file error!!!!!!!!!!!!!!!\n");
        return;
    }
    int cnt = 0;
    for(v_size i = 0; i < g->vCnt; i++) {
        if(clusterId[i] == g->vCnt) outfile << 'n' << ' '<< i <<' '<< g->vCnt << '\n';
        else if(totalEps[i] >= mu) outfile << "c " << i << ' ' << clusterId[i] << ' '<<totalEps[i]<< '\n', cnt++;
        else outfile << "n " << i <<' '<< clusterId[i] << '\n';
    }
    outfile.close();

    printf("total c: %d\n", cnt);
}

#endif /*c++*/