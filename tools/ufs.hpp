#ifndef UFS_HPP
#define UFS_HPP

// #include <numa.h>
#include <fstream>
#include <atomic>
#include <string>
#include <vector>
#include <iostream>

#include <omp.h>
#include "type.hpp"
#include "atomic.hpp"

class Ufs
{
public:
    VertexId *father; //numa-oblivious
    // VertexId * clusterId;
    int vertices;

    Ufs(int vertices) {
        this->vertices = vertices;
        // this->father = (VertexId *)numa_alloc_interleaved( sizeof(VertexId) * vertices );
        this->father = new VertexId[vertices];
        #pragma omp parallel for
        for(int i = 0; i < vertices; i++) {
            this->father[i] = i;
        }
        // clusterId = (VertexId *)numa_alloc_interleaved( sizeof(VertexId) * vertices );
    }

    ~Ufs() {
        delete [] father;
        // numa_free(this->father, sizeof(VertexId) * this->vertices);
        // numa_free(clusterId, sizeof(VertexId) * vertices);
    }

    VertexId findFather(VertexId u, bool findTrueFather=false) {
        if(findTrueFather) {
            VertexId root = u, tmp;
            while(root != this->father[root]) root = this->father[root];
            while(u != root) {
                tmp = this->father[u];
                this->father[u] = root;
                u = tmp;
            }

            return root;
        }
        else {
            // auto isLocalVertex = [this](VertexId u) {
            //     return this->partition_offset[partition_id] <= u && u <= this->partition_offset[partition_id + 1];
            // };
            // VertexId root = u, tmp;
            // while(isLocalVertex(root) && root != this->father[root]) root = this->father[root];
            // while(u != root) {
            //     tmp = this->father[u];
            //     this->father[u] = root;
            //     u = tmp;
            // }

            // return root;
            ;
        }
    }
  
    void unite(VertexId u, VertexId v, bool findTrueFather=false) {
        VertexId ruLocal = findFather(u, findTrueFather), rvLocal = findFather(v, findTrueFather);
        if(ruLocal == rvLocal) return;
        if(ruLocal > rvLocal) this->father[ruLocal] = rvLocal;
        else this->father[rvLocal] = ruLocal; 
    }

    // void save(std::string path) {
    //     std::ofstream outfile(path.c_str(), std::ios::out);
    //     if(!outfile.is_open()) {
    //         printf("create file error!!!!!!!!!!!!!!!\n");
    //         return;
    //     }
    //     for(int i = 0; i < vertices; i++) {
    //         outfile << this->clusterId[i] << std::endl;
    //     }
    //     outfile.close();
    // }
};

// class parallelUfs
// {
// public:
//     std::atomic<VertexId> *father;
//      //numa-oblivious
//     // VertexId * clusterId;
//     int vertices;

//     parallelUfs(int vertices) {
//         this->vertices = vertices;
//         this->father = (std::atomic<VertexId> *)numa_alloc_interleaved( sizeof(std::atomic<VertexId>) * vertices );
//         #pragma omp parallel for
//         for(VertexId i = 0; i < vertices; i++) {
//             this->father[i] = i;
//         }
//     }

//     ~parallelUfs() {
//         numa_free(this->father, sizeof(std::atomic<VertexId>) * this->vertices);
//     }

//     VertexId findFather(VertexId u, bool findTrueFather=false) {
//         if(findTrueFather) {
//             VertexId root = u, tmp1, tmp2;
//             while(root != this->father[root]) root = this->father[root];
//             // int t = 0;
//             while(u != root) {
//                 // t++;
//                 // if(t > vertices) break;
//                 tmp2 = tmp1 = this->father[u];
//                 this->father[u].compare_exchange_strong(tmp1, root);
//                 u = tmp2;
//             }

//             return root;
//         }
//         else {
//             ;
//         }
//     }
  
//     void unite(VertexId u, VertexId v, bool findTrueFather=false) {
//         while(true) {
//             u = findFather(u, findTrueFather), v = findFather(v, findTrueFather);
//             if(u == v) return;
//             if(u > v) std::swap(u, v);
//             if(this->father[v].compare_exchange_strong(v, u)) {
//                 break;
//             }
//         }
//     }
// };

class parallelUfs2
{
public:
    VertexId *father = nullptr;
     //numa-oblivious
    // VertexId * clusterId;
    int vertices;

    parallelUfs2() {};

    void init(VertexId vertices) {
        if(this->father != nullptr) {
            delete [] father;
        }
        // this->father = (VertexId *)numa_alloc_interleaved( sizeof(VertexId) * vertices );
        this->father = new VertexId[vertices ];
        this->vertices = vertices;
        #pragma omp parallel for
        for(VertexId i = 0; i < vertices; i++) {
            this->father[i] = i;
        }
    }

    parallelUfs2(VertexId vertices) {
        this->vertices = vertices;
        this->father = new VertexId[vertices ];
        // this->father = (VertexId *)numa_alloc_interleaved( sizeof(VertexId) * vertices );
        #pragma omp parallel for
        for(VertexId i = 0; i < vertices; i++) {
            this->father[i] = i;
        }
    }

    ~parallelUfs2() {
        delete [] father;
        // numa_free(this->father, sizeof(VertexId) * this->vertices);
    }

    VertexId findFather(VertexId u, bool findTrueFather=false) {
        if(findTrueFather) {
            VertexId root = u, tmp1, tmp2;
            while(root != this->father[root]) root = this->father[root];
            
            while(u > root) {
                tmp2 = tmp1 = this->father[u];
                __sync_bool_compare_and_swap(&this->father[u], tmp1, root);
                // this->father[u].compare_exchange_weak(tmp1, root);
                u = tmp2;
            }

            return root;
        }
        else {
            ;
        }

        return 0;
    }
  
    void unite(VertexId u, VertexId v, bool findTrueFather=false) {
        while(true) {
            u = findFather(u, findTrueFather), v = findFather(v, findTrueFather);
            if(u == v) return;
            if(u > v) std::swap(u, v);
            if(__sync_bool_compare_and_swap(&this->father[v], v, u)) {
                break;
            }
        }
    }

    bool same(VertexId u, VertexId v) {
        for(;;) {
            u = findFather(u, true); v = findFather(v, true);
            if(u == v) return true;
            if(father[u] == u) return false;
        }
    }
};

class parallelUfs3
{
public:
    VertexId *father = nullptr;
    VertexId *rank = nullptr;
     //numa-oblivious
    // VertexId * clusterId;
    int vertices;

    parallelUfs3() {};

    void init(VertexId vertices) {
        if(this->father != nullptr) {
            delete [] father;
            delete [] rank;
            // numa_free(this->father, sizeof(VertexId) * this->vertices);
            // numa_free(this->rank, sizeof(VertexId) * this->vertices);
        }
        // this->father = (VertexId *)numa_alloc_interleaved( sizeof(VertexId) * vertices );
        // this->rank = (VertexId *)numa_alloc_interleaved( sizeof(VertexId) * vertices );
        father = new VertexId[vertices];
        rank = new VertexId[vertices];
        this->vertices = vertices;
        #pragma omp parallel for
        for(VertexId i = 0; i < vertices; i++) {
            this->father[i] = i;
        }
    }

    parallelUfs3(VertexId vertices) {
        this->init(vertices);
        // this->vertices = vertices;
        // this->father = (VertexId *)numa_alloc_interleaved( sizeof(VertexId) * vertices );
        // this->rank = (VertexId *)numa_alloc_interleaved( sizeof(VertexId) * vertices );
        // #pragma omp parallel for
        // for(VertexId i = 0; i < vertices; i++) {
        //     this->father[i] = i;
        // }
    }

    ~parallelUfs3() {
        delete [] father;
        delete [] rank;
        // numa_free(this->father, sizeof(VertexId) * this->vertices);
        // numa_free(this->rank, sizeof(VertexId) * this->vertices);
    }

    VertexId findFather(VertexId u, bool findTrueFather=false) {
        VertexId root = u, tmp1, tmp2;
        while(root != this->father[root]) root = this->father[root];
        
        while(u != root) {
            tmp2 = tmp1 = this->father[u];
            __sync_bool_compare_and_swap(&this->father[u], tmp1, root);
            // this->father[u].compare_exchange_weak(tmp1, root);
            u = tmp2;
        }

        return root;
    }
  
    void unite(VertexId u, VertexId v, bool findTrueFather=false) {
        VertexId rk1, rk2;
        while(true) {
            u = findFather(u, findTrueFather), v = findFather(v, findTrueFather);
            rk1 = this->rank[u];
            rk2 = this->rank[v];

            if(u == v) return;
            
            if(rk1 > rk2 || (rk1 == rk2 && u > v)) {
                std::swap(u, v);
                std::swap(rk1, rk2);
            }

            if(__sync_bool_compare_and_swap(&this->father[u], u, v)) {
                if(rk1 == rk2)  __sync_fetch_and_add(&this->rank[v], 1);
                break;
            }
            // else if(rk1 == rk2) {
            //     if(__sync_bool_compare_and_swap(&this->father[v], v, u)) {
            //         if(rank[u] == rank[v])  __sync_fetch_and_add(&this->rank[u], 1);
            //         break;
            //     }
            // }
        }
    }
};

class parallelUfs4
{
public:
    e_size *father = nullptr;
    int vertices;

    parallelUfs4() {};

    void init(v_size vertices) {
        if(this->father != nullptr) {
            delete [] father;
        }
        this->father = new e_size[vertices];
        this->vertices = vertices;
        #pragma omp parallel for
        for(e_size i = 0; i < vertices; i++) {
            this->father[i] = i;
        }
    }

    parallelUfs4(e_size vertices) {
        this->init(vertices);
    }

    ~parallelUfs4() {
        delete [] father;
    }

    v_size findFather(v_size u, bool findTrueFather=false) {
        auto nxt = [](e_size x)->v_size {
            return v_size(x & 0x00000000ffffffff);
        };
        // auto rnk = [](e_size x)->v_size {
        //     return v_size(x >> 32);
        // };

        v_size root = u;
        e_size tmp2, tmp1, tmp3;

        while(root != nxt(this->father[root])) root = nxt(this->father[root]);
        
        while(u < root) {
            tmp3 = tmp2 = this->father[u];
            tmp1 = (tmp2 & 0xffffffff00000000) | root;
            __sync_bool_compare_and_swap(&this->father[u], tmp2, tmp1);
            // this->father[u].compare_exchange_weak(tmp1, root);
            u = nxt(this->father[nxt(tmp3)]);
        }

        return root;
    }
  
    void unite(v_size u, v_size v, bool findTrueFather=false) {
        v_size rk1, rk2;

        auto nxt = [](e_size x) {
            return v_size(x & 0x00000000ffffffff);
        };
        auto rnk = [](e_size x) {
            return v_size(x >> 32);
        };
        auto cc = [](v_size h32, v_size l32)->e_size {
            return ((e_size)h32<<32)|l32;
        };

        auto updateRoot = [&](e_size x, e_size y)->bool {
            v_size xfa = nxt(x);
            e_size old = this->father[xfa];
            if(old != x) return false;
            return __sync_bool_compare_and_swap(&this->father[xfa], old, y);
        };

        while(true) {
            u = findFather(u), v = findFather(v);
            
            if(u == v) return;

            rk1 = rnk(this->father[u]);
            rk2 = rnk(this->father[v]);
            
            if(rk1 > rk2 || (rk1 == rk2 && u > v)) {
                std::swap(u, v);
                std::swap(rk1, rk2);
            }

            if( updateRoot(cc(rk1, u), cc(rk1, v) ) ) {
                if(rk1 == rk2)  {
                    updateRoot(cc(rk2, v), cc(rk2 + 1, v)); 
                }
                break;
            }
        }
    }
};



/**
 * Lock-free parallel disjoint set data structure (aka UNION-FIND)
 * with path compression and union by rank
 *
 * Supports concurrent find(), same() and unite() calls as described
 * in the paper
 *
 * "Wait-free Parallel Algorithms for the Union-Find Problem"
 * by Richard J. Anderson and Heather Woll
 *
 * In addition, this class supports optimistic locking (try_lock/unlock)
 * of disjoint sets and a combined unite+unlock operation.
 *
 * \author Wenzel Jakob
 */
struct parallelUfs5 {
public:
    parallelUfs5() {}

    parallelUfs5(uint32_t size) {
        init(size);
    }

    void init(uint32_t size) {
        this->father = new e_size[vertices];
        this->vertices = vertices;
        #pragma omp parallel for
        for(e_size i = 0; i < vertices; i++) {
            this->father[i] = i;
        }
        mData = father; 
    }

    uint32_t findFather(uint32_t id, bool f=false) const {
        return find(id);
    }

    void unite(v_size u, v_size v, bool findTrueFather=false) {
        unite2(u, v);
    }

    uint32_t find(uint32_t id) const {
        while (id != parent(id)) {
            uint64_t value = mData[id];
            uint32_t new_parent = parent((uint32_t) value);
            uint64_t new_value =
                (value & 0xFFFFFFFF00000000ULL) | new_parent;
            /* Try to update parent (may fail, that's ok) */
            if (value != new_value)
                // mData[id].compare_exchange_weak(value, new_value);
                __sync_bool_compare_and_swap(&this->mData[id], value, new_value);
            id = new_parent;
        }
        return id;
    }

    bool same(uint32_t id1, uint32_t id2) const {
        for (;;) {
            id1 = find(id1);
            id2 = find(id2);
            if (id1 == id2)
                return true;
            if (parent(id1) == id1)
                return false;
        }
    }

    uint32_t unite2(uint32_t id1, uint32_t id2) {
        for (;;) {
            id1 = find(id1);
            id2 = find(id2);

            if (id1 == id2)
                return id1;

            uint32_t r1 = rank(id1), r2 = rank(id2);

            if (r1 > r2 || (r1 == r2 && id1 < id2)) {
                std::swap(r1, r2);
                std::swap(id1, id2);
            }

            uint64_t oldEntry = ((uint64_t) r1 << 32) | id1;
            uint64_t newEntry = ((uint64_t) r1 << 32) | id2;

            // if (!mData[id1].compare_exchange_strong(oldEntry, newEntry))
            //     continue;
            if(__sync_bool_compare_and_swap(&mData[id1], oldEntry, newEntry))
                continue;

            if (r1 == r2) {
                oldEntry = ((uint64_t) r2 << 32) | id2;
                newEntry = ((uint64_t) (r2+1) << 32) | id2;
                /* Try to update the rank (may fail, that's ok) */
                // mData[id2].compare_exchange_weak(oldEntry, newEntry);
                __sync_bool_compare_and_swap(&mData[id2], oldEntry, newEntry);
            }

            break;
        }
        return id2;
    }

    uint32_t rank(uint32_t id) const {
        return ((uint32_t) (mData[id] >> 32)) & 0x7FFFFFFFu;
    }

    uint32_t parent(uint32_t id) const {
        return (uint32_t) mData[id];
    }

    e_size *father = nullptr, *mData = nullptr;
    v_size vertices;
    // mutable std::vector<std::atomic<uint64_t>> mData;
};


#endif