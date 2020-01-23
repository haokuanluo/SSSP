// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"


/*
GAP Benchmark Suite
Kernel: Single-source Shortest Paths (SSSP)
Author: Scott Beamer
Returns array of distances for all vertices from given source vertex
This SSSP implementation makes use of the ∆-stepping algorithm [1]. The type
used for weights and distances (WeightT) is typedefined in benchmark.h. The
delta parameter (-d) should be set for each input graph.
The bins of width delta are actually all thread-local and of type std::vector
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies their selected
thread-local bin into the shared bin.
Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and it now appears in a lower bin. We find ignoring vertices if
their current distance is less than the min distance for the bin to remove
enough redundant work that this is faster than removing the vertex from older
bins.
[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
    algorithm." Journal of Algorithms, 49(1):114–152, 2003.
*/


using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
const size_t kMaxBin = numeric_limits<size_t>::max()/2;
const int numThreads = 256;

pvector<WeightT> DeltaStep(const WGraph &g, NodeID source, WeightT delta) {
    Timer t;

    cout<<delta<<endl;
    long long weight = 0;
    pvector<WeightT> dist(g.num_nodes(), kDistInf);
    pvector<WeightT> extendedDist(g.num_nodes(), kDistInf);
    pvector<bool> inq(g.num_nodes(),false);
    dist[source] = 0;
    pvector<NodeID> frontier(g.num_edges_directed());
    // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
    size_t shared_indexes[2] = {0, kMaxBin};
    size_t frontier_tails[2] = {1, 0};
    frontier[0] = source;
    t.Start();
#pragma omp parallel
    {
        vector<NodeID>  local_bin;
        long long local_weight = 0;
        size_t iter = 0;
        int dd = 0;
        int tt=0;
        while (frontier_tails[iter&1] != 0) {
            local_weight = 0;

            size_t &curr_frontier_tail = frontier_tails[iter&1];
            size_t &next_frontier_tail = frontier_tails[(iter+1)&1];

#pragma omp for nowait schedule(dynamic, 64)
            for (size_t i=0; i < curr_frontier_tail; i++) {
                NodeID u = frontier[i];
                if (curr_frontier_tail>100 && dist[u]>weight) {
                    local_bin.push_back(u);
                    dd++;
                    continue;
                }
                tt++;
                //inq[u]=0;
                //WeightT old_ext_dist = extendedDist[u];
                //WeightT new_ext_dist = dist[u];
                if (dist[u] < extendedDist[u]) {
                    extendedDist[u]=dist[u];
                    for (WNode wn : g.out_neigh(u)) {
                        WeightT old_dist = dist[wn.v];
                        WeightT new_dist = dist[u] + wn.w;
                        if (new_dist < old_dist) {
                            bool changed_dist = true;
                            while (!compare_and_swap(dist[wn.v], old_dist, new_dist)) {
                                old_dist = dist[wn.v];
                                if (old_dist <= new_dist) {
                                    changed_dist = false;
                                    break;
                                }
                            }
                            if (changed_dist) {
                                local_bin.push_back(wn.v);
                            }
                        }
                    }
                }
            }
            //fetch_and_add(weight,dist[local_bin[local_bin.size()/2]]);


#pragma omp barrier
#pragma omp single
            weight = 0;

#pragma omp single nowait
            curr_frontier_tail = 0;



            size_t copy_start = fetch_and_add(next_frontier_tail,
                                              local_bin.size());

            copy(local_bin.begin(),
                 local_bin.end(), frontier.data() + copy_start);
            for(size_t i=0;i<local_bin.size();i++) local_weight += dist[local_bin[i]];
            local_bin.resize(0);
            fetch_and_add(weight,local_weight);

            iter++;
#pragma omp barrier
#pragma omp single
            if(copy_start > 0)weight = weight / next_frontier_tail;
#pragma omp single
            cout<<next_frontier_tail<<endl;

        }
        //#pragma omp single
        //cout << "took " << iter << " iterations" << endl;
//#pragma omp critical
//        cout<<(double) dd/(double)tt<<endl;
    }
    return dist;
}


void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
    auto NotInf = [](WeightT d) { return d != kDistInf; };
    int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
    cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
}


// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
    // Serial Dijkstra implementation to get oracle distances
    pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
    oracle_dist[source] = 0;
    typedef pair<WeightT, NodeID> WN;
    priority_queue<WN, vector<WN>, greater<WN>> mq;
    mq.push(make_pair(0, source));
    while (!mq.empty()) {
        WeightT td = mq.top().first;
        NodeID u = mq.top().second;
        mq.pop();
        if (td == oracle_dist[u]) {
            for (WNode wn : g.out_neigh(u)) {
                if (td + wn.w < oracle_dist[wn.v]) {
                    oracle_dist[wn.v] = td + wn.w;
                    mq.push(make_pair(td + wn.w, wn.v));
                }
            }
        }
    }
    // Report any mismatches
    bool all_ok = true;
    for (NodeID n : g.vertices()) {
        if (dist_to_test[n] != oracle_dist[n]) {
            cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
            all_ok = false;
        }
    }
    return all_ok;
}


int main(int argc, char* argv[]) {
    CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
    if (!cli.ParseArgs())
        return -1;
    WeightedBuilder b(cli);
    WGraph g = b.MakeGraph();
    SourcePicker<WGraph> sp(g, cli.start_vertex());
    auto SSSPBound = [&sp, &cli] (const WGraph &g) {
        return DeltaStep(g, sp.PickNext(), cli.delta());
    };
    SourcePicker<WGraph> vsp(g, cli.start_vertex());
    auto VerifierBound = [&vsp] (const WGraph &g, const pvector<WeightT> &dist) {
        return SSSPVerifier(g, vsp.PickNext(), dist);
    };
    BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);
    return 0;
}