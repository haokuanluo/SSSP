// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>

#include "omp.h"
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

pvector<WeightT> DeltaStep(const WGraph &g, NodeID source, WeightT delta) {
    Timer t;
    pvector<WeightT> dist(g.num_nodes(), kDistInf);
    dist[source] = 0;
    //pvector<NodeID> frontier(g.num_edges_directed());
    pvector<bool> inq(g.num_nodes(),false);
    // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
    //size_t shared_indexes[2] = {0, kMaxBin};
    //size_t frontier_tails[2] = {1, 0};
    //frontier[0] = source;
    int dealing = 0;

    typedef pair<WeightT, NodeID> WN;
    priority_queue<WN, vector<WN>, greater<WN>> mq;
    mq.push(make_pair(0, source));

    //queue <NodeID> q;
    t.Start();
    //q.push(source);
    //inq[source] = 1;
    omp_lock_t writelock;
    omp_init_lock(&writelock);

#pragma omp parallel
    {

        while(1) {
            bool cont = 0;
            if (dealing == 0 && mq.empty())break; // requires dealing++ before q.pop
            //cout<<dealing<<endl;

            NodeID u=0;
            WeightT td = 0;
            omp_set_lock(&writelock);
            {

                if (!mq.empty()) {
                    dealing++;
                    td = mq.top().first;
                    u = mq.top().second;
                    mq.pop();

                    //u=q.front();
                    //q.pop();
                    //inq[u]=0; // make sure to modify dist before modify inq in the later section
                    cont=1;
                } else cont = 0;

            };
            omp_unset_lock(&writelock);
            if (!cont) {
                continue;
            }

            if(td == dist[u])for (WNode wn : g.out_neigh(u)) {
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

                            omp_set_lock(&writelock);
                            mq.push(make_pair(new_dist, wn.v));

                            omp_unset_lock(&writelock);

                    }
                }
            }
            omp_set_lock(&writelock);
            dealing--;
            omp_unset_lock(&writelock);

        }
    };
    omp_destroy_lock(&writelock);


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
