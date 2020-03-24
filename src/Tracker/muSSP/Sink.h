#pragma once

#include <vector>
#include "Node.h"
#include <algorithm>       /* fabs */
//#include<bits/stdc++.h>
#include <map>
#include <limits>
//#include <iostream>

#define FINF 1000000.0 //numeric_limits<double>::max()
#define FINFHALF FINF/2.0

///
/// \brief The Sink class
///
class Sink
{
public:
    // use set to save sink's precursors' distances
    std::multimap<double, int> sink_precursors;
    std::vector<double> sink_precursor_weights;
    double sink_cost_ = 0; // this can be a vector, in our framework, it it a scaler
    double sink_weight_shift = 0;

    Sink() = default;
    Sink(int n, double sink_cost);

    void sink_update_all(std::vector<Node> &V, std::vector<double> &distance2src, int sink_id, int n);

    void sink_update_all_weight(std::vector<Node> &V, std::vector<double> &distance2src, int sink_id, int n);


    void sink_build_precursormap(std::vector<double> &ancestor_ssd, std::vector<int> &ancestor_node_id, std::vector<int> &parent_node_id, int n);


    void sink_update_all_half(std::vector<double> distance2src, int sink_id, int n);
    void sink_update_subgraph(std::vector<int> update_node_id, std::vector<double> distance2src, int sink_id, int n);
};
