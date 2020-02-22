#pragma once
#include <vector>

class Node
{
public:
    Node() = default;
//    int get_id() const;

//    int node_id = 0;
//    double shortest_path;
    std::vector<int> precursor_idx;
    std::vector<int> precursor_edges_idx;
    std::vector<double> precursor_edges_weights;

    std::vector<int> successor_idx;
    std::vector<int> successor_edges_idx;
    std::vector<double> successor_edges_weights;

    double price = 0;

//    bool visited = false;
    //bool in_tree = false;
//    Node *parent_node = nullptr; //parent node in shortest path tree

    void add_precursor(int pre_id, int pre_edge_id, double weight);
    void add_successor(int succ_id, int succ_edge_id, double weight);
//    void delete_precursor(int pre_id);
//    void delete_successor(int succ_id);
};
