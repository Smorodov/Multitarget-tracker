#include "Node.h"
//#include <algorithm>

//int Node::get_id() const {
//    return node_id;
//}

void Node::add_precursor(int pre_id, int pre_edge_id, double weight) {
    this->precursor_idx.push_back(pre_id);
    this->precursor_edges_idx.push_back(pre_edge_id);
    this->precursor_edges_weights.push_back(weight);
}

void Node::add_successor(int succ_id, int succ_edge_id, double weight) {
    this->successor_idx.push_back(succ_id);
    this->successor_edges_idx.push_back(succ_edge_id);
    this->successor_edges_weights.push_back(weight);
}

//void Node::delete_precursor(int pre_id) {
//
//    auto position = find(this->precursor_idx.begin(), this->precursor_idx.end(), pre_id);
//    long idx = position - this->precursor_idx.begin();
//    this->precursor_idx.erase(position);
//    this->precursor_edges_idx.erase(this->precursor_edges_idx.begin() + idx);
//    this->precursor_edges_weights.erase(this->precursor_edges_weights.begin() + idx);
//}
//
//void Node::delete_successor(int pre_id) {
//
//    auto position = find(this->successor_idx.begin(), this->successor_idx.end(), pre_id);
//    long idx = position - this->successor_idx.begin();
//    this->successor_idx.erase(position);
//    this->successor_edges_idx.erase(this->successor_edges_idx.begin() + idx);
//    this->successor_edges_weights.erase(this->successor_edges_weights.begin() + idx);
//}
