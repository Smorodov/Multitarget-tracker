#include "Sink.h"


Sink::Sink(int n, double sink_cost){
    sink_cost_ = sink_cost;

    sink_precursor_weights.assign(n, FINF);
    //sink_precursor_weights[0] = FINF; // src has 0 weights
//    used_sink_precursor.assign(n, false);
}
/**0 is src and n is sink, 1,3,5... is former node, 2,4,6...is latter node**/
//we update latter nodes cause they link to sink
void Sink::sink_update_all(std::vector<Node> &V, std::vector<double> &distance2src, int sink_id, int n){
    double sink_dist;
    for(int i=2; i<n; i+=2){
        sink_dist = V[i].successor_edges_weights[0] - distance2src[sink_id];
        sink_precursor_weights[i] = sink_dist + distance2src[i];
        if(sink_precursor_weights[i] + distance2src[sink_id] <0)
            sink_precursors.insert(std::make_pair(sink_precursor_weights[i], i));
    }
}

void Sink::sink_update_all_weight(std::vector<Node> &V, std::vector<double> &distance2src, int sink_id, int n){
    double sink_dist;
    for(int i=2; i<n; i+=2){
        sink_dist = V[i].successor_edges_weights[0] - distance2src[sink_id];
        sink_precursor_weights[i] = sink_dist + distance2src[i];
    }
}

void Sink::sink_build_precursormap(std::vector<double> &ancestor_ssd, std::vector<int> &ancestor_node_id, std::vector<int> &parent_node_id, int n){
//    bool* useful_precursor = new bool[n];
//    for(int i=1; i<n; i+=2){
//        useful_precursor[parent_node_id[i]] = true;
//    }
//
//    for(int i=2; i<n; i+=2){
//        if(!useful_precursor[i]) {
//            sink_precursors.insert(std::make_pair(sink_precursor_weights[i], i));
//            //sink_precursors.insert(std::make_pair(sink_precursor_weights[parent_node_id[parent_node_id[i]]], parent_node_id[parent_node_id[i]]));
//        }
//    }

    for(int i=2; i<n; i+=2){
        if(sink_precursor_weights[i] <= ancestor_ssd[ancestor_node_id[i]])
            sink_precursors.insert(std::make_pair(sink_precursor_weights[i], i));
    }
    sink_precursors.insert(sink_precursors.end(), std::make_pair(FINF, 0)); // worst case
}
void Sink::sink_update_all_half(std::vector<double> distance2src, int sink_id, int n){
    double sink_dist = distance2src[sink_id];
    double cur_dist;
    for(int i=1; i<n-1; i++){
        cur_dist = sink_cost_ + distance2src[i] - sink_dist;
        sink_precursors.insert(std::make_pair(cur_dist, i));
        sink_precursor_weights[i] = cur_dist;
    }

}
// this is re-write in Graph class, so not used any more here
void Sink::sink_update_subgraph(std::vector<int> update_node_id, std::vector<double> distance2src, int n, int sink_id){
    sink_weight_shift = sink_weight_shift - distance2src[sink_id];
    std::vector<int>::iterator it;
    for(it = update_node_id.begin(); it != update_node_id.end(); it++){
        int i = *it;
        if ((i)%2 == 0) {
            double cur_dist = sink_precursor_weights[i] + distance2src[i];
            sink_precursors.insert(std::make_pair(cur_dist, i));
        }
    }
}
