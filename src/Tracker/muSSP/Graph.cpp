
#include "Graph.h"
/****shared macros****/
#define MAX(x, y) ( ( (x) > (y) ) ?  x : y )
#define MIN(x, y) ( ( (x) < (y) ) ? x : y )
#define REDUCED_EDGE_WEIGHTS(i, j, e) {\
    edge_weights[e] += distance2src[i];\
    edge_weights[e] -= distance2src[j];\
}\



Graph::Graph(int num_nodes, int num_edges, int src_id, int sink_id, double en_weight, double ex_weight) {

    num_nodes_ = num_nodes;
    num_edges_ = num_edges;
    src_id_ = src_id;
    sink_id_ = sink_id;
    en_weight_ = en_weight;
    ex_weight_ = ex_weight;
    precursor_queue_top_val = FINF;

    V_ = std::vector<Node>(num_nodes);
    for (int i = 0; i < num_nodes; i++) {// indeed this is not needed
        V_[i].price = 0;
    }
    parent_node_id.assign(num_nodes, 0);
    ancestor_node_id.assign(num_nodes, 0);
    distance2src.assign(num_nodes, FINF);
    sink_info = std::make_unique<Sink>(num_nodes, ex_weight);

    node_visited.assign(num_nodes, false);
    // this is used after building sst, so most nodes are visited already
    /**** 0: visited,
     * 1: not visited but in waitinglist,
     * 2: not visited but not in waitinglist,
     * 3: not visited and freshly labelled
     * 4: not visited and possibly will never be used
     * -1: not visitetd and permanently never be used
    *****/
    node_in_visited.assign(num_edges, 0);
    edge_visited.assign(num_edges, false);

    // data save ancestor information
    ancestor_ssd.assign(num_nodes, FINF);
    ancestors_descendants.resize(num_nodes);

    time_test.resize(100, 0);
}

Node &Graph::get_node(int node_id) {
    return V_[node_id];
}

void Graph::add_edge(int tail_id, int head_id, int edge_id, double weight) {

    V_[tail_id].add_successor(head_id, edge_id, weight);
    V_[head_id].add_precursor(tail_id, edge_id, weight);

    if (false) {
        /******
         * for results validation only, no need to use in real application
         * ********/
        edge_weights.push_back(weight);
        //// there will be no collisions for insertion, so complexity is O(1)
        node_id2edge_id.insert({node_key(head_id, tail_id), edge_id});
        node_id2edge_id.insert({node_key(tail_id, head_id), edge_id});

        //// for results validation
        edge_tail_head.emplace_back(std::make_pair(tail_id, head_id));
        edge_org_weights.push_back(weight);

        if (static_cast<int>(edge_weights.size()) - 1 != edge_id)
            std::cout << "we got wrong edge number" << std::endl;
    }
}


/***********
 * remove the edges that are invalid:
 *
 * edge-weigth > in_weight + out_weigth
 *
 *  * *************/
void Graph::invalid_edge_rm(){
    double sink_cost, src_cost;
    int rm_cnt = 0;
    for (int i = 2; i < num_nodes_-1; i+=2){
        sink_cost = V_[i].successor_edges_weights[0];
        for (size_t j = 1; j < V_[i].successor_idx.size(); j++)//(int i = 0; i < this->V_[v].successor_idx.size(); ++i)
        {
            if (V_[i].successor_edges_weights[j] > sink_cost + V_[V_[i].successor_idx[j]].precursor_edges_weights[0]){
                V_[i].successor_edges_weights[j] = FINF;
                rm_cnt++;
            }
        }
    }

    for (int i = 1; i < num_nodes_-1; i+=2){
        src_cost = V_[i].precursor_edges_weights[0];
        for (size_t j = 1; j < V_[i].precursor_idx.size(); j++)//(int i = 0; i < this->V_[v].successor_idx.size(); ++i)
        {
            if (V_[i].precursor_edges_weights[j] > src_cost + V_[V_[i].precursor_idx[j]].successor_edges_weights[0]){
                V_[i].precursor_edges_weights[j] = FINF;
                rm_cnt++;
            }
        }
    }
    std::cout << "# of dummy edges : " << rm_cnt << std::endl;
}
/*******
 * A recursive function used by shortestPath. See below link for details
 * https://www.geeksforgeeks.org/topological-sorting/
 */
void Graph::topologicalSortUtil(int v, std::vector<bool>& visited, std::stack<int> &Stack) {
    // Mark the current node as visited
    visited[v] = true;

    // Recur for all the vertices adjacent to this vertex
    for (auto node_id : V_[v].successor_idx)//(int i = 0; i < this->V_[v].successor_idx.size(); ++i)
    {
        if (!visited[node_id])
            topologicalSortUtil(node_id, visited, Stack);
    }
    // Push current vertex to stack which stores topological sort
    Stack.push(v);
}

/*************************
 * Build the shortest path tree from a DAG. It will only be called once
 * this shortest path tree will be saved in this->parent_node_id. their distance
 * will be saved in distance2src. their ancestors will be saved in ancestor_node_id.
 * The function to find shortest paths from given vertex. It uses recursive
 * topologicalSortUtil() to get topological sorting of given graph.
 *
 * half_flag == true means: we only need to save former node (ancestor is also former nodes)
*************************/
void Graph::shortest_path_dag() {
    std::stack<int> Stack;
    // Mark all the vertices as not visited
    std::vector<bool> visited(num_nodes_, false);

    //// Call the recursive helper function to store Topological Sort starting from all vertices one by one
    for (int i = 0; i < this->num_nodes_; i++)
        if (!visited[i])
            topologicalSortUtil(i, visited, Stack);

    // Initialize distances to all vertices as infinite and distance to source as 0
    for (int i = 1; i<num_nodes_; i+=2)
        distance2src[i] = V_[i].precursor_edges_weights[0];
    for (int i = 2; i<num_nodes_; i+=2)
        distance2src[i] = FINF;

    distance2src[src_id_] = 0;
    ancestor_node_id[src_id_] = 0;
    parent_node_id[src_id_] = 0;
    // Process vertices in topological order
    Stack.pop(); // pop up the source, because every node has been set as en_weight_
    while (!Stack.empty()) {
        // Get the next vertex from topological order
        int cur_node_id = Stack.top();
        Stack.pop();

        ancestor_node_id[cur_node_id] = src_id_ == parent_node_id[cur_node_id] ?
                                        cur_node_id : ancestor_node_id[parent_node_id[cur_node_id]];

        if (cur_node_id != sink_id_)
            ancestors_descendants[ancestor_node_id[cur_node_id]].push_back(cur_node_id);

        // Update distances of all adjacent vertices
        for (size_t i = 0; i < V_[cur_node_id].successor_idx.size(); ++i) {
            int cur_succ_id = V_[cur_node_id].successor_idx[i];
            double cur_distance = distance2src[cur_node_id] + V_[cur_node_id].successor_edges_weights[i];
            if (distance2src[cur_succ_id] - cur_distance > 0.0000001)
            {
                distance2src[cur_succ_id] = cur_distance;
                parent_node_id[cur_succ_id] = cur_node_id;
            }
        }
    }

    //// update price
    for (int i = 0; i < this->num_nodes_; i++){
        V_[i].price -= distance2src[i];
    }
}

// update edge weights of all graph
// can only be used when set precursors == succesors
void Graph::update_allgraph_weights() {

    //// here we can get all precursors for sink node
    sink_info->sink_update_all_weight(V_, distance2src, sink_id_, num_nodes_);
    sink_info->sink_weight_shift = 0;

    //// find shortest path each sub-tree
    for (int i = 2; i < num_nodes_; i += 2) {
        ancestor_ssd[ancestor_node_id[i]] = MIN(ancestor_ssd[ancestor_node_id[i]],
                                                sink_info->sink_precursor_weights[i]);
    }

    sink_info->sink_build_precursormap(ancestor_ssd, ancestor_node_id, parent_node_id, num_nodes_);

    auto curr_best_choice = sink_info->sink_precursors.begin();
    int min4t_node_id = curr_best_choice->second;

    parent_node_id[sink_id_] = min4t_node_id;
    ancestor_node_id[sink_id_] = ancestor_node_id[min4t_node_id];

    /*****
     * output infor of initial shortest path tree to file
    *
    * ****/
    //// output intial shortest path tree to file
//    if (false) {
//        FILE *fp;
//        fp = fopen("sst_initial.txt", "w");
//
//        int num_sub_tree = 0;
//        for (int i = 1; i < num_nodes_ - 1; i += 2) {
//            if (ancestor_node_id[i] == i) {
//                fprintf(fp, "%lf %ld\n", ancestor_ssd[i], ancestors_descendants[i].size());
//                num_sub_tree++;
//            }
//        }
//        fclose(fp);
//
//        fp = fopen("sst_distances.txt", "w");
//        for (int i = 0; i < num_nodes_; i++) {
//            fprintf(fp, "%d, %d, %lf\n", parent_node_id[i], ancestor_node_id[i], distance2src[i]);
//        }
//        fclose(fp);
//
//        //// output the largest sub-tree
//        num_sub_tree = 0;
//        long largest_subtree = 0, t_sub_tree;
//        for (int i = 1; i < num_nodes_ - 1; i += 2) {
//            if (ancestor_node_id[i] == i) {
//                if (ancestors_descendants[i].size() > largest_subtree) {
//                    largest_subtree = ancestors_descendants[i].size();
//                }
//                num_sub_tree++;
//            }
//        }
//        t_sub_tree = ancestors_descendants[ancestor_node_id[num_nodes_ - 1]].size();
//        cout << "Size of largest sub-tree and sub-tree containing t: " << largest_subtree << " " << t_sub_tree << std::endl;
//    }
    ////after updating all shortest distance are 0
    memset(&distance2src[0], 0, distance2src.size() * sizeof(distance2src[0]));
}

void Graph::extract_shortest_path() {
    shortest_path.clear();
    int tmp_node_id = sink_id_;
    while (tmp_node_id != src_id_) {
        shortest_path.push_back(tmp_node_id);
        tmp_node_id = parent_node_id[tmp_node_id];
    }
    shortest_path.push_back(src_id_);
}
/******
 * flip the shortest path and update related nodes' precuror and successors
 *
 * *******/
void Graph::flip_path() { // erase the best one link to sink
    /** for 2 and end-1, specially handled ***/
        // node path(2)
        int node_tmp = shortest_path[shortest_path.size() - 2];// the path currently is from sink to src
        std::vector<int>::iterator edge_id_it;
        std::vector<double>::iterator edge_weight_it;
        double tmp_edge_weight;
        auto it = find(V_[node_tmp].successor_idx.begin(), V_[node_tmp].successor_idx.end(),
                       shortest_path[shortest_path.size() - 3]);
        //// erase
        auto pos = it - V_[node_tmp].successor_idx.begin();
        //V_[node_tmp].successor_edges_idx.erase(V_[node_tmp].successor_edges_idx.begin() + pos);
        tmp_edge_weight = *(V_[node_tmp].successor_edges_weights.begin() + pos);
        V_[node_tmp].successor_edges_weights.erase(V_[node_tmp].successor_edges_weights.begin() + pos);
        V_[node_tmp].successor_idx.erase(it);

        //// add
        it = find(V_[node_tmp].precursor_idx.begin(), V_[node_tmp].precursor_idx.end(), src_id_);
        *it = shortest_path[shortest_path.size() - 3];
        parent_node_id[node_tmp] = *it;
        pos = it - V_[node_tmp].precursor_idx.begin();
        //edge_id_it = V_[node_tmp].precursor_edges_idx.begin() + pos;
        //*edge_id_it = node_id2edge_id[node_key(node_tmp, *it)];
        edge_weight_it = V_[node_tmp].precursor_edges_weights.begin() + pos;
        *edge_weight_it = -tmp_edge_weight;

        //// node path(end-1)
        node_tmp = shortest_path[1];
        it = find(V_[node_tmp].precursor_idx.begin(), V_[node_tmp].precursor_idx.end(), shortest_path[2]);
        pos = it - V_[node_tmp].precursor_idx.begin();
        //// erase
        tmp_edge_weight = *(V_[node_tmp].precursor_edges_weights.begin() + pos);
        V_[node_tmp].precursor_edges_weights.erase(V_[node_tmp].precursor_edges_weights.begin() + pos);

        //V_[node_tmp].precursor_edges_idx.erase(V_[node_tmp].precursor_edges_idx.begin() + pos);
        V_[node_tmp].precursor_idx.erase(it);
        parent_node_id[node_tmp] = -1;
        //// add
        it = find(V_[node_tmp].successor_idx.begin(), V_[node_tmp].successor_idx.end(), sink_id_);
        *it = shortest_path[2];
        pos = it - V_[node_tmp].successor_idx.begin();
        //edge_id_it = V_[node_tmp].successor_edges_idx.begin() + pos;
        //*edge_id_it = node_id2edge_id[node_key(*it, node_tmp)];
        edge_weight_it = V_[node_tmp].successor_edges_weights.begin() + pos;
        *edge_weight_it = -tmp_edge_weight;

        // from 3 to end - 2, reverse their precursor
        for (unsigned long i = shortest_path.size() - 3; i >= 2; i--) {
            node_tmp = shortest_path[i];
            it = find(V_[node_tmp].precursor_idx.begin(), V_[node_tmp].precursor_idx.end(), shortest_path[i + 1]);
            *it = shortest_path[i - 1];
            parent_node_id[node_tmp] = *it;
            pos = it - V_[node_tmp].precursor_idx.begin();

            edge_weight_it = V_[node_tmp].precursor_edges_weights.begin() + pos;
            tmp_edge_weight = *edge_weight_it;

            it = find(V_[node_tmp].successor_idx.begin(), V_[node_tmp].successor_idx.end(), shortest_path[i - 1]);
            *it = shortest_path[i + 1];
            pos = it - V_[node_tmp].successor_idx.begin();

            *edge_weight_it = -*(V_[node_tmp].successor_edges_weights.begin() + pos);
            *(V_[node_tmp].successor_edges_weights.begin() + pos) = -tmp_edge_weight;

        }

    //// after flipping, there is a node that no longer can access sink from itself
    sink_info->sink_precursor_weights[sink_info->sink_precursors.begin()->second] = FINF;
    sink_info->sink_precursors.erase(sink_info->sink_precursors.begin());
}

/*****************************************
 * if we use all nodes in the subgraph
 *
 * **************************************/
void Graph::find_node_set4update(std::vector<int> &update_node_id) {
    update_node_id = ancestors_descendants[shortest_path[shortest_path.size() - 2]];
}

void Graph::topologicalSort_counter_order(int v) {
    if (v == -1)
        return;
    //// Mark the current node as visited
    node_in_visited[v] = 3;
    //// Recur for all the vertices adjacent to this vertex
    if (!node_in_visited[parent_node_id[v]]) {
        topologicalSort_counter_order(parent_node_id[v]);
    }
    //// Push current vertex to stack which stores topological sort
    tplog_vec.push_back(v);
}
/*************
 * sub-routine of update_shortest_path_tree_recursive
 * ****************/
void Graph::recursive_update_successors_distance(int curr_node_id, double curr_dist, int curr_ancestor,
                                                 std::vector<int> &update_node_id4edges)
{
    for (size_t j = 0; j < V_[curr_node_id].successor_idx.size(); j++) {
        int it = V_[curr_node_id].successor_idx[j];
        if (node_in_visited[it] > 0) {
            double cur_edge_weight = V_[curr_node_id].successor_edges_weights[j] - V_[curr_node_id].price + V_[it].price;
            if (abs(cur_edge_weight) < 0.000001) { //// in the shortest path tree, permanently labeled
                node_in_visited[it] = 0;
                parent_node_id[it] = curr_node_id;
                distance2src[it] = curr_dist;
                ancestor_node_id[it] = curr_ancestor;
                update_node_id4edges.push_back(it);
                recursive_update_successors_distance(it, curr_dist, curr_ancestor, update_node_id4edges);
            } else {
                ////edge_upt_waitinglist.push(ie);
                if (cur_edge_weight + curr_dist < distance2src[it]) {
                    ////If this node is not visited and the current parent node distance+distance from there to this node is shorted than the initial distace set to this node, update it
                    parent_node_id[it] = curr_node_id;
                    distance2src[it] = cur_edge_weight + curr_dist;
                    //// no need to update ancestor in this condition
                    //// Set the new distance and add to map
                    node_upt_waitinglist.insert(std::make_pair(cur_edge_weight + curr_dist, it));
                    node_in_visited[it] = 1;
                }
            }
        }
    }
}
/*************
 * main function for shortest path tree updating and shortest path searching
 *
 * use dijkstra algorithm to update the shorttest path tree
 * ****************/
void Graph::update_shortest_path_tree_recursive(std::vector<int> &update_node_id) {
    std::vector<int> update_node_id4edges;
    //// order the nodes as topological order from end of shortest_path ==> start of shortest_path
    for (auto &&i : update_node_id) {
        precursor_queue_top_val = MIN(precursor_queue_top_val, sink_info->sink_precursor_weights[i]);
        if (!node_in_visited[i])////use this function to make all vertices for updating as not visited
            topologicalSort_counter_order(i);
    }
    node_in_visited[0] = 0;
    auto curr_best_choice = sink_info->sink_precursors.begin();

    while (abs(sink_info->sink_precursor_weights[curr_best_choice->second] - curr_best_choice->first) > 0.0000001
           || node_in_visited[curr_best_choice->second] != 0) {
        sink_info->sink_precursors.erase(curr_best_choice);
        curr_best_choice = sink_info->sink_precursors.begin();
    }
    /******
     * postpone the updating if we've already found the next shortest path
     * ******/
    upt_node_num = 0;
    if (precursor_queue_top_val < curr_best_choice->first){
        upt_node_num = tplog_vec.size();
        /***************************
         * updating method:
         * topological ordering update_node_id
         * if distance < min_dist, insert to dijkstra_multi_map, otherwise do not insert
         * ************************/
        double cur_max_distance = cur_path_max_cost - precursor_queue_top_val - sink_info->sink_weight_shift;
        std::stack<int> useless_nodes;
        double re_cal_edge_w;
        //// start
        for (size_t i = 0; i < tplog_vec.size(); i++) {
            int cur_node = tplog_vec[i];
            if (cur_node % 2 == 0) {//// parent_node_id[cur_node] == -1end-1 elements in flipped path, no use
                distance2src[cur_node] = FINF;
            } else {
                double cur_best_distance = MIN(distance2src[parent_node_id[cur_node]], cur_max_distance);
                distance2src[cur_node] = cur_best_distance;

                for (size_t j = 0; j < V_[cur_node].precursor_idx.size(); j++) {
                    int it = V_[cur_node].precursor_idx[j];
                    //// the right thing
                    if (node_in_visited[it] == 0) {
                        re_cal_edge_w = V_[cur_node].precursor_edges_weights[j] - V_[it].price + V_[cur_node].price;
                        if (re_cal_edge_w < distance2src[cur_node]) {
                            distance2src[cur_node] = re_cal_edge_w;
                            parent_node_id[cur_node] = it;
                        }
                    }
                }
                if (distance2src[cur_node] >= cur_max_distance) {
                    node_in_visited[cur_node] = 4;
                    useless_nodes.push(cur_node);
                } else {
                    if (distance2src[cur_node] <= cur_best_distance) {
                        node_upt_waitinglist.insert(std::make_pair(distance2src[cur_node], cur_node));
                    }
                }
            }

        }
        tplog_vec.clear();

        //// check if the pre-best-choice can still be top of the sink_precursors
        std::multimap<double, int>::iterator multi_map_it;
        int curr_node_id;
        double curr_node_dist;
        while (!node_upt_waitinglist.empty()) {
            multi_map_it = node_upt_waitinglist.begin(); ////Current vertex. The shortest distance for this has been found
            curr_node_id = (*multi_map_it).second;
            curr_node_dist = (*multi_map_it).first;

            node_upt_waitinglist.erase(multi_map_it); //// remove the top one

            if (!node_in_visited[curr_node_id]) ////If the vertex is already visited, no point in exploring adjacent vertices
                continue;
            //// update ancestor
            if (parent_node_id[curr_node_id] == src_id_)
                ancestor_node_id[curr_node_id] = curr_node_id;
            else
                ancestor_node_id[curr_node_id] = ancestor_node_id[parent_node_id[curr_node_id]];

            node_in_visited[curr_node_id] = false;

            update_node_id4edges.push_back(curr_node_id);
            recursive_update_successors_distance(curr_node_id, curr_node_dist, ancestor_node_id[curr_node_id],
                                                 update_node_id4edges);
        }

        //// update ancestors_descendants
        for (auto &&i : update_node_id4edges) {
            ancestors_descendants[ancestor_node_id[i]].push_back(i);
            V_[i].price -= distance2src[i];
        }

        while (!useless_nodes.empty()) {
            if (node_in_visited[useless_nodes.top()] == 4) {
                node_in_visited[useless_nodes.top()] = -1; //permanently labelled as no use

            }
            useless_nodes.pop();
        }


        //// re-set precursor_queue_top_val as FINF;
        precursor_queue_top_val = FINF;
    }
    parent_node_id[sink_id_] = -1;
    update_node_id = update_node_id4edges;
}
/******
 * we use a heap to save edges related to sink
 * thus we can decrease the updating to at most n1*log(n)
 * *****/
void Graph::update_sink_info(std::vector<int> update_node_id)
{
    std::multimap<double, int>::iterator it;
    double cur_dist;
    for (auto &&i : update_node_id) {////Set updated node as not visited
        if (i % 2 == 0) { //// 2, 4, 6 ... is the sink's precursors
            if (distance2src[i] < FINFHALF) {
                cur_dist = sink_info->sink_precursor_weights[i] + distance2src[i];
                sink_info->sink_precursor_weights[i] = cur_dist;
                if (cur_dist < ancestor_ssd[ancestor_node_id[i]]) {
                    sink_info->sink_precursors.insert(std::make_pair(cur_dist, i));
                    ancestor_ssd[ancestor_node_id[i]] = cur_dist;
                }
            }
        }
    }

    sink_info->sink_precursor_weights[shortest_path[1]] = FINF; //// set the last but one distance as inf

    auto curr_best_choice = sink_info->sink_precursors.begin();

    while (abs(sink_info->sink_precursor_weights[curr_best_choice->second] - curr_best_choice->first) > 0.0000001
           || node_in_visited[curr_best_choice->second] != 0) {//// if used
        sink_info->sink_precursors.erase(curr_best_choice);
        curr_best_choice = sink_info->sink_precursors.begin();
    }

    double min4t_dist = curr_best_choice->first + sink_info->sink_weight_shift;
    int min4t_node_id = curr_best_choice->second;

    parent_node_id[sink_id_] = min4t_node_id;
    ancestor_node_id[sink_id_] = ancestor_node_id[min4t_node_id];
    distance2src[sink_id_] = min4t_dist;
}

/******
 * we do not need to update edge weight until we need to use them
 * but we need to update the distance labels and shift of sink heap
 * *****/
void Graph::update_subgraph_weights(std::vector<int> &update_node_id)
{
    //// majority of sink's precursors does not change, but they should, we save the change in shift
    sink_info->sink_weight_shift -= distance2src[sink_id_];

    ////reset nodes in sub-tree to 0
    for (auto &&i : update_node_id) {
        distance2src[i] = 0;
    }
}

Graph::~Graph() {
}
