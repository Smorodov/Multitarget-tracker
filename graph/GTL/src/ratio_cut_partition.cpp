/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//	ratio_cut_partition.cpp
//
//==========================================================================
// $Id: ratio_cut_partition.cpp,v 1.9 2001/11/07 13:58:11 pick Exp $

#include <GTL/debug.h>
#include <GTL/dfs.h>
#include <GTL/ratio_cut_partition.h>

#include <iostream>
#include <queue>

#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>

__GTL_BEGIN_NAMESPACE


const ratio_cut_partition::side_type ratio_cut_partition::A = 0;
const ratio_cut_partition::side_type ratio_cut_partition::B = 1;


const ratio_cut_partition::fix_type ratio_cut_partition::FIXA = 0;
const ratio_cut_partition::fix_type ratio_cut_partition::FIXB = 1;
const ratio_cut_partition::fix_type ratio_cut_partition::UNFIXED = 2;


ratio_cut_partition::ratio_cut_partition()
{
    set_vars_executed = false;
    enable_cut_edges_storing = false;
    enable_nodesAB_storing = false;
}


ratio_cut_partition::~ratio_cut_partition()
{
}


void ratio_cut_partition::set_vars(const graph& G,
    const node_map<int>& node_weight, const edge_map<int>& edge_weight)
{
    this->node_weight = node_weight;
    this->edge_weight = edge_weight;
    set_vars_executed = true;
    provided_st = false;
    provided_fix = false;
    this->fixed.init(G, UNFIXED);
    provided_initial_part = false;
    side.init(G);
}


void ratio_cut_partition::set_vars(const graph& G,
    const node_map<int>& node_weight, const edge_map<int>& edge_weight,
    const node source_node, const node target_node)
{
    this->node_weight = node_weight;
    this->edge_weight = edge_weight;
    this->source_node = source_node;
    this->target_node = target_node;
    set_vars_executed = true;
    provided_st = true;
    provided_fix = false;
    this->fixed.init(G, UNFIXED);
    provided_initial_part = false;
    side.init(G);
}


void ratio_cut_partition::set_vars(const graph& G,
    const node_map<int>& node_weight, const edge_map<int>& edge_weight,
    const node source_node, const node target_node,
    const node_map<side_type>& init_side)
{
    this->node_weight = node_weight;
    this->edge_weight = edge_weight;
    this->source_node = source_node;
    this->target_node = target_node;
    this->side = init_side;
    set_vars_executed = true;
    provided_st = true;
    provided_fix = false;
    this->fixed.init(G, UNFIXED);
    provided_initial_part = true;
}


void ratio_cut_partition::set_vars(const graph& G,
    const node_map<int>& node_weight, const edge_map<int>& edge_weight,
    const node source_node, const node target_node,
    const node_map<fix_type>& fixed)
{
    this->node_weight = node_weight;
    this->edge_weight = edge_weight;
    this->source_node = source_node;
    this->target_node = target_node;
    this->fixed = fixed;
    set_vars_executed = true;
    provided_st = true;
    provided_fix = true;
    provided_initial_part = false;
    side.init(G);
}


void ratio_cut_partition::set_vars(const graph& /*G*/,
    const node_map<int>& node_weight, const edge_map<int>& edge_weight,
    const node source_node, const node target_node,
    const node_map<side_type>& init_side, const node_map<fix_type>& fixed)
{
    this->node_weight = node_weight;
    this->edge_weight = edge_weight;
    this->source_node = source_node;
    this->target_node = target_node;
    this->side = init_side;
    this->fixed = fixed;
    set_vars_executed = true;
    provided_st = true;
    provided_fix = true;
    provided_initial_part = true;
}


void ratio_cut_partition::store_cut_edges(const bool set)
{
    enable_cut_edges_storing = set;
}


void ratio_cut_partition::store_nodesAB(const bool set)
{
    enable_nodesAB_storing = set;
}


int ratio_cut_partition::check(graph& G)
{
    if ((!set_vars_executed) || (!G.is_undirected()))
    {
	return GTL_ERROR;
    }

    graph::edge_iterator edge_it = G.edges_begin();
    graph::edge_iterator edges_end = G.edges_end();
    while (edge_it != edges_end)
    {
	if (edge_weight[*edge_it] < 0)
	{
	    return GTL_ERROR;
	}
	++edge_it;
    }
    int real_node_weights = 0;
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (node_weight[*node_it] > 0)
	{
	    ++real_node_weights;
	}
	if (node_weight[*node_it] < 0)
	{
	    return GTL_ERROR;
	}
	++node_it;
    }
    if ((G.number_of_nodes() >= 2) && (real_node_weights < 2))
    {
	return GTL_ERROR;
    }

    if ((provided_st) && (source_node == target_node) &&
	(G.number_of_nodes() > 1))
    {
	return GTL_ERROR;
    }

    if ((provided_initial_part) && ((side[source_node] != A) ||
	(side[target_node] != B)))
    {
	return GTL_ERROR;
    }

    if ((provided_fix) && ((fixed[source_node] == FIXB) ||
	(fixed[target_node] == FIXA)))
    {
	return GTL_ERROR;
    }

    if ((provided_st) && (node_weight[source_node] == 0 ||
	node_weight[target_node] == 0))
    {
	return GTL_ERROR;
    }

    return GTL_OK;
}


int ratio_cut_partition::run(graph& G)
{
    cur_cutsize = 0;
    cur_cutratio = 0.0;
    if (G.number_of_nodes() == 0)
    {
	return GTL_OK;	// nothing to do
    }
    if (G.number_of_nodes() == 1)
    {
	side[*G.nodes_begin()] = A;
	return GTL_OK;
    }

	edges_t artificial_edges;
    if (!G.is_connected())
    {
	make_connected(G, artificial_edges);
    }

    if (provided_fix)
    {
	divide_up(G);
    }

    if (!provided_st)
    {
	determine_source_node(G);
	compute_target_node(G);
    }

    if (provided_initial_part)
    {
	init_variables(G);
	init_data_structure(G);
	direction = LEFT_SHIFT;
	clean_step(G);
    }
    else
    {
	initialization(G);
    }
    iterative_shifting(G);
    group_swapping(G);

    if (enable_cut_edges_storing)
    {
	compute_cut_edges(G);
    }
    if (enable_nodesAB_storing)
    {
	compute_nodesAB(G);
    }
    restore(G, artificial_edges);

    return GTL_OK;
}


int ratio_cut_partition::get_cutsize()
{
    return cur_cutsize;
}


double ratio_cut_partition::get_cutratio()
{
    return cur_cutratio;
}


ratio_cut_partition::side_type
ratio_cut_partition::get_side_of_node(const node& n) const
{
    return side[n];
}


ratio_cut_partition::side_type ratio_cut_partition::operator []
(const node& n) const
{
    return side[n];
}


int ratio_cut_partition::get_weight_on_sideA(const graph& G) const
{
    int nwA = 0;
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (side[*node_it] == A)
	{
	    nwA += node_weight[*node_it];
	}
	++node_it;
    }
    return nwA;
}


int ratio_cut_partition::get_weight_on_sideB(const graph& G) const
{
    int nwB = 0;
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (side[*node_it] == B)
	{
	    nwB += node_weight[*node_it];
	}
	++node_it;
    }
    return nwB;
}


ratio_cut_partition::cut_edges_iterator
ratio_cut_partition::cut_edges_begin() const
{
    return cut_edges.begin();
}


ratio_cut_partition::cut_edges_iterator
ratio_cut_partition::cut_edges_end() const
{
    return cut_edges.end();
}


ratio_cut_partition::nodes_of_one_side_iterator
ratio_cut_partition::nodes_of_sideA_begin() const
{
    return nodesA.begin();
}


ratio_cut_partition::nodes_of_one_side_iterator
ratio_cut_partition::nodes_of_sideA_end() const
{
    return nodesA.end();
}


ratio_cut_partition::nodes_of_one_side_iterator
ratio_cut_partition::nodes_of_sideB_begin() const
{
    return nodesB.begin();
}


ratio_cut_partition::nodes_of_one_side_iterator
ratio_cut_partition::nodes_of_sideB_end() const
{
    return nodesB.end();
}


void ratio_cut_partition::reset()
{
    set_vars_executed = false;
    cut_edges.clear();
    nodesA.clear();
    nodesB.clear();
}


void ratio_cut_partition::divide_up(const graph& G)
{
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (fixed[*node_it] == FIXA)
	{
	    side[*node_it] = A;
	}
	else if (fixed[*node_it] == FIXB)
	{
	    side[*node_it] = B;
	}
	++node_it;
    }
}


void ratio_cut_partition::make_connected(graph& G,
	edges_t& artificial_edges)
{
    dfs conn;
    conn.scan_whole_graph(true);
    conn.check(G);
    conn.run(G);

    // connect dfs roots with zero edges
    dfs::roots_iterator root_it = conn.roots_begin();
    dfs::roots_iterator rootes_end = conn.roots_end();
    while (root_it != rootes_end)
    {
	node edge_start = **root_it;
	++root_it;
	if (root_it != rootes_end)
	{
	    edge ne = G.new_edge(edge_start, **root_it);
	    edge_weight[ne] = 0;	// this edge has no cut costs
	    artificial_edges.push_back(ne);
	}
    }
}


void ratio_cut_partition::restore(graph& G, edges_t& artificial_edges)
{
	edges_t::iterator edge_it = artificial_edges.begin();
	edges_t::iterator edges_end = artificial_edges.end();
    while (edge_it != edges_end)
    {
		G.del_edge(*edge_it);
		++edge_it;
    }
}


void ratio_cut_partition::initialization(const graph& G)
{
    int cutsize_A2B, cutsize_B2A;
    double cutratio_A2B, cutratio_B2A;
    node_map<side_type> side_B2A(G);

    init_variables(G);

    // start with moves from B to A
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (fixed[*node_it] == UNFIXED)
	{
	    side[*node_it] = B;
	}
	++node_it;
    }
    side[source_node] = A;
    side[target_node] = B;
    init_data_structure(G);
    if (fixed[target_node] == UNFIXED)
    {
	bucketB[range_up(gain_value[target_node])].
	    erase(position_in_bucket[target_node]);
	update_max_gain(B);
    }
    left_shift_op(G);
    clean_step(G);
    cutsize_B2A = cur_cutsize;
    cutratio_B2A = cur_cutratio;
    copy_side_node_map(G, side_B2A, side);

    // continue with moves from A to B
    node_it = G.nodes_begin();
    while (node_it != nodes_end)
    {
	if (fixed[*node_it] == UNFIXED)
	{
	    side[*node_it] = A;
	}
	++node_it;
    }
    side[source_node] = A;
    side[target_node] = B;
    init_data_structure(G);
    if (fixed[source_node] == UNFIXED)
    {
	bucketA[range_up(gain_value[source_node])].
	    erase(position_in_bucket[source_node]);
	update_max_gain(A);
    }
    right_shift_op(G);
    clean_step(G);
    cutsize_A2B = cur_cutsize;
    cutratio_A2B = cur_cutratio;

    if (cutratio_B2A < cutratio_A2B)
    {
	copy_side_node_map(G, side, side_B2A);
	cur_cutsize = cutsize_B2A;
	cur_cutratio = cutratio_B2A;
	direction = LEFT_SHIFT;
    }
    else
    {
	// copy_side_node_map(...) not necessary
	cur_cutsize = cutsize_A2B;
	cur_cutratio = cutratio_A2B;
	direction = RIGHT_SHIFT;
    }
}


void ratio_cut_partition::init_data_structure(const graph& G)
{
    aside.init(G);
    bside.init(G);
    unlockedA.init(G);
    unlockedB.init(G);
    cur_cutsize = 0;
    graph::edge_iterator edge_it = G.edges_begin();
    graph::edge_iterator edges_end = G.edges_end();
    while (edge_it != edges_end)
    {
	if ((side[edge_it->source()] == A) &&
	    (side[edge_it->target()] == A))
	{
	    aside[*edge_it] = 2;
	    bside[*edge_it] = 0;
	    unlockedA[*edge_it].push_back(edge_it->source());
	    unlockedA[*edge_it].push_back(edge_it->target());
	}
	else if ((side[edge_it->source()] == B) &&
	    (side[edge_it->target()] == B))
	{
	    aside[*edge_it] = 0;
	    bside[*edge_it] = 2;
	    unlockedB[*edge_it].push_back(edge_it->source());
	    unlockedB[*edge_it].push_back(edge_it->target());
	}
	else if ((side[edge_it->source()] == A) &&
	    (side[edge_it->target()] == B))
	{
	    aside[*edge_it] = 1;
	    bside[*edge_it] = 1;
	    cur_cutsize += edge_weight[*edge_it];
	    unlockedA[*edge_it].push_back(edge_it->source());
	    unlockedB[*edge_it].push_back(edge_it->target());
	}
	else if ((side[edge_it->source()] == B) &&
	    (side[edge_it->target()] == A))
	{
	    aside[*edge_it] = 1;
	    bside[*edge_it] = 1;
	    cur_cutsize += edge_weight[*edge_it];
	    unlockedA[*edge_it].push_back(edge_it->target());
	    unlockedB[*edge_it].push_back(edge_it->source());
	}
	++edge_it;
    }

    bucketA.resize(2 * max_vertex_degree * max_edge_weight + 1);
    bucketB.resize(2 * max_vertex_degree * max_edge_weight + 1);

    init_filling_buckets(G);
    cur_cutratio = cutratio();
}


void ratio_cut_partition::init_filling_buckets(const graph &G)
{
    node_weight_on_sideA = 0;
    node_weight_on_sideB = 0;
    nodes_on_sideA = 0;
    nodes_on_sideB = 0;
    bucketA_empty = true;
    bucketB_empty = true;
    bool first_A_node = true;
    bool first_B_node = true;
    int index;
    //    position_in_bucket.init(G);
    gain_value.init(G);

    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (side[*node_it] == A)
	{
	    node_weight_on_sideA += node_weight[*node_it];
	    ++nodes_on_sideA;
	    gain_value[*node_it] = inital_gain_of_node_on_sideA(*node_it);
	    if (fixed[*node_it] == UNFIXED)
	    {
		if (first_A_node)
		{
		    bucketA_empty = false;
		    max_gainA = gain_value[*node_it];
		    first_A_node = false;
		}
		else
		{
		    if (max_gainA < gain_value[*node_it])
		    {
			max_gainA = gain_value[*node_it];
		    }
		}
		index = range_up(gain_value[*node_it]);
		position_in_bucket[*node_it] = bucketA[index].insert(
		    bucketA[index].begin(), *node_it);
	    }
	}
	else	// side[*node_it] == B
	{
	    node_weight_on_sideB += node_weight[*node_it];
	    ++nodes_on_sideB;
	    gain_value[*node_it] = inital_gain_of_node_on_sideB(*node_it);
	    if (fixed[*node_it] == UNFIXED)
	    {
		if (first_B_node)
		{
		    bucketB_empty = false;
		    max_gainB = gain_value[*node_it];
		    first_B_node = false;
		}
		else
		{
		    if (max_gainB < gain_value[*node_it])
		    {
			max_gainB = gain_value[*node_it];
		    }
		}
		index = range_up(gain_value[*node_it]);
		position_in_bucket[*node_it] = bucketB[index].insert(
		    bucketB[index].begin(), *node_it);
	    }
	}
	++node_it;
    }
}


int ratio_cut_partition::inital_gain_of_node_on_sideA(const node cur_node)
{
    int node_gain = 0;
    node::adj_edges_iterator adj_edge_it = cur_node.adj_edges_begin();
    node::adj_edges_iterator adj_edges_end = cur_node.adj_edges_end();
    while (adj_edge_it != adj_edges_end)
    {
	if (aside[*adj_edge_it] == 1)
	{
	    node_gain += edge_weight[*adj_edge_it];
	}
	if (bside[*adj_edge_it] == 0)
	{
	    node_gain -= edge_weight[*adj_edge_it];
	}
	++adj_edge_it;
    }
    return node_gain;
}


int ratio_cut_partition::inital_gain_of_node_on_sideB(const node cur_node)
{
    int node_gain = 0;
    node::adj_edges_iterator adj_edge_it = cur_node.adj_edges_begin();
    node::adj_edges_iterator adj_edges_end = cur_node.adj_edges_end();
    while (adj_edge_it != adj_edges_end)
    {
	if (bside[*adj_edge_it] == 1)
	{
	    node_gain += edge_weight[*adj_edge_it];
	}
	if (aside[*adj_edge_it] == 0)
	{
	    node_gain -= edge_weight[*adj_edge_it];
	}
	++adj_edge_it;
    }
    return node_gain;
}


void ratio_cut_partition::init_variables(const graph& G)
{
    compute_max_vertex_degree(G);
    bool first_edge_found = true;
    max_edge_weight = 0;
    graph::edge_iterator edge_it = G.edges_begin();
    graph::edge_iterator edges_end = G.edges_end();
    while (edge_it != edges_end)
    {
	if (first_edge_found)
	{
	    max_edge_weight = edge_weight[*edge_it];
	    first_edge_found = false;
	}
	else if (edge_weight[*edge_it] > max_edge_weight)
	{
	    max_edge_weight = edge_weight[*edge_it];
	}
	++edge_it;
    }
}


void ratio_cut_partition::compute_max_vertex_degree(const graph& G)
{
    max_vertex_degree = 0;
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (max_vertex_degree < node_it->degree())
	{
	    max_vertex_degree = node_it->degree();
	}
	++node_it;
    }
}


void ratio_cut_partition::determine_source_node(const graph& G)
{
    srand((unsigned)time(NULL));
    rand();	// necessary, otherwise the next rand() returns always 0 ?-)
    int node_id = (int)floor((((double)rand() / (double)RAND_MAX) *
	(double)(G.number_of_nodes() - 1)) + 0.5);
    graph::node_iterator node_it = G.nodes_begin();
    for (int i = 1; i <= node_id; i++)
    {
	++node_it;
    }
    source_node = *node_it;
    if (node_weight[source_node] == 0)
    {
	node_it = G.nodes_begin();
	while (node_weight[*node_it] == 0)
	{
	    ++node_it;
	}
	source_node = *node_it;
    }
}


void ratio_cut_partition::compute_target_node(const graph& G)
{
    node cur_node, next;
    node_map<bool> visited(G, false);
	std::queue<node> next_nodes;
    next_nodes.push(source_node);
    visited[source_node] = true;

    while (!next_nodes.empty())
    {
	cur_node = next_nodes.front();
	next_nodes.pop();

	node::adj_edges_iterator adj_edge_it = cur_node.adj_edges_begin();
	node::adj_edges_iterator adj_edges_end = cur_node.adj_edges_end();
	while (adj_edge_it != adj_edges_end)
	{
	    if (adj_edge_it->target() != cur_node)
	    {
		next = adj_edge_it->target();
	    }
	    else
	    {
		next = adj_edge_it->source();
	    }
	    if (!visited[next])
	    {
		next_nodes.push(next);
		visited[next] = true;
	    }
	    ++adj_edge_it;
	}
    }
    target_node = cur_node;
    if (node_weight[target_node] == 0)
    {
	graph::node_iterator node_it = G.nodes_begin();
	while ((node_weight[*node_it] == 0) || (*node_it == source_node))
	{
	    ++node_it;
	}
	target_node = *node_it;
    }
}


void ratio_cut_partition::right_shift_op(const graph& G)
{
    int step_number = 0;
    int best_tentative_move = 0;
    int best_bal = node_weight_on_sideA * node_weight_on_sideB;
	std::vector<node> tentative_moves(G.number_of_nodes() + 1);
	std::vector<double> tentative_cutratio(G.number_of_nodes() + 1);
    node moved_node;
    tentative_cutratio[0] = cur_cutratio;
    int best_cutsize = cur_cutsize;

    while (move_vertex_A2B(G, moved_node))
    {
	++step_number;
	tentative_cutratio[step_number] = cur_cutratio;
	tentative_moves[step_number] = moved_node;
	if (tentative_cutratio[best_tentative_move] > cur_cutratio)
	{
	    best_tentative_move = step_number;
	    best_cutsize = cur_cutsize;
	    best_bal = node_weight_on_sideA * node_weight_on_sideB;
	}
	else if (tentative_cutratio[best_tentative_move] == cur_cutratio)
	{
	    if (node_weight_on_sideA * node_weight_on_sideB > best_bal)
	    {
		best_tentative_move = step_number;
		best_cutsize = cur_cutsize;
		best_bal = node_weight_on_sideA * node_weight_on_sideB;
	    }
	}
    }

    for (int i = 1; i <= best_tentative_move; i++)
    {
	if (side[tentative_moves[i]] == A)
	{
	    side[tentative_moves[i]] = B;
	}
	else	// side[tentative_moves[i]] == B
	{
	    side[tentative_moves[i]] = A;
	}
    }
    cur_cutratio = tentative_cutratio[best_tentative_move];
    cur_cutsize = best_cutsize;
}


void ratio_cut_partition::left_shift_op(const graph& G)
{
    int step_number = 0;
    int best_tentative_move = 0;
    int best_bal = node_weight_on_sideA * node_weight_on_sideB;
	std::vector<node> tentative_moves(G.number_of_nodes() + 1);
	std::vector<double> tentative_cutratio(G.number_of_nodes() + 1);
    node moved_node;
    tentative_cutratio[0] = cur_cutratio;
    int best_cutsize = cur_cutsize;

    while (move_vertex_B2A(G, moved_node))
    {
	++step_number;
	tentative_cutratio[step_number] = cur_cutratio;
	tentative_moves[step_number] = moved_node;
	if (tentative_cutratio[best_tentative_move] > cur_cutratio)
	{
	    best_tentative_move = step_number;
	    best_cutsize = cur_cutsize;
	}
	else if (tentative_cutratio[best_tentative_move] == cur_cutratio)
	{
	    if (node_weight_on_sideA * node_weight_on_sideB > best_bal)
	    {
		best_tentative_move = step_number;
		best_cutsize = cur_cutsize;
		best_bal = node_weight_on_sideA * node_weight_on_sideB;
	    }
	}
    }

    for (int i = 1; i <= best_tentative_move; i++)
    {
	if (side[tentative_moves[i]] == A)
	{
	    side[tentative_moves[i]] = B;
	}
	else	// side[tentative_moves[i]] == B
	{
	    side[tentative_moves[i]] = A;
	}
    }
    cur_cutratio = tentative_cutratio[best_tentative_move];
    cur_cutsize = best_cutsize;
}


bool ratio_cut_partition::move_vertex_A2B(const graph &/*G*/, node& moved_node)
{
    if (!bucketA_empty)
    {
	node cons_nodeA =
	    compute_highest_ratio_node(bucketA[range_up(max_gainA)]);
	bucketA[range_up(max_gainA)].erase(position_in_bucket[cons_nodeA]);
	update_data_structure_A2B(cons_nodeA, true);
	moved_node = cons_nodeA;
    }
    else
    {
	return false;	// no more vertices can be moved
    }
    update_max_gain(A);
    return true;
}


bool ratio_cut_partition::move_vertex_B2A(const graph &/*G*/, node& moved_node)
{
    if (!bucketB_empty)
    {
	node cons_nodeB =
	    compute_highest_ratio_node(bucketB[range_up(max_gainB)]);
	bucketB[range_up(max_gainB)].erase(position_in_bucket[cons_nodeB]);
	update_data_structure_B2A(cons_nodeB, true);
	moved_node = cons_nodeB;
    }
    else
    {
	return false;	// no more vertices can be moved
    }
    update_max_gain(B);
    return true;
}


node ratio_cut_partition::compute_highest_ratio_node(nodes_t node_list)
{
    node cons_node = node_list.front();
    double ratio, best_ratio;
    if (side[cons_node] == A)
    {
	best_ratio = ratio_of_node_A2B(cons_node);
    }
    else	// side[cons_node] == B
    {
	best_ratio = ratio_of_node_B2A(cons_node);
    }
	
	nodes_t::iterator node_it = node_list.begin();
	nodes_t::iterator nodes_end = node_list.end();
    while (node_it != nodes_end)
    {
	if (side[cons_node] == A)
	{
	    ratio = ratio_of_node_A2B(*node_it);
	}
	else	// side[cons_node] == B
	{
	    ratio = ratio_of_node_B2A(*node_it);
	}
	if (ratio > best_ratio)	// choose node with highest ratio
	{
	    best_ratio = ratio;
	    cons_node = *node_it;
	}
	++node_it;
    }
    return cons_node;
}


double ratio_cut_partition::cutratio()
{
    double number_of_nodes = (double)(nodes_on_sideA + nodes_on_sideB);
    return ((double)cur_cutsize + number_of_nodes) / (double)
	(node_weight_on_sideA * node_weight_on_sideB);
}


double ratio_cut_partition::ratio_of_node_A2B(const node cur_node)
{
    return (double)gain_value[cur_node] / 
	((double)((node_weight_on_sideB + node_weight[cur_node]) *
	    (node_weight_on_sideA - node_weight[cur_node])));
}


double ratio_cut_partition::ratio_of_node_B2A(const node cur_node)
{
    return (double)gain_value[cur_node] /
	((double)((node_weight_on_sideA + node_weight[cur_node]) *
	    (node_weight_on_sideB - node_weight[cur_node])));
}


inline int ratio_cut_partition::range_up(const int gain_value) const
{
    return gain_value + (max_vertex_degree * max_edge_weight);
}


inline int ratio_cut_partition::range_down(const int index) const
{
    return index - (max_vertex_degree * max_edge_weight);
}


void ratio_cut_partition::update_data_structure_A2B(const node cur_node,
	const bool init_mode)
{
	node_weight_on_sideA -= node_weight[cur_node];
	node_weight_on_sideB += node_weight[cur_node];
	--nodes_on_sideA;
	++nodes_on_sideB;
	cur_cutsize -= gain_value[cur_node];
	cur_cutratio = cutratio();

	// updating gain values
	node::adj_edges_iterator adj_edge_it = cur_node.adj_edges_begin();
	node::adj_edges_iterator adj_edges_end = cur_node.adj_edges_end();
	while (adj_edge_it != adj_edges_end)
	{
		// delete cur_node from side A
#if 1
		unlockedA[*adj_edge_it].remove(cur_node);
#else
		auto& ua = unlockedA[*adj_edge_it];
		ua.erase(std::remove(ua.begin(), ua.end(), cur_node), ua.end());
#endif
		--aside[*adj_edge_it];
		if (aside[*adj_edge_it] == 0)
		{
			nodes_t::iterator node_it = unlockedB[*adj_edge_it].begin();
			nodes_t::iterator nodes_end = unlockedB[*adj_edge_it].end();
			while (node_it != nodes_end)
			{
				update_bucketB(*node_it, gain_value[*node_it],
					gain_value[*node_it] - edge_weight[*adj_edge_it],
					init_mode);
				gain_value[*node_it] -= edge_weight[*adj_edge_it];
				++node_it;
			}
		}
		else if (aside[*adj_edge_it] == 1)
		{
			nodes_t::iterator node_it = unlockedA[*adj_edge_it].begin();
			nodes_t::iterator nodes_end = unlockedA[*adj_edge_it].end();
			while (node_it != nodes_end)
			{
				update_bucketA(*node_it, gain_value[*node_it],
					gain_value[*node_it] + edge_weight[*adj_edge_it],
					init_mode);
				gain_value[*node_it] += edge_weight[*adj_edge_it];
				++node_it;
			}
		}
		// add cur_node to side B
		++bside[*adj_edge_it];
		if (bside[*adj_edge_it] == 1)
		{
			nodes_t::iterator node_it = unlockedA[*adj_edge_it].begin();
			nodes_t::iterator nodes_end = unlockedA[*adj_edge_it].end();
			while (node_it != nodes_end)
			{
				update_bucketA(*node_it, gain_value[*node_it],
					gain_value[*node_it] + edge_weight[*adj_edge_it],
					init_mode);
				gain_value[*node_it] += edge_weight[*adj_edge_it];
				++node_it;
			}
		}
		else if (bside[*adj_edge_it] == 2)
		{
			nodes_t::iterator node_it = unlockedB[*adj_edge_it].begin();
			nodes_t::iterator nodes_end = unlockedB[*adj_edge_it].end();
			while (node_it != nodes_end)
			{
				update_bucketB(*node_it, gain_value[*node_it],
					gain_value[*node_it] - edge_weight[*adj_edge_it],
					init_mode);
				gain_value[*node_it] -= edge_weight[*adj_edge_it];
				++node_it;
			}
		}
		++adj_edge_it;
	}
}


void ratio_cut_partition::update_data_structure_B2A(const node cur_node,
    const bool init_mode)
{
    node_weight_on_sideA += node_weight[cur_node];
    node_weight_on_sideB -= node_weight[cur_node];
    ++nodes_on_sideA;
    --nodes_on_sideB;
    cur_cutsize -= gain_value[cur_node];
    cur_cutratio = cutratio();
	
    // updating gain values
    node::adj_edges_iterator adj_edge_it = cur_node.adj_edges_begin();
    node::adj_edges_iterator adj_edges_end = cur_node.adj_edges_end();
    while (adj_edge_it != adj_edges_end)
    {
	// delete cur_node from side B
#if 1
		unlockedB[*adj_edge_it].remove(cur_node);
#else
		auto& ub = unlockedB[*adj_edge_it];
		ub.erase(std::remove(ub.begin(), ub.end(), cur_node), ub.end());
#endif
	bside[*adj_edge_it] -= 1;
	if (bside[*adj_edge_it] == 0)
	{
		nodes_t::iterator node_it = unlockedA[*adj_edge_it].begin();
		nodes_t::iterator nodes_end = unlockedA[*adj_edge_it].end();
	    while (node_it != nodes_end)
	    {
		update_bucketA(*node_it, gain_value[*node_it],
		    gain_value[*node_it] - edge_weight[*adj_edge_it],
		    init_mode);
		gain_value[*node_it] -= edge_weight[*adj_edge_it];
		++node_it;
	    }
	}
	else if (bside[*adj_edge_it] == 1)
	{
		nodes_t::iterator node_it = unlockedB[*adj_edge_it].begin();
		nodes_t::iterator nodes_end = unlockedB[*adj_edge_it].end();
	    while (node_it != nodes_end)
	    {
		update_bucketB(*node_it, gain_value[*node_it],
		    gain_value[*node_it] + edge_weight[*adj_edge_it],
		    init_mode);
		gain_value[*node_it] += edge_weight[*adj_edge_it];
		++node_it;
	    }
	}
	// add cur_node to side A
	aside[*adj_edge_it] += 1;
	if (aside[*adj_edge_it] == 1)
	{
		nodes_t::iterator node_it = unlockedB[*adj_edge_it].begin();
		nodes_t::iterator nodes_end = unlockedB[*adj_edge_it].end();
	    while (node_it != nodes_end)
	    {
		update_bucketB(*node_it, gain_value[*node_it],
		    gain_value[*node_it] + edge_weight[*adj_edge_it],
		    init_mode);
		gain_value[*node_it] += edge_weight[*adj_edge_it];
		++node_it;
	    }
	}
	else if (aside[*adj_edge_it] == 2)
	{
		nodes_t::iterator node_it = unlockedA[*adj_edge_it].begin();
		nodes_t::iterator nodes_end = unlockedA[*adj_edge_it].end();
	    while (node_it != nodes_end)
	    {
		update_bucketA(*node_it, gain_value[*node_it],
		    gain_value[*node_it] - edge_weight[*adj_edge_it],
		    init_mode);
		gain_value[*node_it] -= edge_weight[*adj_edge_it];
		++node_it;
	    }
	}
	++adj_edge_it;
    }
}


void ratio_cut_partition::update_bucketA(const node cur_node,
    const int old_gain, const int new_gain, const bool init_mode)
{
    if ((init_mode) && (cur_node == source_node))
    {
	return;	// this one needs no update with init_mode
    }
    if (fixed[cur_node] != UNFIXED)
    {
	return;	// fixed nodes need no update
    }
    bucketA[range_up(old_gain)].erase(position_in_bucket[cur_node]);
    bucketA[range_up(new_gain)].push_front(cur_node);
    position_in_bucket[cur_node] = bucketA[range_up(new_gain)].begin();
    if (max_gainA < new_gain)
    {
	max_gainA = new_gain;
    }
}


void ratio_cut_partition::update_bucketB(const node cur_node,
    const int old_gain, const int new_gain, const bool init_mode)
{
    if ((init_mode) && (cur_node == target_node))
    {
	return;	// this one needs no update with init_mode
    }
    if (fixed[cur_node] != UNFIXED)
    {
	return;	// fixed nodes need no update
    }
    bucketB[range_up(old_gain)].erase(position_in_bucket[cur_node]);
    bucketB[range_up(new_gain)].push_front(cur_node);
    position_in_bucket[cur_node] = bucketB[range_up(new_gain)].begin();
    if (max_gainB < new_gain)
    {
	max_gainB = new_gain;
    }
}


void ratio_cut_partition::update_max_gain(const side_type side)
{
    if ((side == A) && (!bucketA_empty))
    {
	while (bucketA[range_up(max_gainA)].begin() ==
	    bucketA[range_up(max_gainA)].end())
	{
	    --max_gainA;
	    if (range_up(max_gainA) < 0)
	    {
		bucketA_empty = true;
		return;
	    }
	}
	bucketA_empty = false;
    }
    if ((side == B) && (!bucketB_empty))
    {
	while (bucketB[range_up(max_gainB)].begin() ==
	    bucketB[range_up(max_gainB)].end())
	{
	    --max_gainB;
	    if (range_up(max_gainB) < 0)
	    {
		bucketB_empty = true;
		return;
	    }
	}
	bucketB_empty = false;
    }
}


void ratio_cut_partition::clean_step(const graph& G)
{
    // clean unlocked* lists
    graph::edge_iterator edge_it = G.edges_begin();
    graph::edge_iterator edges_end = G.edges_end();
    while (edge_it != edges_end)
    {
	unlockedA[*edge_it].clear();
	unlockedB[*edge_it].clear();
	++edge_it;
    }
	
    // clean buckets
    for (int i = 0; i <= 2 * max_vertex_degree * max_edge_weight; i++)
    {
	bucketA[i].clear();
	bucketB[i].clear();
    }
    bucketA.clear();
    bucketB.clear();
}


void ratio_cut_partition::copy_side_node_map(const graph& G,
    node_map<side_type>& dest, const node_map<side_type> source) const
{
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	dest[*node_it] = source[*node_it];
	++node_it;
    }
}


void ratio_cut_partition::iterative_shifting(const graph& G)
{
    bool continue_loop = true;
    double old_cutratio = cur_cutratio;
	
    while (continue_loop)
    {
	if (direction == LEFT_SHIFT)
	{
	    init_data_structure(G);
	    if (fixed[source_node] == UNFIXED)
	    {
		bucketA[range_up(gain_value[source_node])].
		    erase(position_in_bucket[source_node]);
		update_max_gain(A);
	    }
	    right_shift_op(G);
	    clean_step(G);
	    if (cur_cutratio < old_cutratio)
	    {
		continue_loop = true;
		direction = RIGHT_SHIFT;
		old_cutratio = cur_cutratio;
	    }
	    else
	    {
		continue_loop = false;
	    }
	}
	else	// direction == RIGHT_SHIFT
	{
	    init_data_structure(G);
	    if (fixed[target_node] == UNFIXED)
	    {
		bucketB[range_up(gain_value[target_node])].
		    erase(position_in_bucket[target_node]);
		update_max_gain(B);
	    }
	    left_shift_op(G);
	    clean_step(G);
	    if (cur_cutratio < old_cutratio)
	    {
		continue_loop = true;
		direction = LEFT_SHIFT;
		old_cutratio = cur_cutratio;
	    }
	    else
	    {
		continue_loop = false;
	    }
	}
    }
}


void ratio_cut_partition::group_swapping(const graph& G)
{
    bool improved_cutratio;

    do
    {
	init_data_structure(G);
	improved_cutratio = move_manager(G);
	clean_step(G);
    }
    while (improved_cutratio);
}


bool ratio_cut_partition::move_manager(const graph& G)
{
    int step_number = 0;
    int best_tentative_move = 0;
    int best_bal = node_weight_on_sideA * node_weight_on_sideB;
	std::vector<node> tentative_moves(G.number_of_nodes() + 1);
	std::vector<double> tentative_cutratio(G.number_of_nodes() + 1);
    node moved_node;
    tentative_cutratio[0] = cur_cutratio;
    int best_cutsize = cur_cutsize;

    while (move_vertex(G, moved_node))
    {
	++step_number;
	tentative_moves[step_number] = moved_node;
	tentative_cutratio[step_number] = cur_cutratio;
	if (tentative_cutratio[best_tentative_move] > cur_cutratio)
	{
	    best_tentative_move = step_number;
	    best_cutsize = cur_cutsize;
	    best_bal = node_weight_on_sideA * node_weight_on_sideB;
	}
	else if (tentative_cutratio[best_tentative_move] == cur_cutratio)
	{
	    if (node_weight_on_sideA * node_weight_on_sideB > best_bal)
	    {
		best_tentative_move = step_number;
		best_cutsize = cur_cutsize;
		best_bal = node_weight_on_sideA * node_weight_on_sideB;
	    }
	}
    }

    for (int i = 1; i <= best_tentative_move; i++)
    {
	if (side[tentative_moves[i]] == A)
	{
	    side[tentative_moves[i]] = B;
	}
	else	// side[tentative_moves[i]] == B
	{
	    side[tentative_moves[i]] = A;
	}
    }
    cur_cutratio = tentative_cutratio[best_tentative_move];
    cur_cutsize = best_cutsize;
    if (best_tentative_move > 0)	// cutratio improved
    {
	return true;
    }
    return false;	// best_move == 0  -->  cutratio not improved
}


bool ratio_cut_partition::move_vertex(const graph &/*G*/, node& moved_node)
{
    bool consA_ok = false, consB_ok = false;
    node cons_nodeA, cons_nodeB;

    if (!bucketA_empty)
    {
	cons_nodeA =
	    compute_highest_ratio_node(bucketA[range_up(max_gainA)]);
	consA_ok = true;
	if (node_weight_on_sideA - node_weight[cons_nodeA] == 0)
	{
	    node temp_node = cons_nodeA;
	    bucketA[range_up(gain_value[cons_nodeA])].
		erase(position_in_bucket[cons_nodeA]);
	    update_max_gain(A);
	    if (!bucketA_empty)	// nodes with smaller weight available?
	    {
		cons_nodeA = compute_highest_ratio_node
		    (bucketA[range_up(max_gainA)]);
	    }
	    else
	    {
		consA_ok = false;
	    }
	    bucketA_empty = false;
	    bucketA[range_up(gain_value[temp_node])].push_front(temp_node);
	    position_in_bucket[temp_node] =
		bucketA[range_up(gain_value[temp_node])].begin();
	    max_gainA = gain_value[temp_node];
	}
    }
    if (!bucketB_empty)
    {
	cons_nodeB =
	    compute_highest_ratio_node(bucketB[range_up(max_gainB)]);
	consB_ok = true;
	if (node_weight_on_sideB - node_weight[cons_nodeB] == 0)
	{
	    node temp_node = cons_nodeB;
	    bucketB[range_up(gain_value[cons_nodeB])].
		erase(position_in_bucket[cons_nodeB]);
	    update_max_gain(B);
	    if (!bucketB_empty)	// nodes with smaller weight available?
	    {
		cons_nodeB = compute_highest_ratio_node
		    (bucketB[range_up(max_gainB)]);
	    }
	    else
	    {
		consB_ok = false;
	    }
	    bucketB_empty = false;
	    bucketB[range_up(gain_value[temp_node])].push_front(temp_node);
	    position_in_bucket[temp_node] =
		bucketB[range_up(gain_value[temp_node])].begin();
	    max_gainB = gain_value[temp_node];
	}
    }

    if (consA_ok && consB_ok)
    {
	double ratio_A2B = ratio_of_node_A2B(cons_nodeA);
	double ratio_B2A = ratio_of_node_B2A(cons_nodeB);
	if (ratio_A2B > ratio_B2A)
	{
	    moved_node = cons_nodeA;
	    bucketA[range_up(max_gainA)].
		erase(position_in_bucket[cons_nodeA]);
	    update_data_structure_A2B(cons_nodeA, false);
	}
	else	// ratio_A2B <= ratio_B2A
	{
	    moved_node = cons_nodeB;
	    bucketB[range_up(max_gainB)].
		erase(position_in_bucket[cons_nodeB]);
	    update_data_structure_B2A(cons_nodeB, false);
	}
    }
    else if (consA_ok)
    {
	moved_node = cons_nodeA;
	bucketA[range_up(max_gainA)].erase(position_in_bucket[cons_nodeA]);
	update_data_structure_A2B(cons_nodeA, false);
    }
    else if (consB_ok)
    {
	moved_node = cons_nodeB;
	bucketB[range_up(max_gainB)].erase(position_in_bucket[cons_nodeB]);
	update_data_structure_B2A(cons_nodeB, false);
    }
    else
    {
	return false;	// no more vertices can be moved
    }
    update_max_gain(A);
    update_max_gain(B);
    return true;
}


void ratio_cut_partition::compute_cut_edges(const graph& G)
{
    cut_edges.clear();
    graph::edge_iterator edge_it = G.edges_begin();
    graph::edge_iterator edges_end = G.edges_end();
    while (edge_it != edges_end)
    {
	if (side[edge_it->source()] != side[edge_it->target()])
	{
	    cut_edges.push_back(*edge_it);
	}
	++edge_it;
    }
}


void ratio_cut_partition::compute_nodesAB(const graph& G)
{
    nodesA.clear();
    nodesB.clear();
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (side[*node_it] == A)
	{
	    nodesA.push_back(*node_it);
	}
	else	// side[*node_it] == B
	{
	    nodesB.push_back(*node_it);
	}
	++node_it;
    }
}


#ifdef _DEBUG
void ratio_cut_partition::print_bucketA()
{
    GTL_debug::init_debug();
	GTL_debug::os() << std::endl << "bucketA:" << std::endl;
    for (int i = 0; i <= 2 * max_vertex_degree * max_edge_weight; i++)
    {
	GTL_debug::os() << range_down(i) << ": ";
	nodes_t::iterator node_it = bucketA[i].begin();
	nodes_t::iterator nodes_end = bucketA[i].end();
	while (node_it != nodes_end)
	{
	    GTL_debug::os() << *node_it << "  ";
	    ++node_it;
	}
	GTL_debug::os() << std::endl;
    }
	GTL_debug::os() << std::endl;
    GTL_debug::close_debug();
}


void ratio_cut_partition::print_bucketB()
{
    GTL_debug::init_debug();
	GTL_debug::os() << std::endl << "bucketB:" << std::endl;
    for (int i = 0; i <= 2 * max_vertex_degree * max_edge_weight; i++)
    {
	GTL_debug::os() << range_down(i) << ": ";
	nodes_t::iterator node_it = bucketB[i].begin();
	nodes_t::iterator nodes_end = bucketB[i].end();
	while (node_it != nodes_end)
	{
	    GTL_debug::os() << *node_it << "  ";
	    ++node_it;
	}
	GTL_debug::os() << std::endl;
    }
	GTL_debug::os() << std::endl;
    GTL_debug::close_debug();
}
#endif	// _DEBUG


__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
