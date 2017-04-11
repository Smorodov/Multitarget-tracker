/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//	fm_partition.cpp
//
//==========================================================================
// $Id: fm_partition.cpp,v 1.8 2001/11/07 13:58:10 pick Exp $

#include <GTL/debug.h>
#include <GTL/fm_partition.h>

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>

__GTL_BEGIN_NAMESPACE


const fm_partition::side_type fm_partition::A = 0;
const fm_partition::side_type fm_partition::B = 1;


const fm_partition::fix_type fm_partition::FIXA = 0;
const fm_partition::fix_type fm_partition::FIXB = 1;
const fm_partition::fix_type fm_partition::UNFIXED = 2;


fm_partition::fm_partition()
{
    set_vars_executed = false;
    enable_cut_edges_storing = false;
    enable_nodesAB_storing = false;
}


fm_partition::~fm_partition()
{
}


void fm_partition::set_vars(const graph& G,
    const node_map<int>& node_weight, const edge_map<int>& edge_weight)
{
    this->node_weight = node_weight;
    this->edge_weight = edge_weight;
    set_vars_executed = true;
    provided_initial_part = false;
    this->fixed.init(G, UNFIXED);
    provided_fix = false;
    side.init(G);
}


void fm_partition::set_vars(const graph& G,
    const node_map<int>& node_weight, const edge_map<int>& edge_weight,
    const node_map<side_type>& init_side)
{
    this->node_weight = node_weight;
    this->edge_weight = edge_weight;
    this->side = init_side;
    set_vars_executed = true;
    provided_initial_part = true;
    this->fixed.init(G, UNFIXED);
    provided_fix = false;
}


void fm_partition::set_vars(const graph& G,
    const node_map<int>& node_weight, const edge_map<int>& edge_weight,
    const node_map<fix_type>& fixed)
{
    this->node_weight = node_weight;
    this->edge_weight = edge_weight;
    set_vars_executed = true;
    provided_initial_part = false;
    this->fixed = fixed;
    provided_fix = true;
    side.init(G);
}


void fm_partition::set_vars(const graph& /*G*/,
    const node_map<int>& node_weight, const edge_map<int>& edge_weight,
    const node_map<side_type>& init_side, 
    const node_map<fix_type>& fixed)
{
    this->node_weight = node_weight;
    this->edge_weight = edge_weight;
    this->side = init_side;
    set_vars_executed = true;
    provided_initial_part = true;
    this->fixed = fixed;
    provided_fix = true;
}


void fm_partition::store_cut_edges(const bool set)
{
    enable_cut_edges_storing = set;
}


void fm_partition::store_nodesAB(const bool set)
{
    enable_nodesAB_storing = set;
}


int fm_partition::check(graph& G)
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
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (node_weight[*node_it] < 0)
	{
	    return GTL_ERROR;
	}
	++node_it;
    }

    return GTL_OK;
}


int fm_partition::run(graph& G)
{
    init_variables(G);
    if ((provided_initial_part) && (provided_fix))
    {
	divide_up(G);
    }
    if (!provided_initial_part)
    {
	create_initial_bipart(G);
    }
	
    hide_self_loops(G);
    compute_max_vertex_degree(G);

    pass_manager(G);

    if (enable_cut_edges_storing)
    {
	compute_cut_edges(G);
    }
    if (enable_nodesAB_storing)
    {
	compute_nodesAB(G);
    }
    G.restore_graph();

    return GTL_OK;
}


int fm_partition::get_cutsize()
{
    return cur_cutsize;
}


int fm_partition::get_needed_passes()
{
    return no_passes;
}


fm_partition::side_type fm_partition::get_side_of_node(const node& n) const
{
    return side[n];
}


fm_partition::side_type fm_partition::operator [](const node& n) const
{
    return side[n];
}


int fm_partition::get_weight_on_sideA(const graph& G) const
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


int fm_partition::get_weight_on_sideB(const graph& G) const
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


fm_partition::cut_edges_iterator fm_partition::cut_edges_begin() const
{
    return cut_edges.begin();
}


fm_partition::cut_edges_iterator fm_partition::cut_edges_end() const
{
    return cut_edges.end();
}


fm_partition::nodes_of_one_side_iterator
fm_partition::nodes_of_sideA_begin() const
{
    return nodesA.begin();
}


fm_partition::nodes_of_one_side_iterator
fm_partition::nodes_of_sideA_end() const
{
    return nodesA.end();
}


fm_partition::nodes_of_one_side_iterator
fm_partition::nodes_of_sideB_begin() const
{
    return nodesB.begin();
}


fm_partition::nodes_of_one_side_iterator
fm_partition::nodes_of_sideB_end() const
{
    return nodesB.end();
}


void fm_partition::reset()
{
    set_vars_executed = false;
    cut_edges.clear();
    nodesA.clear();
    nodesB.clear();
}


void fm_partition::divide_up(const graph& G)
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


void fm_partition::hide_self_loops(graph& G)
{
    graph::edge_iterator temp_it;
    graph::edge_iterator edge_it = G.edges_begin();
    graph::edge_iterator edges_end = G.edges_end();
    while (edge_it != edges_end)
    {
	if (edge_it->source() == edge_it->target())
	{
	    temp_it = edge_it;
	    ++edge_it;
	    G.hide_edge(*temp_it);
	}
	else
	{
	    ++edge_it;
	}
    }
}


void fm_partition::init_variables(const graph& G)
{
    bool first_edge_found = true;
    bool first_node_found = true;
    max_edge_weight = 0;
    max_node_weight = 0;
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
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    total_node_weight = 0;
    while (node_it != nodes_end)
    {
	total_node_weight += node_weight[*node_it];
	if (first_node_found)
	{
	    max_node_weight = node_weight[*node_it];
	    first_node_found = false;
	}
	else if (node_weight[*node_it] > max_node_weight)
	{
	    max_node_weight = node_weight[*node_it];
	}
	++node_it;
    }
}


void fm_partition::create_initial_bipart(const graph& G)
{
    int i = 0;	// counter
    int no_nodes = G.number_of_nodes();
    node_weight_on_sideA = 0;
    node_weight_on_sideB = 0;

    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
	std::vector<graph::node_iterator> node_vector(G.number_of_nodes());
    while (node_it != nodes_end)
    {
	node_vector[i] = node_it;
	if (fixed[*node_it] == FIXA)
	{
	    side[*node_it] = A;
	    node_weight_on_sideA += node_weight[*node_it];
	}
	else if (fixed[*node_it] == FIXB)
	{
	    side[*node_it] = B;
	    node_weight_on_sideB += node_weight[*node_it];
	}
	else	// fixed[*node_it] == UNFIXED
	{
	    node_weight_on_sideB += node_weight[*node_it];
	    side[*node_it] = B;
	}
	++i;
	++node_it;
    }
    shuffle_vector(no_nodes, node_vector);

    // compute best balance
    int best_bal = node_weight_on_sideA * node_weight_on_sideB;
    int best_pos = -1;
    for (i = 0; i < no_nodes; i++)
    {
	if (fixed[*node_vector[i]] == UNFIXED)
	{
	    node_weight_on_sideA += node_weight[*node_vector[i]];
	    node_weight_on_sideB -= node_weight[*node_vector[i]];
	    if (node_weight_on_sideA * node_weight_on_sideB > best_bal)
	    {
		best_bal = node_weight_on_sideA * node_weight_on_sideB;
		best_pos = i;
	    }
	}
    }

    // create partition with best balance
    for (i = 0; i <= best_pos; i++)
    {
	if (fixed[*node_vector[i]] == UNFIXED)
	{
	    side[*node_vector[i]] = A;
	}
    }
}


void fm_partition::shuffle_vector(const int vector_size,
	std::vector<graph::node_iterator>& node_vector)
{
    srand((unsigned)time(NULL));
    rand();	// necessary, otherwise the next rand() returns always 0 ?-)
    for (int i = 1; i <= vector_size; i++)
    {
	int pos_1 = (int)floor((((double)rand() / (double)RAND_MAX) *
	    (double)(vector_size - 1)) + 0.5);
	int pos_2 = (int)floor((((double)rand() / (double)RAND_MAX) *
	    (double)(vector_size - 1)) + 0.5);
	graph::node_iterator temp_it;
	temp_it = node_vector[pos_1];
	node_vector[pos_1] = node_vector[pos_2];
	node_vector[pos_2] = temp_it;
    }
}


void fm_partition::compute_max_vertex_degree(const graph& G)
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


void fm_partition::pass_manager(const graph& G)
{
    // final pass which doesn't improve cur_cutsize is not counted
    no_passes = -1;
    int best_cutsize = -1;	// = -1 to avoid warning
    node_map<side_type> best_side(G);
    bool improved_cutsize;

    do
    {
	init_data_structure(G);
	if (no_passes == -1)
	{
	    best_cutsize = cur_cutsize;
	    copy_side_node_map(G, best_side, side);
	}
	move_manager(G);
	clean_pass(G);
	improved_cutsize = false;
	if (best_cutsize > cur_cutsize)
	{
	    best_cutsize = cur_cutsize;
	    copy_side_node_map(G, best_side, side);
	    improved_cutsize = true;
	}
	++no_passes;
    }
    while (improved_cutsize);
    cur_cutsize = best_cutsize;
    copy_side_node_map(G, side, best_side);
}


void fm_partition::copy_side_node_map(const graph& G,
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


void fm_partition::init_data_structure(const graph& G)
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
}


void fm_partition::init_filling_buckets(const graph &G)
{
    node_weight_on_sideA = 0;
    node_weight_on_sideB = 0;
    bucketA_empty = true;
    bucketB_empty = true;
    bool first_A_node = true;
    bool first_B_node = true;
    int index;
    // position_in_bucket.init(G);
    gain_value.init(G);

    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (side[*node_it] == A)
	{
	    node_weight_on_sideA += node_weight[*node_it];
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
		    bucketA[index].end(), *node_it);
	    }
	}
	else	// side[*node_it] == B
	{
	    node_weight_on_sideB += node_weight[*node_it];
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
		    bucketB[index].end(), *node_it);
	    }
	}
	++node_it;
    }
}


int fm_partition::inital_gain_of_node_on_sideA(const node cur_node)
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


int fm_partition::inital_gain_of_node_on_sideB(const node cur_node)
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


void fm_partition::move_manager(const graph& G)
{
    int step_number = 0;
    int best_tentative_move = 0;
    int best_bal = node_weight_on_sideA * node_weight_on_sideB;
	std::vector<node> tentative_moves(G.number_of_nodes() + 1);
	std::vector<int> tentative_cutsize(G.number_of_nodes() + 1);
    node moved_node;
    tentative_cutsize[0] = cur_cutsize;

    while (move_vertex(G, moved_node))
    {
	++step_number;
	tentative_cutsize[step_number] = cur_cutsize;
	tentative_moves[step_number] = moved_node;
	if (tentative_cutsize[best_tentative_move] > cur_cutsize)
	{
	    best_tentative_move = step_number;
	    best_bal = node_weight_on_sideA * node_weight_on_sideB;
	}
	else if (tentative_cutsize[best_tentative_move] == cur_cutsize)
	{
	    if (node_weight_on_sideA * node_weight_on_sideB > best_bal)
	    {
		best_tentative_move = step_number;
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
    cur_cutsize = tentative_cutsize[best_tentative_move];
}


bool fm_partition::move_vertex(const graph& G, node& moved_node)
{
    node cons_nodeA;
    if (!bucketA_empty)
    {
	cons_nodeA = bucketA[range_up(max_gainA)].back();
    }
    node cons_nodeB;
    if (!bucketB_empty)
    {
	cons_nodeB = bucketB[range_up(max_gainB)].back();
    }
    if ((!bucketA_empty) && (!bucketB_empty) &&
	(balance_holds(G, cons_nodeA)) && (balance_holds(G, cons_nodeB)))
    {
	if (gain_value[cons_nodeA] > gain_value[cons_nodeB])
	{
	    update_data_structure_A2B(cons_nodeA);
	    moved_node = cons_nodeA;
	}
	else if (gain_value[cons_nodeB] > gain_value[cons_nodeA])
	{
	    update_data_structure_B2A(cons_nodeB);
	    moved_node = cons_nodeB;
	}
	else	// gain_value[cons_nodeB] == gain_value[cons_nodeA]
	{
	    int bal_diff_A2B = abs(node_weight_on_sideA - 2 *
		node_weight[cons_nodeA] - node_weight_on_sideB);
	    int bal_diff_B2A = abs(node_weight_on_sideB - 2 *
		node_weight[cons_nodeB] - node_weight_on_sideA);
	    if (bal_diff_A2B < bal_diff_B2A)
	    {
		update_data_structure_A2B(cons_nodeA);
		moved_node = cons_nodeA;
	    }
	    else if (bal_diff_B2A < bal_diff_A2B)
	    {
		update_data_structure_B2A(cons_nodeB);
		moved_node = cons_nodeB;
	    }
	    else	// break remaining ties as desired [FidMat82]
	    {
		update_data_structure_A2B(cons_nodeA);
		moved_node = cons_nodeA;
	    }
	}
    }
    else if ((!bucketA_empty) && (balance_holds(G, cons_nodeA)))
    {
	update_data_structure_A2B(cons_nodeA);
	moved_node = cons_nodeA;
    }
    else if ((!bucketB_empty) && (balance_holds(G, cons_nodeB)))
    {
	update_data_structure_B2A(cons_nodeB);
	moved_node = cons_nodeB;
    }
    else
    {
	return false;	// no more vertices can be moved
    }
    update_max_gain(A);
    update_max_gain(B);
    return true;
}


bool fm_partition::balance_holds(const graph& /*G*/, const node cur_node)
{
    if (side[cur_node] == A)
    {
	if ((double)node_weight_on_sideB + (double)node_weight[cur_node]
	    <= ((double)total_node_weight / 2.0) + (double)max_node_weight)
	{
	    return true;
	}
    }
    else	// side[cur_node] == B
    {
	if ((double)node_weight_on_sideA + (double)node_weight[cur_node]
	    <= ((double)total_node_weight / 2.0) + (double)max_node_weight)
	{
	    return true;
	}
    }
    return false;
}


void fm_partition::update_data_structure_A2B(const node cur_node)
{
    bucketA[range_up(max_gainA)].pop_back();
    node_weight_on_sideA -= node_weight[cur_node];
    node_weight_on_sideB += node_weight[cur_node];
    cur_cutsize -= gain_value[cur_node];
	
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
		    gain_value[*node_it] - edge_weight[*adj_edge_it]);
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
		    gain_value[*node_it] + edge_weight[*adj_edge_it]);
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
		    gain_value[*node_it] + edge_weight[*adj_edge_it]);
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
		    gain_value[*node_it] - edge_weight[*adj_edge_it]);
		gain_value[*node_it] -= edge_weight[*adj_edge_it];
		++node_it;
	    }
	}
	++adj_edge_it;
    }
}


void fm_partition::update_data_structure_B2A(const node cur_node)
{
    bucketB[range_up(max_gainB)].pop_back();
    node_weight_on_sideA += node_weight[cur_node];
    node_weight_on_sideB -= node_weight[cur_node];
    cur_cutsize -= gain_value[cur_node];
	
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
		    gain_value[*node_it] - edge_weight[*adj_edge_it]);
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
		    gain_value[*node_it] + edge_weight[*adj_edge_it]);
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
		    gain_value[*node_it] + edge_weight[*adj_edge_it]);
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
		    gain_value[*node_it] - edge_weight[*adj_edge_it]);
		gain_value[*node_it] -= edge_weight[*adj_edge_it];
		++node_it;
	    }
	}
	++adj_edge_it;
    }
}


void fm_partition::update_bucketA(const node cur_node, const int old_gain,
    const int new_gain)
{
    if (fixed[cur_node] != UNFIXED)
    {
	return;	// fixed nodes need no update
    }

    bucketA[range_up(old_gain)].erase(position_in_bucket[cur_node]);
    position_in_bucket[cur_node] = bucketA[range_up(new_gain)].insert(
	bucketA[range_up(new_gain)].end(), cur_node);

    if (max_gainA < new_gain)
    {
	max_gainA = new_gain;
    }
}


void fm_partition::update_bucketB(const node cur_node, const int old_gain,
    const int new_gain)
{
    if (fixed[cur_node] != UNFIXED)
    {
	return;	// fixed nodes need no update
    }

    bucketB[range_up(old_gain)].erase(position_in_bucket[cur_node]);
    position_in_bucket[cur_node] = bucketB[range_up(new_gain)].insert(
	bucketB[range_up(new_gain)].end(), cur_node);

    if (max_gainB < new_gain)
    {
	max_gainB = new_gain;
    }
}


void fm_partition::update_max_gain(const side_type side)
{
    if ((side == A) && (!bucketA_empty))
    {
	while (bucketA[range_up(max_gainA)].empty())
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
	while (bucketB[range_up(max_gainB)].empty())
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


inline int fm_partition::range_up(const int gain_value) const
{
    return gain_value + (max_vertex_degree * max_edge_weight);
}


inline int fm_partition::range_down(const int index) const
{
    return index - (max_vertex_degree * max_edge_weight);
}


void fm_partition::clean_pass(const graph& G)
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


void fm_partition::compute_cut_edges(const graph& G)
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


void fm_partition::compute_nodesAB(const graph& G)
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
void fm_partition::print_bucketA()
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


void fm_partition::print_bucketB()
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
