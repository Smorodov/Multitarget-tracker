/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   maxflow_pp.cpp 
//
//==========================================================================
// $Id: maxflow_pp.cpp,v 1.7 2001/11/07 13:58:10 pick Exp $

#include <GTL/maxflow_pp.h>

#include <cstdlib>
#include <iostream>
#include <cassert>

__GTL_BEGIN_NAMESPACE

maxflow_pp::maxflow_pp()
{
    max_graph_flow = 0.0;
    set_vars_executed = false;
}


maxflow_pp::~maxflow_pp()
{
}


void maxflow_pp::set_vars(const edge_map<double>& edge_capacity)
{
    this->edge_capacity = edge_capacity;
    artif_source_target = true;
    max_graph_flow = 0.0;
    set_vars_executed = true;
}


void maxflow_pp::set_vars(const edge_map<double>& edge_capacity, 
    const node& net_source, const node& net_target)
{
    this->edge_capacity = edge_capacity;
    this->net_source = net_source;
    this->net_target = net_target;
    artif_source_target = false;
    max_graph_flow = 0.0;
    set_vars_executed = true;
}


int maxflow_pp::check(graph& G)
{
    if (!set_vars_executed)
    {
	return(GTL_ERROR);
    }
    graph::edge_iterator edge_it = G.edges_begin();
    graph::edge_iterator edges_end = G.edges_end();
    while (edge_it != edges_end)
    {
	if (edge_capacity[*edge_it] < 0)
	{
	    return(GTL_ERROR);
	}
	++edge_it;
    }
    // G.is_acyclic may be false
    if ((G.number_of_nodes() <= 1) || (!G.is_connected()) || (G.is_undirected()))
    {
	return(GTL_ERROR);
    }
    if (artif_source_target)
    {
	bool source_found = false;
	bool target_found = false;
	graph::node_iterator node_it = G.nodes_begin();
	graph::node_iterator nodes_end = G.nodes_end();
	while (node_it != nodes_end)
	{
	    if (node_it->indeg() == 0)
	    {
		source_found = true;
	    }
	    if (node_it->outdeg() == 0)
	    {
		target_found = true;
	    }
	    ++node_it;
	}
	if (!(source_found && target_found))
	{
	    return(GTL_ERROR);
	}
    }
    else
    {
	if (net_source == net_target)
	{
	    return(GTL_ERROR);
	}
    }
    return(GTL_OK);	// ok
}


int maxflow_pp::run(graph& G)
{
    // init
    if (artif_source_target)
    {
	create_artif_source_target(G);
    }
    prepare_run(G);

    double flow_value = 0;
    node min_tp_node;
    while(leveling(G) == TARGET_FROM_SOURCE_REACHABLE)
    {
	hide_unreachable_nodes(G);
	min_throughput_node(G, min_tp_node, flow_value);
	push(G, min_tp_node, flow_value);
	pull(G, min_tp_node, flow_value);
	comp_rem_net(G);
    }

    restore_graph(G);
    return(GTL_OK);
}


double maxflow_pp::get_max_flow(const edge& e) const
{
    return(edge_max_flow[e]);
}


double maxflow_pp::get_max_flow() const
{
    return(max_graph_flow);
}


double maxflow_pp::get_rem_cap(const edge& e) const
{
    return(edge_capacity[e] - edge_max_flow[e]);
}


void maxflow_pp::reset()
{
}


void maxflow_pp::create_artif_source_target(graph& G)
{
    net_source = G.new_node();
    net_target = G.new_node();
    edge e;
    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if (*node_it != net_source && node_it->indeg() == 0)
	{
	    e = G.new_edge(net_source, *node_it);
	    edge_capacity[e] = 1.0;	// 1.0 prevents e from hiding
	    node::out_edges_iterator out_edge_it = node_it->out_edges_begin();
	    node::out_edges_iterator out_edges_end = node_it->out_edges_end();
	    while (out_edge_it != out_edges_end)
	    {
		edge_capacity[e] += edge_capacity[*out_edge_it];
		++out_edge_it;
	    }
	}
	if (*node_it != net_target && node_it->outdeg() == 0)
	{
	    e = G.new_edge(*node_it, net_target);
	    edge_capacity[e] = 1.0;	// 1.0 prevents e from hiding
	    node::in_edges_iterator in_edge_it = node_it->in_edges_begin();
	    node::in_edges_iterator in_edges_end = node_it->in_edges_end();
	    while (in_edge_it != in_edges_end)
	    {
		edge_capacity[e] += edge_capacity[*in_edge_it];
		++in_edge_it;
	    }
	}
	++node_it;
    }
}


void maxflow_pp::prepare_run(const graph& G)
{
    flow_update.init(G, 0.0);
    edge_max_flow.init(G, 0.0);
    edge_org.init(G, true);
    back_edge_exists.init(G, false);
    max_graph_flow = 0.0;
    full_edges.clear();
    temp_unvisible_nodes.clear();
    temp_unvisible_edges.clear();
}


int maxflow_pp::leveling(graph& G)
{
    bool source_target_con = false;
    node_map<int> level(G, -1);	// -1 means no level yet!
	std::queue<node> next_nodes;
    next_nodes.push(net_source);
    level[net_source] = 0;
    node cur_node;
    while (!next_nodes.empty())
    {
	cur_node = next_nodes.front();
	next_nodes.pop();
	node::out_edges_iterator out_edge_it = cur_node.out_edges_begin();
	node::out_edges_iterator out_edges_end = cur_node.out_edges_end();
	while (out_edge_it != out_edges_end)
	{
	    if (level[out_edge_it->target()] == -1)
	    {
		if (out_edge_it->target() == net_target)
		{
		    source_target_con = true;
		}
		level[out_edge_it->target()] = level[cur_node] + 1;
		next_nodes.push(out_edge_it->target());
		++out_edge_it;
	    }
	    else if (level[out_edge_it->target()] <= level[cur_node])
	    {
		node::out_edges_iterator temp_it = out_edge_it;
		++out_edge_it;
		temp_unvisible_edges.push_back(*temp_it);
		G.hide_edge(*temp_it);
	    }
	    else
	    {
		++out_edge_it;
	    }
	}
    }
    if (source_target_con)
    {
	return(TARGET_FROM_SOURCE_REACHABLE);
    }
    else
    {
	return(TARGET_FROM_SOURCE_NOT_REACHABLE);
    }
}


void maxflow_pp::hide_unreachable_nodes(graph& G)
{
    node_map<bool> reachable_from_net_source(G, false);
    node_map<bool> reachable_from_net_target(G, false);
	std::queue<node> next_nodes;
    node cur_node;

    next_nodes.push(net_source);
    reachable_from_net_source[net_source] = true;
    while (!next_nodes.empty())
    {
	cur_node = next_nodes.front();
	next_nodes.pop();
	node::out_edges_iterator out_edge_it = cur_node.out_edges_begin();
	node::out_edges_iterator out_edges_end = cur_node.out_edges_end();
	while (out_edge_it != out_edges_end)
	{
	    node next = out_edge_it->target();
	    if (!reachable_from_net_source[next])
	    {
		next_nodes.push(next);
		reachable_from_net_source[next] = true;
	    }
	    ++out_edge_it;
	}
    }

    next_nodes.push(net_target);
    reachable_from_net_target[net_target] = true;
    while (!next_nodes.empty())
    {
	cur_node = next_nodes.front();
	next_nodes.pop();
	node::in_edges_iterator in_edge_it = cur_node.in_edges_begin();
	node::in_edges_iterator in_edges_end = cur_node.in_edges_end();
	while (in_edge_it != in_edges_end)
	{
	    node next = in_edge_it->source();
	    if (!reachable_from_net_target[next])
	    {
		next_nodes.push(next);
		reachable_from_net_target[next] = true;
	    }
	    ++in_edge_it;
	}
    }

    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    while (node_it != nodes_end)
    {
	if ((!reachable_from_net_source[*node_it]) || 
	    (!reachable_from_net_target[*node_it]))
	{
	    graph::node_iterator temp_it = node_it;
	    ++node_it;
	    temp_unvisible_nodes.push_back(*temp_it);
	    store_temp_unvisible_edges(*temp_it);
	    G.hide_node(*temp_it);
	}
	else
	{
	    ++node_it;
	}
    }
}


void maxflow_pp::store_temp_unvisible_edges(const node& cur_node)
{
    node::in_edges_iterator in_it = cur_node.in_edges_begin();
    node::in_edges_iterator in_edges_end = cur_node.in_edges_end();
    while (in_it != in_edges_end)
    {
	temp_unvisible_edges.push_back(*in_it);
	++in_it;
    }
    node::out_edges_iterator out_it = cur_node.out_edges_begin();
    node::out_edges_iterator out_edges_end = cur_node.out_edges_end();
    while (out_it != out_edges_end)
    {
	temp_unvisible_edges.push_back(*out_it);
	++out_it;
    }
}


void maxflow_pp::min_throughput_node(const graph& G, node& min_tp_node, 
    double& flow_value)
{
    min_tp_node = net_source;
    flow_value = comp_min_throughput(min_tp_node);

    graph::node_iterator node_it = G.nodes_begin();
    graph::node_iterator nodes_end = G.nodes_end();
    double cur_tp;
    while (node_it != nodes_end)
    {
	cur_tp = comp_min_throughput(*node_it);
	if (cur_tp < flow_value)
	{
	    min_tp_node = *node_it;
	    flow_value = cur_tp;
	}
	++node_it;
    }
}


double maxflow_pp::comp_min_throughput(const node cur_node) const
{
    double in_flow = 0.0;
    double out_flow = 0.0;
    node::in_edges_iterator in_it = cur_node.in_edges_begin();
    node::in_edges_iterator in_edges_end = cur_node.in_edges_end();
    while (in_it != in_edges_end)
    {
	in_flow += edge_capacity[*in_it] - edge_max_flow[*in_it];
	++in_it;
    }
    node::out_edges_iterator out_it = cur_node.out_edges_begin();
    node::out_edges_iterator out_edges_end = cur_node.out_edges_end();
    while (out_it != out_edges_end)
    {
	out_flow += edge_capacity[*out_it] - edge_max_flow[*out_it];
	++out_it;
    }
    if (cur_node == net_source)
    {
	return(out_flow);
    }
    if (cur_node == net_target)
    {
	return(in_flow);
    }
    return(in_flow < out_flow ? in_flow : out_flow);
}


void maxflow_pp::get_sp_ahead(const graph& G, const node& start_node, 
    node_map<edge>& last_edge)
{
	std::queue<node> next_nodes;
    node_map<bool> visited(G, false);
    next_nodes.push(start_node);
    visited[start_node] = true;

    node cur_node;
    while (!next_nodes.empty())
    {
	cur_node = next_nodes.front();
	next_nodes.pop();
		
	node::out_edges_iterator out_edge_it = cur_node.out_edges_begin();
	node::out_edges_iterator out_edges_end = cur_node.out_edges_end();
	while (out_edge_it != out_edges_end)
	{
	    node next = out_edge_it->target();
	    if (!visited[next])
	    {
		last_edge[next] = *out_edge_it;
		if (next == net_target)
		{
		    return;	// sp found
		}
		next_nodes.push(next);
		visited[next] = true;
	    }
	    ++out_edge_it;
	}
    }
}


void maxflow_pp::get_sp_backwards(const graph& G, const node& start_node, 
    node_map<edge>& prev_edge)
{
    std::queue<node> next_nodes;
    node_map<bool> visited(G, false);
    next_nodes.push(start_node);
    visited[start_node] = true;

    node cur_node;
    while (!next_nodes.empty())
    {
	cur_node = next_nodes.front();
	next_nodes.pop();
		
	node::in_edges_iterator in_edge_it = cur_node.in_edges_begin();
	node::in_edges_iterator in_edges_end = cur_node.in_edges_end();
	while (in_edge_it != in_edges_end)
	{
	    node next = in_edge_it->source();
	    if (!visited[next])
	    {
		prev_edge[next] = *in_edge_it;
		if (next == net_source) 
		{
		    return;	// sp found
		}
		next_nodes.push(next);
		visited[next] = true;
	    }
	    ++in_edge_it;
	}
    }
}


void maxflow_pp::push(graph& G, const node& start_node, const double flow_value)
{
    node_map<edge> last_edge;
    double cur_flow = flow_value;
    double min_value = 0.0;

    if (start_node == net_target)
    {
	return;	// no push necessary
    }
    do
    {
	get_sp_ahead(G, start_node, last_edge);
	min_value = extra_charge_ahead(start_node, last_edge);
	if (min_value > cur_flow)
	{
	    min_value = cur_flow;
	}
	node cur_node = net_target;
	do
	{
	    if (edge_org[last_edge[cur_node]])
	    {
		edge_max_flow[last_edge[cur_node]] += min_value;
		if (back_edge_exists[last_edge[cur_node]])
		{
		    flow_update[back_edge[last_edge[cur_node]]] += min_value;
		}
	    }
	    else
	    {
		edge_capacity[last_edge[cur_node]] -= min_value;
		flow_update[back_edge[last_edge[cur_node]]] += min_value;
	    }
	    if (edge_capacity[last_edge[cur_node]] <= 
		edge_max_flow[last_edge[cur_node]])
	    {
		full_edges.push_back(last_edge[cur_node]);
		G.hide_edge(last_edge[cur_node]);
	    }
	    cur_node = last_edge[cur_node].source();
	}
	while (cur_node != start_node);
	cur_flow -= min_value;
	if (cur_flow < 1e-015)	// quite hacky ;-)
	{
	    cur_flow = 0.0;	// to avoid rounding errors
	}
    } while (cur_flow > 0.0);
}


void maxflow_pp::pull(graph& G, const node& start_node, const double flow_value)
{
    node_map<edge> prev_edge;
    double cur_flow = flow_value;
    double min_value = 0.0;
	
    if (start_node == net_source)
    {
	return;	// pull not necessary
    }
    do
    {
	get_sp_backwards(G, start_node, prev_edge);
	min_value = extra_charge_backwards(start_node, prev_edge);
	if (min_value > cur_flow)
	{
	    min_value = cur_flow;
	}
	node cur_node = net_source;
	do
	{
	    if (edge_org[prev_edge[cur_node]])
	    {
		edge_max_flow[prev_edge[cur_node]] += min_value;
		if (back_edge_exists[prev_edge[cur_node]])
		{
		    flow_update[back_edge[prev_edge[cur_node]]] += min_value;
		}
	    }
	    else
	    {
		edge_capacity[prev_edge[cur_node]] -= min_value;
		flow_update[back_edge[prev_edge[cur_node]]] += min_value;
	    }
	    if (edge_capacity[prev_edge[cur_node]] <= 
		edge_max_flow[prev_edge[cur_node]])
	    {
		full_edges.push_back(prev_edge[cur_node]);
		G.hide_edge(prev_edge[cur_node]);
	    }
	    cur_node = prev_edge[cur_node].target();
	}
	while (cur_node != start_node);
	cur_flow -= min_value;
	if (cur_flow < 1e-015)	// quite hacky ;-)
	{
	    cur_flow = 0.0;	// to avoid rounding errors
	}
    } while (cur_flow > 0.0);
}


void maxflow_pp::comp_rem_net(graph& G)
{
    // update back_edges
    graph::edge_iterator edge_it = G.edges_begin();
    graph::edge_iterator edges_end = G.edges_end();
    while (edge_it != edges_end)
    {
	single_edge_update(G, *edge_it);
	++edge_it;
    }
	edges_t::iterator list_it = full_edges.begin();
	edges_t::iterator list_end = full_edges.end();
    while (list_it != list_end)
    {
	G.restore_edge(*list_it);
	if (flow_update[*list_it] > 0.0)
	{
	    single_edge_update(G, *list_it);
		edges_t::iterator temp_it = list_it;
	    ++list_it;
	    full_edges.erase(temp_it);	// now it's visible again
	}
	else
	{
	    if (!back_edge_exists[*list_it])
	    {
		create_back_edge(G, *list_it);
		edge_capacity[back_edge[*list_it]] = edge_max_flow[*list_it];
	    }
	    G.hide_edge(*list_it);
	    ++list_it;
	}
    }

	
    // make hidden levels visible again
	nodes_t::iterator temp_un_node_it = temp_unvisible_nodes.begin();
	nodes_t::iterator temp_un_nodes_end = temp_unvisible_nodes.end();
    while (temp_un_node_it != temp_un_nodes_end)
    {
	G.restore_node(*temp_un_node_it);
	++temp_un_node_it;
    }
	edges_t::iterator temp_un_edge_it = temp_unvisible_edges.begin();
	edges_t::iterator temp_un_edges_end = temp_unvisible_edges.end();
    while (temp_un_edge_it != temp_un_edges_end)
    {
	G.restore_edge(*temp_un_edge_it);
	if (flow_update[*temp_un_edge_it] > 0.0)
	{
	    single_edge_update(G, *temp_un_edge_it);
	}
	++temp_un_edge_it;
    }
    temp_unvisible_nodes.clear();
    temp_unvisible_edges.clear();
}


void maxflow_pp::single_edge_update(graph& G, edge cur_edge)
{
    if(edge_org[cur_edge])
    {
	edge_max_flow[cur_edge] -= flow_update[cur_edge];
	flow_update[cur_edge] = 0.0;
	if (!back_edge_exists[cur_edge])
	{
	    if (edge_max_flow[cur_edge] > 0.0)
	    {
		create_back_edge(G, cur_edge);
		edge_capacity[back_edge[cur_edge]] = edge_max_flow[cur_edge];
	    }
	}
    }
    else
    {
	edge_capacity[cur_edge] += flow_update[cur_edge];
	flow_update[cur_edge] = 0.0;
    }
}


double maxflow_pp::extra_charge_ahead(const node& start_node, 
    const node_map<edge>& last_edge) const
{
    node cur_node = net_target;
    double min_value = edge_capacity[last_edge[cur_node]] 
	- edge_max_flow[last_edge[cur_node]];
    double cur_capacity;

    do
    {
	cur_capacity = edge_capacity[last_edge[cur_node]] 
	    - edge_max_flow[last_edge[cur_node]];
	if (cur_capacity < min_value) min_value = cur_capacity;
	cur_node = last_edge[cur_node].source();
    }
    while (cur_node != start_node);
    return(min_value);
}


double maxflow_pp::extra_charge_backwards(const node& start_node, const node_map<edge>& prev_edge) const
{
    node cur_node = net_source;
    double min_value = edge_capacity[prev_edge[cur_node]] 
	- edge_max_flow[prev_edge[cur_node]];
    double cur_capacity;

    do
    {
	cur_capacity = edge_capacity[prev_edge[cur_node]] 
	    - edge_max_flow[prev_edge[cur_node]];
	if (cur_capacity < min_value) min_value = cur_capacity;
	cur_node = prev_edge[cur_node].target();
    }
    while (cur_node != start_node);
    return(min_value);
}


void maxflow_pp::create_back_edge(graph& G, const edge& org_edge)
{
    edge be = G.new_edge(org_edge.target(), org_edge.source());
    edge_org[be] = false;
    edges_not_org.push_back(be);
    back_edge[org_edge] = be;
    back_edge[be] = org_edge;
    edge_max_flow[be] = 0.0;
    edge_capacity[be] = 0.0;
    back_edge_exists[org_edge] = true;
    back_edge_exists[be] = true;	// a back edge always has a org. edge ;-)
    flow_update[be] = 0.0;
}


void maxflow_pp::comp_max_flow(const graph& /*G*/)
{
    max_graph_flow = 0.0;

    node::out_edges_iterator out_edge_it = net_source.out_edges_begin();
    node::out_edges_iterator out_edges_end = net_source.out_edges_end();
    while (out_edge_it != out_edges_end)
    {
	max_graph_flow += edge_max_flow[*out_edge_it];
	++out_edge_it;
    }
}


void maxflow_pp::restore_graph(graph& G)
{
    G.restore_graph();
    while (!edges_not_org.empty())
    {
	G.del_edge(edges_not_org.front());
	edges_not_org.pop_front();
    }
    comp_max_flow(G);
    if (artif_source_target)
    {
	G.del_node(net_source);
	G.del_node(net_target);
    }
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
