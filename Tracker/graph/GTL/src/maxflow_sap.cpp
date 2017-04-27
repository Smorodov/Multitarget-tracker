/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   maxflow_sap.cpp 
//
//==========================================================================
// $Id: maxflow_sap.cpp,v 1.6 2001/11/07 13:58:10 pick Exp $

#include <GTL/maxflow_sap.h>
#include <cstdlib>
#include <iostream>
#include <cassert>

__GTL_BEGIN_NAMESPACE

maxflow_sap::maxflow_sap()
{
    max_graph_flow = 0.0;
    set_vars_executed = false;
}


maxflow_sap::~maxflow_sap()
{
}


void maxflow_sap::set_vars(const edge_map<double>& edge_capacity)
{
    this->edge_capacity = edge_capacity;
    artif_source_target = true;
    max_graph_flow = 0.0;
    set_vars_executed = true;
}


void maxflow_sap::set_vars(const edge_map<double>& edge_capacity, 
    const node& net_source, const node& net_target)
{
    this->edge_capacity = edge_capacity;
    this->net_source = net_source;
    this->net_target = net_target;
    artif_source_target = false;
    max_graph_flow = 0.0;
    set_vars_executed = true;
}


int maxflow_sap::check(graph& G)
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
    if ((G.number_of_nodes() <= 1) || (!G.is_connected())
		|| (G.is_undirected()))
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
    return(GTL_OK);	// everything ok
}


int maxflow_sap::run(graph& G)
{
    // init
    if (artif_source_target)
    {
		create_artif_source_target(G);
    }
	bool go_on = true;
	node_map<edge> last_edge(G);
	node cur_node;
	int number_of_nodes = G.number_of_nodes();
	std::vector<int> numb(number_of_nodes, 0);
    prepare_run(G);

	comp_dist_labels(G, numb);
	cur_node = net_source;

	while (go_on)
	{
		if (has_an_admissible_arc(cur_node))
		{
			advance(cur_node, last_edge);
			if (cur_node == net_target)
			{
				augment(G, last_edge);
				cur_node = net_source;
			}
		}
		else	// only inadmissible edges
		{
			go_on = retreat(number_of_nodes, cur_node, last_edge, numb);
		}
	}

    restore_graph(G);
    return(GTL_OK);
}


double maxflow_sap::get_max_flow(const edge& e) const
{
    return(edge_max_flow[e]);
}


double maxflow_sap::get_max_flow() const
{
    return(max_graph_flow);
}


double maxflow_sap::get_rem_cap(const edge& e) const
{
    return(edge_capacity[e] - edge_max_flow[e]);
}


void maxflow_sap::reset()
{
    max_graph_flow = 0.0;
    set_vars_executed = false;
}


void maxflow_sap::create_artif_source_target(graph& G)
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
		    node::out_edges_iterator out_edge_it =
				node_it->out_edges_begin();
		    node::out_edges_iterator out_edges_end =
				node_it->out_edges_end();
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
		    node::in_edges_iterator in_edge_it =
				node_it->in_edges_begin();
			node::in_edges_iterator in_edges_end =
				node_it->in_edges_end();
		    while (in_edge_it != in_edges_end)
		    {
				edge_capacity[e] += edge_capacity[*in_edge_it];
				++in_edge_it;
		    }
		}
		++node_it;
    }
}


void maxflow_sap::prepare_run(const graph& G)
{
    edge_max_flow.init(G, 0.0);
    edge_org.init(G, true);
    back_edge_exists.init(G, false);
    max_graph_flow = 0.0;
}


void maxflow_sap::comp_dist_labels(const graph& G, std::vector<int>& numb)
{
	std::queue<node> next_nodes;
    node_map<bool> visited(G, false);
    next_nodes.push(net_target);
    visited[net_target] = true;
	dist_label[net_target] = 0;
	numb[0] = 1;	// only one sink
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
			    next_nodes.push(next);
				visited[next] = true;
				dist_label[next] = dist_label[cur_node] + 1;
				++numb[dist_label[next]];
			}
			++in_edge_it;
		}
	}
}


bool maxflow_sap::has_an_admissible_arc(const node cur_node)
{
	node::out_edges_iterator out_edge_it = cur_node.out_edges_begin();
	node::out_edges_iterator out_edges_end = cur_node.out_edges_end();
	while (out_edge_it != out_edges_end)
	{
		if (dist_label[cur_node] == dist_label[out_edge_it->target()] + 1)
		{
			return true;
		}
		++out_edge_it;
	}
	return false;
}


void maxflow_sap::advance(node& cur_node, node_map<edge>& last_edge)
{
	node::out_edges_iterator out_edge_it = cur_node.out_edges_begin();
	node::out_edges_iterator out_edges_end = cur_node.out_edges_end();
	while (out_edge_it != out_edges_end)
	{
		if (dist_label[cur_node] == dist_label[out_edge_it->target()] + 1)
		{
			last_edge[out_edge_it->target()] = *out_edge_it;
			cur_node = out_edge_it->target();
		}
		++out_edge_it;
	}
}


void maxflow_sap::augment(graph& G, const node_map<edge>& last_edge)
{
    double additional_flow = free_capacity(last_edge);
	node cur_node = net_target;

	do
	{
		if (edge_org[last_edge[cur_node]])
			// shortest path runs over a org. edge
		{
			if (!back_edge_exists[last_edge[cur_node]])	// create back edge
		    {
				create_back_edge(G, last_edge[cur_node]);
			}
		    edge_max_flow[last_edge[cur_node]] += additional_flow;
		    G.restore_edge(back_edge[last_edge[cur_node]]);
			edge_capacity[back_edge[last_edge[cur_node]]] +=
				additional_flow;
		}
		else	// shortest path runs over a inserted back edge
		{
		    edge oe = back_edge[last_edge[cur_node]];
			G.restore_edge(oe);
		    edge_max_flow[oe] -= additional_flow;
		    edge_capacity[last_edge[cur_node]] -= additional_flow;
		}
		if (edge_capacity[last_edge[cur_node]] <=
			edge_max_flow[last_edge[cur_node]])
		{
		    G.hide_edge(last_edge[cur_node]);
		}
		cur_node = last_edge[cur_node].source();
	}
	while (cur_node != net_source);
}


bool maxflow_sap::retreat(const int number_of_nodes,
						  node& cur_node,
						  const node_map<edge>& last_edge,
						  std::vector<int>& numb)
{
	--numb[dist_label[cur_node]];
	if (numb[dist_label[cur_node]] == 0)
	{
		return false;
	}
	else
	{
		dist_label[cur_node] =
			min_neighbour_label(number_of_nodes, cur_node) + 1;
		++numb[dist_label[cur_node]];
		if (cur_node != net_source)
		{
			cur_node = last_edge[cur_node].source();
		}
		return true;
	}
}


int maxflow_sap::min_neighbour_label(const int number_of_nodes,
									 const node cur_node) const
{
	int min_value = number_of_nodes;	// if no out edge exists

	node::out_edges_iterator out_edge_it = cur_node.out_edges_begin();
    node::out_edges_iterator out_edges_end = cur_node.out_edges_end();
    while (out_edge_it != out_edges_end)
    {
		if (min_value > dist_label[out_edge_it->target()])
		{
			min_value = dist_label[out_edge_it->target()];
		}
		++out_edge_it;
    }
	return min_value;
}


double maxflow_sap::free_capacity(const node_map<edge>& last_edge) const
{
    node cur_node = net_target;
    double min_value = 
		edge_capacity[last_edge[cur_node]] -
		edge_max_flow[last_edge[cur_node]];
    double cur_capacity;

    do
    {
		cur_capacity = 
			edge_capacity[last_edge[cur_node]] -
			edge_max_flow[last_edge[cur_node]];

		if (cur_capacity < min_value)
		{
			min_value = cur_capacity;
		}
		cur_node = last_edge[cur_node].source();
    }
    while (cur_node != net_source);
    return(min_value);
}


void maxflow_sap::create_back_edge(graph& G, const edge& org_edge)
{
    edge be = G.new_edge(org_edge.target(), org_edge.source());
    edge_org[be] = false;
    edges_not_org.push_back(be);
    back_edge[org_edge] = be;
    back_edge[be] = org_edge;
    edge_max_flow[be] = 0.0;
    edge_capacity[be] = 0.0;
    back_edge_exists[org_edge] = true;
    back_edge_exists[be] = true;	// a back edge always has a org. edge
}


void maxflow_sap::comp_max_flow(const graph& /*G*/)
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


void maxflow_sap::restore_graph(graph& G)
{
    G.restore_graph();	// hidden edges can not be deleted!
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
