/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   biconnectivity.cpp
//
//==========================================================================
// $Id: biconnectivity.cpp,v 1.20 2002/02/28 15:40:52 raitner Exp $

#include <GTL/biconnectivity.h>

__GTL_BEGIN_NAMESPACE

biconnectivity::biconnectivity() : dfs()
{
	add_edges = false;
	store_preds(true);
	store_comp = false;
	scan_whole_graph(true);
	num_of_components = 0;
}

void biconnectivity::reset()
{
	dfs::reset();

	if (store_comp) {
		while (!node_stack.empty()) {
			node_stack.pop();
		}

		while (!edge_stack.empty()) {
			edge_stack.pop();
		}

		components.erase(components.begin(), components.end());
	}

	if (add_edges) {
		additional.erase(additional.begin(), additional.end());
	}

	cut_points.erase(cut_points.begin(), cut_points.end());
	num_of_components = 0;
}

int biconnectivity::check(graph& G)
{
	return G.is_undirected() && preds &&
		dfs::check(G) == GTL_OK ? GTL_OK : GTL_ERROR;
}


//--------------------------------------------------------------------------
//   Handler
//--------------------------------------------------------------------------


void biconnectivity::init_handler(graph& G)
{
	if (add_edges) {
		dfs D;
		D.scan_whole_graph(true);
		D.check(G);
		D.run(G);

		roots_iterator it, end;
		it = D.roots_begin();
		end = D.roots_end();
		start = *(*it);
		++it;

		for (; it != end; ++it) {
			additional.push_back(G.new_edge(start, *(*it)));
		}

		first_child.init(G, node());
	}

	low_num.init(G);
	in_component.init(G);
	cut_count.init(G, 0);

	//
	// Detect self loops and hide them.
	// 

	assert(self_loops.empty());
	graph::edge_iterator eit = G.edges_begin(),
		eend = G.edges_end();

	while (eit != eend) {
		edge e = *eit;
		eit++;
		if (e.target() == e.source()) {
			self_loops.push_back(e);
			G.hide_edge(e);
		}
	}
}

void biconnectivity::entry_handler(graph& /*G*/, node& curr, node& father)
{
	if (add_edges) {
		if (father != node()) {
			if (first_child[father] == node()) {
				first_child[father] = curr;
			}
		}
	}

	low_num[curr] = dfs_number[curr];
}

void biconnectivity::new_start_handler(graph& /*G*/, node& st)
{
	cut_count[st] = -1;

	//
	// If this node has no adjacent edges, we
	// must write down the component right here. This is because
	// then the method after_recursive_call_handle is never
	// executed.
	//
	// 28/2/2002 MR
	// 

	if (st.degree() == 0) {
		++num_of_components;

		if (store_comp) {
			component_iterator li = components.insert(
				components.end(),
				std::pair<nodes_t, edges_t>(nodes_t(), edges_t()));

			li->first.push_back(st);
			in_component[st] = li;
		}
	}
}

void biconnectivity::before_recursive_call_handler(graph& /*G*/, edge& /*e*/, node& n)
{
	if (store_comp) {
		node_stack.push(n);
	}
}


void biconnectivity::after_recursive_call_handler(graph& G, edge& e, node& n)
{
	node curr = n.opposite(e);

	if (low_num[n] < low_num[curr]) {
		low_num[curr] = low_num[n];
	}

	if (low_num[n] >= dfs_num(curr)) {
		//
		// Component found
		// 

		if (store_comp) {
			component_iterator li = components.insert(
				components.end(),
				std::pair<nodes_t, edges_t>(nodes_t(), edges_t()));

			nodes_t& component = li->first;
			edges_t& co_edges = li->second;

			//
			// Nodes of biconnected component
			// 

			node tmp = node_stack.top();

			while (dfs_num(tmp) >= dfs_num(n)) {
				node_stack.pop();
				component.push_back(tmp);
				in_component[tmp] = li;
				if (node_stack.empty()) break;
				else tmp = node_stack.top();
			}

			component.push_back(curr);

			//
			// edges of biconnected component
			//

			edge ed = edge_stack.top();

			while ((dfs_num(ed.source()) >= dfs_num(n) &&
				dfs_num(ed.target()) >= dfs_num(n)) ||
				(dfs_num(ed.source()) == dfs_num(curr) &&
				dfs_num(ed.target()) >= dfs_num(n)) ||
				(dfs_num(ed.source()) >= dfs_num(n) &&
				dfs_num(ed.target()) == dfs_num(curr))) {
				edge_stack.pop();
				co_edges.push_back(ed);
				if (edge_stack.empty()) break;
				else ed = edge_stack.top();
			}
		}


		++num_of_components;

		//
		// curr is cut point; increase counter
		// 

		++cut_count[curr];

		if (add_edges) {
			node father = (*preds)[curr];
			node first = first_child[curr];

			if (father != node() && n == first) {
				additional.push_back(G.new_edge(father, first));
			}

			if (n != first) {
				additional.push_back(G.new_edge(n, first));
			}
		}

	}
}

void biconnectivity::old_adj_node_handler(graph& /*G*/, edge& e, node& n)
{
	node curr = n.opposite(e);

	//
	// Store backedges at lower endpoint
	//

	if (store_comp) {
		if (dfs_num(curr) > dfs_num(n)) {
			edge_stack.push(e);
		}
	}

	if (dfs_num(n) < low_num[curr]) {
		low_num[curr] = dfs_number[n];
	}
}

void biconnectivity::leave_handler(graph& /*G*/, node& n, node& /*f*/)
{
	if (cut_count[n] > 0)
	{
		cut_points.push_back(n);
	}
}

void biconnectivity::end_handler(graph& G)
{
	edges_t::iterator it = self_loops.begin();
	edges_t::iterator end = self_loops.end();

	while (it != end)
	{
		G.restore_edge(*it);
		if (store_comp)
		{
			component_iterator cit = in_component[it->target()];
			cit->second.push_back(*it);
		}

		it = self_loops.erase(it);
	}
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
