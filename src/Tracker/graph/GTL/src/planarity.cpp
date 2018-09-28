/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   planarity.cpp 
//
//==========================================================================
// $Id: planarity.cpp,v 1.28 2008/02/03 18:12:07 chris Exp $

#include <GTL/planarity.h>
#include <GTL/pq_tree.h>
#include <GTL/biconnectivity.h>
#include <GTL/debug.h>

#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>

__GTL_BEGIN_NAMESPACE

//--------------------------------------------------------------------------
//   Planarity Test
//--------------------------------------------------------------------------


planarity::planarity() : 
	algorithm (), emp (false), kup (false), bip (true) 
{
#ifdef _DEBUG  
    GTL_debug::init_debug();
#endif
}

planarity::~planarity()
{
#ifdef _DEBUG  
    GTL_debug::close_debug();
#endif
}


int planarity::check (graph& /*G*/) 
{
    return algorithm::GTL_OK;
}

bool planarity::run_on_biconnected (graph& G, planar_embedding& em)
{

    if (G.number_of_edges() == 0) return algorithm::GTL_OK;
	
    st_number st_;
    
    //
    // The graph may have self loops. Make sure that we 
    // choose a normal edge for st.
    //

    graph::edge_iterator 
	edge_it = G.edges_begin(), 
	edge_end = G.edges_end();

    edge st;

    while (edge_it != edge_end) {
	if (edge_it->source() != edge_it->target()) {
	    st = *edge_it;
	    break;
	}
	++edge_it;
    }

    //
    // G has only selfloops
    //

    if (st == edge()) {
	if (emp) {
	    em.init (G);
	    edge_it = G.edges_begin();
	    edge_end = G.edges_end();
	
	    for (;edge_it != edge_end; ++edge_it) {
		em.self.push_back (*edge_it);
	    }
	}

	return algorithm::GTL_OK;
    }

    st_.st_edge (st);
    st_.s_node (st.source());
    int res = st_.check(G);
    assert (res == algorithm::GTL_OK);
    res = st_.run(G);
    assert (res == algorithm::GTL_OK);
    int size = G.number_of_nodes();

    if (emp) {
	em.init (G);
    }
	
    std::list<pq_leaf*> neighbors;
    st_number::iterator st_it = st_.begin();
    node curr = *st_it;
    node::out_edges_iterator o_it = curr.out_edges_begin();
    node::out_edges_iterator o_end = curr.out_edges_end();
    node::in_edges_iterator i_it = curr.in_edges_begin();
    node::in_edges_iterator i_end = curr.in_edges_end();
	edges_t self_loops;
    node opp;
    node_map<int> visited_from (G, 0);
    pq_leaf* tmp_leaf;
	std::vector< std::list<pq_leaf*> > leaves(size);
    
    for (; o_it != o_end; ++o_it) {
	opp = curr.opposite (*o_it);

	if (opp != curr) {
	    if (visited_from[opp] == st_[curr] && emp) {
		em.multi.push_back (*o_it);
	    } else {
		visited_from[opp] = st_[curr];
		tmp_leaf = new pq_leaf (st_[opp], st_[curr], *o_it, opp);
		leaves[st_[opp]-1].push_back (tmp_leaf);
		neighbors.push_back (tmp_leaf);
	    }

	} else if (emp) {
	    em.self.push_back (*o_it);
	}
    }
	
    for (; i_it != i_end; ++i_it) {
	opp = curr.opposite (*i_it);

	if (opp != curr) {
	    if (visited_from[opp] == st_[curr] && emp) {
		em.multi.push_back (*i_it);
	    } else {
		visited_from[opp] = st_[curr];
		tmp_leaf = new pq_leaf (st_[opp], st_[curr], *i_it, opp);
		leaves[st_[opp]-1].push_back (tmp_leaf);
		neighbors.push_back (tmp_leaf);
	    }
	} 
    }

    node_map<std::list<direction_indicator> > dirs;

    //
    // There is a problem with node/edge maps of iterators with Visual C++
    // which I don´t fully understand at the moment. Fortunatly the init for the 
    // maps below is only needed to allocate memory, which is done anyway, when
    // values are assigned to it.
    //

#ifndef __GTL_MSVCC
    dirs.init (G);
#endif
    pq_tree PQ (st_[curr], curr, neighbors);
    neighbors.erase (neighbors.begin(), neighbors.end());
    ++st_it;
    curr = *st_it;
	
    while (st_[curr] < size) {
		
#ifdef _DEBUG
	char filename[10] = "out";
	char buffer[12];
#ifdef __GTL_MSVCC
	_snprintf (buffer, 12, "%s%d.gml", filename, st_[curr]);
#else 
	snprintf (buffer, 12, "%s%d.gml", filename, st_[curr]);
#endif
	std::ofstream os(buffer, std::ios::out | std::ios::trunc);
	os << PQ << std::endl;
	os.close();
	bool ret_flag = PQ.integrity_check();
	assert(ret_flag);
#endif
		
	if (!PQ.reduce (leaves[st_[curr]-1])) {
#ifdef _DEBUG
		os.open("fail.gml", std::ios::out | std::ios::trunc);
		os << PQ << std::endl;
	    os.close ();
#endif	    	
	    if (kup) {
		examine_obstruction (G, st_, curr, 
		    PQ.get_fail(), PQ.is_fail_root(), em, dirs, &PQ);
	    }

	    PQ.reset();
	    return false;
	}

	
	//
	// It seems to be not very comfortable to use in and out iterators to
	// go through the adjacency of a node. For graphs without selfloops this
	// could be replaced by using adj_iterator, but if there are selfloops 
	// they will occur in both the list of outgoing and the one of incoming 
	// edges, and thus two times in the adjacency.
	//

	o_it = curr.out_edges_begin();
	o_end = curr.out_edges_end();
	i_it = curr.in_edges_begin();
	i_end = curr.in_edges_end();
		
	for (; o_it != o_end; ++o_it) {
	    opp = curr.opposite (*o_it);
			
	    if (st_[opp] > st_[curr]) {
		if (visited_from[opp] == st_[curr] && emp) {
		    em.multi.push_back (*o_it);
		} else {
		    visited_from[opp] = st_[curr];
		    tmp_leaf = new pq_leaf (st_[opp], st_[curr], *o_it, opp);
		    leaves[st_[opp]-1].push_back (tmp_leaf);
		    neighbors.push_back (tmp_leaf);
		}

	    } else if (st_[opp] == st_[curr] && emp) {
		em.self.push_back (*o_it);
	    }
	}
	
	for (; i_it != i_end; ++i_it) {
	    opp = curr.opposite (*i_it);
			
	    if (st_[opp] > st_[curr]) {
		if (visited_from[opp] == st_[curr] && emp) {
		    em.multi.push_back (*i_it);
		} else {
		    visited_from[opp] = st_[curr];
		    tmp_leaf = new pq_leaf (st_[opp], st_[curr], *i_it, opp);
		    leaves[st_[opp]-1].push_back (tmp_leaf);
		    neighbors.push_back (tmp_leaf);
		}
	    }
	}
		
	if (emp) {
	    PQ.replace_pert (st_[curr], curr, neighbors, &em, &(dirs[curr]));
#ifdef _DEBUG
	    GTL_debug::os() << "Embedding of " << st_[curr] << ":: ";
	    planar_embedding::iterator adit, adend;
	    for (adit = em.adj_edges_begin (curr), adend = em.adj_edges_end (curr); adit != adend; ++adit) {
		GTL_debug::os() << "[" << st_[curr.opposite (*adit)] << "]";
	    }
		GTL_debug::os() << std::endl;
	    GTL_debug::os() << "Direction Indicators for: " << st_[curr] << ":: ";
	    std::list<direction_indicator>::iterator dit, dend;

	    for (dit = dirs[curr].begin(), dend = dirs[curr].end(); dit != dend; ++dit) {
		GTL_debug::os() << "[";
		if (dit->direction)
		    GTL_debug::os() << ">> " << dit->id << " >>";
		else 
		    GTL_debug::os() << "<< " << dit->id << " <<";
		GTL_debug::os() << "]";
	    }
		GTL_debug::os() << std::endl;
#endif

	} else {
	    PQ.replace_pert (st_[curr], curr, neighbors);	    
	}

	PQ.reset();


	neighbors.erase (neighbors.begin(), neighbors.end());
	++st_it;
	curr = *st_it;
    }
	
    if (emp) {
	PQ.get_frontier (em, dirs[curr]);
    

	//
	// get self_loops for last node
	//

	o_it = curr.out_edges_begin();
	o_end = curr.out_edges_end();
    
	for (; o_it != o_end; ++o_it) {
	    if (o_it->target() == o_it->source()) {
		em.self.push_back (*o_it);
	    }
	}
    
	//
	// some adjcacency list of the embedding obtained so far have to be
	// turned.
	// 

	correct_embedding(em, st_, dirs);
		
	node_map<int> mark;
	mark.init (G, 0);	
	node_map<symlist<edge>::iterator > upward_begin;
	upward_begin.init (G);
	node tmp;
    
	forall_nodes (tmp, G) {
	    upward_begin[tmp] = em.adjacency(tmp).begin();
	} 

	extend_embedding(curr, em, mark, upward_begin);
    }

    return true;
}


int planarity::run (graph& G) 
{	   
    bool directed = false;
    
    if (G.is_directed()) {
	G.make_undirected();
	directed = true;
    }
    
    biconnectivity biconn;
    
    if (bip) {
	biconn.make_biconnected (true);
    } else {
	biconn.store_components (true);
    }

    biconn.check (G);
    biconn.run (G);
	
    if (emp) {
	embedding.init (G);
    }

    planar_embedding em;

    if (!biconn.is_biconnected() && !bip) {
	biconnectivity::component_iterator c_it, c_end;

	for (c_it = biconn.components_begin(), c_end = biconn.components_end(); 
	c_it != c_end; ++c_it) {
	    
	    switch_to_component (G, c_it);
	    
#ifdef _DEBUG
		GTL_debug::os() << "Component is: " << std::endl;
		GTL_debug::os() << G << std::endl;
#endif
	    if (!run_on_biconnected (G, em)) {
		if (directed) {
		    G.make_directed();
		}
		
		G.restore_graph();
		planar = false;
		return algorithm::GTL_OK;
	    }
	 
	    if (emp) {
		add_to_embedding (G, em);
	    }
	}
	
	G.restore_graph();

    } else {

	//
	// G is already biconnected
	//
	
	GTL_debug::debug_message ("graph is biconnected\n");

	if (!run_on_biconnected (G, embedding)) {
	    if (directed) {
		G.make_directed();
	    }
	    
	    planar = false;
	    return algorithm::GTL_OK;
	}	
    }

    if (bip) {
		edges_t::iterator it, end;
	it = biconn.additional_begin();
	end = biconn.additional_end();
	
	for (; it != end; ++it) {

	    if (emp) {
		node s = it->source();
		node t = it->target();
		embedding.adj[s].erase (embedding.s_pos[*it]);
		embedding.adj[t].erase (embedding.t_pos[*it]);
	    }

	    G.del_edge (*it);
	}
    }

    if (directed) {
	G.make_directed();
    }

    planar = true;
    return algorithm::GTL_OK;
}

void planarity::add_to_embedding (graph& G, planar_embedding& em) 
{
    node n;
    forall_nodes (n, G) {
	planar_embedding::iterator it = em.adj_edges_begin (n);
	planar_embedding::iterator end = em.adj_edges_end (n);
		
	for (; it != end; ++it) {
	    embedding.pos (n, *it) = em.pos (n, *it);
	}
		
	embedding.adjacency(n).splice (
	    embedding.adj_edges_end (n),
	    em.adj_edges_begin (n),
	    em.adj_edges_end (n));
    }

    embedding.self.splice (
	embedding.self.end(), 
	em.self, em.self.begin(), em.self.end());
    embedding.multi.splice (
	embedding.multi.end(), 
	em.multi, em.multi.begin(), em.multi.end());
}


void planarity::reset () 
{
    ob_edges.erase (ob_edges.begin(), ob_edges.end());
    ob_nodes.erase (ob_nodes.begin(), ob_nodes.end());
}


void planarity::correct_embedding (
    planar_embedding& em, 
    st_number& st_, 
    node_map<std::list<direction_indicator> >& dirs) 
{
    st_number::reverse_iterator it = st_.rbegin();
    st_number::reverse_iterator end = st_.rend();
    bool* turn = new bool[st_[*it]];
	
    for (int i = 0; i < st_[*it]; ++i) {
	turn[i] = false;
    } 
	
    while (it != end) {
	node curr = *it;
		
	if (turn[st_[curr] - 1]) {
	    em.adjacency(curr).reverse();
	} 
		
	std::list<direction_indicator>::iterator d_it = dirs[curr].begin();
		
	while (!dirs[curr].empty()) {
			
	    if (d_it->direction && turn[st_[curr] - 1] || 
		!d_it->direction && !turn[st_[curr] - 1]) {
		turn[d_it->id - 1] = true;
	    }
	    
	    d_it = dirs[curr].erase (d_it);
	}
		
	++it;
    }
	
    delete[] turn;
}


void planarity::extend_embedding (	
    node n, 
    planar_embedding& em, 
    node_map<int>& mark,
    node_map<symlist<edge>::iterator >& upward_begin)
{
    mark[n] = 1;
    
    symlist<edge>::iterator it = upward_begin[n];
    symlist<edge>::iterator end = em.adjacency(n).end();
    node other;

    for (; it != end; ++it) {
	em.pos (n, *it) = it;
	other = n.opposite (*it);
	em.pos (other, *it) = em.push_front (other, *it);

	if (mark[other] == 0) {
	    extend_embedding (other, em, mark, upward_begin);
	}
    }
}

void planarity::switch_to_component (graph& G, 
				     biconnectivity::component_iterator c_it)
{
    //
    // hide all nodes 
    //

	nodes_t dummy;
    G.induced_subgraph (dummy);

    //
    // Restore nodes in this component.
    // 

	nodes_t::iterator it = c_it->first.begin();
    nodes_t::iterator end = c_it->first.end();

    for (; it != end; ++it) {
	G.restore_node (*it);
    }

    //
    // Restore edges in this component.
    //

	edges_t::iterator e_it = c_it->second.begin();
	edges_t::iterator e_end = c_it->second.end();
  
    for (; e_it != e_end; ++e_it) {
	G.restore_edge (*e_it);
    }
}

void planarity::examine_obstruction (graph& G, 
				     st_number& st_, 
				     node act, 
				     pq_node* fail, 
				     bool is_root,
				     planar_embedding& em,
				     node_map<std::list<direction_indicator> >& dirs,
				     pq_tree* PQ)
{
    node_map<int> used (G, 0);
    node_map<edge> to_father (G);

    //
    // Create a dfs-tree of the so called bush form. This is basically a normal dfs 
    // applied to the induced subgraph of G consisting only of the nodes with st_number 
    // 1, ..., st_[act] - 1. The only difference is that edges are always directed from 
    // the lower numbered vertex to higher numbered one.
    //

    dfs_bushform (st_.s_node(), used, st_, st_[act], to_father);

    if (fail->kind() == pq_node::Q_NODE) {

	//
	// In case the reduction failed at a Q-Node we need to know the edges that 
	// form the boundary of the biconnected component, which this Q-Node represents.
	// These can easily be obtained from the embedding we got so far.
	//

	q_node* q_fail = fail->Q();

	pq_tree::sons_iterator s_it = q_fail->sons.begin();
	pq_tree::sons_iterator s_end = q_fail->sons.end();
	node greatest = fail->n;

	while (s_it != s_end)  {
	    if ((*s_it)->kind() == pq_node::DIR) {
		direction_indicator* dir = (*s_it)->D();
		pq_tree::sons_iterator tmp = s_it;
		
		if (++tmp == ++(dir->pos)) {
		    dir->direction = true;
		} else {
		    dir->direction = false;
		}

		dirs[act].push_back (*dir);

		//
		// chris 2/3/2008:
		//
		// To avoid a memory leak, it is not sufficient to erase it from the
		// PQ-tree (-node). The direction indicator object also has to be
		// deleted. Since it is then not a member of the pertinent subtree any
		// more, it must not be cleared by PQ->reset(). The instance in the
		// dirs node map is a clone!
		//

		// s_it = q_fail->sons.erase (s_it);
		s_it = PQ->remove_dir_ind(q_fail, s_it);
	    } else {
		if (st_[(*s_it)->up] > st_[greatest]) {
		    greatest = (*s_it)->up;
		}

		++s_it;
	    }
	}
	
	correct_embedding (em, st_, dirs);
	node_map<int> mark;
	mark.init (G, 0);	
	node_map<symlist<edge>::iterator > upward_begin;
	upward_begin.init (G);
	node tmp;

	em.adjacency(fail->n).erase (
	    em.adjacency(fail->n).begin(), 
	    em.adjacency(fail->n).end());

	forall_nodes (tmp, G) {
	    upward_begin[tmp] = em.adjacency(tmp).begin();
	}

	//
	// chris 2/3/2008:
	//
	// With the code of MR 11/27/2001 the component of the failing Q-node is not found
	// correctly.
	//

	extend_embedding(greatest, em, mark, upward_begin);
	
	/*
	//
	// MR 11/27/2001:
	//
	// This is important! We restricted building the embedding to the nodes in
	// the biconnected component which the Q-node fail refers to. But the st-number 
	// obtained for the whole graph restricted to these nodes will not be a st-numbering 
	// for this biconnected component. 
	//

	st_number::reverse_iterator st_it, st_end;

	for (st_it = st_.rbegin(), st_end = st_.rend();
	     st_it != st_end;
	     ++st_it) 
	{
	    if (mark[*st_it] == 0) {
		extend_embedding (*st_it, em, mark, upward_begin);
	    }
	}
	*/
#ifdef _DEBUG
	GTL_debug::os() << "Embedding so far (st_numbered): " << std::endl;
	em.write_st (GTL_debug::os(), st_);
#endif 

	attachment_cycle (fail->n, em);

	if (!q_fail->pert_cons) {

	    //
	    // the reduction failed because there was more than one block 
	    // of pertinent children. The reduction in this case assures that 
	    // pert_begin and pert_end lie in different blocks and that 
	    // --pert_end is empty and lies between these two blocks.
	    //
	    // This is one of the two cases that may apply when the reduction 
	    // fails already in bubble up. The reduction takes care of this.
	    //

	    GTL_debug::debug_message ("CASE C (non consecutive pertinent children)\n");
	    pq_tree::sons_iterator tmp = q_fail->pert_begin;
	    pq_leaf* leaves[3];
	    node nodes[3];
	    leaves[0] = search_full_leaf (*tmp);
	    nodes[0] = (*tmp)->up;

	    tmp = q_fail->pert_end;
	    leaves[2] = search_full_leaf (*tmp);
	    nodes[2] = (*tmp)->up;
	    
	    --tmp;
	    while ((*tmp)->kind() == pq_node::DIR) {
		--tmp;
	    }

	    leaves[1] = search_empty_leaf (*tmp);
	    nodes[1] = (*tmp)->up;
	    
	    case_C (nodes, leaves, st_, to_father, G, q_fail);

	} else if (!(*(q_fail->pert_end))->is_endmost && !is_root) {

	    GTL_debug::debug_message ("CASE D (non-root q-node with both endmost sons empty)\n");
	    pq_tree::sons_iterator tmp = q_fail->sons.begin();
	    pq_leaf* leaves[3];
	    node nodes[3];
	    leaves[0] = search_empty_leaf (*tmp);
	    nodes[0] = (*tmp)->up;

	    tmp = --(q_fail->sons.end());
	    leaves[2] = search_empty_leaf (*tmp);
	    nodes[2] = (*tmp)->up;
	    
	    tmp = q_fail->pert_begin;
	    leaves[1] = search_full_leaf (*tmp);
	    nodes[1] = (*tmp)->up;

	    case_D (nodes, leaves, st_, to_father, G, q_fail);	       

	} else if (q_fail->partial_count == 1) {
	    if (q_fail->partial_pos[0] == q_fail->pert_end) {		
		GTL_debug::debug_message ("CASE D (non-root q-node with partial child at end of pertinent children\n");
		pq_tree::sons_iterator tmp = q_fail->sons.begin();
		pq_leaf* leaves[3];
		node nodes[3];
		leaves[0] = search_empty_leaf (*tmp);
		nodes[0] = (*tmp)->up;
		
		tmp = q_fail->pert_end;
		leaves[2] = search_empty_leaf (*tmp);
		nodes[2] = (*tmp)->up;
		
		tmp = q_fail->pert_begin;
		leaves[1] = search_full_leaf (*tmp);
		nodes[1] = (*tmp)->up;
		
		case_D (nodes, leaves, st_, to_father, G, q_fail);	    
	    } else {
		GTL_debug::debug_message ("CASE C (q-node with partial children surrounded by pertinent children)\n");
		pq_tree::sons_iterator tmp = q_fail->pert_begin;
		pq_leaf* leaves[3];
		node nodes[3];
		leaves[0] = search_full_leaf (*tmp);
		nodes[0] = (*tmp)->up;
		
		tmp = q_fail->pert_end;
		leaves[2] = search_full_leaf (*tmp);
		nodes[2] = (*tmp)->up;
		
		tmp = q_fail->partial_pos[0];
		leaves[1] = search_empty_leaf (*tmp);
		nodes[1] = (*tmp)->up;
		
		
		case_C (nodes, leaves, st_, to_father, G, q_fail);
	    }
	    
	} else if ((q_fail->partial_pos[0] == q_fail->pert_begin ||
		    q_fail->partial_pos[0] == q_fail->pert_end) && 
		   (q_fail->partial_pos[1] == q_fail->pert_begin ||
		    q_fail->partial_pos[1] == q_fail->pert_end)) {

	    if (++(q_fail->sons.begin()) == --(q_fail->sons.end())) {

		//
		// q_node with two children, which are both partial.
		//

		pq_tree::sons_iterator tmp = q_fail->sons.begin();
		pq_leaf* leaves[4];
		node nodes[2];
		leaves[0] = search_empty_leaf (*tmp);
		nodes[0] = (*tmp)->up;
		leaves[1] = search_full_leaf (*tmp);
		
		++tmp;
		leaves[2] = search_empty_leaf (*tmp);
		nodes[1] = (*tmp)->up;
		leaves[3] = search_full_leaf (*tmp);

		case_E (nodes, leaves, st_, to_father, G, q_fail);	    

	    } else if (q_fail->partial_count == 2) {
		GTL_debug::debug_message ("CASE D (non-root q_node with first and last pertinent children partial)\n");

		//
		// sons.begin() is empty, pert_begin is partial, pert_end is partial
		//

		pq_tree::sons_iterator tmp = q_fail->sons.begin();
		pq_leaf* leaves[3];
		node nodes[3];
		leaves[0] = search_empty_leaf (*tmp);
		nodes[0] = (*tmp)->up;
		
		tmp = q_fail->pert_end;
		leaves[2] = search_empty_leaf (*tmp);
		nodes[2] = (*tmp)->up;
		
		tmp = q_fail->pert_begin;

		if (tmp == q_fail->sons.begin()) {
		    ++tmp;
		}

		leaves[1] = search_full_leaf (*tmp);
		nodes[1] = (*tmp)->up;
		
		case_D (nodes, leaves, st_, to_father, G, q_fail);	    

	    } else {
		GTL_debug::debug_message ("CASE C (q_node with at least three partial children)\n");

		//
		// There must be at least one other partial child among the pertinent.
		//
		
		pq_tree::sons_iterator tmp = q_fail->pert_begin;
		pq_leaf* leaves[3];
		node nodes[3];
		leaves[0] = search_full_leaf (*tmp);
		nodes[0] = (*tmp)->up;
		
		tmp = q_fail->pert_end;
		leaves[2] = search_full_leaf (*tmp);
		nodes[2] = (*tmp)->up;
		
		tmp = q_fail->partial_pos[2];
		leaves[1] = search_empty_leaf (*tmp);
		nodes[1] = (*tmp)->up;
				
		case_C (nodes, leaves, st_, to_father, G, q_fail);
	    }

	} else {
	    
	    //
	    // At least one partial son is in between the pertinent sons.
	    //

	    GTL_debug::debug_message ("CASE C (q_node with at least two partial children, at least one surrounded by pertinent)\n");
	    pq_tree::sons_iterator tmp = q_fail->pert_begin;
	    pq_leaf* leaves[3];
	    node nodes[3];
	    leaves[0] = search_full_leaf (*tmp);
	    nodes[0] = (*tmp)->up;
	    
	    tmp = q_fail->pert_end;
	    leaves[2] = search_full_leaf (*tmp);
	    nodes[2] = (*tmp)->up;
	    
	    tmp = q_fail->partial_pos[0];
	    
	    if (tmp == q_fail->pert_begin || tmp == q_fail->pert_end) {
		tmp = q_fail->partial_pos[1];
	    }
	    
	    leaves[1] = search_empty_leaf (*tmp);
	    nodes[1] = (*tmp)->up;
	    
	    case_C (nodes, leaves, st_, to_father, G, q_fail);
	}

    } else {

	//
	// pert_root is a P-Node ==> at least two partial children.
 	//

	p_node* p_fail = fail->P();

	if (p_fail->partial_count == 2) {
	    GTL_debug::debug_message ("CASE B (non-root p-node with two partial children)\n");
	    case_B (p_fail, act, st_, to_father, G);
	
	} else {

	    //
	    // We have at least three partial children 
	    //

	    GTL_debug::debug_message ("CASE A (p-node with at least three partial children)\n");
	    case_A (p_fail, act, st_, to_father, G);
	}	
    }   
}



void planarity::case_A (p_node* p_fail,
			node act,
			st_number& st_, 
			node_map<edge> to_father, 
			graph& G)
{
    node art = p_fail->n;
    ob_nodes.push_back (art);
    ob_nodes.push_back (act);
    node_map<int> mark (G, 0);
    mark[art] = 1;
    pq_leaf* empty[3];
    pq_tree::sons_iterator part_pos = p_fail->partial_sons.begin();
    int i;
    
    for (i = 0; i < 3; ++i) {
	q_node* q_part = (*part_pos)->Q();
	empty[i] = run_through_partial (q_part, mark, to_father, art);
	++part_pos;
    }
    
    node t_node = st_.s_node().opposite (st_.st_edge());
    mark[t_node] = 1;
    node tmp[3];
    
    for (i = 0; i < 3; ++i) {
	tmp[i] = up_until_marked (empty[i]->n, mark, st_);
    }
    
    assert (tmp[0] == t_node);
    node tmp_node;
    
    //
    // The three paths found meet at the nodes tmp[1] and tmp[2]; the one
    // one with the higher st_number is the one we are looking for. Since the
    // first path always ends at t and it may happen that the paths meet below 
    // t, we might have to delete some of the edges found in the first path.
    //
    
    if (st_[tmp[1]] < st_[tmp[2]]) {
	tmp_node = tmp[2];
	ob_nodes.push_back (tmp[1]);
    } else {
	tmp_node = tmp[1];
	ob_nodes.push_back (tmp[2]);
    }
    
    
    if (tmp_node != t_node) {
		edges_t::iterator it, end;
	int max_st = st_[tmp_node];
	
	it = ob_edges.begin();
	end = ob_edges.end();
	
	while (it != end) {
	    edge cur = *it;
	    
	    if (st_[cur.source()] > max_st || st_[cur.target()] > max_st) {
		it = ob_edges.erase (it);
	    } else {
		++it;
	    }
	}
    }
}



void planarity::case_B (p_node* p_fail,
			node act,
			st_number& st_, 
			node_map<edge> to_father, 
			graph& G)
{
    //
    // P-Node, which is not the root of the pertinent subtree, but has 
    // two partial children. 
    // 

    node art = p_fail->n;
    ob_nodes.push_back (art);
    ob_nodes.push_back (act);
    node_map<int> mark (G, 0);
    node_map<int> below (G, 0);
    mark[art] = 1;
    
    //
    // mark edges leading to full leaves from full sons.
    //
    
    pq_tree::sons_iterator it, end;
    for (it = p_fail->full_sons.begin(), end = p_fail->full_sons.end(); it != end; ++it) {
	mark_all_neighbors_of_leaves (*it, below);
    }
    
    //
    // search paths from one full and one empty leaf to the articulation point 
    // in TBk. mark edges leading to full leaves from pertinent sons of part.
    //
    
    pq_tree::sons_iterator part_pos = p_fail->partial_sons.begin();
    q_node* q_part = (*part_pos)->Q();
    pq_leaf* empty1 = run_through_partial (q_part, mark, to_father, art);
    
    for (it = q_part->pert_begin, end = ++(q_part->pert_end); it != end; ++it) {
	mark_all_neighbors_of_leaves (*it, below);
    }
    
    //
    // search paths from one full and one empty leaf to the articulation point 
    // in TBk. mark edges leading to full leaves from pertinent sons of part.
    //
    
    ++part_pos;
    q_part = (*part_pos)->Q();
    pq_leaf* empty2 = run_through_partial (q_part, mark, to_father, art);

    
    for (it = q_part->pert_begin, end = ++(q_part->pert_end); it != end; ++it) {
	mark_all_neighbors_of_leaves (*it, below);
    }

    //
    // now that all the adjacent edges of act, that lead to art in TBk have been
    // marked search an unmarked adj. edge of act, 
    //
    
    node::adj_edges_iterator a_it, a_end;
    
    for (a_it = act.adj_edges_begin(), a_end = act.adj_edges_end(); a_it != a_end; ++a_it) {
	if (below[act.opposite (*a_it)] == 0 && st_[act.opposite (*a_it)] < st_[act]) {
	    break;
	}
    }
    
    assert (a_it != a_end);
    mark[st_.s_node()] = 1;
    mark[art] = 0;
    node tmp = up_until_marked (art, mark, to_father);
    assert (tmp == st_.s_node());
    tmp = up_until_marked (act.opposite (*a_it), mark, to_father);
    assert(tmp != art);
    ob_nodes.push_back (tmp);
    ob_edges.push_back (*a_it);
    ob_edges.push_back (st_.st_edge());
    
    //
    // search from empty1 and empty2 to t.
    //
    
    node t_node = st_.s_node().opposite (st_.st_edge());
    mark[t_node] = 1;
    tmp = up_until_marked (empty1->n, mark, st_);
    assert (tmp == t_node); 
    tmp = up_until_marked (empty2->n, mark, st_);
    ob_nodes.push_back (tmp);
}


void planarity::case_C (node* nodes,
			pq_leaf** leaves,
			st_number& st_, 
			node_map<edge> to_father, 
			graph& G,
			q_node* q_fail) 
{
    int i;
    node_map<int> mark (G, 0);
    node y_0 = q_fail->n;

    for (i = 0; i < 3; ++i) {
	mark[nodes[i]] = 1;    
	edge tmp_edge = leaves[i]->e;
	node tmp_node = leaves[i]->n;
	ob_edges.push_back (tmp_edge);
	tmp_node = up_until_marked (tmp_node.opposite (tmp_edge), mark, to_father);
	assert (tmp_node == nodes[i]);
	ob_nodes.push_back (nodes[i]);
    }

    ob_nodes.push_back (y_0);
    mark[st_.s_node()] = 1;
    node tmp = up_until_marked (y_0, mark, to_father);    
    assert (tmp == st_.s_node ());

    ob_nodes.push_back (leaves[2]->n);
    ob_edges.push_back (st_.st_edge());
    
    node t_node = st_.s_node().opposite (st_.st_edge());
    mark[t_node] = 1;
    tmp = up_until_marked (leaves[1]->n, mark, st_);
    assert (tmp == t_node); 
    tmp = up_until_marked (leaves[2]->n, mark, st_);
    ob_nodes.push_back (tmp);
}


void planarity::case_D (node* nodes,
			pq_leaf** leaves,
			st_number& st_, 
			node_map<edge> to_father, 
			graph& G,
			q_node* q_fail) 
{
    //
    // Mark all edges from leaves leading to this component.
    //
    
    node y_0 = q_fail->n;
    pq_tree::sons_iterator it, end;
    node_map<int> below (G, 0);
    node act = leaves[1]->n;
       
    for (it = q_fail->sons.begin(), end = q_fail->sons.end(); it != end; ++it) {
	mark_all_neighbors_of_leaves (*it, below);
    }
    
    node::adj_edges_iterator a_it, a_end;
    
    for (a_it = act.adj_edges_begin(), a_end = act.adj_edges_end(); a_it != a_end; ++a_it) {
	if (below[act.opposite (*a_it)] == 0 && st_[act.opposite (*a_it)] < st_[act]) {
	    break;
	}
    }
    

    //
    // Since q_fail can't be the root of the pertinent subtree, there must 
    // be at least one edge from act not leading to the component described by 
    // q_fail.
    //
    
    assert (a_it != a_end);

    int i;
    node_map<int> mark (G, 0);

    for (i = 0; i < 3; ++i) {
	mark[nodes[i]] = 1;    
	edge tmp_edge = leaves[i]->e;
	node tmp_node = leaves[i]->n;
	ob_edges.push_back (tmp_edge);
	tmp_node = up_until_marked (tmp_node.opposite (tmp_edge), mark, to_father);
	assert (tmp_node == nodes[i]);
	ob_nodes.push_back (nodes[i]);
    }

    ob_nodes.push_back (y_0);
    mark[st_.s_node()] = 1; 
    node tmp = up_until_marked (y_0, mark, to_father);    
    assert (tmp == st_.s_node ());
    ob_edges.push_back (*a_it);
    tmp = up_until_marked (act.opposite (*a_it), mark, to_father);


    //
    // The paths from y_0 and from act meet in tmp. If tmp != s_node we have 
    // to delete some edges.
    //

    if (tmp != st_.s_node()) {
		edges_t::iterator it, end;
	int min_st = st_[tmp];
	it = ob_edges.begin();
	end = ob_edges.end();
	
	while (it != end) {
	    edge cur = *it;
	    
	    if (st_[cur.source()] < min_st || st_[cur.target()] < min_st) {
		it = ob_edges.erase (it);
	    } else {
		++it;
	    }
	}
    }

    ob_nodes.push_back (act);

    node t_node = st_.s_node().opposite (st_.st_edge());
    mark[t_node] = 1;
    node tmp_nodes[3];
    
    for (i = 0; i < 3; ++i) {
	tmp_nodes[i] = up_until_marked (leaves[i]->n, mark, st_);
    }
    
    assert (tmp_nodes[0] == t_node);
    
    //
    // The three paths found meet at the nodes tmp[1] and tmp[2]; the one
    // one with the higher st_number is the one we are looking for. Since the
    // first path always ends at t and it may happen that the paths meet below 
    // t, we might have to delete some of the edges found in the first path.
    //
    
    if (st_[tmp_nodes[1]] < st_[tmp_nodes[2]]) {
	tmp = tmp_nodes[2];
	ob_nodes.push_back (tmp_nodes[1]);
    } else {
	tmp = tmp_nodes[1];
	ob_nodes.push_back (tmp_nodes[2]);
    }
    
    
    if (tmp != t_node) {
		edges_t::iterator it, end;
	int max_st = st_[tmp];
	it = ob_edges.begin();
	end = ob_edges.end();
	
	while (it != end) {
	    edge cur = *it;
	    
	    if (st_[cur.source()] > max_st || st_[cur.target()] > max_st) {
		it = ob_edges.erase (it);
	    } else {
		++it;
	    }
	}	
    }
}


void planarity::case_E (node* nodes,
			pq_leaf** leaves,
			st_number& st_, 
			node_map<edge> to_father, 
			graph& G,
			q_node* q_fail) 
{

    //
    // Mark all edges from the act node leading to this component.
    //
    
    node y_0 = q_fail->n;
    pq_tree::sons_iterator it, end;
    node_map<int> below (G, 0);
    node act = leaves[1]->n;
    
    for (it = q_fail->pert_begin, end = ++(q_fail->pert_end); it != end; ++it) {
	mark_all_neighbors_of_leaves (*it, below);
    }
    
    node::adj_edges_iterator a_it, a_end;
    
    for (a_it = act.adj_edges_begin(), a_end = act.adj_edges_end(); a_it != a_end; ++a_it) {
	if (below[act.opposite (*a_it)] == 0 && st_[act.opposite (*a_it)] < st_[act]) {
	    break;
	}
    }
    

    //
    // Since q_fail can't be the root of the pertinent subtree, there must 
    // be at least one edge from act not leading to the component described by 
    // q_fail.
    //
    
    assert (a_it != a_end);
    
    //
    // The list ob_edges at the moment contains the boundary. we need to know the paths 
    // from y_0 to nodes[0] ( = y_1), from nodes[0] to nodes[1] ( = y_2 ) and from nodes[1]
    // back to y_0, because some of them will be eventually deleted later.
    //

	edges_t::iterator paths_begin[3];
	edges_t::iterator l_it, l_end;
    node next = y_0;

    for (l_it = ob_edges.begin(), l_end = ob_edges.end(); l_it != l_end; ++l_it) {
	next = next.opposite (*l_it);

	if (next == nodes[1]) {
	    node tmp = nodes[1];
	    nodes[1] = nodes[0];
	    nodes[0] = tmp;
	    pq_leaf* tmp_leaf = leaves[2];
	    leaves[2] = leaves[0];
	    leaves[0] = tmp_leaf;
	    tmp_leaf = leaves[3];
	    leaves[3] = leaves[1];
	    leaves[1] = tmp_leaf;

	    paths_begin[0] = l_it;
	    ++paths_begin[0];
	    break;
	} else if (next == nodes[0]) {
	    paths_begin[0] = l_it;
	    ++paths_begin[0];
	    break;
	}
    }

    assert (l_it != l_end);
    ++l_it;
    assert (l_it != l_end);

    for (; l_it != l_end; ++l_it) {
	next = next.opposite (*l_it);

	if (next == nodes[1]) {
	    paths_begin[1] = l_it;
	    ++paths_begin[1];
	    break;
	}
    }

    assert (l_it != l_end);

    paths_begin[2] = --l_end;

    node y[3];
    int i;
    node_map<int> mark (G, 0);
	edges_t from_act[3];
	edges_t::iterator pos;

    for (i = 0; i < 2; ++i) {
	mark[nodes[i]] = 1;    
	edge tmp_edge = leaves[2 * i]->e;
	node tmp_node = leaves[2 * i]->n;
	ob_edges.push_back (tmp_edge);
	tmp_node = up_until_marked (tmp_node.opposite (tmp_edge), mark, to_father);
	assert (tmp_node == nodes[i]);
	tmp_edge = leaves[2 * i + 1]->e;
	tmp_node = leaves[2 * i + 1]->n;
	pos = ob_edges.insert (ob_edges.end(), tmp_edge);
	y[i + 1] = up_until_marked (tmp_node.opposite (tmp_edge), mark, to_father);
	from_act[i + 1].splice (from_act[i + 1].begin(), ob_edges, pos, ob_edges.end());
    }

    mark[st_.s_node()] = 1; 
    node tmp = up_until_marked (y_0, mark, to_father);    
    assert (tmp == st_.s_node ());
    pos = ob_edges.insert (ob_edges.end(), *a_it);
    y[0] = up_until_marked (act.opposite (*a_it), mark, to_father);
    from_act[0].splice (from_act[0].begin(), ob_edges, pos, ob_edges.end());

    node t_node = st_.s_node().opposite (st_.st_edge());
    mark[t_node] = 1;
    node tmp_nodes[3];
    node_map<int> from_where (G, 0);

    for (i = 0; i < 2; ++i) {
	pos = --(ob_edges.end());
	tmp_nodes[i] = up_until_marked (leaves[2 * i]->n, mark, st_);
	for (l_it = ++pos, l_end = ob_edges.end(); l_it != l_end; ++l_it) {
	    from_where[l_it->source()] = i + 1;
	    from_where[l_it->target()] = i + 1;
	}
    }
    
    assert (tmp_nodes[0] == t_node);

    if (y_0 != y[0]) {
	ob_nodes.push_back (y_0);
	ob_nodes.push_back (y[0]);
	ob_nodes.push_back (y[1]);
	ob_nodes.push_back (y[2]);
	ob_nodes.push_back (act);
	ob_nodes.push_back (tmp_nodes[1]);

	l_it = paths_begin[0];
	l_end = paths_begin[1];
	ob_edges.erase (l_it, l_end);

	for (i = 0; i < 3; ++i) {
	    ob_edges.splice (ob_edges.end(), from_act[i], from_act[i].begin(), from_act[i].end());
	}

	GTL_debug::debug_message ("CASE E(i)\n");

    } else if (nodes[0] != y[1]) {
	ob_nodes.push_back (y_0);
	ob_nodes.push_back (y[1]);
	ob_nodes.push_back (nodes[0]);
	ob_nodes.push_back (y[2]);
	ob_nodes.push_back (act);
	ob_nodes.push_back (tmp_nodes[1]);
	l_it = paths_begin[1];
	l_end = paths_begin[2];
	++l_end;
	ob_edges.erase (l_it, l_end);

	for (i = 0; i < 3; ++i) {
	    ob_edges.splice (ob_edges.end(), from_act[i], from_act[i].begin(), from_act[i].end());
	}

	GTL_debug::debug_message ("CASE E(ii)\n");

    } else if (nodes[1] != y[2]) {
	ob_nodes.push_back (y_0);
	ob_nodes.push_back (y[1]);
	ob_nodes.push_back (nodes[1]);
	ob_nodes.push_back (y[2]);
	ob_nodes.push_back (act);
	ob_nodes.push_back (tmp_nodes[1]);
	l_it = ob_edges.begin();
	l_end = paths_begin[0];
	ob_edges.erase (l_it, l_end);

	for (i = 0; i < 3; ++i) {
	    ob_edges.splice (ob_edges.end(), from_act[i], from_act[i].begin(), from_act[i].end());
	}

	GTL_debug::debug_message ("CASE E(ii)\n");

    } else {
	tmp_nodes[2] = up_until_marked (leaves[1]->n, mark, st_);
	ob_nodes.push_back (y_0);
	ob_nodes.push_back (y[1]);
	ob_nodes.push_back (tmp_nodes[1]);
	ob_nodes.push_back (y[2]);
	ob_nodes.push_back (act);
	
	if (st_[tmp_nodes[1]] < st_[tmp_nodes[2]]) {
	    ob_nodes.push_back (tmp_nodes[2]);
	    l_it = paths_begin[0];
	    l_end = paths_begin[1];
	    ob_edges.erase (l_it, l_end);
	    
	    for (i = 1; i < 3; ++i) {
		ob_edges.splice (ob_edges.end(), from_act[i], from_act[i].begin(), from_act[i].end());
	    }

	    GTL_debug::debug_message ("CASE E(iii) (1)\n");

	} else if (st_[tmp_nodes[1]] > st_[tmp_nodes[2]]) {
	    edge last_edge = *(--(ob_edges.end()));
	    ob_nodes.push_back (tmp_nodes[2]);
	    ob_edges.splice (ob_edges.end(), from_act[0], from_act[0].begin(), from_act[0].end());
	    int from;

	    if (from_where[last_edge.source()] > 0) {
		from = from_where[last_edge.source()];
	    } else {
		from = from_where[last_edge.target()];
	    }
	    
	    assert (from > 0);

	    if (from == 1) {
		ob_edges.splice (ob_edges.end(), from_act[2], from_act[2].begin(), from_act[2].end());

		l_it = paths_begin[1];
		l_end = paths_begin[2];
		++l_end;
		ob_edges.erase (l_it, l_end);

	    } else {
		ob_edges.splice (ob_edges.end(), from_act[1], from_act[1].begin(), from_act[1].end());

		l_it = ob_edges.begin();
		l_end = paths_begin[0];
		ob_edges.erase (l_it, l_end);
	    }

	    GTL_debug::debug_message ("CASE E(iii) (2)\n");

	} else {
	    for (i = 0; i < 3; ++i) {
		ob_edges.splice (ob_edges.end(), from_act[i], from_act[i].begin(), from_act[i].end());
	    }

	    GTL_debug::debug_message ("CASE E(iii) (3)\n");
	}
    }   

    ob_edges.push_back (st_.st_edge());
}		


pq_leaf* planarity::search_full_leaf (pq_node* n) 
{
    switch (n->kind()) {
    case pq_node::LEAF:
	return n->L();   
	
    case pq_node::P_NODE:
    case pq_node::Q_NODE:
	return search_full_leaf (*(--(n->sons.end())));

    default:
	assert (false);
	return 0;
    }
}


pq_leaf* planarity::search_empty_leaf (pq_node* n) 
{
    switch (n->kind()) {
    case pq_node::LEAF:
	return n->L();   
	
    case pq_node::Q_NODE:
    case pq_node::P_NODE:
	return search_empty_leaf (*(n->sons.begin()));
	
    default:
	assert (false);
	return 0;
    }
}



void planarity::mark_all_neighbors_of_leaves (pq_node* act, node_map<int>& mark)
{
    if (act->kind() == pq_node::LEAF) {
        mark[((pq_leaf*)act)->e.opposite(((pq_leaf*)act)->n)] = 1;
    } else {
        pq_tree::sons_iterator it, end;

        for (it = act->sons.begin(), end = act->sons.end(); it != end; ++it) {
            mark_all_neighbors_of_leaves (*it, mark);
        }
    }
}


pq_leaf* planarity::run_through_partial (q_node* part, node_map<int>& mark, node_map<edge>& to_father, node v) 
{
    pq_leaf* tmp = search_full_leaf (part);
    edge tmp_edge = tmp->e;
    node tmp_node = tmp->n;
    ob_edges.push_back (tmp_edge);
    tmp_node = up_until_marked (tmp_node.opposite (tmp_edge), mark, to_father);

    tmp = search_empty_leaf (part);
    tmp_node = tmp->n;
    tmp_edge = tmp->e;
    ob_edges.push_back (tmp_edge);
    tmp_node = up_until_marked (tmp_node.opposite (tmp_edge), mark, to_father);
    assert (tmp_node != v);
    ob_nodes.push_back (tmp_node);
    
    return tmp->L();
}


node planarity::up_until_marked (node act, node_map<int>& mark, node_map<edge>& to_father) 
{
    while (mark[act] == 0) {
	mark[act] = 1;
	edge next = to_father[act];
	ob_edges.push_back (next);
	act = act.opposite (next);
    }

    return act;
}

node planarity::up_until_marked (node act, node_map<int>& mark, st_number& st_) 
{
    while (mark[act] == 0) {
	mark[act] = 1;
	node opp;
	node::adj_edges_iterator it, end;

	for (it = act.adj_edges_begin(), end = act.adj_edges_end(); it != end; ++it) {
	    opp = act.opposite (*it);
	    if (st_[opp] > st_[act]) {
		break;
	    }
	}

	assert (it != end);
	ob_edges.push_back (*it);
	act = opp;
    }

    return act;
}

void planarity::attachment_cycle (node start, planar_embedding& em) 
{
    edge act = em.adjacency(start).front();
    node next = start.opposite (act);
    ob_edges.push_back (act);

    while (next != start) {
	act = em.cyclic_next (next, act);
	next = next.opposite (act);
	ob_edges.push_back (act);
    }
}


void planarity::dfs_bushform (node n, 
			      node_map<int>& used, 
			      st_number& st_, 
			      int stop, 
			      node_map<edge>& to_father) 
{    
    used[n] = 1;
    node::adj_edges_iterator it, end;
    
    for (it = n.adj_edges_begin(), end = n.adj_edges_end(); it != end; ++it) {
	edge act = *it;
	node other = n.opposite(act);
	
	if (used[other] == 0 && st_[other] < stop) {
	    to_father[other] = *it;
	    dfs_bushform (other, used, st_, stop, to_father);
	}
    }
}

#ifdef _DEBUG

void planarity::write_node(std::ostream& os, int id, int label, int mark) {
	os << "node [\n" << "id " << id << std::endl;
    os << "label \"" << label << "\"\n";
    os << "graphics [\n" << "x 100\n" << "y 100 \n";
    if (mark == 1) {
        os << "outline \"#ff0000\"\n";
    }
    os << "]\n";    
	os << "]" << std::endl;
}
#endif

#ifdef _DEBUG
void planarity::write_bushform(graph& G, st_number& st_, int k, const char* name, const node_map<int>& mark,
                               const node_map<edge>& to_father) 
{
    // create the bushform Bk for the k where the test failed 
    st_number::iterator st_it, st_end; 
    int id = 0;
    node_map<int> ids (G);
	std::ofstream os(name, std::ios::out | std::ios::trunc);

	os << "graph [\n" << "directed 1" << std::endl;
    
    for (st_it = st_.begin(), st_end = st_.end(); st_it != st_end && st_[*st_it] <= k; ++st_it) {
        write_node(os, id, st_[*st_it], mark[*st_it]);
        ids[*st_it] = id;
        id++;
    }

    for (st_it = st_.begin(), st_end = st_.end(); st_it != st_end && st_[*st_it] <= k; ++st_it) {
        node::adj_edges_iterator ait, aend;
        
        for (ait = st_it->adj_edges_begin(), aend = st_it->adj_edges_end(); ait != aend; ait++) {
            node other = ait->opposite(*st_it);
            int other_id;
            if (st_[*st_it] < st_[other]) {
                if(st_[other] > k) {
                    write_node(os, id, st_[other], mark[other]);
                    other_id = id;
                    id++;
                } else {
                    other_id = ids[other];
                }

				os << "edge [\n" << "source " << ids[*st_it] << "\ntarget " << other_id << std::endl;
                if (*ait == to_father[*st_it] || *ait == to_father[other]) {
					os << "graphics [\n" << "fill \"#0000ff\"" << std::endl;
					os << "width 4.0\n]" << std::endl;
                } 
				os << "\n]" << std::endl;
            }
        }
    }

	os << "]" << std::endl;
}

#endif

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
