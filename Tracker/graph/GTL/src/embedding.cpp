/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   embedding.cpp 
//
//==========================================================================
// $Id: embedding.cpp,v 1.18 2002/10/04 08:07:36 chris Exp $

#include <GTL/embedding.h>

__GTL_BEGIN_NAMESPACE

planar_embedding::planar_embedding (const planar_embedding& em) 
{
    init (*(em.G));
	
    node n;
    forall_nodes (n, *G) {		
	adj_list::const_iterator it = em.adj[n].begin();
	adj_list::const_iterator end = em.adj[n].end();
		
	for (; it != end; ++it) {
	    pos (n, *it) = push_back (n, *it);
	}
    }

    self.insert (self.begin(), em.self.begin(), em.self.end());
    multi.insert (multi.begin(), em.multi.begin(), em.multi.begin());
}


planar_embedding&
planar_embedding::operator= (const planar_embedding& em) 
{
    node n;
    if (G != 0) {
	forall_nodes (n, *G) {    
	    adj[n].erase (adj[n].begin(), adj[n].end());
	}
    }
	
    self.erase (self.begin(), self.end());
    multi.erase (multi.begin(), multi.end());

    init (*(em.G));
    
    forall_nodes (n, *G) {    
	adj_list::const_iterator it = em.adjacency(n).begin();
	adj_list::const_iterator end = em.adjacency(n).end();
		
	for (; it != end; ++it) {
	    pos (n, *it) = push_back (n, *it);
	}
    }
	
    self.insert (self.begin(), em.self.begin(), em.self.end());
    multi.insert (multi.begin(), em.multi.begin(), em.multi.begin());
		
    return *this;
}


void 
planar_embedding::init (graph& my_G) 
{
    adj.init (my_G);

    //
    // There is a problem with node/edge maps of iterators with Visual C++
    // which I don´t fully understand at the moment. Anyway the init for the 
    // maps below is only needed to allocate memory, which is done anyway, when
    // values are assigned to it. 
    //

#ifndef __GTL_MSVCC
    s_pos.init (my_G);
    t_pos.init (my_G);
#endif
    G = &my_G;
}


symlist<edge>::iterator 
planar_embedding::push_back (node n, edge e) 
{
    return adj[n].insert (adj[n].end(), e);
}


symlist<edge>::iterator 
planar_embedding::push_front (node n, edge e) 
{
    return adj[n].insert (adj[n].begin(), e);
}


symlist<edge>::iterator& 
planar_embedding::pos (node n, edge e)
{
    if (e.source() == n) {
	return s_pos[e];
    } else if (e.target() == n) {
	return t_pos[e];
    } else {
	assert (false);
	// this should not happen.
	return s_pos[e];
    }
}


void 
planar_embedding::insert_selfloop (edge e) 
{
    node n = e.source();
    s_pos[e] = t_pos[e] = adj[n].insert (adj[n].begin(), e);
}


void 
planar_embedding::turn (node n)
{
    adj[n].reverse();
}


edge 
planar_embedding::cyclic_next (node n, edge e)
{
    iterator it = pos (n, e);    
    ++it;
    
    if (it == adj[n].end()) {
	++it;
    }
	
    return *it;
} 


edge 
planar_embedding::cyclic_prev (node n, edge e)
{
    iterator it = pos (n, e);
    --it;
	
    if (it == adj[n].end()) {
	--it;
    }
	
    return *it;
}

bool
planar_embedding::check ()
{    
    node n;
    forall_nodes (n ,*G) {
	iterator it, end;

	for (it = adj[n].begin(), end = adj[n].end(); it != end; ++it) {
	    edge curr = *it;
	    node other = n.opposite (curr);

	    edge prev = cyclic_prev (n, curr);
	    edge next = cyclic_next (n, prev);
	    assert (next == curr);

	    while (other != n) {
		curr = cyclic_next (other, curr);
		other = other.opposite (curr);
	    }
	    if (curr != prev) {
		return false;
	    }
	    
	}
    }

    return true;
}


void 
planar_embedding::write_st(std::ostream& os, st_number& st)
{
    st_number::iterator n_it = st.begin();
    st_number::iterator n_end = st.end();
    iterator it, end;
	
    for (; n_it != n_end; ++n_it) {
	node n = *n_it;
	os << "[" << st[n] << "]::";
		
	it = adj[n].begin();
	end = adj[n].end();
		
	for (; it != end; ++it) {
	    os << "[" << st[n.opposite (*it)] << "]";
	}
		
	os << std::endl;
    }
    
	os << "SELFLOOPS:" << std::endl;
	edges_t::iterator e_it, e_end;
    for (e_it = self.begin(), e_end = self.end(); e_it != e_end; ++e_it)
	{
		os << st[e_it->source()] << "---" << st[e_it->target()] << std::endl;
    }
    
	os << "MULTIPLE EDGES:" << std::endl;
    for (e_it = multi.begin(), e_end = multi.end(); e_it != e_end; ++e_it)
	{
		os << st[e_it->source()] << "---" << st[e_it->target()] << std::endl;
    }
}

GTL_EXTERN std::ostream& operator<< (std::ostream& os, planar_embedding& em)
{
    graph::node_iterator n_it = em.G->nodes_begin();
    graph::node_iterator n_end = em.G->nodes_end();
    symlist<edge>::iterator it, end;
	
    for (; n_it != n_end; ++n_it) {
	node n = *n_it;
	os << n << ":: ";
		
	it = em.adj[n].begin();
	end = em.adj[n].end();
		
	for (; it != end; ++it) {
	    os << n.opposite (*it) << "*";
	}
		
	os << std::endl;
    }

	os << "SELFLOOPS:" << std::endl;
	edges_t::iterator e_it, e_end;
    for (e_it = em.self.begin(), e_end = em.self.end(); e_it != e_end; ++e_it)
	{
		os << *e_it << std::endl;
    }

	os << "MULTIPLE EDGES:" << std::endl;
    for (e_it = em.multi.begin(), e_end = em.multi.end(); e_it != e_end; ++e_it)
	{
		os << *e_it << std::endl;
    }

    return os;
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
