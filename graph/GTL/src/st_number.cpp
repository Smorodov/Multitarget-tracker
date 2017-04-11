/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   st_number.cpp 
//
//==========================================================================
// $Id: st_number.cpp,v 1.10 2001/11/07 13:58:11 pick Exp $

#include <GTL/st_number.h>

#include <cassert>

__GTL_BEGIN_NAMESPACE

pathfinder::pathfinder (const graph& G, edge st, node s) 
{
    node t = s.opposite (st);
    dfs_num.init (G, 0);
    low_num.init (G);
	tree.init(G, edges_t());
	back.init(G, edges_t());
	forward.init(G, edges_t());

    //
    // There is a problem with node/edge maps of iterators with Visual C++
    // which I don´t fully understand at the moment. Anyway the init for the 
    // maps below is only needed to allocate memory, which is done anyway, when
    // values are assigned to it.
    //
    
#ifndef __GTL_MSVCC	
    to_low.init (G);
    to_father.init (G);
    pos.init (G);
#endif
    
    used.init (G,0);
    act_dfs_num  = 1;
    new_nodes = G.number_of_nodes();
    is_biconn = true;
    
    //
    // Do DFS with biconnectivity extensions.
    //
    
    dfs_num[t] = act_dfs_num++;
    low_num[t] = dfs_num[t];
    new_nodes--;
	
    dfs_sub (s, t);
	
    if (new_nodes != 0) {
	is_biconn = false;
    } 
	
    used[t] = used[s] = 1;
}


void pathfinder::dfs_sub (node& curr, node& father) 
{
    low_num[curr] = dfs_num[curr] = act_dfs_num++;
    new_nodes--;
	
    node::adj_edges_iterator it = curr.adj_edges_begin();
    node::adj_edges_iterator end = curr.adj_edges_end();
    
    while (it != end) {
	edge adj = *it;
	node opp = curr.opposite(adj);
		
	if (dfs_num[opp] == 0) {	    
			
		edges_t::iterator tmp = tree[curr].insert (tree[curr].end(), adj);
	    to_father[opp] = tmp;
			
	    dfs_sub (opp, curr);
			
	    if (low_num[opp] < low_num[curr]) {
		low_num[curr] = low_num[opp];
		to_low[curr] = tmp;
	    } 
			
	    if (low_num[opp] >= dfs_num[curr]) {
		is_biconn = false;
	    }
			
	} else if (opp != father && dfs_num[opp] < dfs_num[curr]) { 
		edges_t::iterator back_pos = back[curr].insert (back[curr].end(), adj);
		edges_t::iterator forward_pos = forward[opp].insert (forward[opp].end(), adj);
	    pos[adj] = pos_pair (forward_pos, back_pos);
			
	    if (dfs_num[opp] < low_num[curr]) {
		low_num[curr] = dfs_num[opp];
		to_low[curr] = back_pos;
	    }
	}
		
	++it;
    }
}


//--------------------------------------------------------------------------
//   ITERATOR
//--------------------------------------------------------------------------

pathfinder::const_iterator::const_iterator (pathfinder& _pf, node n) : 
	pf (_pf) 
{
    if (!pf.back[n].empty()) {
	edge back = pf.back[n].front();
	curr = n.opposite (back);
	pf.used[curr] = 1;
	pf.back[n].pop_front();
	pf.forward[curr].erase (pf.pos[back].first);
	state = END;
		
    } else if (!pf.tree[n].empty()) {
	curr = n.opposite (pf.tree[n].front());
	pf.used[curr] = 1;
	pf.tree[n].pop_front();
	state = DOWN;
		
    } else if (!pf.forward[n].empty()) {
	edge forward = pf.forward[n].front();
	curr = n.opposite (forward);
	pf.forward[n].pop_front();
	pf.back[curr].erase (pf.pos[forward].second); 
		
	if (pf.used[curr]) {
	    state = END;
	} else {
	    pf.used[curr] = 1;
	    state = UP;
	}
    }
}

pathfinder::const_iterator& pathfinder::const_iterator::operator++ () 
{
	edges_t::iterator tmp;
    edge adj;
    node opp;
	
    switch (state) {
	case END :
	    curr = node();
	    break;
		
	case UP :
	    tmp = pf.to_father[curr];
	    curr = curr.opposite (*tmp);
	    pf.tree[curr].erase (tmp);
		
	    if (pf.used[curr]) {
		state = END;
	    } else {
		pf.used[curr] = 1;
	    }
		
	    break;
		
	case DOWN :
	    tmp = pf.to_low[curr];
	    adj = *tmp;
	    opp = curr.opposite (adj);
		
	    if (pf.used[opp]) {
		pf.forward[opp].erase (pf.pos[adj].first);
		pf.back[curr].erase (tmp);
		state = END;
	    } else {
		pf.tree[curr].erase (tmp);
		pf.used[opp] = 1;
	    }
		
	    curr = opp;
	    break;
		
	default:
	    assert (0);
    }
    
    return *this;
}


pathfinder::const_iterator pathfinder::const_iterator::operator++ (int)
{
    const_iterator tmp = *this;
    operator++();
    return tmp;
}


//--------------------------------------------------------------------------
//   ST-NUMBER
//--------------------------------------------------------------------------

int st_number::check (graph& G)
{
    if (G.is_directed()) return GTL_ERROR;
    
    pf = new pathfinder (G, st, s);
	
    return pf->is_valid() ? GTL_OK : GTL_ERROR;
}


int st_number::run (graph& /*G*/) 
{
	nodes_t order;
    node t = s.opposite (st);
    order.push_back (t);
    node tmp = s;
    pathfinder::const_iterator end = pf->end();
    int act_st = 1;
	
	while (tmp != t)
	{
		pathfinder::const_iterator it = pf->path(tmp);
		nodes_t::iterator pos;

		if (it == end)
		{
			st_num[tmp] = act_st++;
			st_ord.push_back(tmp);
			tmp = order.back();
			order.pop_back();

		}
		else
		{
			pos = order.end();

			while (it != end)
			{
				pos = order.insert(pos, *it);
				++it;
			}

			order.erase(pos);
		}
	}
	
    st_num[t] = act_st;
    st_ord.push_back (t);
    
    delete pf;
	
    return GTL_OK;
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
