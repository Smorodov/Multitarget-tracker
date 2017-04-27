/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   bellman_ford.cpp
//
//==========================================================================
// $Id: bellman_ford.cpp,v 1.4 2003/01/30 17:50:56 raitner Exp $

#include <GTL/bellman_ford.h>

__GTL_BEGIN_NAMESPACE

bellman_ford::bellman_ford()
{
    vars_set = false;
    preds = 0;
}

bellman_ford::~bellman_ford()
{
    if (preds) delete preds;
}

void bellman_ford::store_preds (bool set)
{
    if (set && !preds) {
	preds = new node_map<edge>;
    } else if (!set && preds) {
	delete preds;
	preds = 0;
    }
}


int bellman_ford::check(graph& G)
{
    if (!vars_set) 
    {
	return algorithm::GTL_ERROR;
    }

    if (G.nodes_begin() == G.nodes_end()) 
    {
	return algorithm::GTL_ERROR;
    }

    return algorithm::GTL_OK;
}

int bellman_ford::run(graph& G)
{
    if (s == node()) 
    {
	s = *(G.nodes_begin());
    }

    //----------------------------------------------------------------------
    //   initialize
    //----------------------------------------------------------------------

    inf.init (G, true);
    
    if (preds) {
	preds->init (G, edge());
    }

    inf[s] = false;
    d[s] = 0;
    cycle = false;

    //----------------------------------------------------------------------
    //   relax
    //----------------------------------------------------------------------

    graph::edge_iterator it, end;

    for (int i = 1; i < G.number_of_nodes(); ++i)
    {	
	for (it = G.edges_begin(), end = G.edges_end(); it != end; ++it)
	{
            relax (*it, true);

            if (G.is_undirected())
            {
                relax(*it, false);
            }
	}
    }

    //----------------------------------------------------------------------
    //   cycle detection
    //----------------------------------------------------------------------    

    for (it = G.edges_begin(), end = G.edges_end(); it != end; ++it)
    {
	node u = it->source();
	node v = it->target();

	if(!inf[u] && !inf[v]) 
	{
	    if (d[v] > d[u] + w[*it]) 
	    {
		cycle = true;
	    }
	}
    }

    return algorithm::GTL_OK;
}

void bellman_ford::reset()
{
}

void bellman_ford::relax(const edge& e, bool dir )
{
    node u = e.source();
    node v = e.target();

    if (!dir) {
        node tmp = u;
        u = v;
        v = tmp;
    }        
    
    if (!inf[u] && (inf[v] || (d[v] > d[u] + w[e]))) 
    {
	d[v] = d[u] + w[e];
	inf[v] = false;
	
	if (preds) 
	{
	    (*preds)[v] = e;
	} 
    }
}



__GTL_END_NAMESPACE
