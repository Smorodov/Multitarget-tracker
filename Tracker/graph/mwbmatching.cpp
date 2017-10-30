// $Id: mwbmatching.cpp,v 1.3 2007/10/28 08:47:20 rdmp1c Exp $

#include "mwbmatching.h"


#include <cstdlib>
#include <cassert>
#include <iostream>
#include <queue>
#include <list>
#include <map>
#include <set>
#include <stack>

#ifdef __GNUC__
#include <algorithm>
#endif

#ifdef __BORLANDC__
	#include <values.h>
#endif
#if (defined __MWERKS__) || (defined __GNUC__)
	#include <climits>
	#define MAXINT INT_MAX
#endif


mwbmatching::mwbmatching ()
    :
      algorithm(),
      mwbm(0),
      set_vars_executed(false),
      pq(NULL)
{
}


mwbmatching::~mwbmatching ()
{
}

void mwbmatching::set_vars(const edge_map<int>& edge_weight)
{
    this->edge_weight = edge_weight;
	mwbm = 0;
    set_vars_executed = true;
}

int mwbmatching::check (graph& G) 
{
	if (!set_vars_executed)
    {
		return(GTL_ERROR);
    }
	if ((G.number_of_nodes() <= 1) || (!G.is_connected()) || (!G.is_directed()))
    {
		return(GTL_ERROR);
    }
    return GTL_OK;
}


int mwbmatching::run(graph& G)
{
	// Initialise
	pot.init (G, 0);
	free.init (G, true);
	dist.init (G, 0);
	
	nodes_t A;
	nodes_t B;
	node n;
	// Partition graph based on direction of edges
	forall_nodes (n, G)
	{
		if (n.outdeg() == 0)
		{
			B.push_back (n);
		}
		else
		{
			A.push_back(n);
		}
		
		node_from_id[n.id()] = n;
	}
	
	// Simple heuristic
	int C = 0;
	edge e;
	forall_edges (e, G)
	{
		edge_from_id[e.id()] = e;
		if (edge_weight[e] > C) 
			C = edge_weight[e];	
	}

	nodes_t::iterator it = A.begin();
	nodes_t::iterator end = A.end();
	while (it != end)
	{
		pot[*it] = C;
		it++;
	}
	
	it = A.begin();
	
	while (it != end)
	{
		if (free[*it])
		{
			augment (G, *it);
		}
		it++;
	}

	
	// Get edges in matching
	it = B.begin();
	end = B.end();
	while (it != end)
	{
		edge e;
		forall_out_edges (e, *it)
		{
			result.push_back (e);	
			mwbm += edge_weight[e];
		}
		it++;
	}


    return(GTL_OK);
}



int mwbmatching::augment(graph& G, node a)
{
	// Initialise
	pred.init(G, -1);
	pq = fh_alloc(G.number_of_nodes());
	
	dist[a] = 0;
	node best_node_in_A = a;
	long minA = pot[a];
	long delta;
	
	std::stack<node, std::vector<node> > RA;
	RA.push(a);
	std::stack<node, std::vector<node> > RB;
	
	node a1 = a;
	edge e;
	
	// Relax
	forall_adj_edges (e, a1)
	{
		const node& b = e.target_();
		long db = dist[a1] + (pot[a1] + pot[b] - edge_weight[e]);
		if (pred[b] == -1)
		{
			dist[b] = db;
			pred[b] = e.id();
			RB.push(b);
			
			fh_insert (pq, b.id(), db);	
		}
		else
		{
			if (db < dist[b])
			{
				dist[b] = db;
				pred[b] = e.id();

				fh_decrease_key (pq, b.id(), db);  
			}
		}
	}

	for (;;)
	{
		// Find node with minimum distance db
		int node_id;
		long db = 0;
		if (pq->n == 0)
		{
			node_id = -1;
		}
		else
		{
			node_id = fh_delete_min (pq);
			db = dist[node_from_id[node_id]];
		}
		
		if (node_id == -1 || db >= minA)
		{
			delta = minA;
			// augmentation by best node in A
      		augment_path_to (G, best_node_in_A);  
      		free[a] = false; 
      		free[best_node_in_A] = true; 
			break;
		}
		else
		{
			node b = node_from_id[node_id];
			if (free[b])
			{
				delta = db;
				// augmentation by path to b, so a and b are now matched
				augment_path_to (G, b);
        		free[a] = false;
        		free[b] = false;
		        break;
			}
			else
			{
				// continue shortest path computation
				edge e = (*b.adj_edges_begin()); 
				const node& a1 = e.target_();
				pred[a1] = e.id(); 
				RA.push(a1);
				dist[a1] = db; 
					  
				if (db + pot[a1] < minA)
				{ 
					best_node_in_A = a1;
				  	minA = db + pot[a1];
				}
				
				// Relax
				forall_adj_edges (e, a1)
				{
					const node& b = e.target_();
					long db = dist[a1] + (pot[a1] + pot[b] - edge_weight[e]);
					if (pred[b] == -1)
					{
						dist[b] = db;
						pred[b] = e.id();
						RB.push(b);
						
						fh_insert (pq, b.id(), db);	
					}
					else
					{
						if (db < dist[b])
						{
							dist[b] = db;
							pred[b] = e.id();

							fh_decrease_key (pq, b.id(), db);  
						}
					}
				}
								
			}
		}
	}
	
	
	while (!RA.empty())
	{
		node a = RA.top();
		RA.pop();
		pred[a] = -1;
		long pot_change = delta - dist[a];
		if (pot_change <= 0) continue;
		pot[a] = pot[a] - pot_change;		
	}
	while (!RB.empty())
	{
		node b = RB.top();
		RB.pop();
		pred[b] = -1;
		
		long pot_change = delta - dist[b];
		if (pot_change <= 0) continue;
		pot[b] = pot[b] + pot_change;		
	}
	
	// Clean up
	fh_free(pq);

	return 0;
}

void mwbmatching::augment_path_to (graph &/*G*/, node v)
{
	int i = pred[v];
	while (i != -1)
	{
		edge e = edge_from_id[i];
		e.reverse();
		i = pred[e.target()];
	}	

}

edges_t MAX_WEIGHT_BIPARTITE_MATCHING(graph &G, edge_map<int> weights)
{
	edges_t L;

	mwbmatching mwbm;
	mwbm.set_vars(weights);
	
	//if (mwbm.check(G) != algorithm::GTL_OK)
	//{
	//	cout << "Maximum weight bipartite matching algorithm check failed" << endl;
		//exit(1);
	//}
	//else
	{
		if (mwbm.run(G) != algorithm::GTL_OK)
		{
			std::cout << "Error running maximum weight bipartite matching algorithm" << std::endl;
			//exit(1);
		}
		else
		{
			L = mwbm.get_match();			
		}
	}
	return L;

}




