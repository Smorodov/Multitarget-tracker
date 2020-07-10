// $Id: mygraph.cpp,v 1.6 2005/03/16 09:51:26 rdmp1c Exp $

#include "mygraph.h"

#ifdef __GNUC__
	#if __GNUC__ == 3
		#include <iterator>
	#endif
#endif
#include <sstream>
#include <cstring>

void MyGraph::load_edge_info_handler (GTL::edge e, GTL::GML_pair* list)
{
	if (list)
	{
		// Iterate over the list of GML_pair values
		struct GTL::GML_pair *p = list;
		while (p)
		{
			switch (p->kind)
			{
				case GTL::GML_STRING:
					store_edge_string (e, p->key, p->value.str);
					break;
				case GTL::GML_INT:
					store_edge_integer (e, p->key, p->value.integer);
					break;
				case GTL::GML_DOUBLE:
					store_edge_double (e, p->key, p->value.floating);
					break;
				default:
					break;	
			}		
			p = p->next;
		}
	}
}

void MyGraph::store_edge_double (GTL::edge /*e*/, char * /*key*/, double /*value*/)
{
}


void MyGraph::store_edge_integer (GTL::edge e, char *key, int value)
{
	if (strcmp (key, "weight") == 0)
	{
		weight[e] = value;
	}

}

void MyGraph::store_edge_string (GTL::edge e, char *key, char *value)
{
	if (strcmp (key, "label") == 0)
	{
		if (labels_as_weights)
		{
			// Treat edge label as a weight
			std::istringstream iss(value);
			iss >> weight[e];
		}
		else
		{
			// Store edge label as a label
			edge_label[e] = value;
		}
	}
}


void MyGraph::load_node_info_handler(GTL::node n, GTL::GML_pair* list )
{
	if (list)
	{
		// Iterate over the list of GML_pair values
		struct GTL::GML_pair *p = list;
		while (p)
		{
			switch (p->kind)
			{
				case GTL::GML_STRING:
					store_node_string (n, p->key, p->value.str);
					break;
				case GTL::GML_INT:
					store_node_integer (n, p->key, p->value.integer);
					break;
				case GTL::GML_DOUBLE:
					store_node_double (n, p->key, p->value.floating);
					break;
				default:
					break;	
			}		
			p = p->next;
		}
	}
}

void MyGraph::store_node_double (GTL::node /*n*/, char * /*key*/, double /*value*/)
{
}


void MyGraph::store_node_integer (GTL::node /*n*/, char * /*key*/, int /*value*/)
{
}

void MyGraph::store_node_string (GTL::node n, char *key, char *value)
{
	if (strcmp (key, "label") == 0)
	{
		label[n] = value;
	}
}


//------------------------------------------------------------------------------
void MyGraph::save_edge_info_handler(std::ostream *os, GTL::edge e) const
{
	graph::save_edge_info_handler (os, e);	
	*os << "weight " << weight[e] << std::endl;
	*os << "label \"" << edge_label[e] << "\"" << std::endl;

	// Line width 1 pt
	*os << "graphics [" << std::endl;
	*os << "width 1.0" << std::endl;
	*os << "]" << std::endl;
	
	// Use standard Postscript font
	*os << "LabelGraphics [" << std::endl;
	*os << "type \"text\"" << std::endl;
	*os << "font \"Helvetica\"" << std::endl;
	*os << "]" << std::endl;
}

//------------------------------------------------------------------------------
void MyGraph::save_node_info_handler(std::ostream *os, GTL::node n) const
{
	graph::save_node_info_handler (os, n);
	if (label[n] != "")	
		*os << "label \"" << label[n] << "\"" << std::endl;

	// Use standard Postscript font
	*os << "LabelGraphics [" << std::endl;
	*os << "type \"text\"" << std::endl;
	*os << "font \"Helvetica\"" << std::endl;
	*os << "]" << std::endl;

}


//------------------------------------------------------------------------------
void MyGraph::save_dot (char *fname, bool weights)
{
	std::ofstream f (fname);
	save_dot (f, weights);
	f.close ();
}

//------------------------------------------------------------------------------
void MyGraph::save_dot(std::ostream &f, bool weights)
{
	GTL::node_map <int> index;
	graph::node_iterator nit = nodes_begin();
	graph::node_iterator nend = nodes_end();
	int count = 0;
	while (nit != nend)
	{
		index[*nit] = count++;
		nit++;
	}
	
	if (is_directed())
		f << "digraph";
	else
		f << "graph";
	
	f << " G {" << std::endl;

	// Try and make the graph look nice
	f << "   node [width=.2,height=.2,fontsize=10];" << std::endl;
	f << "   edge [fontsize=10,len=2];" << std::endl;
	
	// Write node labels
	nit = nodes_begin();
	while (nit != nend)
	{
		f << "   " << index[*nit] << " [label=\"" << label[*nit] << "\"";
		
		if (node_colour[*nit] != "white")
		{
			f << ", color=" << node_colour[*nit] << ", style=filled";
		}
		f << "];" << std::endl;
		
		nit++;
	}
	
	
	// Write edges 
	graph::edge_iterator it = edges_begin();
	graph::edge_iterator end = edges_end();
	while (it != end)
	{
		f << "   " << index[it->source()];
		if (is_directed())
			f << " -> ";
		else
			f << " -- ";
		f << index[it->target()];
		
		f << " [";
		
		if (weights)
		{
			f << "label=\"" << weight[*it] << "\", ";
		}
		else
		{
			std::string s = edge_label[*it];
			if (s != "")
				f << "label=\"" << s << "\", ";
		}
		
		f << " color=" << edge_colour[*it] << "];" << std::endl;
		
		
		
		it++;
	}
	
	f << "}" << std::endl;
}

//------------------------------------------------------------------------------
bool MyGraph::edge_exists (GTL::node n1, GTL::node n2)
{
	bool result = false;
	
	if (is_undirected ())
	{	
		// Visit all edges adjacent to n1 and ask whether any
		// is connect to n2
		GTL::node::adj_edges_iterator eit = n1.adj_edges_begin();
		GTL::node::adj_edges_iterator eend = n1.adj_edges_end();
		GTL::node found = n1;
		while ((found == n1) && (eit != eend))
		{
			if (n1.opposite (*eit) == n2)
				found = n2;
			else
				eit++;
		}
		if (found == n2)
		{
			result = true;		
		}
	}
	else
	{
		// Visit all edges that have n1 as their source and ask 
		// whether any is connected to n2
		GTL::node::out_edges_iterator eit = n1.out_edges_begin();
		GTL::node::out_edges_iterator eend = n1.out_edges_end();
		GTL::node found = n1;
		while ((found == n1) && (eit != eend))
		{
			if (n1.opposite (*eit) == n2)
				found = n2;
			else
				eit++;
		}
		if (found == n2)
		{
			result = true;		
		}
	}
	
	return result;
}

//------------------------------------------------------------------------------
void MyGraph::delete_edge (GTL::node n1, GTL::node n2)
{
	GTL::edge e;
	bool exists = false;
	
	if (is_undirected ())
	{	
		// Visit all edges adjacent to n1 and ask whether any
		// is connect to n2
		GTL::node::adj_edges_iterator eit = n1.adj_edges_begin();
		GTL::node::adj_edges_iterator eend = n1.adj_edges_end();
		GTL::node found = n1;
		while ((found == n1) && (eit != eend))
		{
			if (n1.opposite (*eit) == n2)
			{
				found = n2;
				e = *eit;
			}
			else
				eit++;
		}
		exists = (found == n2);
	}
	else
	{
		// Visit all edges that have n1 as their source and ask 
		// whether any is connected to n2
		GTL::node::out_edges_iterator eit = n1.out_edges_begin();
		GTL::node::out_edges_iterator eend = n1.out_edges_end();
		GTL::node found = n1;
		while ((found == n1) && (eit != eend))
		{
			if (n1.opposite (*eit) == n2)
			{
				found = n2;
				e = *eit;
			}
			else
				eit++;
		}
		exists = (found == n2);
	}
	
	if (exists)
		del_edge(e);
}


double MyGraph::node_cliqueishness (GTL::node &n)
{
	double c = 1.0;
	
	int numneighbours = n.degree();
	int possconnections = numneighbours * (numneighbours - 1) / 2;
	int actualconnections = 0;
	if (possconnections > 0)
	{
		// Build list of all nodes adjacent to n (n's neighbours)
		GTL::nodes_t neighbours;
		
		GTL::node::adj_nodes_iterator nit = n.adj_nodes_begin ();
		GTL::node::adj_nodes_iterator nend = n.adj_nodes_end ();
		while (nit != nend)
		{
			GTL::node ne = (*nit);
			neighbours.push_back (ne);
			nit++;
		}
		
		// Count number of edges between neighbours
		GTL::nodes_t::iterator i = neighbours.begin();
		GTL::nodes_t::iterator iend = neighbours.end();
		while (i != iend)
		{
			GTL::nodes_t::iterator j = i;
			j++;
			
			while (j != iend)
			{
				if (edge_exists (*i, *j))
					actualconnections++;
				j++;	
			}
			i++;
		}
		c = (double) actualconnections / (double) possconnections;			
	}
	return c;
}

double MyGraph::cliqueishness ()
{
	double sum = 0.0;
	graph::node_iterator nit = nodes_begin();
	graph::node_iterator nend = nodes_end();
	
	while (nit != nend)
	{	
		GTL::node n = (*nit);
		sum += node_cliqueishness (n);
		nit++;
	}
	
	return ( sum / (double)number_of_nodes() );
}




