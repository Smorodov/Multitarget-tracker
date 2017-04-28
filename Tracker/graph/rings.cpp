// $Id: rings.cpp,v 1.1 2005/01/07 23:01:20 rdmp1c Exp $

// Output RINGS representation of a GML tree
// See "RINGS: A Technique for Visualizing Large Hierarchies"
// http://www.cs.ucdavis.edu/~ma/papers/graph2.pdf


#include <iostream>
#include <fstream>
#include <queue>

#include <GTL/GTL.h>
#include <GTL/bfs.h>
#include <GTL/dfs.h>

#include "gport.h"

#include "mygraph.h"
#include "mytree.h"

#include <time.h>

GPostscriptPort g;


clock_t t0;
clock_t t1;

void ShowTimeUsed (clock_t &t0, clock_t &t1);

void ShowTimeUsed (clock_t &t0, clock_t &t1)
{
	cout << "CPU time used = " << (float)(t1 - t0)/(float)CLOCKS_PER_SEC << " seconds" << endl;
}

node_map<int> num_children;
node_map<int> num_grandchildren;
node_map<double> R1;
node_map<GPoint> pt;
node_map<bool> outer_ring;


class CompareNodes {
public:
	bool operator() (node x, node y)
	{
		return num_children[x] < num_children[y];
	}
};


// Get numbers of children and grandchildren
class mydfs : public dfs
{
public:
	mydfs () : dfs () {  };
	virtual void entry_handler (graph &G, node &n, node &f)
	{
		num_children[n] = n.outdeg();
		num_grandchildren[n] = 0;
	}
	virtual void leave_handler (graph &G, node &n, node &f)
	{
		if (n.outdeg() > 0)
		{
			node::adj_nodes_iterator it = n.adj_nodes_begin();
			while (it != n.adj_nodes_end())
			{
				num_grandchildren[n] += num_children[(*it)];
				it++;
			}
		}
	}
};

#ifndef M_PI
	#define M_PI		3.14159265358979323846	// define pi
#endif

#define SQR(X) ((X) * (X)) 


double f (double &R1, double &R2, int n);

double f (double &R1, double &R2, int n)
{
	double theta = M_PI/double(n);
	double fn = SQR(1 - sin(theta)) /  SQR (1 + sin(theta));
	R2 = sqrt (fn * SQR(R1));
	return fn;
}


// Compute and draw layout ussing a breadth first search
class mybfs : public bfs
{
public:
	mybfs () : bfs () {  };
	virtual void popped_node_handler (graph &G, node &n)
	{		
		if ((n.outdeg() > 0) && (num_grandchildren[n] > 0))
		{		
			// Colour map by level in graph
			switch (level(n))
			{
				case 0: g.SetFillColorRGB (125,126,190); break;
				case 1: g.SetFillColorRGB (202,154,152); break;
				case 2: g.SetFillColorRGB (178,219,178); break;
				case 3: g.SetFillColorRGB (255,179,179); break;
				case 4: g.SetFillColorRGB (225,224,179); break;
				case 5: g.SetFillColorRGB (255,178,255); break;
				default: g.SetFillColorRGB (192,192,192); break;
			}
		
		
			// Create a list of the children sorted by their number of children
			double total_grandchildren = 0.0;
			priority_queue <node, vector<node>, CompareNodes> p;
			node::adj_nodes_iterator it = n.adj_nodes_begin();
			while (it != n.adj_nodes_end())
			{
				p.push(*it);
				outer_ring[*it] = false;
				total_grandchildren += (double)num_children[*it];
				it++;
			}
			
			
			// Find how many children to put in the two rings
			double R2;
			double sum_grandchildren = 0.0;
			int count = 0;
			int outer = num_children[n];	// number in outer ring
			int inner = 0;					// number in inner ring
			
			// We look at the sorted list of children and
			// put the split between inner and outer rings at the point where
			// the fraction of children yet to be included is less than the
			// amount of space available in an inner ring if count rings are
			// in the outer ring. It is possible that there won't be an inner ring,
			// in which case inner is 0.
			// We use the node map outer_ring to classify children by which ring they
			// are assigned to. 
			while (!p.empty())
			{

				node x = p.top();
				outer_ring[x] = (inner == 0);
				count++;
				
				sum_grandchildren += (double)num_children[x];
				
				double fraction = 1.0 - (sum_grandchildren / total_grandchildren);
			
				double fk = f(R1[n], R2, count);
				
				if ((count > 2) && (inner == 0))
				{
					if (fraction < fk) 
					{
						inner = outer - count;
						outer = count;
					}
				}
				p.pop();
			}
			
			// Compute radius of children in outer ring
			double fn = f(R1[n], R2, outer);
			double r_outer = (R1[n] - R2)/2.0;
			double theta_outer = M_PI/double(outer);
			
			
			// Compute radius of children in inner ring (if any)
			double r_inner = 0.0;
			double R3 = 0.0;
			double theta_inner = 0.0;
			if (inner > 0)
			{
				fn = f(R2, R3, inner);
				r_inner = (R2 - R3)/2.0;
				theta_inner = M_PI/double(inner);
			}
			int inner_count = 0;
			int outer_count = 0;
			it = n.adj_nodes_begin();
			while (it != n.adj_nodes_end())
			{
				if (outer_ring[*it])
				{
					R1[*it] = r_outer;
					int offset_x = (int)((R1[*it] + R2) * cos(2 * theta_outer * outer_count));
					int offset_y = (int)((R1[*it] + R2) * sin(2 * theta_outer * outer_count));
					outer_count++;
					
					// Draw!!!
					GPoint p = pt[n];
					p.Offset (offset_x, offset_y);
					pt[*it] = p;
					g.DrawLinePts (pt[n], pt[*it]);
					//g.DrawCircle (pt[*it], R1[*it]);
					
					// Centre
					pt[*it] = p;
					
					
					
				}
				else
				{
				
					R1[*it] = r_inner;
					int offset_x = (int)((R1[*it] + R3) * cos(2 * theta_inner * inner_count));
					int offset_y = (int)((R1[*it] + R3) * sin(2 * theta_inner * inner_count));
					inner_count++;
					
					// Draw!!!
					GPoint p = pt[n];
					p.Offset (offset_x, offset_y);
					pt[*it] = p;
					g.DrawLinePts (pt[n], pt[*it]);
					//g.DrawCircle (pt[*it], R1[*it]);
					
					// Centre
					pt[*it] = p;
					
				}

				
				it++;
			}


			
		}
		else
		{
			// leaf, or node with no grandchildren (i.e., a star)
			if (n.outdeg() > 0)
			{
				//g.DrawCircle (pt[n], R1[n]);
				
				
				double theta = (2 * M_PI)/double(n.outdeg());
	
				double radius = R1[n] * 0.9;
	
				double gap = fabs (sin(theta) * radius);
				
				// If the gap between two edges is too small to be easily visible
				// we draw a filled circle
				if ((gap < 2.0) && (n.outdeg() > 1))
				{
					g.FillCircle (pt[n], (int)radius);
				}
				else
				{
					int count = 0;
					node::adj_nodes_iterator it = n.adj_nodes_begin();
					while (it != n.adj_nodes_end())
					{
						int offset_x = (int)(radius * cos(theta*count));
						int offset_y = (int)(radius * sin(theta*count));
						GPoint p = pt[n];
						p.Offset (offset_x, offset_y);
						pt[*it] = p;
						g.DrawLinePts (pt[n], pt[*it]);
						count++;
						it++;
					}
				
				}
			}
		
		}
	}
	virtual void finished_handler (graph &G, node &n)
	{
	}
};




int main (int argc, const char * argv[]) 
{
	if (argc < 2)
	{
		cout << "Usage: graph <file-name>" << endl;
		exit(1);
	}
	char filename[256];
	strcpy (filename, argv[1]);

  	// ---------------------------------------------------------	
  	// Read graph	

	MyTree G;
	
	G.read_labels_as_weights();
	t0 = clock();
	GML_error err  = G.load (filename);
	t1 = clock();
	if (err.err_num != GML_OK)
	{
		cerr << "Error (" << err.err_num << ") loading graph from file \"" << filename << "\"";
		switch (err.err_num)
		{
			case GML_FILE_NOT_FOUND: cerr << "A file with that name doesn't exist."; break;
			case GML_TOO_MANY_BRACKETS: cerr << "A mismatch of brackets was detected, i.e. there were too many closing brackets (])."; break;
			case GML_OPEN_BRACKET: cerr << "Now, there were too many opening brackets ([)";  break;
			case GML_TOO_MANY_DIGITS: cerr << "The number of digits a integer or floating point value can have is limited to 1024, this should be enough :-)";  break;
			case GML_PREMATURE_EOF: cerr << "An EOF occured, where it wasn't expected, e.g. while scanning a string."; break;
			case GML_SYNTAX: cerr << "The file isn't a valid GML file, e.g. a mismatch in the key-value pairs."; break;
			case GML_UNEXPECTED: cerr << "A character occured, where it makes no sense, e.g. non-numerical characters"; break;
			case GML_OK: break;
		}
		cerr << endl;
		exit(1);
	}
	else
	{
		cout << "Graph read from file \"" << filename << "\" has " << G.number_of_nodes() << " nodes and " << G.number_of_edges() << " edges" << endl;
	}
	ShowTimeUsed (t0, t1);
	
  	// ---------------------------------------------------------		
	// Test that it is a tree
	if (is_tree (G))
	{
		cout << "Is a tree" << endl;
	}
	else
	{
		cout << "Graph is not a tree" << endl;
		  node v;
  		forall_nodes(v,G)
    		if ( v.indeg () < 1 ) cout << G.get_node_label(v) << " has no parent" << endl;
		if (!G.is_connected() ) cout << "Not connected";

		exit(1);
	}
	
	node root = G.root();
	cout << "Root = " << root << " " << "\"" << G.get_node_label (root) << "\"" << endl;
	
	cout << "Computing layout..." << endl;
	t0 = clock();
	
    bfs b;
	b.start_node (G.root());
	b.calc_level(true);
    if (b.check(G) != algorithm::GTL_OK) 
	{
		cerr << "bfs check failed at " << __LINE__ << " in " << __FILE__  << endl;
		exit(1);
    } 
	else 
	{
		if (b.run(G) != algorithm::GTL_OK) 
		{
	    	cerr << "bfs algorithm failed at " << __LINE__ << " in " << __FILE__ << endl;
			exit(1);
		} 
	}
	
	// dfs
    mydfs d;
	d.start_node (G.root());
    if (d.check(G) != algorithm::GTL_OK) 
	{
		cerr << "dfs check failed at " << __LINE__ << " in " << __FILE__  << endl;
		exit(1);
    } 
	else 
	{
		if (d.run(G) != algorithm::GTL_OK) 
		{
	    	cerr << "dfs algorithm failed at " << __LINE__ << " in " << __FILE__ << endl;
			exit(1);
		} 
	}

	char picture_filename[256];
	strcpy (picture_filename, filename);
	strcat (picture_filename, ".ps");
    g.StartPicture (picture_filename);    
	g.SetPenWidth(1);

	

	R1[G.root()] = 200.0;
	GPoint centre(200,200);
	pt[G.root()] = centre;

    mybfs layout;
	layout.start_node (G.root());
	layout.calc_level(true);
    if (layout.check(G) != algorithm::GTL_OK) 
	{
		cerr << "bfs check failed at " << __LINE__ << " in " << __FILE__  << endl;
		exit(1);
    } 
	else 
	{
		if (layout.run(G) != algorithm::GTL_OK) 
		{
	    	cerr << "bfs algorithm failed at " << __LINE__ << " in " << __FILE__ << endl;
			exit(1);
		} 
	}

    g.EndPicture ();

	t1 = clock();
	ShowTimeUsed (t0, t1);
	
   return 0;
}



