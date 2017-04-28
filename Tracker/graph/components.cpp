// $Id: components.cpp,v 1.1 2006/01/18 17:40:47 rdmp1c Exp $

//
// Simple program to extract all the components of a GML format graph
//


#include <iostream>
#include <fstream>

#include <GTL/GTL.h>
#include <GTL/components.h>

#include "mygraph.h"



int main (int argc, const char * argv[]) 
{
	if (argc < 2)
	{
		cout << "Usage: components <file-name>" << endl;
		exit(1);
	}
	char filename[256];
	strcpy (filename, argv[1]);

  	// ---------------------------------------------------------	
  	// Read graph	

 	MyGraph G;
	
	GML_error err  = G.load (filename);
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
	
	// Components
	G.make_undirected();
	
	if (!G.is_connected())
	{
		// 2. Get components
	    components cp;
	    if (cp.check(G) != algorithm::GTL_OK) 
		{
			cerr << "component check failed at line " << __LINE__ << endl;
			exit(1);
	    } 
		else 
		{
			if (cp.run(G) != algorithm::GTL_OK) 
			{
		    	cerr << "component algorithm failed at line " << __LINE__ << endl;
				exit(1);
			} 
			else 
			{
				cout << "Graph has " << cp.number_of_components() << " components" << endl;
				
				G.make_directed();
			
				// Dump components
				int count = 0;
				components::component_iterator it = cp.components_begin ();
				components::component_iterator end = cp.components_end ();
				while (it != end)
				{
				
					list<node> comp = (*it).first;
					
					G.induced_subgraph (comp);
					
					char buf[64];
					sprintf (buf, "%d.%d.gml", comp.size(), count);
					
					G.save(buf);
					
					count++;
					
					G.restore_graph();
					
					it++;
				}

				
			}
		}
	}
		
	
	return 0;
}
