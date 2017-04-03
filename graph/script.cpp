// $Id: script.cpp,v 1.1 2005/03/16 09:52:54 rdmp1c Exp $

//
// Simple program to apply an edit script to aa graph
//


#include <iostream>
#include <fstream>
#include <map>

#include <GTL/GTL.h>
#include <GTL/dfs.h>

#include "mygraph.h"
#include "tokenise.h"



int main (int argc, const char * argv[]) 
{
	if (argc < 2)
	{
		cout << "Usage: script <file-name>" << endl;
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
	
	
	// Output starting graph
	G.save_dot ("start.dot");


	// 1. Get map between labels and nodes
	std::map < std::string, GTL::node, std::less<std::string> > l; 
	node n;
	forall_nodes (n, G)
	{
		l[G.get_node_label(n)] = n;
	}
	
	// Read edit script
	{
		ifstream f ("script.txt");
		char buf[512];
		
		while (f.good())
		{
			f.getline (buf, sizeof(buf));
			
			if (f.good())
			{
				
				// Break line into tokens
				std::vector<std::string> tokens;
				std::string s = buf;
				std::string delimiters = "|";
				Tokenise (s, delimiters, tokens);
				
				if (tokens[0] == "delete")
				{
					if (tokens[1] == "node")
					{
						string name = tokens[2];
						if (l.find(name) == l.end())
						{
							cout << "Node labelled \"" << name << "\" not found" << endl;
							exit(1);
						}
						else
						{
							cout << "Node \"" << name << "\" deleted" << endl;
							G.del_node(l[name]);
							l.erase(name);
						}
					}
					else if (tokens[1] == "branch")
					{
						string source = tokens[2];
						string target = tokens[3];
						if (l.find(source) == l.end())
						{
							cout << "Node labelled \"" << source << "\" not found" << endl;
							exit(1);
						}
						if (l.find(target) == l.end())
						{
							cout << "Node labelled \"" << target << "\" not found" << endl;
							exit(1);
						}
						cout << "Edge \"" << source << "\"-->\"" << target << "\" deleted" << endl;
						G.delete_edge(l[source], l[target]);
					}
				}
				if (tokens[0] == "insert")
				{
					if (tokens[1] == "node")
					{
						string name = tokens[2];
						node n = G.new_node();
						G.set_node_label (n, name);
						G.set_node_colour (n, "green");
						l[name] = n;
						cout << "Node \"" << name << "\" inserted" << endl;
					}
					else if (tokens[1] == "branch")
					{
						string source = tokens[2];
						string target = tokens[3];
						if (l.find(source) == l.end())
						{
							cout << "Node labelled \"" << source << "\" not found" << endl;
							exit(1);
						}
						if (l.find(target) == l.end())
						{
							cout << "Node labelled \"" << target << "\" not found" << endl;
							exit(1);
						}
						
						edge e = G.new_edge (l[source], l[target]);
						G.set_edge_colour (e, "green");
						
						cout << "Edge \"" << source << "\"-->\"" << target << "\" added" << endl;
					}
				}

			}			

		}
		f.close();
	}

	G.save_dot ("end.dot");


	
   return 0;
}
