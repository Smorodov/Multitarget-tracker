// $Id: gml2dot.cpp,v 1.1 2004/03/30 21:45:45 rdmp1c Exp $

//
// Simple program to convert a GML format graph into a DOT format graph
// for display by GraphViz
//


#include <iostream>
#include <fstream>

#include <GTL/GTL.h>
#include <GTL/dfs.h>

#include "mygraph.h"



int main (int argc, const char * argv[]) 
{
	if (argc < 2)
	{
		cout << "Usage: gml2dot <file-name>" << endl;
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
	
	char dotfilename[256];
	strcpy (dotfilename, filename);
	strcat (dotfilename, ".dot");
	G.save_dot (dotfilename);
	
   return 0;
}
