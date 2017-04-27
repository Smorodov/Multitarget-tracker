// $Id: gml2nestedsql.cpp,v 1.2 2006/03/24 15:06:57 rdmp1c Exp $


#include <iostream>
#include <fstream>

#include <GTL/GTL.h>
#include <GTL/dfs.h>

#include "mygraph.h"
#include "mytree.h"

#include <time.h>


void Tokenise (std::string s, std::string delimiters, std::vector<std::string> &tokens);

void Tokenise (std::string s, std::string delimiters, std::vector<std::string> &tokens)
{
	tokens.erase (tokens.begin(), tokens.end());
	int start, stop;
	int n = s.length();
	start = s.find_first_not_of (delimiters);
	while ((start >= 0) && (start < n))
	{
		stop = s.find_first_of (delimiters, start);
		if ((stop < 0) || (stop > n)) stop = n;
		tokens.push_back (s.substr(start, stop - start));
		start = stop + delimiters.length();
	}
}

string escape_string (string s);

//-------------------------------------------------
// Escape string so it can be INSERTed into a SQL database
string escape_string (string s)
{
	if (s.size() == 0) return s;
	
	string result = "";
	for (unsigned int i = 0;i<s.size();i++)
	{
    	switch (s[i]) 
    	{
    		case '\0':
    			result += '\\';
    			result += '0';
    			break;
    		case '\'':
    			result += '\\';
    			result += "'";
    			break;
    		case '\n':
    			result += '\\';
    			result += 'n';
    			break;
    		case '\r':
    			result += '\\';
    			result += 'r';
    			break;
    		case '\\':
    			result += '\\';
    			result += '\\';
    			break;
    		case '"':
    			result += '\\';
    			result += '"';
    			break;
    		default:
    			result += s[i];
    			break;
    	}
    }
    return result;
}
    
clock_t t0;
clock_t t1;

void ShowTimeUsed (clock_t &t0, clock_t &t1);

void ShowTimeUsed (clock_t &t0, clock_t &t1)
{
	cout << "CPU time used = " << (float)(t1 - t0)/(float)CLOCKS_PER_SEC << " seconds" << endl;
}

node_map <int> left_visitation;
node_map <int> right_visitation;
node_map <int> num_children;
node_map <int> child_number;
node_map <string> path;
node_map <int> depth;

// All purpose traversal of tree (ugly)
class mydfs : public dfs
{
public:
	mydfs () : dfs () { visit = 1; current_path = ""; height = 0; };
	virtual void entry_handler (graph &G, node &n, node &f)
	{
		// Depth of node
		depth[n] = height;
		height++;
		
		// SQL visitation number
		left_visitation[n] = visit++;
		
		// Path as a string of child numbers
		string p = current_path;
		if (n.indeg() > 0) 
		{	
			char buf[32];
			sprintf (buf, "/%d", child_number[n]);
			p += buf;
			path[n] = p;
		}
		current_path = p;
		
	}
	virtual void leave_handler (graph &G, node &n, node &f)
	{
		height--;
		right_visitation[n] = visit++;
		
		// Parent
		if (n.indeg() == 0) 
			current_path = "";
		else
		{	
			edge e = (*n.in_edges_begin());
		 	current_path = path[e.source()];
		}
	}
protected:
	int visit;
	int height;
	string current_path;
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
    	if (!G.is_connected() ) 
    		cout << "Not connected";

		exit(1);
	}
	
	node root = G.root();

  	// ---------------------------------------------------------		
	// Assign ids to nodes
	node_map<int> id (G,0);
  	
  	// Extract id's from name
	{
		node v;
  		forall_nodes(v,G)
  		{
  			string s = G.get_node_label(v);
  			int n = atol (s.c_str());
			id[v] = n;
  		}
  	}
  	
  	// ---------------------------------------------------------		
	// Compute number of children for each node, and 
	// assign each node its child number
	{
		node v;
  		forall_nodes(v,G)
  		{
  			num_children[v] = 0;
			node::adj_nodes_iterator it = v.adj_nodes_begin();
			node::adj_nodes_iterator end = v.adj_nodes_end();
			int j = 0;
			while (it != end)
			{
				j++;
				child_number[(*it)] = j;
				num_children[v]++;
				it++;
			}
		}
  	}
	
  	// ---------------------------------------------------------		
	// Get visitation numbers for SQL queries

    mydfs d;
	d.start_node (G.root());
    if (d.check(G) != algorithm::GTL_OK) 
	{
		cerr << "dfs check failed at " << __LINE__ << " in " << __FILE__  << endl;
    } 
	else 
	{
		if (d.run(G) != algorithm::GTL_OK) 
		{
	    	cerr << "dfs algorithm failed at " << __LINE__ << " in " << __FILE__ << endl;
		} 
	}
	
  	// ---------------------------------------------------------		
	// Ensure root path = /
	path[G.root()] = "/";
	
	
  	// ---------------------------------------------------------		
	// SQL Dump
	
	ofstream sql("tree.sql");
  	node v;
  	forall_nodes(v,G)
  	{
  		sql << "INSERT INTO ncbi_tree (tax_id, parent_id, left_id, right_id, path) ";
  		sql << "VALUES (" << id[v] << ", ";
		
		// Ensure the node label is acceptable to SQL (NCBI names may have quotes, etc.)
//		string s = G.get_node_label(v);
//		sql << "'" << escape_string (s) << "', ";
  		// For Oracle to work we need to ensure the root of the tree has a NULL parent
  		if (G.parent(v) == v)
  			sql << "NULL";
  		else
		{
  			sql << id[G.parent(v)];
		}
  		
  		sql <<  ", " << left_visitation[v] << ", " << right_visitation[v] << ", '" << path[v] << "');" << endl;
  	}
	sql.close();

	return 0;
}
