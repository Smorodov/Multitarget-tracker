 // $Id: mygraph.h,v 1.6 2006/01/02 16:03:14 rdmp1c Exp $

#ifndef MYGRAPH_H
#define MYGRAPH_H

// STL
#include <iostream>
#include <fstream>
#include <map>
#include <set>

// GTL
#include <GTL/graph.h>



/**
 * @class MyGraph
 * MyGrap extends the GTL class graph to provide support for graphs
 * with weighted edges. 
 *
 */
class MyGraph : public graph
{
public:
	MyGraph () { labels_as_weights = false; };
	
	int get_edge_weight (edge e) const { return weight[e]; };	
	void set_edge_weight (edge e, int w) { weight[e] = w; };


	std::string get_edge_label(edge e) const { return edge_label[e]; };
	void set_edge_label(edge e, std::string s) { edge_label[e] = s; };
	
	/**
	 * Returns true if an edge exists between a pair of nodes.
	 *
     * @param <code>n1</code> a node.
     * @param <code>n2</code> another node.
     */
	virtual bool edge_exists (node n1, node n2);
	
	
	/**
	 * Delete the edge (if it exists) between a pair of nodes.
	 *
     * @param <code>n1</code> a node.
     * @param <code>n2</code> another node.
     */
	virtual void delete_edge (node n1, node n2);

	
	/**
	 * Sets the labels_as_weights flag to true (the default is false).
	 * If this flag is set the <code>load_edge_info_handler</code> will
	 * read any labels associated with an edge in a GML file an integer
	 * weight. This is useful if you want to import a GML graph generated
	 * by LEDA.
	 *
	 */
	void read_labels_as_weights () { labels_as_weights = true; };
	
	/**
	 * Extends graph::load_edge_info_handler to read in edge weights. These are
	 * stored in the list of key-value-pairs, where the key is the sttring "weight"
	 * and the value is the integer weight of that edge.
	 *
     * @param <code>e</code> edge parsed 
     * @param <code>list</code> pointer to the list of key-value-pairs of
     *                          this edge.
     * @see graph#load_edge_info_handler 
     */
	
    virtual void load_edge_info_handler (edge e, GML_pair* list);
    
	virtual void store_edge_double (edge e, char *key, double value);
	
	/**
	 * Handles an integer value associated with an edge. By default, it
	 * process the "weight" key by storing <code>value</code> in the
	 * <code>weight</code> edge map. This method is called by 
	 * <code>load_edge_info_handler</code>.
	 * @param n the node
	 * @param key the name of the item (e.g., "weight")
	 * @param value the contents of the key (e.g., "5")
	 *
	 */
	virtual void store_edge_integer (edge e, char *key, int value);
	/**
	 * Handles a string value associated with an edge. By default, it
	 * process the "label" key by storing <code>value</code> in the
	 * <code>label</code> edge map. If <code>labels_as_weights</code>
	 * is <code>true</true>, then converts label to integer and
	 * sets the edge weight to that value. This method is called by 
	 * <code>load_edge_info_handler</code>.
	 * @param e the edge
	 * @param key the name of the item (e.g., "label")
	 * @param value the contents of the key (e.g., "branch")
	 *
	 */
	virtual void store_edge_string (edge e, char *key, char *value);

	
    /**
     * Extends graph::post_new_edge_handler to ensure by default edge has weight 1,
     * and edge_label is "".
	 *
     * @param <code>e</code> created edge 
     * @see graph#new_edge
     */
    virtual void post_new_edge_handler(edge /*e*/) {
		//weight[e] = 1;
		//edge_label[e] = "";
		//edge_colour[e] = "black";
	}

	/**
	 * Extends graph::save_edge_info_handler to write the weight of the edge
	 * as a label when saving the graph to a GML file.
	 * @param ostream the stream being written to
	 * @param e the edge
	 * 
	 */
	virtual void save_edge_info_handler(std::ostream *os, edge e) const;


	/**
	 * Extends graph::load_node_info_handler iterator over the list
	 * of values associated with node <code>n</code> (if any) in the GML file. After
	 * determining the type of the associated value (integer, floating
	 * point, or string), the method calls the appropriate handler from
	 * store_node_double, store_node_double, or store_node_string.
	 * @param n the node being read
	 * @param list pointer to the list of paired values
	 * 
	 */
	virtual void load_node_info_handler(node n, GML_pair* list );
	
	
	virtual void store_node_double (node n, char *key, double value);
	
	virtual void store_node_integer (node n, char *key, int value);
	/**
	 * Handles a string value associated with a node. By default, it
	 * process the "label" key by storing <code>value</code> in the
	 * <code>label</code> node map. This method is called by 
	 * <code>load_node_info_handler</code>.
	 * @param n the node
	 * @param key the name of the item (e.g., "label")
	 * @param value the contents of the key (e.g., "root")
	 *
	 */
	virtual void store_node_string (node n, char *key, char *value);

    /**
     * Extends graph::post_new_node_handler to ensure by default node label is "". 
	 *
     * @param <code>n</code> created node 
     * @see graph#new_node
     */
    virtual void post_new_node_handler(node /*n*/)
	{
		//label[n] = "";
		//node_colour[n] = "white";
	}

	virtual void save_node_info_handler(std::ostream *os, node n) const;
	
	/**
	 * @param f output stream
	 *
	 * Write the graph in dot format (as used by programs in the
	 * <a href="http://www.research.att.com/sw/tools/graphviz/">GraphViz</A>
	 * package.
	 */
	virtual void save_dot(std::ostream &f, bool weights = false);
	
	/**
	 * @param fname output file name
	 *
	 * Write the graph in dot format (as used by programs in the
	 * <a href="http://www.research.att.com/sw/tools/graphviz/">GraphViz</A>
	 * package.
	 */
	virtual void save_dot (char *fname, bool weights = false);
	
	std::string get_node_label(node n) { return label[n]; };
	void set_node_label(node n, std::string s) { label[n] = s; };
	
	void set_edge_colour(edge e, std::string colour) { edge_colour[e] = colour; };
	void set_node_colour(node n, std::string colour) { node_colour[n] = colour; };


	double node_cliqueishness (node &n);
	double cliqueishness ();

			
protected:
	/** 
	 * A map between edges and an integer weight, being the weight of that edge
	 * in the graph. 
	 *
	 */
	edge_map<int> weight;
	edge_map<std::string> edge_label;
	node_map<std::string> label;
	
	bool labels_as_weights;
	
	// Styles
	node_map<std::string> node_colour;
	edge_map<std::string> edge_colour;


	
};	

#endif
