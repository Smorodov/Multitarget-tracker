/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   graph.cpp
//
//==========================================================================
// $Id: graph.cpp,v 1.58 2003/01/14 16:47:14 raitner Exp $

#include <GTL/graph.h>
#include <GTL/node_data.h>
#include <GTL/edge_data.h>
#include <GTL/node_map.h>

#include <GTL/dfs.h>
#include <GTL/topsort.h>

#include <cassert>
#include <cstdio>

#include <algorithm>
#include <queue>
#include <set>
#include <map>

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

__GTL_BEGIN_NAMESPACE

//--------------------------------------------------------------------------
//   Con-/Destructors
//--------------------------------------------------------------------------

graph::graph() :
    directed(true),
    nodes_count(0), edges_count(0),
    hidden_nodes_count(0), hidden_edges_count(0),
    free_node_ids_count(0), free_edge_ids_count(0)
{
}

graph::graph(const graph &G) :
    directed(G.directed),
    nodes_count(0), edges_count(0),
    hidden_nodes_count(0), hidden_edges_count(0),
    free_node_ids_count(0), free_edge_ids_count(0)
{
    copy (G, G.nodes.begin(), G.nodes.end());
}


graph::graph(const graph& G, const nodes_t& nod) :
    directed(G.directed),
    nodes_count(0), edges_count(0),
    hidden_nodes_count(0), hidden_edges_count(0),
    free_node_ids_count(0), free_edge_ids_count(0)
{
    copy (G, nod.begin(), nod.end());
}


graph::graph (const graph& G, 
	nodes_t::const_iterator it,
	nodes_t::const_iterator end) :
    directed(G.directed),
    nodes_count(0), edges_count(0),
    hidden_nodes_count(0), hidden_edges_count(0),
    free_node_ids_count(0), free_edge_ids_count(0)
{
    copy (G, it, end);
}

void graph::copy (const graph& G, 
	nodes_t::const_iterator it,
	nodes_t::const_iterator end)
{
    node_map<node> copy (G, node());    
	nodes_t::const_iterator n_it;
	nodes_t::const_iterator n_end;

    for(n_it = it, n_end = end; n_it != n_end; ++n_it)
    {
	copy[*n_it] = new_node();
    }

    for(n_it = it, n_end = end; n_it != n_end; ++n_it)
    {
	node::out_edges_iterator e_it, e_end;
	
	for (e_it = n_it->out_edges_begin(), e_end = n_it->out_edges_end();
	     e_it != e_end; ++e_it) {

	    if (copy[e_it->target()] != node()) {
		new_edge(copy[e_it->source()], copy[e_it->target()]);
	    }
	}
    }    
}



graph::~graph() 
{
    clear();
}

//-------------------------------------------------------------------------
// Output 
//------------------------------------------------------------------------- 

GTL_EXTERN std::ostream& operator<< (std::ostream& os, const graph& G) {
    node n;
    edge out;
	std::string conn;

    if (G.is_directed()) 
	conn = "-->"; 
    else 
	conn = "<-->"; 

    forall_nodes (n, G) {
	os << n << ":: ";

	forall_adj_edges (out, n) {
	    os << conn << n.opposite (out);
	}
	
	os << std::endl;
    }
    
    return  os;
}

//--------------------------------------------------------------------------
//   Directed/Undirected
//--------------------------------------------------------------------------

void graph::make_directed()
{
    if (!directed)
    {
	pre_make_directed_handler();
	directed = true;
	post_make_directed_handler();
    }
}

void graph::make_undirected()
{
    if (directed)
    {
	pre_make_undirected_handler();
	directed = false;
	post_make_undirected_handler();
    }
}

bool graph::is_directed() const
{
    return directed;
}

bool graph::is_undirected() const
{
    return !directed;
}

//--------------------------------------------------------------------------
//   Creation
//--------------------------------------------------------------------------

node graph::new_node()
{
    pre_new_node_handler();

    // create node
    
    node n;
    n.data = new node_data;

    // set data variables

    n.data->id = new_node_id();
    n.data->owner = this;
    n.data->pos = nodes.insert(nodes.end(), n);
    n.data->hidden = false;
    ++nodes_count;

    // done
    
    post_new_node_handler(n);

    return n;
}

edge graph::new_edge(node source, node target)
{
    assert(source.data);
    assert(target.data);
    assert(source.data->owner == this);
    assert(target.data->owner == this);

    pre_new_edge_handler(source, target);

    // create edge
    
    edge e;
    e.data = new edge_data;
    
    // set id

    e.data->owner = this;
    e.data->id = new_edge_id();	
    
    // set sources and targets
    
    e.data->nodes[0].push_back(source);
    e.data->nodes[1].push_back(target);

    // set pos
    
    e.data->pos = edges.insert(edges.end(), e);
    e.data->hidden = false;
    ++edges_count;

    // set adj_pos
    
	edges_t& source_adj = source.data->edges[1];
	edges_t& target_adj = target.data->edges[0];
    
    e.data->adj_pos[0].push_back(source_adj.insert(source_adj.begin(), e));
    e.data->adj_pos[1].push_back(target_adj.insert(target_adj.begin(), e));

    // done
    
    post_new_edge_handler(e);

    return e;
}

edge graph::new_edge(const nodes_t &/*sources*/, const nodes_t &/*targets*/)
{
    // not implemented

    return edge();
}

//--------------------------------------------------------------------------
//   Deletion
//--------------------------------------------------------------------------

void graph::del_node(node n)
{
    assert (n.data);
    assert (n.data->owner == this);

    // delete edges

    while(n.in_edges_begin() != n.in_edges_end())
    {
	del_edge (*n.in_edges_begin());
    }

    while(n.out_edges_begin() != n.out_edges_end())
    {
	del_edge (*n.out_edges_begin());
    }

    //
    // delete hidden edges adjacent to n.  
    // 
    // [ TODO ] This is only a quick fix and should be thought
    // over some time or the other.
    // 

	edges_t::iterator it = hidden_edges.begin();
	edges_t::iterator end = hidden_edges.end();

    while (it != end)
    { 
	if (it->source() == n || it->target() == n)
	{
	    delete it->data;
	    it = hidden_edges.erase (it);
	}
	else
	{ 
	    ++it;
	}
    }

    // delete node

    pre_del_node_handler(n);

    nodes.erase(n.data->pos);
    --nodes_count;
    free_node_ids.push_back(n.data->id);
    ++free_node_ids_count;
    delete n.data;

    post_del_node_handler();
}

void graph::del_edge(edge e)
{
    assert (e.data->owner == this);
    assert (e.data->owner == this);

    pre_del_edge_handler(e);
    node s = e.source();
    node t = e.target();

    e.remove_from(0);
    e.remove_from(1);
    edges.erase(e.data->pos);
    --edges_count;
    free_edge_ids.push_back(e.data->id);
    ++free_edge_ids_count;
    delete e.data;

    post_del_edge_handler(s, t);
}

void graph::clear()
{
    pre_clear_handler();

    del_list(edges);
    del_list(hidden_edges);
    del_list(nodes);
    del_list(hidden_nodes);

    free_node_ids.clear();
    free_edge_ids.clear();

    nodes_count = edges_count = 0;
    hidden_nodes_count = hidden_edges_count = 0;
    free_node_ids_count = free_edge_ids_count = 0;
    
    post_clear_handler();
}

void graph::del_all_nodes()
{
    assert(false);
    // not fully implemented:
    //  * update id lists !!!
    //  * call handlers

    del_list(edges);
    del_list(nodes);

    nodes_count = edges_count = 0;
}

void graph::del_all_edges()
{
    assert(false);
    // not fully implemented:
    //  * update id lists !!!
    //  * call handlers
    del_list(edges);

    edges_count = 0;
    
	nodes_t::iterator it = nodes.begin();
	nodes_t::iterator end = nodes.end();

    while(it != end)
    {
	it->data->edges[0].clear();
	it->data->edges[1].clear();
    }
}

//--------------------------------------------------------------------------
//   Informations
//--------------------------------------------------------------------------



bool graph::is_bidirected(edge_map<edge>& rev) const {
    edge e1;
    node target, source;
    bool bidirected = true;
    node::out_edges_iterator it;
    node::out_edges_iterator end;

    forall_edges (e1, *this) {
	target = e1.target ();
	source = e1.source ();
	end = target.out_edges_end ();
	it = target.out_edges_begin ();

	//
	// Search all out-edges of target if they are connected to the actual
	// edges source.
	//

	while (it != end) { 
	    if (it->target () == source) {
		break;
	    }
	    ++it;
	}

	if (it == end) {
	    bidirected = false;
	    rev[e1] = edge ();
	} else {
	    rev[e1] = *it;
	}
    }
    
    return bidirected;
}

bool graph::is_connected() const
{
    bool save_directed = directed;
    directed = false;
    
    dfs d;
    d.run(*const_cast<graph *>(this));

    directed = save_directed;

    return d.number_of_reached_nodes() == number_of_nodes();
}

bool graph::is_acyclic() const
{
    topsort t;
    t.run(*const_cast<graph *>(this));

    return t.is_acyclic();
}

int graph::number_of_nodes() const
{
    return nodes_count - hidden_nodes_count;
}

int graph::number_of_edges() const
{
    return edges_count - hidden_edges_count;
}

node graph::center() const 
{
    int min_excentricity = number_of_nodes()+1;
    node n, center;
    forall_nodes(n, *this) 
    {
	int excentricity = n.excentricity();
	if(excentricity < min_excentricity) 
	{
	    center = n;
	    min_excentricity = excentricity;
	}
    }
    return center;
}

//--------------------------------------------------------------------------
//   Iterators
//--------------------------------------------------------------------------

graph::node_iterator graph::nodes_begin() const
{
    return nodes.begin();
}

graph::node_iterator graph::nodes_end() const
{
    return nodes.end();
}

graph::edge_iterator graph::edges_begin() const
{
    return edges.begin();
}

graph::edge_iterator graph::edges_end() const
{
    return edges.end();
}

//--------------------------------------------------------------------------
//   Node/Edge lists
//--------------------------------------------------------------------------

nodes_t graph::all_nodes() const
{
    return nodes;
}

edges_t graph::all_edges() const
{
    return edges;
}

//--------------------------------------------------------------------------
//   Hide
//   If an edge is already hidden (this really happens :-), it will not be 
//   hidden for the second time
//--------------------------------------------------------------------------

void graph::hide_edge (edge e) 
{
    assert (e.data->owner == this);
    assert (e.data->owner == this);

    pre_hide_edge_handler (e);
    
    if (!e.is_hidden()) {

	//
	// remove e from all sources and targets adjacency lists
	//
	e.remove_from(0);
	e.remove_from(1);
    
	//
	// clear the list of positions
	//
	e.data->adj_pos[0].erase 
	    (e.data->adj_pos[0].begin(), e.data->adj_pos[0].end()); 
	e.data->adj_pos[1].erase 
	    (e.data->adj_pos[1].begin(), e.data->adj_pos[1].end());

	//
	// remove e from the list of all edges
	//
	edges.erase (e.data->pos);

	//
	// insert e in hidden edges list
	//
	e.data->pos = hidden_edges.insert(hidden_edges.end(), e);
	e.data->hidden = true;
	++hidden_edges_count;
    }

    post_hide_edge_handler (e);
}

//--------------------------------------------------------------------------
//   restore_edge
//   An edge will be restored only if it is hidden (sounds wise, hmm ...)
//--------------------------------------------------------------------------

void graph::restore_edge (edge e)
{
    assert (e.data->owner == this);
    assert (e.data->owner == this);

    pre_restore_edge_handler (e);

    if (e.is_hidden()) {
	//
	// remove e from hidden edges list
	//
	hidden_edges.erase (e.data->pos);
	--hidden_edges_count;

	//
	// for each source of e insert e in its list of out-edges and store
	// the position in e's list of positions
	//
	nodes_t::iterator it;
	nodes_t::iterator end = e.data->nodes[0].end();

	for (it = e.data->nodes[0].begin (); it != end; ++it)
	{
		edges_t& adj = it->data->edges[1];
	    e.data->adj_pos[0].push_back(adj.insert(adj.begin(), e));
	}

	//
	// for each target of e insert e in its list of in-edges and store 
	// the pos
	//
	end = e.data->nodes[1].end();
    
	for (it = e.data->nodes[1].begin (); it != end; ++it)
	{
		edges_t& adj = it->data->edges[0];
	    e.data->adj_pos[1].push_back(adj.insert(adj.begin(), e));
	}
    
	e.data->pos = edges.insert(edges.end(), e);
	e.data->hidden = false;
    }    
    
    post_restore_edge_handler (e);
}

//--------------------------------------------------------------------------
//   Hide
//   If an node is already hidden (this really happens :-), it will not be 
//   hidden for the second time
//   Note: also all adjacent edges will be hidden
//--------------------------------------------------------------------------

edges_t graph::hide_node(node n)
{
    assert (n.data->owner == this);

    pre_hide_node_handler (n);
	edges_t implicitly_hidden_edges;
    
    if (!n.is_hidden()){
	// hide all connected egdes
	for (int i = 0; i <= 1; ++i)
	{
		edges_t::iterator end = n.data->edges[i].end();
		edges_t::iterator edge = n.data->edges[i].begin();
	    while (edge != end)
	    {
		implicitly_hidden_edges.push_back(*edge);
		hide_edge(*edge);
		edge = n.data->edges[i].begin();
	    }
	}

	// hide node
	hidden_nodes.push_back(n);
	nodes.erase(n.data->pos);
	n.data->hidden = true;
	++hidden_nodes_count;
    }

    post_hide_node_handler (n);

    return implicitly_hidden_edges;
}

//--------------------------------------------------------------------------
//   restore_node
//   A node will be restored only if it is hidden (sounds wise, hmm ...)
//   connected nodes won't be restored automatically !
//--------------------------------------------------------------------------

void graph::restore_node (node n)
{
    assert (n.data->owner == this);

    pre_restore_node_handler(n);

	if (n.is_hidden())
	{
		// node is hidden

		nodes.push_back(n);
		n.data->pos = --nodes.end();

#if 1
		hidden_nodes.remove(n);
#else
		hidden_nodes.erase(std::remove(hidden_nodes.begin(), hidden_nodes.end(), n), hidden_nodes.end());
#endif
		n.data->hidden = false;
		--hidden_nodes_count;
	}

    post_restore_node_handler (n);
}


void graph::induced_subgraph(nodes_t& sub_nodes)
{
    node_map<int> in_sub (*this, 0);
	nodes_t::iterator it, end, tmp;

    for (it = sub_nodes.begin(), end = sub_nodes.end(); it != end; ++it) {
	in_sub[*it] = 1;
    }

    it = nodes.begin();
    end = nodes.end();
    
    while (it != end) {
	tmp = it;
	++tmp;

	if (!in_sub[*it]) {
	    hide_node (*it);
	}

	it = tmp;
    }
}

void graph::restore_graph () 
{
	nodes_t::iterator it, end, tmp;

    it = hidden_nodes.begin();
    end = hidden_nodes.end();

	while (it != end)
	{
		tmp = it;
		++tmp;
		restore_node(*it);
		it = tmp;
	}

	edges_t::iterator e_it = hidden_edges.begin();
	edges_t::iterator e_end = hidden_edges.end();

    while (e_it != e_end)
	{
		edges_t::iterator e_tmp = e_it;
		++e_tmp;
		restore_edge (*e_it);
		e_it = e_tmp;
    }
}

//--------------------------------------------------------------------------
//   Node/edge numbering
//--------------------------------------------------------------------------

int graph::number_of_ids(node) const
{
    return
	free_node_ids_count +
	nodes_count;
}

int graph::number_of_ids(edge) const
{
    return
	free_edge_ids_count +
	edges_count; 
}

int graph::new_node_id()
{
    if(free_node_ids.empty())
	return nodes_count;
   
    int id = free_node_ids.back();
    free_node_ids.pop_back();
    --free_node_ids_count;
    return id;
}

int graph::new_edge_id()
{
    if(free_edge_ids.empty())
	return edges_count;
   
    int id = free_edge_ids.back();
    free_edge_ids.pop_back();
    --free_edge_ids_count;
    return id;
}

//--------------------------------------------------------------------------
//   Utilities
//--------------------------------------------------------------------------

void graph::del_list(nodes_t& l)
{
	nodes_t::const_iterator it = l.begin();
	nodes_t::const_iterator end = l.end();

    while(it != end)
    {
	delete it->data;
	++it;
    }
    
    l.clear();
}

void graph::del_list(edges_t& l)
{
	edges_t::const_iterator it = l.begin();
	edges_t::const_iterator end = l.end();

    while(it != end)
    {
	delete it->data;
	++it;
    }
    
    l.clear();
}

//--------------------------------------------------------------------------
//   Others
//--------------------------------------------------------------------------

edges_t graph::insert_reverse_edges() {
	edges_t rev;
    edge e;

    node::out_edges_iterator it, end;

    forall_edges (e, *this) {
	it = e.target().out_edges_begin();
	end = e.target().out_edges_end();

	while (it != end) {
	    if (it->target() == e.source()) 
		break;
	    ++it;
	}
	
	if (it == end) {
	    rev.push_back(new_edge (e.target(), e.source()));
	}
    }

    return rev;
}

node graph::choose_node () const
{
    // Well, probably doesn't guarantee uniform distribution :-)
    return nodes.empty() ? node() : nodes.front();
}

//--------------------------------------------------------------------------
//   I/O 
//--------------------------------------------------------------------------

GML_error graph::load (const char* filename, bool preserve_ids) {

    GML_stat stat;
    stat.key_list = NULL;
    GML_pair* key_list;
    GML_pair* orig_list;
    
    FILE* file = fopen (filename, "r");
    
    if (!file) {
	stat.err.err_num = GML_FILE_NOT_FOUND;
	return stat.err;
    } 

    GML_init ();
    key_list = GML_parser (file, &stat, 0);
    fclose (file);

    if (stat.err.err_num != GML_OK) {
	GML_free_list (key_list, stat.key_list);
	return stat.err;
    }
    
    //
    // This file is a valid GML-file, let's build the graph.
    // 

    clear();
    orig_list = key_list;

    

    //
    // get the first entry with key "graph" in the list
    // 

    while (key_list) {
	if (!strcmp ( "graph", key_list->key)) {
	    break;
	}
	
	key_list = key_list->next;
    }

    assert (key_list);

    key_list = key_list->value.list;
    GML_pair* graph_list = key_list;

    GML_pair* tmp_list;
    // GML_pair* node_entries = 0;
    // GML_pair* edge_entries = 0;
    
    std::list<std::pair<int,GML_pair*> > node_entries;
	std::list<std::pair<std::pair<int, int>, GML_pair*> > edge_entries;
    
    int num_nodes = 0; 

    bool target_found;
    bool source_found;

    //
    // Node and edge keys may come in arbitrary order, so sort them such
    // that all nodes come before all edges.
    //
    
    while (key_list) {
	if (!strcmp (key_list->key, "node")) {

	    //
	    // Search the list associated with this node for the id
	    //

	    assert (key_list->kind == GML_LIST);
	    tmp_list = key_list->value.list;
		std::pair<int, GML_pair*> n;
	    n.second = tmp_list;

	    while (tmp_list) {
		if (!strcmp (tmp_list->key, "id")) {
		    assert (tmp_list->kind == GML_INT);
		    n.first = tmp_list->value.integer;
		    break;
		}

		tmp_list = tmp_list->next;
	    }

	    assert (tmp_list);
	    node_entries.push_back(n);
	    ++num_nodes;
	    
	} else if (!strcmp (key_list->key, "edge")) {

	    //
	    // Search for source and target entries
	    //

	    assert (key_list->kind == GML_LIST);
	    tmp_list = key_list->value.list;
	    source_found = false;
	    target_found = false;
		std::pair<std::pair<int, int>, GML_pair*> e;
	    e.second = tmp_list;

	    while (tmp_list) {
		if (!strcmp (tmp_list->key, "source")) {
		    assert (tmp_list->kind == GML_INT);
		    source_found = true;
		    e.first.first = tmp_list->value.integer;
		    if (target_found) break;

		} else if (!strcmp (tmp_list->key, "target")) {
		    assert (tmp_list->kind == GML_INT);
		    target_found = true;
		    e.first.second = tmp_list->value.integer;
		    if (source_found) break;
		}

		tmp_list = tmp_list->next;
	    }

	    assert (source_found && target_found);
	    edge_entries.push_back (e);

	} else if (!strcmp (key_list->key, "directed")) {	    
	    directed = (key_list->value.integer != 0);
	}	

	key_list = key_list->next;
    }

    //
    // make this graph the graph decribed in list
    //

	std::map<int, node> id_2_node;
    node source, target;
    node tmp_node;
    edge tmp_edge;
    std::list<std::pair<int,GML_pair*> >::iterator it, end;
	std::vector<int> node_ids;
    node_ids.reserve(num_nodes);

    for (it = node_entries.begin(), end = node_entries.end();
	 it != end; ++it) {
	tmp_node = new_node ();
	if (preserve_ids) {
	    tmp_node.data->id = it->first;
	    node_ids.push_back(it->first);
	}
	id_2_node[it->first] = tmp_node;	
	load_node_info_handler (tmp_node, it->second);
    }

	std::list<std::pair<std::pair<int, int>, GML_pair*> >::iterator eit, eend;
    for (eit = edge_entries.begin(), eend = edge_entries.end();
	 eit != eend; ++eit) {
	source = id_2_node[eit->first.first];
	target = id_2_node[eit->first.second];
	tmp_edge = new_edge (source, target);
	load_edge_info_handler (tmp_edge, eit->second);
    }

    load_graph_info_handler (graph_list);
    top_level_key_handler (orig_list);

    sort(node_ids.begin(),node_ids.end());

	std::vector<int>::iterator iit, iend;
    int prev = 0;

    for (iit = node_ids.begin(), iend = node_ids.end();
	 iit != iend; ++iit)
    {
	if (iit != node_ids.begin()) {
	    free_node_ids_count += *iit - prev - 1;  
	} else {
	    free_node_ids_count += *iit;
	}
	prev = *iit;
    }

    GML_free_list (orig_list, stat.key_list);
    stat.err.err_num = GML_OK;
    return stat.err;
}

void graph::load_node_info_handler (node /*n*/, GML_pair* /*li*/) {
}


void graph::load_edge_info_handler (edge /*e*/, GML_pair* /*li*/) {
}

void graph::load_graph_info_handler (GML_pair* /*li*/) {
}

void graph::top_level_key_handler (GML_pair* /*li*/) {
}


void graph::save(std::ostream* file) const {
    pre_graph_save_handler (file);
	(*file) << "graph [" << std::endl;
	(*file) << "directed " << (directed ? "1" : "0") << std::endl;

    node_iterator it = nodes_begin();
    node_iterator end = nodes_end();

    for (;it != end; ++it) {
	(*file) << "node [\n" << "id " << it->id() << "\n";
	save_node_info_handler (file, *it);
	(*file) << " ]" << std::endl;
    }

    edge_iterator e_it = edges_begin();
    edge_iterator e_end = edges_end();
    
    for (; e_it != e_end; ++e_it) {
	(*file) << "edge [\n" << "source " << e_it->source().id() << "\n";
	(*file) << "target " << e_it->target().id() << "\n";
	save_edge_info_handler (file, *e_it);
	(*file) << " ]" << std::endl;
    }

    save_graph_info_handler (file);

	(*file) << "]" << std::endl;
    after_graph_save_handler (file);
}

int graph::save (const char* filename) const {
    
	std::ofstream file(filename);
    if (!file) return 0;
    
    save (&file);

    return 1;
}


// void graph::top_level_key_handler (GML_pair_list::const_iterator it,
//     GML_pair_list::const_iterator end) 
// {
//     cout << "TOP_LEVEL_HANDLER" << endl;

//     for (; it != end; ++it) {
// 	cout << *it << endl;
//     }
// }

// void graph::load_graph_info_handler (GML_pair_list::const_iterator it,
//     GML_pair_list::const_iterator end)
// {
//     cout << "GRAPH_INFO_HANDLER" << endl;

//     for (; it != end; ++it) {
// 	cout << *it << endl;
//     }
// }

// void graph::load_node_info_handler (node n, GML_pair_list::const_iterator it,
//     GML_pair_list::const_iterator end)
// {
//     cout << "NODE_INFO_HANDLER for " << n << endl;

//     for (; it != end; ++it) {
// 	cout << *it << endl;
//     }
// }

// void graph::load_edge_info_handler (edge e, GML_pair_list::const_iterator it,
//     GML_pair_list::const_iterator end)
// {
//     cout << "EDGE_INFO_HANDLER for " << e.source() << "-->" 
// 	 << e.target()  << endl;

//     for (; it != end; ++it) {
// 	cout << *it << endl;
//     }
// }

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
