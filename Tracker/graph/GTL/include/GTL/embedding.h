/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   embedding.h
//
//==========================================================================
// $Id: embedding.h,v 1.20 2003/06/11 11:28:21 raitner Exp $

#ifndef __EMBEDDING__H
#define __EMBEDDING__H

#include <GTL/GTL.h>
#include <GTL/graph.h>
#include <GTL/st_number.h>
#include <GTL/symlist.h>

__GTL_BEGIN_NAMESPACE

/**
 * @brief Ordered adjacency lists as a result of planarity testing.
 *
 * It is known that if a graph is planar the adjacency list of every node
 * can be ordered in such a way that it reflects the order the adjacent
 * edges will have in a planar drawing around the node. Although the tested
 * graph might have been directed the planar embedding one gets will always
 * correspond to the underlying undirected graph, i.e. an edge from @c n1 to
 * @c n2 will occurr in both adjacency lists.
 */
class GTL_EXTERN planar_embedding 
{
public:
    /**
     * @internal
     */
    typedef symlist<edge> adj_list;

    /**
     * @internal
     */
    typedef symlist<edge>::iterator iterator;
private:
    /**
     * @internal
     * Creates an empty planar embedding not related to any graph.
     * @note At the moment planar embedding are thought as an output of
     * planarity testing, this why they can't be constructed from scratch. 
     */
    planar_embedding() : G(0)
    {
    }
public:
    /**
     * 
     * Make this object a copy of @p em.
     *
     * @param em planar embedding
     */
    planar_embedding(const planar_embedding& em);

    /**
     * 
     * Destructor.
     */
    virtual ~planar_embedding()
    {
    }

    /**
     * 
     * Assigns @p em to this object. All former information in this object
     * will be deleted.
     *
     * @param em 
     *
     * @return reference to this object
     */
    planar_embedding& operator=(const planar_embedding& em); 
private:
    /**
     * @internal
     * Initializes adjacency lists.
     * 
     * @param G graph 
     */
    void init(graph& G);

    /**
     * @internal
     * Turns adjacency list of node @p n.
     *
     * @param n node.
     */
    void turn(node n);

    /**
     * @internal
     * Insert edge @p e at the end of adjacency list of @p n.
     *
     * @param n node 
     * @param e edge to be inserted
     *
     * @return iterator to position of insertion
     */
    iterator push_back(node n, edge e);

    /**
     * @internal
     * Insert edge @p e at the beginning of adjacency list of @p n.
     *
     * @param n node
     * @param e edge to be inserted
     *
     * @return iterator to position of insertion
     */
    iterator push_front(node n, edge e);

    /**
     * @internal
     * Insert selfloop @p e.
     *
     * @param @p e selfloop 
     */
    void insert_selfloop (edge e);

    /**
     * @internal
     * Returns position of edge @p e in adjacency list of node @p n.
     *
     * @param n node
     * @param e adjacent edge
     *
     * @return position of @p e
     */
    iterator& pos (node, edge);
public:
    /**
     * 
     * Returns reference to ordered adjacency list of node @p n.
     *
     * @param n node
     *
     * @return ordered adjacency list
     */
    adj_list& adjacency(node n)
    {
	return adj[n];
    }

    /**
     * 
     * Returns reference to ordered adjacency list of node @p n.
     *
     * @param n node
     *
     * @return ordered adjacency list
     */
    const adj_list& adjacency(node n) const
    {
	return adj[n];
    }

    /**
     * 
     * Start iteration through adjacency list of node @p n.
     *
     * @param n node
     *
     * @return start iterator
     */
    iterator adj_edges_begin(node n)
    {
	return adj[n].begin();
    } 

    /**
     * 
     * End of iteration through adjacency list of node @p n.
     *
     * @param @p n node
     *
     * @return one-past the end iterator
     */
    iterator adj_edges_end(node n)
    {
	return adj[n].end();
    }

    /**
     * 
     * Returns the cyclic successor of edge @p e in the adjacency list of
     * node @p n.
     *
     * @param n node
     * @param e edge adjacent to @p n
     *
     * @return edge following @p e in adjacency of @p n
     */
    edge cyclic_next(node n, edge e);

    /**
     * 
     * Returns the cyclic predecessor of edge @p e in the adjacency list of
     * node @p n.
     *
     * @param n node
     * @param e edge adjacent to @p n
     *
     * @return edge preceding @p e in adjacency of @p n
     */
    edge cyclic_prev(node n, edge e);


    /**
     * 
     * Writes embedding with st-numbers as given by @p st to @p os.
     *
     * @param os output stream
     *
     * @param st st-numbers
     */
	void write_st(std::ostream& os, st_number& st);

    /**
     * 
     * Returns list of selfloops contained in the graph. These will not
     * occur in the adjacency lists.
     *
     * @retval list of selfloops
     */
	edges_t& selfloops()
    {
	return self;
    }

    /**
     * 
     * Returns list of selfloops contained in the graph. These will not
     * occur in the adjacency lists.
     *
     * @retval list of selfloops
     */
	const edges_t& selfloops() const
    {
	return self;
    }

    /**
     * 
     * Returns list of multiple edges contained in the graph. These are
     * edges for which there is already another edge connecting the same
     * endpoints is contained in the adjacency lists. Please note that the
     * notion "connecting" is meant in an undirected sense. These edges will
     * not occur it the adjacency lists.
     *
     * @retval list of multiple edges
     */
	edges_t& multiple_edges()
    {
	return multi;
    }

    /**
     * 
     * Returns list of multiple edges contained in the graph. These are
     * edges for which there is already another edge connecting the same
     * endpoints is contained in the adjacency lists. Please note that the
     * notion "connecting" is meant in an undirected sense. These edges will
     * not occur it the adjacency lists.
     *
     * @retval list of multiple edges
     */
	const edges_t& multiple_edges() const
    {
	return multi;
    }

    /**
     * 
     * Used for debugging only. Checks whether this is a correct planar 
     * embedding by checking the faces of the graph, i.e. at any node
     * starting with an arbitrary adjacent edge and advancing along @c
     * cyclic_next the start node must be met through the edge given by @c
     * cyclic_prev of the edge we started with. 
     *
     * @retval true iff embedding is correct
     */
    bool check();

    /**
     * @internal
     */
    friend class planarity;

    /**
     * @internal
     */
    friend class pq_tree;

    /**
     * @internal
     */
	GTL_EXTERN friend std::ostream& operator<< (std::ostream&, planar_embedding&);
private:
    /**
     * @internal
     * Graph.
     */
    graph* G;

    /**
     * @internal
     * Adjacency lists.
     */
    node_map<adj_list> adj;

    /**
     * @internal
     * Positions of edges in its source's adjacency list.
     */
    edge_map<adj_list::iterator> s_pos;

    /**
     * @internal
     * Positions of edges in its target's adjacency list.
     */
    edge_map<adj_list::iterator> t_pos;

    /**
     * @internal
     * Selfloops.
     */
	edges_t self;

    /**
     * @internal
     * Multiple edges.
     */
	edges_t multi;
};


// class face
// {    
// public:
//     face (planar_embedding& em, node n, edge e) : embed (em),
// 	start (n), first (e) { }
//     virtual ~face () { }

// private:    
//     planar_embedding& embed;
//     node start;
//     edge first;

//     friend class planar_embedding;
// };

// struct _face_iterator
// {


//     face& _face;
// };

__GTL_END_NAMESPACE

#endif

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
