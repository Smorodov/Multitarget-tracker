/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   st_number.h
//
//==========================================================================
// $Id: st_number.h,v 1.17 2002/12/20 08:26:08 chris Exp $

#ifndef GTL_ST_NUMBER_H
#define GTL_ST_NUMBER_H

#include <GTL/GTL.h>
#include <GTL/graph.h>
#include <GTL/node_map.h>
#include <GTL/edge_map.h>
#include <GTL/algorithm.h>

#include <list>
#include <utility>

__GTL_BEGIN_NAMESPACE

/**
* @internal
*/
class GTL_EXTERN pathfinder 
{
public:
    //---------------------------------------------------------- CONSTRUCTOR
	
    /**
     * @internal
     */
    pathfinder(const graph& G, edge st, node s);    
	
    /**
     * @internal
     */
    bool is_valid()
    {
	return is_biconn;
    }
	
    //------------------------------------------------------------- ITERATOR
	
    /**
     * @internal
     */
    class const_iterator 
    {
    public:
        /**
	 * @internal
	 */
	const_iterator(pathfinder& _pf) : pf (_pf)
	{
	}

        /**
	 * @internal
	 */
	const_iterator(pathfinder& _pf, node n);	
		
        /**
	 * @internal
	 */
	const_iterator& operator++();
        /**
	 * @internal
	 */
	const_iterator operator++(int);
        /**
	 * @internal
	 */
	const node& operator*() const 
	{
	    return curr;
	}
		
        /**
	 * @internal
	 */
	bool operator==(const const_iterator& it) 
	{
	    return curr == it.curr;
	}

        /**
	 * @internal
	 */
	bool operator!=(const const_iterator& it)
	{
	    return curr != it.curr;
	}
    private:
        /**
	 * @internal
	 */
	enum iteration_state {END, UP, DOWN};

        /**
	 * @internal
	 */
	iteration_state state;

        /**
	 * @internal
	 */
	node curr;

        /**
	 * @internal
	 */
	pathfinder& pf;
    };
	
    /**
     * @internal
     */
    const_iterator path(node n)
    {
	return const_iterator(*this, n);
    }
	
    /**
     * @internal
     */
    const_iterator end()
    {
	return const_iterator (*this);
    }
	
private:
    //------------------------------------------------------------ FUNCTIONS
		
    /**
     * @internal
     */
    void dfs_sub (node&, node&);
		
    //-------------------------------------------------------------- MEMBERS 
		
    /**
     * @internal
     */
    node_map<int> dfs_num;

    /**
     * @internal
     */
    node_map<int> low_num;

    /**
     * @internal
     */
    node_map<edges_t> tree;

    /**
     * @internal
     */
    node_map<edges_t> back;

    /**
     * @internal
     */
    node_map<edges_t> forward;

    /**
     * @internal
     */
    node_map<edges_t::iterator> to_low;

    /**
     * @internal
     */
    node_map<edges_t::iterator> to_father;
		
    /**
     * @internal
     */
    typedef std::pair<edges_t::iterator, edges_t::iterator> pos_pair;

    /**
     * @internal
     */
    edge_map<pos_pair> pos;

    /**
     * @internal
     */
    node_map<int> used;
		
    /**
     * @internal
     */
    int act_dfs_num;

    /**
     * @internal
     */
    int new_nodes;

    /**
     * @internal
     */
    bool is_biconn;
		
    /**
     * @internal
     * Allows const_iterator private access.
     */
    friend class const_iterator;
};

/**
 * @brief ST-number algorithm.
 *
 * Encapsulates the st-number algorithm together with all the data produced
 * by it. 
 * <p>
 * Assigns an integer <tt>st[n]</tt> to each node @c n of a undirected,
 * biconnected graph, such that each node is connected with at least one
 * node having a smaller and with at least one having a larger number than
 * itself. The only exception to this rule are the endpoints of edge @a st
 * connecting nodes @a s (st-number 1) and @c t (highest st-number).
 * <p>
 * The following options are supported:
 * - #st_edge sets/retrieves the edge that connects the node with the lowest
 *   number to that with the highest.
 * - #s_node sets/retrieves that endpoints of the @a st_edge, which gets
 *   number 1.
 */
class GTL_EXTERN st_number : public algorithm 
{
public:
    /**
     * @brief Default constructor.
     * Creates st-number object. Please note that there are no reasonable 
     * default settings for the parameters, i.e. the edge @s st connecting
     * the lowest with highest numbers node and which of its endpoints
     * should get number 1 (= node @a s) has to be specified always.
     */
    st_number() : algorithm()
    {
    }

    /**
     * @brief Destructor
     */
    virtual ~st_number()
    {
    }
	
    /**
     * @brief Sets edge @a st for the next run. 
     *
     * @param e edge @a st
     */
    void st_edge(edge e)
    {
	st = e;
    }

    /**
     * @brief Get edge @a st.
     *
     * @retval edge @a st
     */
    edge st_edge() const
    {
	return st;
    }

    /**
     * @brief Sets node @a s for next run.
     *
     * This must be one of the endpoints of edge @a st. This node will get
     * st-number 1 and thus the other endpoint will get the highest
     * st-number.
     *
     * @param n node @a s
     */
    void s_node(node n) 
    {
	s = n;
    }

    /**
     * @brief Get node @a s.
     *
     * @retval node @a s
     */
    node s_node() const
    {
	return s;
    }

    /**
     * @brief Returns st-number of node @p n as determined in the last run.
     *
     * @param n node
     *
     * @return st-number of @p n
     */
    int& operator[](const node& n)
    {
	return st_num[n];
    }
    
    /**
     * @internal
     */
    typedef nodes_t::iterator iterator;

    /**
     * @internal
     */
    typedef nodes_t::reverse_iterator reverse_iterator;
	
    /**
     * @brief Iteration through the nodes of graph st-numbered in last
     *	      run in st-number order, i.e. from 1 to highest st-number.
     *
     * @return start of iteration through nodes in st-number order 
     */
    iterator begin()
    {
	return st_ord.begin();
    }	

    /**
     * @brief Iteration through nodes of graph in st-number order.
     *
     * @return end of iteration through nodes of graph in st-number order
     */
    iterator end()
    {
	return st_ord.end();
    }
	
    /**
     * @brief Iteration through the nodes of graph st-numbered in last run
     *	      in reverse st-number order, i.e. from highest st-number down
     *	      to 1.
     *
     * @return start of iteration through nodes in reverse st-number order 
     */
    reverse_iterator rbegin()
    {
	return st_ord.rbegin();
    }
	
    /**
     * @brief End of iteration through nodes of graph in reverse st-number
     *	      order.
     *
     * @return end of iteration through nodes in reverse st-number order 
     */
    reverse_iterator rend()
    {
	return st_ord.rend();
    }
	

    /**
     * @brief Checks whether st-number algorithm can be applied to @p G.
     *
     * Besides from the trivial preconditions that edge @a st and node @a s
     * lie in @p G and @a s is really an endpoint of @a st (which isn't
     * checked), @p G must be undirected and biconnected.
     * @note As for all algorithms in GTL, #check must be called, because it
     * might do some initialization.
     *
     * @param G graph
     *
     * @retval algorithm::GTL_OK iff st-number algorithm may be applied
     *
     * @sa algorithm::check
     */
    int check(graph& G);


    /**
     * @brief Runs st-number algorithm on graph @p G.
     *
     * It is assumed that #check was called previously and returned
     * algorithm::GTL_OK.
     *
     * @param G graph
     *
     * @return algorithm::GTL_OK iff @p G could be correctly st-numbered
     *
     * @sa algorithm::run
     */
    int run(graph& G);


    /**
     * @brief Resets algorithm in order to be applied to the next graph.
     *
     * This will delete most of the information obtained in the last run.
     * 
     * @sa algorithm::reset
     */
    void reset()
    {
	st_ord.erase (st_ord.begin(), st_ord.end());
    }
protected:
    /**
     * @internal
     */    
    edge st;

    /**
     * @internal
     */
    node s;

    /**
     * @internal
     */
    pathfinder* pf;

    /**
     * @internal
     */
    nodes_t st_ord;

    /**
     * @internal
     */
    node_map<int> st_num;
};

__GTL_END_NAMESPACE

#endif // GTL_ST_NUMBER_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
