// $Id : $

/**
 * @file fheap.h
 *
 * Fibonacci heap
 *
 */
 
#ifndef FHEAP_H
#define FHEAP_H

// rdmp
#ifdef __cplusplus
extern "C" {
#endif


/*
This code comes from <A HREF="http://www.cosc.canterbury.ac.nz/research/reports/HonsReps/1999/hons_9907.pdf">
A Comparison of Data Structures for Dijkstra's 
Single Source Shortest Path Algorithm</A> by Shane Saunders (Department 
of Computer Science, University of Canterbury, NZ).
The code itself is available from Tadao Takaoka's <A HREF="http://www.cosc.canterbury.ac.nz/~tad/alg/heaps/heaps.html">
 Algorithm Repository Home Page</A>
*/

/*** Header File for the Fibonacci Heap Implementation ***/
/*
 *   Shane Saunders
 */



/* Option to allow printing of debugging information.  Use 1 for yes, or 0 for
 * no.
 */
#define FHEAP_DUMP 0

#if FHEAP_DUMP
	#include <stdio.h>
#endif 



/*** Definitions of structure types. ***/

/* The structure type for Fibonacci heap nodes.
 *
 * Nodes have the following pointers:
 * parent      - a pointer to the nodes parent node (if any).
 * child       - a pointer to a child node (typically the highest rank child).
 * left, right - sibling pointers which provide a circular doubly linked list
 *               containing all the parents nodes children.
 *
 * The remaining structure fields are:
 * rank        - the nodes rank, that is, the number of children it has.
 * `key'       - the nodes key.
 * vertex_no   - the number of the graph vertex that the node corresponds to.
 *               Vertex numbering in the graph should be:
 *                    1, 2, 3, ... max_vertex.
 */
typedef struct fheap_node {
    struct fheap_node *parent;
    struct fheap_node *left, *right;
    struct fheap_node *child;
    int rank;
    int marked;
    long key;
    int vertex_no;
} fheap_node_t;

/* The structure type for a Fibonacci heap.
 *
 * trees - An array of pointers to trees at root level in the heap.  Entry i
 *         in the array points to the root node of a tree that has nodes of
 *         dimension i on the main trunk.
 * nodes - An array of pointers to nodes in the heap.  Nodes are indexed
 *         according to their vertex number.  This array can then be used to
 *         look up the node for corresponding to a vertex number, and is
 *         useful when freeing space taken up by the heap.
 * max_nodes - The maximum number of nodes allowed in the heap.
 * max_trees - The maximum number of trees allowed in the heap (calculated from
 *             max_nodes).
 * n     - The current number of nodes in the heap.
 * value - The binary value represented by trees in the heap.
 *         By maintaining this it is easy to keep track of the maximum rank
 *         tree in the heap.
 * key_comps - can be used for experimental purposes when counting the number
 *             of key comparisons.
 */
typedef struct fheap {
    fheap_node_t **trees;
    fheap_node_t **nodes;
    int max_nodes, max_trees, n, value;
    long key_comps;
} fheap_t;



/*** Function prototypes. ***/

/* Fibonacci heap functions. */
/* Note that fheap_t pointers are used so that function definitions are compatible
 * with those of other heaps.  This allows any type heap to be given as an
 * argument to a particular algorithm.  It is up to the user to ensure the
 * correct heap type is passed to the given functions.
 */

/* fh_alloc() - creates and and returns a pointer to a F-heap which can contain
 * up to max_nodes nodes.
 */
fheap_t *fh_alloc(int max_nodes);

/* fh_free() - destroys the heap pointed to by h, freeing up any space that was
 * used by it.
 */
void fh_free(fheap_t *h);

/* fh_insert() - creates and inserts new a node representing vertex_no with key
 * k into the heap h.
 */
void fh_insert(fheap_t *h, int vertex_no, long k);

/* fh_delete_min() - deletes the minimum node from the heap pointed to by h and
 * returns its vertex number.
 */
int fh_delete_min(fheap_t *h);

/* fh_decrease_key() - decreases the key used for vertex, vertex_no, to
 * new_value.  No check is made to ensure that new_value is in-fact less than
 * the current value so it is up to the user of this function to ensure that
 * it is.
 */
void fh_decrease_key(fheap_t *h, int vertex_no, long new_value);

/* Debugging functions. */
#if FHEAP_DUMP
void fh_dump(fheap_t *h);
#endif


// rdmp
#ifdef __cplusplus
}
#endif


#endif
