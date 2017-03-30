#===========================================================================
#
#   src/Makefile.am
#
#===========================================================================
# $Id: Makefile.am,v 1.37 2003/01/31 08:58:09 chris Exp $

lib_LTLIBRARIES = libGTL.la

INCLUDES = \
	-I$(top_srcdir)/include \
	-I$(top_builddir)/include

libGTL_la_LDFLAGS = \
	-version-info $(GENERIC_LIBRARY_VERSION) 

libGTL_la_SOURCES =        \
        gml_scanner.cpp    \
        gml_parser.cpp     \
        edge.cpp           \
        graph.cpp          \
        node.cpp           \
	dfs.cpp 	   \
	biconnectivity.cpp \
	bfs.cpp            \
	topsort.cpp        \
	st_number.cpp      \
	embedding.cpp	   \
	pq_node.cpp	   \
	pq_tree.cpp	   \
	planarity.cpp 	   \
	maxflow_ff.cpp     \
	maxflow_pp.cpp     \
	maxflow_sap.cpp    \
	debug.cpp          \
	components.cpp     \
	fm_partition.cpp   \
	ratio_cut_partition.cpp \
	min_tree.cpp \
	dijkstra.cpp \
	bellman_ford.cpp	\
	bid_dijkstra.cpp

#---------------------------------------------------------------------------
#   end of file
#---------------------------------------------------------------------------

