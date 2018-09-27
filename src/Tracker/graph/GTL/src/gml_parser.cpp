/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   gml_parser.cpp - parser for the GML-file-format specified in:
//                    Michael Himsolt, GML: Graph Modelling Language,
//                    21.01.1997 
//
//==========================================================================
// $Id: gml_parser.cpp,v 1.9 2001/11/07 13:58:10 pick Exp $

#include <GTL/gml_parser.h>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string.h>

__GTL_BEGIN_NAMESPACE

struct GML_pair* GML_parser (FILE* source, struct GML_stat* stat, int open) {
    
    struct GML_token token;
    struct GML_pair* pair;
    struct GML_pair* list;
    struct GML_pair* tmp = NULL;
    struct GML_list_elem* tmp_elem;

    assert (stat);

    pair = (struct GML_pair*) malloc (sizeof (struct GML_pair));
    pair->next = NULL;
    list = pair;

    for (;;) {
	token = GML_scanner (source);
	
	if (token.kind == GML_END) {
	    if (open) {
		stat->err.err_num = GML_OPEN_BRACKET;
		stat->err.line = GML_line;
		stat->err.column = GML_column;
		free (pair);

		if (tmp == NULL) {
		  return NULL;
		} else {
		  tmp->next = NULL;
		  return list;
		}
	    }

	    break;

	} else if (token.kind == GML_R_BRACKET) {
	    if (!open) {
		stat->err.err_num = GML_TOO_MANY_BRACKETS;
		stat->err.line = GML_line;
		stat->err.column = GML_column;
		free (pair);

		if (tmp == NULL) {
		  return NULL;
		} else {
		  tmp->next = NULL;
		  return list;
		}
	    }

	    break;

	} else if (token.kind == GML_ERROR) {
	    stat->err.err_num = token.value.err.err_num;
	    stat->err.line = token.value.err.line;
	    stat->err.column = token.value.err.column;
	    free (pair);
	      
	    if (tmp == NULL) {
		return NULL;
	    } else {
		tmp->next = NULL;
		return list;
	    }

	} else if (token.kind != GML_KEY) {
	    stat->err.err_num = GML_SYNTAX;
	    stat->err.line = GML_line;
	    stat->err.column = GML_column;
	    free (pair);
	   
	    if (token.kind == GML_STRING) {
		free (token.value.str);
	    }

	    if (tmp == NULL) {
		return NULL;
	    } else {
		tmp->next = NULL;
		return list;
	    }
	}
	   
	if (!stat->key_list) {
	    stat->key_list = (struct GML_list_elem*) 
		malloc (sizeof (struct GML_list_elem));
	    stat->key_list->next = NULL;
	    stat->key_list->key = token.value.str;
	    pair->key = token.value.str;
	
	} else {
	    tmp_elem = stat->key_list;
	    
	    while (tmp_elem) {
		if (!strcmp (tmp_elem->key, token.value.str)) {
		    free (token.value.str);
		    pair->key = tmp_elem->key;
		    break;
		}
		
		tmp_elem = tmp_elem->next;
	    }
	
	    if (!tmp_elem) {
		tmp_elem = (struct GML_list_elem*)
		    malloc (sizeof (struct GML_list_elem));
		tmp_elem->next = stat->key_list;
		stat->key_list = tmp_elem;
		tmp_elem->key = token.value.str;
		pair->key = token.value.str;
	    }
	}
	
	token = GML_scanner (source);

	switch (token.kind) {
	case GML_INT:
	    pair->kind = GML_INT;
	    pair->value.integer = token.value.integer;
	    break;

	case GML_DOUBLE:
	    pair->kind = GML_DOUBLE;
	    pair->value.floating = token.value.floating;
	    break;

	case GML_STRING:
	    pair->kind = GML_STRING;
	    pair->value.str = token.value.str;
	    break;

	case GML_L_BRACKET:
	    pair->kind = GML_LIST;
	    pair->value.list = GML_parser (source, stat, 1);
	    
	    if (stat->err.err_num != GML_OK) {
		return list;
	    }

	    break;

	case GML_ERROR:
	    stat->err.err_num = token.value.err.err_num;
	    stat->err.line = token.value.err.line;
	    stat->err.column = token.value.err.column;
	    free (pair);
		
	    if (tmp == NULL) {
		return NULL;
	    } else {
		tmp->next = NULL;
		return list;
	    }

	default:    
	    stat->err.line = GML_line;
	    stat->err.column = GML_column;
	    stat->err.err_num = GML_SYNTAX;
	    free (pair);

	    if (tmp == NULL) {
		return NULL;
	    } else {
		tmp->next = NULL;
		return list;
	    }
	}

	tmp = pair;
	pair = (struct GML_pair*) malloc (sizeof (struct GML_pair));
	tmp->next = pair;
	pair->next = NULL;
    }

    stat->err.err_num = GML_OK;
    free (pair);
    
    if (tmp == NULL) {
	return NULL;
    } else {
	tmp->next = NULL;
	return list;
    }
}
	

void GML_free_list (struct GML_pair* list, struct GML_list_elem* keys) {
    
    struct GML_pair* tmp = list;
    struct GML_list_elem* tmp_key;

    while (keys) {
	free (keys->key);
	tmp_key = keys->next;
	free (keys);
	keys = tmp_key;
    }

    while (list) {
	
	switch (list->kind) {
	case GML_LIST:
	    GML_free_list (list->value.list, NULL);
	    break;

	case GML_STRING:
	    free (list->value.str);
	    break;

	default:
	    break;
	}
	
	tmp = list->next;
	free (list);
	list = tmp;
    }
}
	


void GML_print_list (struct GML_pair* list, int level) {
    
    struct GML_pair* tmp = list;
    int i;

    while (tmp) {
	
	for (i = 0; i < level; i++) {
	    printf ("    ");
	}

	printf ("*KEY* : %s", tmp->key);

	switch (tmp->kind) {
	case GML_INT:
	    printf ("  *VALUE* (long) : %ld \n", tmp->value.integer);
	    break;

	case GML_DOUBLE:
	    printf ("  *VALUE* (double) : %f \n", tmp->value.floating);
	    break;

	case GML_STRING:
	    printf ("  *VALUE* (string) : %s \n", tmp->value.str);
	    break;
	    
	case GML_LIST:
	    printf ("  *VALUE* (list) : \n");
	    GML_print_list (tmp->value.list, level+1);
	    break;

	default:
	    break;
	}
	
	tmp = tmp->next;
    }
}
	
__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
