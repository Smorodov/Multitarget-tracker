/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   gml_parser.h 
//
//==========================================================================
// $Id: gml_parser.h,v 1.7 2000/01/05 16:32:36 raitner Exp $

#ifndef GTL_GML_PARSER_H
#define GTL_GML_PARSER_H

#include <GTL/GTL.h>
#include <GTL/gml_scanner.h>

__GTL_BEGIN_NAMESPACE

/**
 * @internal
 */
union GTL_EXTERN GML_pair_val {
    long integer;
    double floating;
    char* str;
    struct GML_pair* list;
};

/**
 * @internal
 */
struct GTL_EXTERN GML_pair {
    char* key;
    GML_value kind;
    union GML_pair_val value;
    struct GML_pair* next;
};

/**
 * @internal
 */
struct GTL_EXTERN GML_list_elem {
    char* key;
    struct GML_list_elem* next;
};

/**
 * @internal
 */
struct GTL_EXTERN GML_stat {
    struct GML_error err;
    struct GML_list_elem* key_list;
};

/*
 * returns list of KEY - VALUE pairs. Errors and a pointer to a list
 * of key-names are returned in GML_stat. Previous information contained
 * in GML_stat, i.e. the key_list, will be *lost*. 
 */

GTL_EXTERN GML_pair* GML_parser (FILE*, GML_stat*, int);

/*
 * free memory used in a list of GML_pair
 */

GTL_EXTERN void GML_free_list (GML_pair*, GML_list_elem*);


/*
 * debugging 
 */

GTL_EXTERN void GML_print_list (GML_pair*, int);

__GTL_END_NAMESPACE

#endif // GTL_GML_PARSER_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
