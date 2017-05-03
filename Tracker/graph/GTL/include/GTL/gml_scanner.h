/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   gml_scanner.h 
//
//==========================================================================
// $Id: gml_scanner.h,v 1.11 2000/03/06 15:16:52 raitner Exp $

#ifndef GTL_GML_SCANNER_H
#define GTL_GML_SCANNER_H

#include <GTL/GTL.h>

#include <cstdio>

__GTL_BEGIN_NAMESPACE

/*
 * start-size of buffers for reading strings. If too small it will be enlarged
 * dynamically
 */

#define INITIAL_SIZE 1024

GTL_EXTERN typedef enum {
    GML_KEY, GML_INT, GML_DOUBLE, GML_STRING, GML_L_BRACKET, 
    GML_R_BRACKET, GML_END, GML_LIST, GML_ERROR
} GML_value; 


/**
 * Possible errors while parsing a GML file. 
 */
GTL_EXTERN typedef enum {
    GML_UNEXPECTED, GML_SYNTAX, GML_PREMATURE_EOF, GML_TOO_MANY_DIGITS,
    GML_OPEN_BRACKET, GML_TOO_MANY_BRACKETS, GML_OK, GML_FILE_NOT_FOUND
} GML_error_value;


/**
 * @short Reason and position of an error in a GML file. 
 *
 * When an error occurs while parsing the structure of a GML file 
 * <code>GML_error</code> is used to return the type and position 
 * of the error detected. Position is specified by
 * <code>line</code> and <code>column</code>, but might be
 * somewhat imprecise. However at least the line number should
 * not differ too much from the real position. 
 *
 * @see graph#load
 */
struct GTL_EXTERN GML_error {
    /**
     * Contains the error description as symbolic constant:
     * <ul> 
     *   <li><code>GML_FILE_NOT_FOUND</code>: A file with that name
     *       doesn't exist.</li>
     *   <li><code>GML_OK</code>: No error :-)</li>
     *   <li><code>GML_TOO_MANY_BRACKETS</code>: A mismatch of
     *       brackets was detected, i.e. there were too many closing 
     *       brackets (<code>]</code>).</li>
     *   <li><code>GML_OPEN_BRACKET</code>: Now, there were too many
     *       opening brackets (<code>[</code>)</li>
     *   <li><code>GML_TOO_MANY_DIGITS</code>: The number of digits a 
     *       integer or floating point value can have is limited to
     *       1024, this should be enough :-)</li>
     *   <li><code>GML_PREMATURE_EOF</code>: An EOF occured, where it 
     *       wasn't expected, e.g. while scanning a string.</li>
     *   <li><code>GML_SYNTAX</code>: The file isn't a valid GML file,
     *       e.g. a mismatch in the key-value pairs.</li>
     *   <li><code>GML_UNEXPECTED</code>: A character occured, where
     *       it makes no sense, e.g. non-numerical characters in
     *       numbers or keys beginning with numbers</li>
     * </ul>
     * 
     */
    GML_error_value err_num;
    
    /**
     * Contains the line, where the error was detected. This will
     * usually be near the line where the error really is
     * located.
     */
    int line;

    /**
     * Contains the column, where the error was detected.
     */
    int column;
};


union GTL_EXTERN GML_tok_val {
    long integer;
    double floating;
    char* str;
    struct GML_error err;
};

/**
 * @internal
 */
struct GTL_EXTERN GML_token { 
    GML_value kind;
    union GML_tok_val value;
};

/*
 * global variables
 */

GTL_EXTERN extern unsigned int GML_line;
GTL_EXTERN extern unsigned int GML_column;

/*
 * if you are interested in the position where an error occured it is a good
 * idea to set GML_line and GML_column back. 
 * This is what GML_init does.
 */
 
GTL_EXTERN void GML_init ();

/*
 * returns the next token in file. If an error occured it will be stored in 
 * GML_token.
 */

GTL_EXTERN struct GML_token GML_scanner (FILE*);

GTL_EXTERN extern const char* GML_table[];

__GTL_END_NAMESPACE

#endif // GTL_GML_SCANNER_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
