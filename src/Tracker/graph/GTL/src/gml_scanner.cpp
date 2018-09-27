/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   gml_scanner.cpp - Scanner for the GML - file format
//
//==========================================================================
// $Id: gml_scanner.cpp,v 1.10 2001/11/07 13:58:10 pick Exp $

#include <GTL/gml_scanner.h>

#include <cstdlib>
#include <cctype>
#include <cassert>
#include <cstring>

__GTL_BEGIN_NAMESPACE

/*
 * ISO8859-1 coding of chars >= 160
 */

const char* GML_table[] = {
    "&nbsp;",     /* 160 */
    "&iexcl;",
    "&cent;",
    "&pound;",
    "&curren;",
    "&yen;",
    "&brvbar;",
    "&sect;",
    "&uml;",
    "&copy;",
    "&ordf;",     /* 170 */
    "&laquo;",
    "&not;",
    "&shy;",
    "&reg;",
    "&macr;",
    "&deg;",
    "&plusmn;",
    "&sup2;",
    "&sup3;",     /* 180 */
    "&acute;",
    "&micro;",
    "&para;",
    "&middot;",
    "&cedil;",
    "&sup1;",
    "&ordm;",
    "&raquo;",
    "&frac14;",
    "&frac12;",   
    "&frac34;",   /* 190 */
    "&iquest;",
    "&Agrave;",
    "&Aacute;",
    "&Acirc;",
    "&Atilde;",
    "&Auml;",
    "&Aring;",
    "&AElig;",
    "&Ccedil;",
    "&Egrave;",   /* 200 */
    "&Eacute;",
    "&Ecirc;",
    "&Euml;",
    "&Igrave;",
    "&Iacute;",
    "&Icirc;",
    "&Iuml;",
    "&ETH;",
    "&Ntilde;",
    "&Ograve;",   /* 210 */
    "&Oacute;",
    "&Ocirc;",
    "&Otilde;",
    "&Ouml;",
    "&times;",
    "&Oslash;",
    "&Ugrave;",
    "&Uacute;",
    "&Ucirc;",
    "&Uuml;",     /* 220 */
    "&Yacute;",
    "&THORN;",
    "&szlig;",
    "&agrave;",
    "&aacute;",
    "&acirc;",
    "&atilde;",
    "&auml;",
    "&aring;",
    "&aelig;",    /* 230 */
    "&ccedil;",
    "&egrave;",
    "&eacute;",
    "&ecirc;",
    "&euml;",
    "&igrave;",
    "&iacute;",
    "&icirc;",
    "&iuml;",
    "&eth;",      /* 240 */
    "&ntilde;",
    "&ograve;",
    "&oacute;",
    "&ocirc;",
    "&otilde;",
    "&ouml;",
    "&divide;",
    "&oslash;",
    "&ugrave;",
    "&uacute;",   /* 250 */
    "&ucirc;",
    "&uuml;",
    "&yacute;",
    "&thorn;",
    "&yuml;"
}; 


unsigned int GML_line;
unsigned int GML_column;


int GML_search_ISO (char* str, int len) {
  
  int i;
  int ret = '&'; 

  //
  // First check the extraordinary ones
  // 

  if (!strncmp (str, "&quot;", len)) {
    return 34;
  } else if (!strncmp (str, "&amp;", len)) {
    return 38;
  } else if (!strncmp (str, "&lt;", len)) {
    return 60;
  } else if (!strncmp (str, "&gt;", len)) {
    return 62;
  }

  for (i = 0; i < 96; i++) {
	if (!strncmp (str, GML_table[i], len)) {
	  ret = i + 160;
	  break;
	}
  }

  return ret;
}


void GML_init () {

    GML_line = 1;
    GML_column = 1;
}



struct GML_token GML_scanner (FILE* source) {
    
    int cur_max_size = INITIAL_SIZE;
    static char buffer[INITIAL_SIZE];
    char* tmp = buffer;
    char* ret = tmp;
    struct GML_token token;
    int is_float = 0;
    int count = 0;
    int next;
	char ISO_buffer[8];
	int ISO_count;

    assert (source != NULL);
    
    /* 
     * eliminate preceeding white spaces
     */
    
    do {
	next = fgetc (source);
	GML_column++;

	if (next == '\n') {
	    GML_line++;
	    GML_column = 1;
	} else if (next == EOF) {
	    token.kind = GML_END;
	    return token;
	}
    } while (isspace (next));
    
    if (isdigit (next) || next == '.' || next == '+' || next == '-') {
	
	/* 
	 * floating point or integer 
	 */
	    
	do {
	    if (count == INITIAL_SIZE - 1) {
		token.value.err.err_num = GML_TOO_MANY_DIGITS;
		token.value.err.line = GML_line;
		token.value.err.column = GML_column + count;
		token.kind = GML_ERROR;
		return token;
	    }

	    if (next == '.' || next == 'E') {
		is_float = 1;
	    }

	    buffer[count] = next;
	    count++;
	    next = fgetc (source);

	} while (!isspace(next) && next != ']');

	if (next == ']') {
	   ungetc (next, source);
	}

	buffer[count] = 0;
	
	if (next == '\n') {
	    GML_line++;
	    GML_column = 1;
	} else {
	    GML_column += count;
	}
	    
	if (is_float) {
	    token.value.floating = atof (tmp);
	    token.kind = GML_DOUBLE;
	} else {
	    token.value.integer = atol (tmp);
	    token.kind = GML_INT;
	}

	return token;
	
    } else if (isalpha (next) || next == '_') {
	
	/*
	 * key
	 */
	        
	do {
	    if (count == cur_max_size - 1) {
		*tmp = 0;
		tmp =  (char*) malloc(2 * cur_max_size * sizeof (char));
		strcpy (tmp, ret);
		
		if (cur_max_size > INITIAL_SIZE) {
		    free (ret);
		}

		ret = tmp;
		tmp += count;
		cur_max_size *= 2;
	    }

	    if (!isalnum (next) && next != '_') {
		token.value.err.err_num = GML_UNEXPECTED;
		token.value.err.line = GML_line;
		token.value.err.column = GML_column + count;
		token.kind = GML_ERROR;
		
		if (cur_max_size > INITIAL_SIZE) {
		    free (ret);
		}

		return token;
	    }

	    *tmp++ = next;
	    count++;
	    next = fgetc (source);
	} while (!isspace (next) && next != EOF);

	if (next == '\n') {
	    GML_line++;
	    GML_column = 1;
	} else {
	    GML_column += count;
	}

	*tmp = 0;
	token.kind = GML_KEY;
	token.value.str = (char*) malloc((count+1) * sizeof (char));
	strcpy (token.value.str, ret);
	
	if (cur_max_size > INITIAL_SIZE) {
	    free (ret);
	}
    
	return token;

    } else {
	/*
	 * comments, brackets and strings
	 */

	switch (next) {
	case '#':
	    do {
		next = fgetc (source);
	    } while (next != '\n' && next != EOF);
		
	    GML_line++;
	    GML_column = 1;
	    return GML_scanner (source);

	case '[':
	    token.kind = GML_L_BRACKET;
	    return token;

	case ']':
	    token.kind = GML_R_BRACKET;
	    return token;
		
	case '"':
	    next = fgetc (source);
	    GML_column++;

	    while (next != '"') {
		
		if (count >= cur_max_size - 8) {
		    *tmp = 0;
		    tmp = (char*) malloc (2 * cur_max_size * sizeof(char));
		    strcpy (tmp, ret);
		    
		    if (cur_max_size > INITIAL_SIZE) {
			free (ret);
		    }

		    ret = tmp;
		    tmp += count;
		    cur_max_size *= 2;
		}
	    
		if (next == '&') {
		  ISO_count = 0;

		  while (next != ';') {
			if (next == '"' || next == EOF) {
			  ungetc (next, source);
			  ISO_count = 0;
			  break;
			}
			
			if (ISO_count < 8) {
			  ISO_buffer[ISO_count] = next;
			  ISO_count++;
			}

			next = fgetc (source);
		  }
			
		  if (ISO_count == 8) {
			ISO_count = 0;
		  }

		  if (ISO_count) {
			ISO_buffer[ISO_count] = ';';
			ISO_count++;
			next = GML_search_ISO (ISO_buffer, ISO_count);
			ISO_count = 0;
		  } else {
			next = '&';
		  }
		} 

		*tmp++ = next;
		count++;
		GML_column++;
			
		next = fgetc (source);

		if (next == EOF) {
		    token.value.err.err_num = GML_PREMATURE_EOF;
		    token.value.err.line = GML_line;
		    token.value.err.column = GML_column + count;
		    token.kind = GML_ERROR;
		    
		    if (cur_max_size > INITIAL_SIZE) {
			free (ret);
		    }
			
		    return token;
		}

		if (next == '\n') {
		    GML_line++;
		    GML_column = 1;
		}
	    }

	    *tmp = 0;
	    token.kind = GML_STRING;
	    token.value.str = (char*) malloc((count+1) * sizeof (char));
	    strcpy (token.value.str, ret);
	    
	    if (cur_max_size > INITIAL_SIZE) {
		free (ret);
	    }

	    return token;
	    
	default:
	    token.value.err.err_num = GML_UNEXPECTED;
	    token.value.err.line = GML_line;
	    token.value.err.column = GML_column;
	    token.kind = GML_ERROR;
	    return token;
	}		
    }
}	    

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
